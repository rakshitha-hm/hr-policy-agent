import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import ollama

from guardrails import check_access, restricted_connection

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "hr.db"

LLM_MODEL = "qwen2.5:7b"
MAX_ROWS = 50          # never return more rows than this
MAX_RETRIES = 1        # one self-correction attempt after an SQL error
TIMEOUT_S = 2.0        # stop queries that run too long
NUM_CTX = 8192         # same as rag.py: a different num_ctx makes Ollama reload the model

ALLOWED_TABLES = {"employees", "leave_balances", "leave_requests", "expense_claims"}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex|grant)\b",
    re.IGNORECASE)

SCHEMA_DOC = """SQLite database of Northwind Analytics (HR data). Today is 2026-10-01.

employees(employee_id INTEGER PK, first_name, last_name, email, department, job_title,
          level, country, office, work_mode, fte REAL, hire_date TEXT 'YYYY-MM-DD', manager_id -> employees)
  country: 'NL' | 'DE' | 'IN'   office: 'Amsterdam' | 'Berlin' | 'Bengaluru'
  department: 'Engineering' | 'Data Science' | 'Sales' | 'Marketing' | 'Finance' | 'People'
  level: 'L1'..'L6'   work_mode: 'office' | 'hybrid' | 'remote'   fte: 1.0 = full-time

leave_balances(employee_id -> employees, year INTEGER, entitled_days REAL, carried_over_days REAL, used_days REAL)
  annual leave only (no leave_type column); remaining annual leave = entitled_days + carried_over_days - used_days
  carried_over_days = days carried from the previous year INTO this year, stored in this year's row

leave_requests(request_id PK, employee_id -> employees, leave_type, start_date TEXT, end_date TEXT, days REAL, status)
  leave_type: 'annual' | 'sick' | 'care' | 'study'   status: 'approved' | 'pending' | 'rejected'

expense_claims(claim_id PK, employee_id -> employees, category, amount REAL, currency, submitted_date TEXT, status)
  category: 'travel' | 'meals' | 'training' | 'home_office'   currency: 'EUR' | 'INR'
  status: 'approved' | 'pending' | 'rejected'   never add EUR and INR amounts together"""

EXAMPLES = """Question: How many employees work in Berlin?
{"sql": "SELECT COUNT(*) AS n FROM employees WHERE office = 'Berlin'"}

Question: How many annual leave days does employee 1001 have left in 2026?
{"sql": "SELECT entitled_days + carried_over_days - used_days AS remaining_days FROM leave_balances WHERE employee_id = 1001 AND year = 2026"}"""

SYSTEM_PROMPT = f"""You translate HR questions into one SQLite SELECT query.
{SCHEMA_DOC}

Rules:
- Write exactly one read-only SELECT statement. Never modify data.
- Use only the tables and columns above. Return only the columns needed to answer.
- Only filter on fte if the question says part-time or full-time. "Fully remote" means work_mode = 'remote'.
- If the question cannot be answered with this data, or asks to change data, return {{"sql": null}}.
- Reply with JSON only: {{"sql": "..."}}

Examples:
{EXAMPLES}"""


class UnsafeSQL(Exception):
    def __init__(self, reason, retryable=False):
        super().__init__(reason)
        self.retryable = retryable   # a wrong table name is a mistake, not an attack: allow one retry


def validate_sql(sql):
    """Static safety checks before anything touches the database. Returns the cleaned SQL."""
    if not sql or not sql.strip():
        raise UnsafeSQL("empty query")
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise UnsafeSQL("more than one statement")
    if "--" in sql or "/*" in sql:
        raise UnsafeSQL("comments are not allowed")
    if re.search(r"\[[^\]]*\]", sql):            # e.g. WHERE employee_id = [employee_id]
        raise UnsafeSQL("placeholder in square brackets, write the real value", retryable=True)
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise UnsafeSQL("only SELECT queries are allowed")
    match = FORBIDDEN.search(sql)
    if match:
        raise UnsafeSQL(f"forbidden keyword: {match.group(0).upper()}")
    tables = {t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)}
    cte_names = {n.lower() for n in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql, re.IGNORECASE)}
    unknown = tables - ALLOWED_TABLES - cte_names
    if unknown:
        raise UnsafeSQL(f"table not allowed: {', '.join(sorted(unknown))}",
                        retryable=not any(t.startswith("sqlite_") for t in unknown))
    return sql


def run_query(sql, employee_id=None):
    """Execute on a READ-ONLY connection with a row limit and a timeout.
    With employee_id, the connection only shows that employee's personal rows (guardrails.py)."""
    if employee_id is None:                       # HR analyst: full read-only access
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    else:                                         # employee: own rows only
        conn = restricted_connection(DB_PATH, employee_id)
    deadline = time.perf_counter() + TIMEOUT_S
    conn.set_progress_handler(lambda: time.perf_counter() > deadline, 10_000)  # non-zero = abort
    try:
        cur = conn.execute(f"SELECT * FROM ({sql}) LIMIT {MAX_ROWS}")
        columns = [d[0] for d in cur.description]
        return columns, cur.fetchall()
    finally:
        conn.close()


def generate_sql(question, feedback=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"}]
    if feedback:
        messages += [{"role": "assistant", "content": json.dumps({"sql": feedback["sql"]})},
                     {"role": "user", "content": f"That query failed with: {feedback['error']}. "
                                                 "Fix it and reply with JSON only."}]
    resp = ollama.chat(model=LLM_MODEL, messages=messages, format="json", options={"temperature": 0, "num_ctx": NUM_CTX})
    try:
        return json.loads(resp["message"]["content"]).get("sql")
    except json.JSONDecodeError:
        return None


class SQLTool:
    """Question -> SQL (LLM) -> safety checks -> read-only execution, with one self-correction retry."""

    def run(self, question, employee_id=None):
        start = time.perf_counter()
        result = {"question": question, "sql": None, "columns": [], "rows": [],
                  "error": None, "blocked": False, "access_denied": False, "attempts": 0}
        feedback = None
        for _ in range(1 + MAX_RETRIES):
            result["attempts"] += 1
            sql = generate_sql(question, feedback)
            if sql is None:
                result["error"] = "The question cannot be answered from the HR database."
                break
            try:
                sql = validate_sql(sql)
            except UnsafeSQL as e:
                result.update(sql=sql, error=f"Blocked: {e}", blocked=not e.retryable)
                if not e.retryable:
                    break                               # never retry an unsafe query
                feedback = {"sql": sql, "error": str(e)}
                continue
            if employee_id is not None:
                reason = check_access(sql, employee_id)
                if reason:
                    result.update(sql=sql, error=f"Access denied: {reason}", blocked=True, access_denied=True)
                    break                               # never retry around an access rule
            try:
                result["columns"], result["rows"] = run_query(sql, employee_id)
                result.update(sql=sql, error=None)
                break
            except sqlite3.Error as e:
                result.update(sql=sql, error=str(e))
                feedback = {"sql": sql, "error": str(e)}  # give the error back to the LLM once
        result["latency_s"] = time.perf_counter() - start
        return result


def main():
    question = " ".join(sys.argv[1:]) or "How many employees work in Amsterdam?"
    r = SQLTool().run(question)
    print(f"Q: {question}\nSQL: {r['sql']}\n")
    if r["error"]:
        print(f"Error: {r['error']}")
    else:
        print(r["columns"])
        for row in r["rows"][:10]:
            print(row)
        if len(r["rows"]) > 10:
            print(f"... {len(r['rows'])} rows")
    print(f"\n{r['attempts']} attempt(s), {r['latency_s']:.1f} s")


if __name__ == "__main__":
    main()