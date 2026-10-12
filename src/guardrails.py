import re
import sqlite3

MAX_QUESTION_CHARS = 500

# ---------- 1. Input guardrail: prompt injection and abuse ----------
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\b.{0,30}\b(rules|instructions|prompt|guidelines)\b",
    r"\b(reveal|show|print|repeat|tell me)\b.{0,20}\b(system prompt|your prompt|your instructions)\b",
    r"\byou are now\b|\bpretend (to be|you are)\b|\bact as (an? )?(admin|administrator|hr manager|unrestricted)\b",
    r"\b(jailbreak|developer mode)\b",
    r"\b(drop|delete|truncate|wipe)\b.{0,40}\b(table|database|records|rows|employees)\b",
]
BLOCKED_INPUT = "I can't help with that request. Please ask a question about Northwind HR policies or your own HR data."


def check_input(question):
    """Return a reason string if the question must be blocked, else None."""
    if len(question) > MAX_QUESTION_CHARS:
        return f"question longer than {MAX_QUESTION_CHARS} characters"
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            return f"matched injection pattern: {pattern}"
    return None


# ---------- 2. PII redaction (input and output) ----------
PII_PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z]{4}\d{10}\b",
    "CARD": r"\b(?:\d[ -]?){13,19}\b",
    "PHONE": r"\+\d{1,3}[\s-]?\d[\d\s-]{7,}\d",
}


PUBLIC_EMAILS = {"dpo@northwind-analytics.example", "people@northwind-analytics.example"}  # team mailboxes


def redact_pii(text):
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, lambda m: m.group(0) if m.group(0).lower() in PUBLIC_EMAILS else f"[{label}]", text)
    return text


# ---------- 3. Data access guardrail: an employee sees only their own records ----------
PERSONAL_TABLES = ("leave_balances", "leave_requests", "expense_claims")
SENSITIVE_COLUMNS = ("fte", "hire_date", "level", "email")
ACCESS_DENIED = "You can only view your own HR records. For other employees or company-wide figures, please contact HR."


def check_access(sql, employee_id):
    """Layer A (static check): personal data must be filtered on the user's own employee_id."""
    s = sql.lower()
    personal = any(re.search(rf"\b{t}\b", s) for t in PERSONAL_TABLES + SENSITIVE_COLUMNS)
    if not personal:
        return None                                    # directory data (names, offices, departments): allowed
    other_ids = {int(n) for n in re.findall(r"employee_id\s*=\s*(\d+)", s)} - {employee_id}
    if other_ids:
        return f"query reads other employees: {sorted(other_ids)}"
    if not re.search(rf"\b{employee_id}\b", s):
        return "company-wide query on personal data"
    return None


def restricted_connection(db_path, employee_id):
    """Layer B (enforced by the database): TEMP views with the same names hide other employees' rows.
    SQLite looks in the temp schema first, so 'leave_balances' now means 'my leave_balances'."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    eid = int(employee_id)
    for table in PERSONAL_TABLES:
        conn.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM main.{table} WHERE employee_id = {eid}")
    masked = ", ".join(f"CASE WHEN employee_id = {eid} THEN {c} END AS {c}" for c in SENSITIVE_COLUMNS)
    conn.execute(f"CREATE TEMP VIEW employees AS SELECT employee_id, first_name, last_name, department, "
                 f"job_title, country, office, work_mode, manager_id, {masked} FROM main.employees")
    return conn