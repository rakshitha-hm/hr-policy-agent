import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
DB_PATH = ROOT / "data" / "hr.db"

QUERIES = {
    "Employees per country and work mode": """
        SELECT country, work_mode, COUNT(*) AS n
        FROM employees GROUP BY country, work_mode ORDER BY country, work_mode""",
    "Remaining annual leave (5 employees)": """
        SELECT e.employee_id, e.first_name, e.country,
               b.entitled_days + b.carried_over_days - b.used_days AS remaining_days
        FROM employees e JOIN leave_balances b ON b.employee_id = e.employee_id
        WHERE b.year = 2026 ORDER BY e.employee_id LIMIT 5""",
    "Eligible for home office allowance but not claimed yet": """
        SELECT COUNT(*) AS n
        FROM employees e
        WHERE e.work_mode IN ('hybrid', 'remote')
          AND julianday('2026-10-01') - julianday(e.hire_date) >= 91
          AND NOT EXISTS (SELECT 1 FROM expense_claims c
                          WHERE c.employee_id = e.employee_id AND c.category = 'home_office')""",
    "Approved training spend per country": """
        SELECT e.country, c.currency, ROUND(SUM(c.amount), 2) AS total
        FROM expense_claims c JOIN employees e ON e.employee_id = c.employee_id
        WHERE c.category = 'training' AND c.status = 'approved'
        GROUP BY e.country, c.currency""",
}


def policy_stats():
    print("POLICY DOCUMENTS")
    total = 0
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        words = len(text.split())
        sections = len(re.findall(r"^## ", text, flags=re.MULTILINE))
        tables = len(re.findall(r"^\| ---", text, flags=re.MULTILINE))
        total += words
        print(f"  {path.name:35s} {title:45s} {words:5d} words  {sections:2d} sections  {tables} tables")
    print(f"  Total: {total} words (about {round(total * 1.3)} tokens)\n")


def db_stats():
    print("DATABASE")
    with sqlite3.connect(DB_PATH) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            print(f"  {table:15s} {n:4d} rows  columns: {', '.join(cols)}")
        for name, sql in QUERIES.items():
            cur = conn.execute(sql)
            header = [d[0] for d in cur.description]
            print(f"\n  {name}")
            print("   ", header)
            for row in cur.fetchall():
                print("   ", row)


if __name__ == "__main__":
    policy_stats()
    db_stats()