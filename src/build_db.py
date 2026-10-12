import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "hr.db"
SEED = 42
N_EMPLOYEES = 200
YEAR = 2026
TODAY = date(2026, 10, 1)

COUNTRIES = {
    "NL": {"office": "Amsterdam", "locale": "nl_NL", "base_leave": 25, "currency": "EUR", "weight": 0.40},
    "DE": {"office": "Berlin", "locale": "de_DE", "base_leave": 28, "currency": "EUR", "weight": 0.25},
    "IN": {"office": "Bengaluru", "locale": "en_IN", "base_leave": 21, "currency": "INR", "weight": 0.35},
}
DEPARTMENTS = {
    "Engineering": ["Software Engineer", "Data Engineer", "ML Engineer", "QA Engineer"],
    "Data Science": ["Data Scientist", "Data Analyst"],
    "Sales": ["Account Executive", "Sales Manager"],
    "Marketing": ["Marketing Specialist", "Content Manager"],
    "Finance": ["Financial Analyst", "Accountant"],
    "People": ["HR Business Partner", "Recruiter"],
}
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]
LEVEL_WEIGHTS = [0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
WORK_MODES = ["office", "hybrid", "remote"]
WORK_MODE_WEIGHTS = [0.2, 0.6, 0.2]

SCHEMA = """
CREATE TABLE employees (
    employee_id     INTEGER PRIMARY KEY,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    department      TEXT NOT NULL,
    job_title       TEXT NOT NULL,
    level           TEXT NOT NULL,
    country         TEXT NOT NULL,
    office          TEXT NOT NULL,
    work_mode       TEXT NOT NULL,
    fte             REAL NOT NULL,
    hire_date       TEXT NOT NULL,
    manager_id      INTEGER REFERENCES employees(employee_id)
);
CREATE TABLE leave_balances (
    employee_id        INTEGER NOT NULL REFERENCES employees(employee_id),
    year               INTEGER NOT NULL,
    entitled_days      REAL NOT NULL,
    carried_over_days  REAL NOT NULL,
    used_days          REAL NOT NULL,
    PRIMARY KEY (employee_id, year)
);
CREATE TABLE leave_requests (
    request_id   INTEGER PRIMARY KEY,
    employee_id  INTEGER NOT NULL REFERENCES employees(employee_id),
    leave_type   TEXT NOT NULL,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    days         REAL NOT NULL,
    status       TEXT NOT NULL
);
CREATE TABLE expense_claims (
    claim_id        INTEGER PRIMARY KEY,
    employee_id     INTEGER NOT NULL REFERENCES employees(employee_id),
    category        TEXT NOT NULL,
    amount          REAL NOT NULL,
    currency        TEXT NOT NULL,
    submitted_date  TEXT NOT NULL,
    status          TEXT NOT NULL
);
"""


def full_years(start, end):
    return end.year - start.year - ((end.month, end.day) < (start.month, start.day))


def round_up_half(x):
    return math.ceil(x * 2) / 2


def annual_entitlement(country, hire_date, fte):
    """Policy 01: base by country + 1 day per 5 years (max 5), pro rata FTE and join month."""
    years = full_years(hire_date, date(YEAR, 1, 1)) if hire_date.year < YEAR else 0
    days = COUNTRIES[country]["base_leave"] + min(5, years // 5)
    if hire_date.year == YEAR:
        full_months = 12 - hire_date.month + (1 if hire_date.day == 1 else 0)
        days = days * full_months / 12
    return round_up_half(days * fte)


def working_days(start, n_days):
    """Return the end date after n working days (Mon-Fri) starting at start."""
    current, counted = start, 0
    while True:
        if current.weekday() < 5:
            counted += 1
            if counted == n_days:
                return current
        current += timedelta(days=1)


def random_workday(rng, start, end):
    while True:
        d = start + timedelta(days=rng.randint(0, (end - start).days))
        if d.weekday() < 5:
            return d


def make_employees(rng):
    fakers = {c: Faker(info["locale"]) for c, info in COUNTRIES.items()}
    for f in fakers.values():
        f.seed_instance(SEED)
    employees, emails = [], set()
    for emp_id in range(1001, 1001 + N_EMPLOYEES):
        country = rng.choices(list(COUNTRIES), weights=[c["weight"] for c in COUNTRIES.values()])[0]
        fake = fakers[country]
        first, last = fake.first_name(), fake.last_name()
        email = f"{first}.{last}".lower().replace(" ", "") + "@northwind-analytics.example"
        while email in emails:
            email = email.replace("@", f"{rng.randint(1, 9)}@")
        emails.add(email)
        department = rng.choice(list(DEPARTMENTS))
        hire = date(2012, 1, 1) + timedelta(days=rng.randint(0, (date(2026, 9, 1) - date(2012, 1, 1)).days))
        employees.append({
            "employee_id": emp_id, "first_name": first, "last_name": last, "email": email,
            "department": department, "job_title": rng.choice(DEPARTMENTS[department]),
            "level": rng.choices(LEVELS, weights=LEVEL_WEIGHTS)[0],
            "country": country, "office": COUNTRIES[country]["office"],
            "work_mode": rng.choices(WORK_MODES, weights=WORK_MODE_WEIGHTS)[0],
            "fte": rng.choices([1.0, 0.8, 0.6], weights=[0.85, 0.1, 0.05])[0],
            "hire_date": hire, "manager_id": None,
        })
    # Managers: a more senior L4+ employee in the same department
    for e in employees:
        candidates = [m for m in employees if m["department"] == e["department"]
                      and m["level"] in ("L4", "L5", "L6") and m["employee_id"] != e["employee_id"]
                      and LEVELS.index(m["level"]) > LEVELS.index(e["level"])]
        if candidates:
            e["manager_id"] = rng.choice(candidates)["employee_id"]
    return employees


def make_leave(rng, employees):
    balances, requests, request_id = [], [], 1
    for e in employees:
        entitled = annual_entitlement(e["country"], e["hire_date"], e["fte"])
        carried = rng.choice([0, 0, 1, 2, 3, 4, 5]) if e["hire_date"].year < YEAR else 0
        year_start = max(date(YEAR, 1, 1), e["hire_date"])
        used = 0.0
        for _ in range(rng.randint(0, 6)):
            leave_type = rng.choices(["annual", "sick", "care", "study"], weights=[0.65, 0.25, 0.05, 0.05])[0]
            n = rng.randint(1, 10) if leave_type == "annual" else rng.randint(1, 3)
            start = random_workday(rng, year_start, date(YEAR, 12, 15))
            end = working_days(start, n)
            if leave_type != "annual":
                status = "approved"
            elif start > TODAY:
                status = "pending"
            else:
                status = rng.choices(["approved", "rejected"], weights=[0.92, 0.08])[0]
            if leave_type == "annual" and status == "approved":
                if used + n > entitled + carried:
                    continue
                used += n
            requests.append({"request_id": request_id, "employee_id": e["employee_id"],
                             "leave_type": leave_type, "start_date": start, "end_date": end,
                             "days": float(n), "status": status})
            request_id += 1
        balances.append({"employee_id": e["employee_id"], "year": YEAR, "entitled_days": entitled,
                         "carried_over_days": float(carried), "used_days": used})
    return balances, requests


def make_expenses(rng, employees):
    claims, claim_id = [], 1
    amounts = {"travel": (80, 900), "meals": (15, 60), "training": (100, 1500)}
    for e in employees:
        eur = COUNTRIES[e["country"]]["currency"] == "EUR"
        rate = 1 if eur else 80  # rough EUR -> INR factor for realistic INR amounts
        start = max(date(YEAR, 1, 1), e["hire_date"])
        for _ in range(rng.randint(0, 5)):
            category = rng.choice(list(amounts))
            low, high = amounts[category]
            claims.append({"claim_id": claim_id, "employee_id": e["employee_id"], "category": category,
                           "amount": round(rng.uniform(low, high) * rate, 2),
                           "currency": COUNTRIES[e["country"]]["currency"],
                           "submitted_date": random_workday(rng, start, TODAY) if start <= TODAY else TODAY,
                           "status": rng.choices(["approved", "pending", "rejected"], weights=[0.8, 0.15, 0.05])[0]})
            claim_id += 1
        # Home office allowance (policy 04): hybrid/remote, after 3 months; about half have claimed it
        eligible = e["work_mode"] in ("hybrid", "remote") and (TODAY - e["hire_date"]).days >= 91
        if eligible and rng.random() < 0.5:
            claims.append({"claim_id": claim_id, "employee_id": e["employee_id"], "category": "home_office",
                           "amount": 500.0 if eur else 25000.0, "currency": COUNTRIES[e["country"]]["currency"],
                           "submitted_date": random_workday(rng, max(e["hire_date"] + timedelta(days=91),
                                                                     date(2020, 1, 1)), TODAY),
                           "status": "approved"})
            claim_id += 1
    return claims


def insert(conn, table, rows):
    cols = list(rows[0])
    placeholders = ", ".join("?" for _ in cols)
    values = [[r[c].isoformat() if isinstance(r[c], date) else r[c] for c in cols] for r in rows]
    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", values)


def main():
    rng = random.Random(SEED)
    employees = make_employees(rng)
    balances, requests = make_leave(rng, employees)
    claims = make_expenses(rng, employees)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        insert(conn, "employees", employees)
        insert(conn, "leave_balances", balances)
        insert(conn, "leave_requests", requests)
        insert(conn, "expense_claims", claims)

    print(f"Created {DB_PATH}")
    for table, rows in [("employees", employees), ("leave_balances", balances),
                        ("leave_requests", requests), ("expense_claims", claims)]:
        print(f"  {table:15s} {len(rows):5d} rows")


if __name__ == "__main__":
    main()