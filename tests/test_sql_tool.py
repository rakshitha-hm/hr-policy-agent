import pytest

import sql_tool
from sql_tool import SQLTool, UnsafeSQL, run_query, validate_sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "DELETE FROM employees",
    "SELECT * FROM employees; DROP TABLE employees",
    "SELECT * FROM employees -- comment",
    "PRAGMA table_info(employees)",
    "SELECT name FROM sqlite_master",
    "WITH x AS (SELECT 1) DELETE FROM employees",
    "SELECT used_days FROM leave_balances WHERE employee_id = [employee_id]",
])
def test_validator_blocks(sql):
    with pytest.raises(UnsafeSQL):
        validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM employees",
    "WITH t AS (SELECT employee_id, COUNT(*) AS n FROM expense_claims GROUP BY employee_id) SELECT MAX(n) FROM t",
    "SELECT COUNT(*) FROM employees WHERE last_name = 'Updated'",
])
def test_validator_allows(sql):
    assert validate_sql(sql + ";") == sql             # trailing semicolon is stripped


def test_hr_mode_sees_everyone():
    _, rows = run_query("SELECT COUNT(*) FROM leave_balances")
    assert rows == [(200,)]


def test_employee_mode_sees_only_own_rows():
    _, rows = run_query("SELECT COUNT(*) FROM leave_balances", employee_id=1005)
    assert rows == [(1,)]


def fake_llm(*answers):
    """Replace the LLM with a list of SQL answers, returned one per call."""
    answers = iter(answers)
    return lambda question, feedback=None: next(answers)


def test_retry_fixes_a_placeholder(monkeypatch):
    monkeypatch.setattr(sql_tool, "generate_sql", fake_llm(
        "SELECT used_days FROM leave_balances WHERE employee_id = [employee_id]",
        "SELECT used_days FROM leave_balances WHERE employee_id = 1005"))
    r = SQLTool().run("How many days have I used?", employee_id=1005)
    assert r["attempts"] == 2 and r["error"] is None and r["rows"] == [(11.0,)]


def test_unsafe_sql_is_not_retried(monkeypatch):
    monkeypatch.setattr(sql_tool, "generate_sql", fake_llm("DELETE FROM employees", "SELECT 1"))
    r = SQLTool().run("Delete everyone")
    assert r["blocked"] and r["attempts"] == 1


def test_other_employee_is_access_denied(monkeypatch):
    monkeypatch.setattr(sql_tool, "generate_sql", fake_llm(
        "SELECT used_days FROM leave_balances WHERE employee_id = 1010"))
    r = SQLTool().run("How many days has 1010 used?", employee_id=1005)
    assert r["access_denied"] and r["rows"] == []


def test_model_refusal(monkeypatch):
    monkeypatch.setattr(sql_tool, "generate_sql", fake_llm(None))
    r = SQLTool().run("What is the weather?")
    assert r["sql"] is None and "cannot be answered" in r["error"]