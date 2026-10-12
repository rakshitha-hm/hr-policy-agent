import pytest
from fastapi.testclient import TestClient

import api

POLICY_STATE = {"question": "How long is maternity leave?", "answer": "16 weeks [1].", "route": "policy",
                "policy": {"citations": [{"source": "03_parental_and_family_leave.md", "section": "3"}]}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "build_agent", lambda: object())     # no models loaded in tests
    monkeypatch.setattr(api, "warm_up", lambda: True)
    monkeypatch.setattr(api, "ask", lambda agent, question, employee_id: dict(POLICY_STATE))
    with TestClient(api.app) as c:
        yield c


def ask(client, question="How long is maternity leave?", employee="1005"):
    headers = {"X-Employee-Id": employee} if employee else {}
    return client.post("/ask", json={"question": question}, headers=headers)


def test_ask_returns_answer_and_citations(client):
    r = ask(client)
    body = r.json()
    assert r.status_code == 200
    assert body["route"] == "policy" and body["citations"][0]["source"].startswith("03_")
    assert set(body) == {"request_id", "answer", "route", "citations", "latency_ms"}


@pytest.mark.parametrize("question, employee, status", [
    ("", "1005", 422),                 # empty question
    ("x" * 501, "1005", 422),          # too long
    ("hi", None, 422),                 # missing header
    ("hi", "9999", 401),               # unknown employee
])
def test_bad_requests(client, question, employee, status):
    assert ask(client, question, employee).status_code == status


def test_llm_down_returns_503(client, monkeypatch):
    def broken(*args):
        raise ConnectionError("Ollama is not running")
    monkeypatch.setattr(api, "ask", broken)
    assert ask(client).status_code == 503


def test_health_reports_degraded_without_llm(client, monkeypatch):
    def broken():
        raise ConnectionError
    monkeypatch.setattr(api.ollama, "list", broken)
    assert client.get("/health").json()["status"] == "degraded"