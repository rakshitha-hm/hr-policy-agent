import pytest
import requests

import app as web


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    def json(self):
        return {"request_id": "abc", "answer": "16 weeks [1].", "route": "policy", "citations": [], "latency_ms": 5}


@pytest.fixture
def client():
    web.app.config["TESTING"] = True
    return web.app.test_client()


def test_chat_requires_login(client):
    assert client.post("/chat", json={"question": "hi"}).status_code == 401


def test_identity_comes_from_the_session_not_the_browser(client, monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(headers)
        return FakeResponse()

    monkeypatch.setattr(web.requests, "post", fake_post)
    client.post("/login", data={"employee_id": "1005"})
    r = client.post("/chat", json={"question": "hi", "employee_id": 1001})    # browser tries to be 1001
    assert r.status_code == 200 and sent["X-Employee-Id"] == "1005"


def test_unknown_user_cannot_log_in(client):
    client.post("/login", data={"employee_id": "9999"})
    assert client.post("/chat", json={"question": "hi"}).status_code == 401


def test_api_down_returns_502(client, monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError

    monkeypatch.setattr(web.requests, "post", down)
    client.post("/login", data={"employee_id": "1005"})
    assert client.post("/chat", json={"question": "hi"}).status_code == 502