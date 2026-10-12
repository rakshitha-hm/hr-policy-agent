import os

import requests
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

API_URL = os.environ.get("API_URL", "http://localhost:8000")     # the FastAPI service from Day 9
TIMEOUT_S = 120

# Demo "login": in production this would be company SSO. The browser never sends the employee id itself.
DEMO_USERS = {
    1005: "Ayden (Amsterdam, hybrid)",
    1001: "Felicitas (Berlin, hybrid)",
    1010: "Karl Heinz (Berlin, part-time)",
    1020: "Ayushman (Bengaluru, office)",
    1033: "Linde (Amsterdam, hybrid)",
    1042: "Jari (Amsterdam, part-time)",
}
EXAMPLES = [
    "How many leave days do I have left, and can I carry them over?",
    "How long is maternity leave?",
    "Who is my manager?",
    "How much of my training budget have I used this year?",
]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")   # signs the session cookie


@app.get("/")
def index():
    employee_id = session.get("employee_id")
    return render_template("index.html", users=DEMO_USERS, employee_id=employee_id,
                           user_name=DEMO_USERS.get(employee_id), examples=EXAMPLES)


@app.post("/login")
def login():
    employee_id = request.form.get("employee_id", type=int)
    if employee_id in DEMO_USERS:
        session["employee_id"] = employee_id                     # identity lives in the signed session
    return redirect(url_for("index"))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.post("/chat")
def chat():
    """Browser -> Flask -> FastAPI. Flask adds the identity from the session, never from the browser."""
    employee_id = session.get("employee_id")
    if employee_id is None:
        return jsonify(error="Please log in first."), 401
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Please type a question."), 400
    try:
        resp = requests.post(f"{API_URL}/ask", json={"question": question},
                             headers={"X-Employee-Id": str(employee_id)}, timeout=TIMEOUT_S)
    except requests.RequestException:
        return jsonify(error="The HR assistant service is not reachable."), 502
    if resp.status_code != 200:
        detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else None
        return jsonify(error=f"The assistant could not answer ({resp.status_code}). {detail or ''}".strip()), 502
    return jsonify(resp.json())


@app.get("/health")
def health():
    try:
        api = requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        api = {"status": "unreachable"}
    return jsonify(web="ok", api=api)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)