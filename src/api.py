import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent import ask, build_agent
from guardrails import MAX_QUESTION_CHARS
from rag import LLM_MODEL, NUM_CTX
from sql_tool import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hr-agent")
logging.getLogger("httpx").setLevel(logging.WARNING)     # hide the library's per-request logs
STATE = {}                                     # objects created once at startup, shared by all requests                                   # objects created once at startup, shared by all requests


def warm_up():
    """Load the LLM into GPU memory now, with the same num_ctx as real calls, so request 1 isn't slow."""
    try:
        ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "hi"}],
                    options={"num_ctx": NUM_CTX, "num_predict": 1})
        return True
    except Exception as e:                     # Ollama not running: start anyway, /health reports it
        log.warning("LLM warm-up failed: %s", e)
        return False


@asynccontextmanager
async def lifespan(app):
    start = time.perf_counter()
    STATE["agent"] = build_agent()             # embedder + Chroma + SQL tool, loaded once
    STATE["llm_ready"] = warm_up()
    log.info("startup done in %.1f s (llm_ready=%s)", time.perf_counter() - start, STATE["llm_ready"])
    yield
    STATE.clear()


app = FastAPI(title="Northwind HR Policy Agent",
              description="Answers HR questions from the policy handbook (RAG) and the HR database (text-to-SQL).",
              version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS,
                          examples=["How many leave days do I have left, and can I carry them over?"])


class Citation(BaseModel):
    source: str
    section: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    route: str
    citations: list[Citation]
    latency_ms: int


def employee_exists(employee_id):
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return conn.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)).fetchone() is not None


@app.get("/health")
def health():
    try:
        ollama.list()
        llm = "ok"
    except Exception:
        llm = "unreachable"
    return {"status": "ok" if llm == "ok" else "degraded", "model": LLM_MODEL,
            "llm": llm, "agent_loaded": "agent" in STATE}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest,
                 x_employee_id: int = Header(description="Logged-in employee. In production this comes from SSO.")):
    # 'def' (not 'async def'): the agent is blocking code, so FastAPI runs it in a thread pool
    if not employee_exists(x_employee_id):
        raise HTTPException(status_code=401, detail="Unknown employee")
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        s = ask(STATE["agent"], body.question, x_employee_id)
    except (ConnectionError, ollama.ResponseError) as e:
        log.error("request=%s llm error: %s", request_id, e)
        raise HTTPException(status_code=503, detail="The language model is not available. Please try again later.")
    latency_ms = int((time.perf_counter() - start) * 1000)
    citations = s["policy"]["citations"] if "policy" in s else []
    # s["question"] was redacted by the guard node, so no PII ends up in the logs
    log.info("request=%s employee=%s route=%s latency_ms=%d question=%r",
             request_id, x_employee_id, s["route"], latency_ms, s["question"])
    return AskResponse(request_id=request_id, answer=s["answer"], route=s["route"],
                       citations=citations, latency_ms=latency_ms)