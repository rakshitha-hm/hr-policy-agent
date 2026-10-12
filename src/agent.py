import json
import operator
import re
import sqlite3
import sys
import time
from typing import Annotated, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from guardrails import ACCESS_DENIED, BLOCKED_INPUT, check_input, redact_pii
from rag import LLM_MODEL, NUM_CTX, RAG, REFUSAL
from sql_tool import DB_PATH, SQLTool

ROUTES = ("policy", "data", "both", "out_of_scope")
OUT_OF_SCOPE = "I can only help with questions about Northwind HR policies and your HR data."
COUNTRY_NAMES = {"NL": "Netherlands", "DE": "Germany", "IN": "India"}

ROUTER_PROMPT = """You route questions for the HR assistant of Northwind Analytics.
Sources:
- policy: the HR handbook with rules, entitlements, limits and procedures. Example: "How long is maternity leave?"
- data: the HR database with facts about employees: leave balances, leave taken, expense claims,
  hire dates, departments, managers, counts. Example: "How many leave days do I have left?"
- both: the question needs a rule from the handbook AND a fact from the database.
  Example: "How many leave days do I have left, and can I carry them over?"
- out_of_scope: anything that is not about HR at Northwind.

About the user: {profile}
Make each sub-question standalone: write employee {employee_id} instead of I, my or me,
and the user's country instead of "my country". Never write a placeholder like [employee_id].
Split the question: policy_question gets only the rule part, data_question gets only the fact part.
A general "Can I ...?" or "Am I allowed ...?" question about a rule is policy, unless it needs the user's own records.

Example for employee {employee_id}:
Question: "How much have I spent on travel, and what is the hotel limit in Berlin?"
{{"route": "both", "policy_question": "What is the hotel limit per night in Berlin?",
 "data_question": "What is the total amount of travel expense claims of employee {employee_id}?"}}

Reply with JSON only:
{{"route": "policy" | "data" | "both" | "out_of_scope", "policy_question": "..." or null, "data_question": "..." or null}}"""

ANSWER_PROMPT = f"""You are the HR assistant of Northwind Analytics. Answer the employee's question
using ONLY the evidence below. Personal facts come from the user profile and the database result,
rules come from the policy answer.
Rules:
- Be brief: 1 to 3 sentences. Address the employee as "you".
- Keep the policy citations exactly as they appear, for example [2]. Facts from the database get no citation.
- If the evidence does not answer the question, reply exactly: {REFUSAL}"""


class AgentState(TypedDict, total=False):
    question: str
    employee_id: int
    profile: str                              # facts about the user, loaded by code
    route: str
    policy_question: str
    data_question: str
    policy: dict                              # output of RAG.answer
    data: dict                                # output of SQLTool.run
    answer: str
    blocked: bool                             # stopped by the input guardrail
    trace: Annotated[list, operator.add]      # reducer: parallel nodes append instead of overwrite


def llm_json(system, user):
    resp = ollama.chat(model=LLM_MODEL, format="json", options={"temperature": 0, "num_ctx": NUM_CTX},
                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        return json.loads(resp["message"]["content"])
    except json.JSONDecodeError:
        return {}


def load_profile(employee_id):
    """Facts about the logged-in user, read by code (not the LLM), so the agent knows who 'I' is."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT first_name, country, office, work_mode, fte, hire_date, department, level "
                           "FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return f"employee {employee_id} (not found in the HR database)"
    name, country, office, mode, fte, hired, dept, level = row
    return (f"employee {employee_id} ({name}), country {COUNTRY_NAMES.get(country, country)}, office {office}, "
            f"work mode {mode}, fte {fte}, hired {hired}, department {dept}, level {level}")


def format_evidence(state):
    parts = [f"User profile: {state['profile']}"]
    if "policy" in state:
        p = state["policy"]
        sources = ", ".join(f"{c['source']} > {c['section']}" for c in p["citations"]) or "none"
        parts.append(f"Policy answer (question: {state['policy_question']}):\n{p['answer']}\nSources: {sources}")
    if "data" in state:
        d = state["data"]
        if d["error"]:
            parts.append(f"Database result (question: {state['data_question']}): no result ({d['error']})")
        else:
            parts.append(f"Database result (question: {state['data_question']}):\n"
                         f"columns: {d['columns']}\nrows: {d['rows'][:20]}")
    return "\n\n".join(parts)


def build_agent(force_route=None):
    """Build the LangGraph agent. force_route='both' turns off the router decision (used as a baseline)."""
    rag, sql_tool = RAG(), SQLTool()

    def guard(state):
        question = redact_pii(state["question"])       # PII never reaches the prompts or the trace
        reason = check_input(question)
        if reason:
            return {"question": question, "blocked": True, "answer": BLOCKED_INPUT,
                    "trace": [f"guard: blocked ({reason})"]}
        return {"question": question, "blocked": False, "trace": ["guard: ok"]}

    def after_guard(state):
        return "end" if state["blocked"] else "router"

    def router(state):
        profile = load_profile(state["employee_id"])
        prompt = ROUTER_PROMPT.format(employee_id=state["employee_id"], profile=profile)
        out = llm_json(prompt, state["question"])
        route = out.get("route") if out.get("route") in ROUTES else "both"   # unsure -> use both tools
        if force_route:
            route = force_route
        question, eid = state["question"], str(state["employee_id"])
        # identity is added by code, never left to the LLM
        data_question = f"{out.get('data_question') or question} (The user asking is employee {eid}.)"
        policy_question = out.get("policy_question") or question
        return {"profile": profile, "route": route, "policy_question": policy_question,
                "data_question": data_question, "trace": [f"router -> {route}"]}

    def pick_tools(state):
        return {"policy": ["policy"], "data": ["data"], "both": ["policy", "data"]}.get(state["route"], ["refuse"])

    def policy(state):
        out = rag.answer(state["policy_question"])
        return {"policy": out, "trace": [f"policy question: {state['policy_question']}"]}

    def data(state):
        out = sql_tool.run(state["data_question"], employee_id=state["employee_id"])
        return {"data": out, "trace": [f"data question: {state['data_question']}", f"data sql: {out['sql']}"]}

    def compose(state):
        if state.get("data", {}).get("access_denied"):  # guardrail decision: no LLM involved
            return {"answer": ACCESS_DENIED, "trace": ["compose: access denied"]}
        if state["route"] == "policy":                  # RAG already wrote a cited answer: no extra LLM call
            return {"answer": state["policy"]["answer"], "trace": ["compose: pass-through"]}
        user = f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state)}"
        resp = ollama.chat(model=LLM_MODEL, options={"temperature": 0, "num_ctx": NUM_CTX},
                           messages=[{"role": "system", "content": ANSWER_PROMPT},
                                     {"role": "user", "content": user}])
        answer = resp["message"]["content"].strip()
        if "policy" not in state:                       # no policy passages -> any [n] is invented
            answer = re.sub(r"\s*\[\d+\]", "", answer)
        return {"answer": redact_pii(answer), "trace": ["compose: LLM"]}

    def refuse(state):
        return {"answer": OUT_OF_SCOPE, "trace": ["refuse"]}

    graph = StateGraph(AgentState)
    graph.add_node("guard", guard)
    graph.add_node("router", router)
    graph.add_node("policy", policy)
    graph.add_node("data", data)
    graph.add_node("compose", compose)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", after_guard, {"router": "router", "end": END})
    graph.add_conditional_edges("router", pick_tools, ["policy", "data", "refuse"])
    graph.add_edge("policy", "compose")
    graph.add_edge("data", "compose")
    graph.add_edge("compose", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def ask(agent, question, employee_id):
    start = time.perf_counter()
    state = agent.invoke({"question": question, "employee_id": employee_id, "trace": []})
    state.setdefault("route", "blocked")              # the input guardrail stops before the router
    state["latency_s"] = time.perf_counter() - start
    return state


def main():
    agent = build_agent()
    if sys.argv[1:] == ["--graph"]:
        print(agent.get_graph().draw_mermaid())
        return
    employee_id = int(sys.argv[1]) if len(sys.argv) > 2 else 1005
    question = " ".join(sys.argv[2:]) or "How many leave days do I have left, and can I carry them over?"
    s = ask(agent, question, employee_id)
    print(f"Employee {employee_id}: {question}\n")
    for step in s["trace"]:
        print(f"  {step}")
    print(f"\nA: {s['answer']}")
    if "policy" in s:
        for c in s["policy"]["citations"]:
            print(f"  source: {c['source']} > {c['section']}")
    print(f"\n{s['latency_s']:.1f} s")


if __name__ == "__main__":
    main()