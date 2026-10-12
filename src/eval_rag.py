import csv
import json
from pathlib import Path

import ollama

from rag import LLM_MODEL, NUM_CTX, RAG, FullHandbook

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval" / "rag_questions.csv"
REPORT_DIR = ROOT / "reports"

JUDGE_PROMPT = """You check whether an answer is supported by the context.
Reply with JSON only: {"supported": true} if every fact in the answer appears in the context,
otherwise {"supported": false}."""


def is_refusal(answer):
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()


def has_key_fact(answer, key_facts):
    answer = answer.lower()
    return any(fact.strip().lower() in answer for fact in key_facts.split(";") if fact.strip())


def is_supported(context, answer):
    """LLM-as-judge faithfulness check (the same local model judges its own answers)."""
    resp = ollama.chat(model=LLM_MODEL, format="json",
                       options={"temperature": 0, "num_ctx": NUM_CTX},
                       messages=[{"role": "system", "content": JUDGE_PROMPT},
                                 {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}])
    try:
        return bool(json.loads(resp["message"]["content"]).get("supported"))
    except (json.JSONDecodeError, AttributeError):
        return False


def evaluate(name, system, rows):
    details = []
    for i, row in enumerate(rows, start=1):
        out = system.answer(row["question"])
        refused = is_refusal(out["answer"])
        answerable = row["answerable"] == "1"
        record = {"system": name, "question": row["question"], "answerable": answerable,
                  "answer": out["answer"], "refused": refused,
                  "cited_sources": ";".join(sorted({c["source"] for c in out["citations"]})),
                  "prompt_tokens": out["prompt_tokens"], "latency_s": round(out["latency_s"], 2)}
        if answerable:
            record["correct"] = (not refused) and has_key_fact(out["answer"], row["key_facts"])
            record["citation_ok"] = row["source"] in record["cited_sources"]
            record["supported"] = is_supported(out["context"], out["answer"]) if not refused else None
        else:
            record["correct"] = refused
        details.append(record)
        print(f"  [{name}] {i}/{len(rows)} {'OK ' if record['correct'] else 'BAD'} {row['question']}")
    return details


def summarise(details):
    ans = [d for d in details if d["answerable"]]
    unans = [d for d in details if not d["answerable"]]
    answered = [d for d in ans if not d["refused"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "system": details[0]["system"],
        "answer_accuracy": mean([d["correct"] for d in ans]),
        "false_refusals": mean([d["refused"] for d in ans]),
        "citation_accuracy": mean([d["citation_ok"] for d in answered]),
        "faithfulness": mean([d["supported"] for d in answered]),
        "unanswerable_refused": mean([d["correct"] for d in unans]),
        "avg_prompt_tokens": round(mean([d["prompt_tokens"] for d in details])),
        "avg_latency_s": round(mean([d["latency_s"] for d in details]), 2),
    }


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} questions ({sum(r['answerable'] == '1' for r in rows)} answerable)\n")

    all_details, summaries = [], []
    for name, system in [("rag_top5", RAG()), ("full_handbook", FullHandbook())]:
        details = evaluate(name, system, rows)
        all_details += details
        summaries.append(summarise(details))

    print("\n" + " | ".join(f"{k}" for k in summaries[0]))
    for s in summaries:
        print(" | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in s.values()))

    print("\nRAG answers that were wrong:")
    for d in all_details:
        if d["system"] == "rag_top5" and not d["correct"]:
            print(f"- {d['question']}\n    -> {d['answer'][:200]}")

    REPORT_DIR.mkdir(exist_ok=True)
    with open(REPORT_DIR / "day5_rag_details.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["system", "question", "answerable", "answer", "refused", "correct",
                  "citation_ok", "supported", "cited_sources", "prompt_tokens", "latency_s"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_details)
    with open(REPORT_DIR / "day5_rag_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()