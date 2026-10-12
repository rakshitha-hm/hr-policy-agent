import csv
from pathlib import Path

from agent import ask, build_agent

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval" / "agent_questions.csv"
REPORT_DIR = ROOT / "reports"


def is_refusal(answer):
    a = answer.lower()
    return "don't know" in a or "do not know" in a or "only help with" in a


def has_facts(answer, key_facts):
    """key_facts: groups separated by | (all needed), alternatives separated by ; (any one is enough)."""
    a = answer.lower().replace(",", "")
    return all(any(alt.strip().lower().replace(",", "") in a for alt in group.split(";") if alt.strip())
               for group in key_facts.split("|"))


def evaluate(name, agent, rows):
    details = []
    for i, row in enumerate(rows, start=1):
        s = ask(agent, row["question"], int(row["employee_id"]))
        expected = row["expected_route"]
        if expected == "out_of_scope":
            correct = is_refusal(s["answer"])
        else:
            correct = (not is_refusal(s["answer"])) and has_facts(s["answer"], row["key_facts"])
        details.append({"system": name, "question": row["question"], "employee_id": row["employee_id"],
                        "expected_route": expected, "route": s["route"],
                        "route_ok": s["route"] == expected, "correct": correct, "answer": s["answer"],
                        "trace": " | ".join(s["trace"]), "latency_s": round(s["latency_s"], 2)})
        print(f"  [{name}] {i:2d}/{len(rows)} {'OK ' if correct else 'BAD'} "
              f"route={s['route']:12s} (expected {expected}) {row['question']}")
    return details


def summarise(details):
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    name = details[0]["system"]
    print(f"\n{name}")
    print(f"  {'expected route':15s} {'n':>3s} {'route acc':>9s} {'answer acc':>10s} {'latency s':>9s}")
    for route in ("policy", "data", "both", "out_of_scope"):
        sub = [d for d in details if d["expected_route"] == route]
        print(f"  {route:15s} {len(sub):3d} {mean([d['route_ok'] for d in sub]):9.3f} "
              f"{mean([d['correct'] for d in sub]):10.3f} {mean([d['latency_s'] for d in sub]):9.2f}")
    print(f"  {'all':15s} {len(details):3d} {mean([d['route_ok'] for d in details]):9.3f} "
          f"{mean([d['correct'] for d in details]):10.3f} {mean([d['latency_s'] for d in details]):9.2f}")


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} questions\n")

    all_details = []
    for name, force in [("agent (router)", None), ("always both tools", "both")]:
        details = evaluate(name, build_agent(force_route=force), rows)
        summarise(details)
        all_details += details

    print("\nAgent answers that were wrong:")
    for d in all_details:
        if d["system"] == "agent (router)" and not d["correct"]:
            print(f"- [{d['employee_id']}] {d['question']}\n    route: {d['route']} (expected {d['expected_route']})"
                  f"\n    trace: {d['trace']}\n    -> {d['answer'][:250]}")

    REPORT_DIR.mkdir(exist_ok=True)
    with open(REPORT_DIR / "day7_agent_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_details[0]))
        writer.writeheader()
        writer.writerows(all_details)


if __name__ == "__main__":
    main()