import json
import time

import ollama

MODEL = "qwen2.5:7b"


def ask(messages, temperature=0.0, fmt=None):
    start = time.perf_counter()
    resp = ollama.chat(model=MODEL, messages=messages,
                       options={"temperature": temperature}, format=fmt)
    seconds = time.perf_counter() - start
    tokens = resp["eval_count"]
    tok_per_s = tokens / (resp["eval_duration"] / 1e9)
    return resp["message"]["content"], tokens, tok_per_s, seconds


def show(title, text, tokens=None, tok_per_s=None, seconds=None):
    print(f"\n=== {title} ===")
    print(text.strip())
    if tokens is not None:
        print(f"[{tokens} tokens, {tok_per_s:.0f} tok/s, {seconds:.1f} s]")


def main():
    # 1. Plain question
    text, n, tps, s = ask([{"role": "user", "content": "In two sentences, what is annual leave?"}])
    show("1. Plain question", text, n, tps, s)

    # 2. System prompt with a rule
    messages = [
        {"role": "system", "content": "You are an HR assistant. Always answer in exactly one short sentence."},
        {"role": "user", "content": "What is sick leave?"},
    ]
    show("2. System prompt", ask(messages)[0])

    # 3. Temperature: 0 is repeatable, 1.0 varies
    q = [{"role": "user", "content": "Suggest a name for an HR chatbot. Reply with the name only."}]
    for t in (0.0, 0.0, 1.0, 1.0):
        show(f"3. Temperature {t}", ask(q, temperature=t)[0])

    # 4. Hallucination: a company that does not exist
    question = "What is the parental leave policy at Zentrovia Industries?"
    show("4a. No rule", ask([{"role": "user", "content": question}])[0])
    messages = [
        {"role": "system", "content": "Answer only from the provided context. "
                                      "If the context does not contain the answer, reply exactly: I don't know."},
        {"role": "user", "content": f"Context: (empty)\n\nQuestion: {question}"},
    ]
    show("4b. With 'I don't know' rule", ask(messages)[0])

    # 5. Structured JSON output
    messages = [{"role": "user", "content":
                 "Extract the leave type and number of days from this request as JSON with keys "
                 "'leave_type' and 'days': 'I would like to take 3 days of sick leave next week.'"}]
    raw = ask(messages, fmt="json")[0]
    data = json.loads(raw)
    show("5. JSON output", f"{raw}\nParsed: leave_type={data.get('leave_type')}, days={data.get('days')}")


if __name__ == "__main__":
    main()