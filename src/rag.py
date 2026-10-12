import re
import sys
import time
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
POLICY_DIR = ROOT / "data" / "policies"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODEL = "qwen2.5:7b"
TOP_K = 5          # chosen on Day 4: hit@5 = 0.978
NUM_CTX = 8192     # context window we ask Ollama for (its default is smaller)
REFUSAL = "I don't know based on the HR policies."

SYSTEM_PROMPT = f"""You are the HR policy assistant of Northwind Analytics.
Answer the employee's question using ONLY the numbered context passages.
Rules:
- Be brief: 1 to 3 sentences.
- After each fact, cite the passage number in square brackets, for example [2].
- If the context does not contain the answer, reply exactly: {REFUSAL}
- Do not use outside knowledge and do not guess."""


def generate(context, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    resp = ollama.chat(model=LLM_MODEL, messages=messages,
                       options={"temperature": 0, "num_ctx": NUM_CTX})
    prompt_tokens = getattr(resp, "prompt_eval_count", 0) or 0
    return resp["message"]["content"].strip(), prompt_tokens


def cited_numbers(text, n_passages):
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return sorted(n for n in numbers if 1 <= n <= n_passages)


class RAG:
    """Retrieval-augmented answering: top-5 policy chunks from ChromaDB as context."""

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("policies")

    def retrieve(self, question, k=TOP_K):
        vec = self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True, show_progress_bar=False)
        res = self.collection.query(query_embeddings=vec.tolist(), n_results=k)
        return [{"text": text, "source": meta["source"], "section": meta["section"], "similarity": 1 - dist}
                for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def answer(self, question):
        start = time.perf_counter()
        chunks = self.retrieve(question)
        context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
        text, prompt_tokens = generate(context, question)
        citations = [chunks[n - 1] for n in cited_numbers(text, len(chunks))]
        return {"answer": text, "context": context, "prompt_tokens": prompt_tokens,
                "citations": [{"source": c["source"], "section": c["section"]} for c in citations],
                "latency_s": time.perf_counter() - start}


class FullHandbook:
    """Baseline: no retrieval, all 8 policy documents in every prompt."""

    def __init__(self):
        self.docs = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(POLICY_DIR.glob("*.md"))]
        self.context = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(self.docs, start=1))

    def answer(self, question):
        start = time.perf_counter()
        text, prompt_tokens = generate(self.context, question)
        citations = [{"source": self.docs[n - 1][0], "section": "whole document"}
                     for n in cited_numbers(text, len(self.docs))]
        return {"answer": text, "context": self.context, "prompt_tokens": prompt_tokens,
                "citations": citations, "latency_s": time.perf_counter() - start}


def main():
    question = " ".join(sys.argv[1:]) or "How long is maternity leave?"
    result = RAG().answer(question)
    print(f"Q: {question}\n\nA: {result['answer']}\n")
    for c in result["citations"]:
        print(f"  source: {c['source']} > {c['section']}")
    print(f"\n{result['prompt_tokens']} prompt tokens, {result['latency_s']:.1f} s")


if __name__ == "__main__":
    main()