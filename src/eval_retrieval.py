import csv
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunking import load_chunks

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "eval" / "retrieval_questions.csv"
REPORT_DIR = ROOT / "reports"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K = 5            # how many chunks we score
CANDIDATES = 10  # how many chunks the reranker re-orders


def normalise(text):
    return re.sub(r"\s+", " ", text).lower()


class Retriever:
    def __init__(self, embedder, chunks, use_prefix=True, reranker=None):
        self.embedder, self.chunks = embedder, chunks
        self.use_prefix, self.reranker = use_prefix, reranker
        self.vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def search(self, question, k=K):
        query = (QUERY_PREFIX if self.use_prefix else "") + question
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ q_vec                      # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:CANDIDATES if self.reranker else k]
        candidates = [self.chunks[i] for i in top]
        if self.reranker:
            ce_scores = self.reranker.predict([(question, c["text"]) for c in candidates])
            candidates = [candidates[i] for i in np.argsort(-ce_scores)]
        return candidates[:k]


def first_relevant_rank(results, answer_text):
    """Rank (1-based) of the first chunk containing the answer, or None if not in the top K."""
    for rank, chunk in enumerate(results, start=1):
        if normalise(answer_text) in normalise(chunk["text"]):
            return rank
    return None


def evaluate(name, retriever, rows):
    ranks, misses = [], []
    start = time.perf_counter()
    for row in rows:
        rank = first_relevant_rank(retriever.search(row["question"]), row["answer_text"])
        ranks.append(rank)
        if rank != 1:
            misses.append((row["question"], rank))
    ms = (time.perf_counter() - start) * 1000 / len(rows)
    n = len(rows)
    result = {
        "config": name,
        "chunks": len(retriever.chunks),
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr@5": sum(1 / r for r in ranks if r is not None) / n,
        "ms_per_query": round(ms, 1),
    }
    return result, misses


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Evaluation questions: {len(rows)}\n")

    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    section = load_chunks(POLICY_DIR, "section")

    configs = [
        ("section (baseline)", Retriever(embedder, section)),
        ("section, no query prefix", Retriever(embedder, section, use_prefix=False)),
        ("fixed 120 words / 30 overlap", Retriever(embedder, load_chunks(POLICY_DIR, "fixed", size=120, overlap=30))),
        ("fixed 60 words / 15 overlap", Retriever(embedder, load_chunks(POLICY_DIR, "fixed", size=60, overlap=15))),
        ("section + reranker", Retriever(embedder, section, reranker=reranker)),
    ]

    results, all_misses = [], {}
    for name, retriever in configs:
        result, misses = evaluate(name, retriever, rows)
        results.append(result)
        all_misses[name] = misses

    header = f"{'config':30s} {'chunks':>6s} {'hit@1':>6s} {'hit@3':>6s} {'hit@5':>6s} {'mrr@5':>6s} {'ms/q':>6s}"
    print(header)
    for r in results:
        print(f"{r['config']:30s} {r['chunks']:6d} {r['hit@1']:6.3f} {r['hit@3']:6.3f} "
              f"{r['hit@5']:6.3f} {r['mrr@5']:6.3f} {r['ms_per_query']:6.1f}")

    for name in ("section (baseline)", "section + reranker"):
        print(f"\nNot at rank 1 with '{name}':")
        for question, rank in all_misses[name]:
            print(f"  rank {rank if rank else '>5'}: {question}")

    REPORT_DIR.mkdir(exist_ok=True)
    with open(REPORT_DIR / "day4_retrieval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()