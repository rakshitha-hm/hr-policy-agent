import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUESTIONS = [
    "How many days off do I get per year in Germany?",
    "How long is maternity leave?",
    "Can I expense a parking ticket?",
    "Do I get paid when I'm ill in Bangalore?",
    "What happens to unused vacation days at the end of the year?",
    "Do I have to pay back my course fees if I resign?",
    "Can a client give me a gift card?",
]


def search(collection, model, question, k=3):
    query_vec = model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
    result = collection.query(query_embeddings=query_vec.tolist(), n_results=k)
    return list(zip(result["metadatas"][0], result["distances"][0], result["documents"][0]))


def main():
    model = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("policies")
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUESTIONS
    for question in questions:
        print(f"\nQ: {question}")
        for rank, (meta, distance, text) in enumerate(search(collection, model, question), start=1):
            preview = text.split("\n", 1)[1][:90].replace("\n", " ")
            print(f"  {rank}. sim {1 - distance:.3f} | {meta['doc_title']} > {meta['section']}")
            print(f"     {preview}...")


if __name__ == "__main__":
    main()