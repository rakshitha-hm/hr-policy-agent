import statistics
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import load_chunks

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
CHROMA_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "policies"


def main():
    chunks = load_chunks(POLICY_DIR, strategy="section")
    lengths = [len(c["text"].split()) for c in chunks]
    print(f"Chunks: {len(chunks)} | words per chunk: min {min(lengths)}, "
          f"median {statistics.median(lengths):.0f}, max {max(lengths)}")

    model = SentenceTransformer(EMBED_MODEL)
    start = time.perf_counter()
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    print(f"Embedded in {time.perf_counter() - start:.1f} s, vector size {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"], "doc_title": c["doc_title"], "section": c["section"]}
                   for c in chunks],
    )
    print(f"Stored {collection.count()} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()