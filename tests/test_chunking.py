from pathlib import Path

from chunking import fixed_chunks, load_chunks

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


def test_section_chunks():
    chunks = load_chunks(POLICY_DIR, "section")
    assert len(chunks) == 69
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)          # ids are unique
    for c in chunks:
        assert c["text"].startswith(f"{c['doc_title']} > ")              # contextual header


def test_fixed_chunks_overlap():
    path = sorted(POLICY_DIR.glob("*.md"))[0]
    chunks = fixed_chunks(path, size=60, overlap=15)
    words = [c["text"].split("\n", 1)[1].split() for c in chunks]      # drop the title line
    assert len(words[0]) == 60
    assert words[0][-15:] == words[1][:15]                              # 15 words shared