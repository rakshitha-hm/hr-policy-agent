import re
from pathlib import Path

HEADING = re.compile(r"^(#{2,3})\s+(.*)$")


def read_policy(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()
    return title, lines[1:]


def section_chunks(path):
    """One chunk per ## or ### section, prefixed with 'Title > Section' for context."""
    title, lines = read_policy(path)
    chunks, buffer = [], []
    h2, h3 = "Document information", None

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            section = f"{h2} > {h3}" if h3 else h2
            chunks.append({"source": Path(path).name, "doc_title": title, "section": section,
                           "text": f"{title} > {section}\n{body}"})

    for line in lines:
        match = HEADING.match(line)
        if match:
            flush()
            buffer = []
            if len(match.group(1)) == 2:
                h2, h3 = match.group(2).strip(), None
            else:
                h3 = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return chunks


def fixed_chunks(path, size=120, overlap=30):
    """Sliding windows of `size` words with `overlap` words shared between neighbours."""
    title, lines = read_policy(path)
    words = "\n".join(lines).split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        chunks.append({"source": Path(path).name, "doc_title": title,
                       "section": f"words {start}-{start + len(piece)}",
                       "text": f"{title}\n" + " ".join(piece)})
        if start + size >= len(words):
            break
    return chunks


def load_chunks(policy_dir, strategy="section", **kwargs):
    chunk_fn = section_chunks if strategy == "section" else fixed_chunks
    all_chunks = []
    for path in sorted(Path(policy_dir).glob("*.md")):
        for i, chunk in enumerate(chunk_fn(path, **kwargs)):
            chunk["chunk_id"] = f"{chunk['source']}#{i}"
            all_chunks.append(chunk)
    return all_chunks