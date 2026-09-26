"""
ISDO - Knowledge Base Indexer
-----------------------------
1. Reads every .md file in data/kb/
2. Splits each article into chunks at level-2 (##) headings
3. Stores the chunks in a persistent ChromaDB collection called 'isdo_kb'
4. Runs 4 sample queries and prints the best-matching article + confidence

Dependency: chromadb only (uses Chroma's built-in default embedding model,
all-MiniLM-L6-v2, which is downloaded automatically on first run).

Run from anywhere:  python kb_setup.py
"""

from pathlib import Path
import re

import chromadb

# ---------------------------------------------------------------- config ---
def find_project_root(start: Path) -> Path:
    """Walk up from the script's folder until a folder containing data/kb is found,
    so the script works from the project root or any subfolder (e.g. labs/C1)."""
    for folder in [start, *start.parents]:
        if (folder / "data" / "kb").is_dir():
            return folder
    raise FileNotFoundError(f"Could not find a data/kb folder above {start}")


BASE_DIR = find_project_root(Path(__file__).resolve().parent)
KB_DIR = BASE_DIR / "data" / "kb"
DB_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "VPN says authentication failed after I changed my password",
    "My account got locked after too many wrong password attempts",
    "Whole finance team gets DBCON_FAIL error when logging into SAP",
    "Outlook on my phone stopped syncing emails",
]

H2_SPLIT = re.compile(r"^(?=## )", flags=re.MULTILINE)   # split *before* each "## "
H1_TITLE = re.compile(r"^# (.+)$", flags=re.MULTILINE)


# ------------------------------------------------------------- chunking ---
def chunk_article(path: Path) -> list[dict]:
    """Split one markdown article into chunks at '## ' headings.

    The text before the first '##' (title + metadata) becomes an 'Overview'
    chunk. '###' sub-headings stay inside their parent '##' chunk.
    Each chunk is prefixed with the article title so it embeds with context.
    """
    text = path.read_text(encoding="utf-8").strip()
    title_match = H1_TITLE.search(text)
    title = title_match.group(1).strip() if title_match else path.stem

    chunks = []
    for i, part in enumerate(H2_SPLIT.split(text)):
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            section = part.splitlines()[0][3:].strip()
        else:
            section = "Overview"
        chunks.append(
            {
                "id": f"{path.stem}::{i:02d}",
                "document": f"{title}\n\n{part}",
                "metadata": {
                    "article": path.name,
                    "title": title,
                    "section": section,
                    "chunk_index": i,
                },
            }
        )
    return chunks


def load_kb(kb_dir: Path) -> list[dict]:
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    all_chunks = []
    for f in files:
        chunks = chunk_article(f)
        print(f"  {f.name:<28} -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


# -------------------------------------------------------------- storage ---
def build_collection(chunks: list[dict]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild from scratch each run so edits/deletions in data/kb are reflected.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # distance in [0, 2]; similarity = 1 - distance
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


# ------------------------------------------------------------- querying ---
def best_article(collection: chromadb.Collection, query: str, n_results: int = 5) -> dict:
    """Return the best article for a query.

    Confidence = cosine similarity of the best-matching chunk (1 - distance),
    clamped to [0, 1]. Results from several chunks are grouped by article and
    each article keeps its highest-scoring chunk.
    """
    res = collection.query(query_texts=[query], n_results=n_results)
    best: dict[str, dict] = {}
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        score = max(0.0, min(1.0, 1.0 - dist))
        art = meta["article"]
        if art not in best or score > best[art]["confidence"]:
            best[art] = {"article": art, "title": meta["title"],
                         "section": meta["section"], "confidence": score}
    ranked = sorted(best.values(), key=lambda r: r["confidence"], reverse=True)
    return ranked[0]


def main() -> None:
    print(f"Reading KB articles from {KB_DIR}")
    chunks = load_kb(KB_DIR)

    collection = build_collection(chunks)
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}' ({DB_DIR})\n")

    print("Sample query results")
    print("=" * 78)
    for q in SAMPLE_QUERIES:
        r = best_article(collection, q)
        print(f"Query      : {q}")
        print(f"Best match : {r['article']}  ({r['title']})")
        print(f"Section    : {r['section']}")
        print(f"Confidence : {r['confidence']:.2%}")
        print("-" * 78)


if __name__ == "__main__":
    main();