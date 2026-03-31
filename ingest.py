"""
ingest.py — Load PDFs and text files into the vector database.
Usage: python ingest.py file1.pdf file2.txt ...
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions
import pypdf
from pathlib import Path

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ingest_file(path: str, collection):
    p = Path(path)
    print(f"Ingesting {p.name}...")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = load_pdf(str(p))
    elif suffix in (".txt", ".md"):
        text = load_text(str(p))
    else:
        print(f"  Skipping unsupported file type: {suffix}")
        return

    chunks = chunk_text(text)
    if not chunks:
        print(f"  No text extracted from {p.name}")
        return

    ids = [f"{p.name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": p.name, "chunk": i} for i in range(len(chunks))]

    # Delete existing chunks from this file to allow re-ingestion
    try:
        existing = collection.get(where={"source": p.name})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"  Added {len(chunks)} chunks")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file1.pdf> [file2.txt] ...")
        sys.exit(1)

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    for file_path in sys.argv[1:]:
        if os.path.exists(file_path):
            ingest_file(file_path, collection)
        else:
            print(f"File not found: {file_path}")

    print(f"\nDone! Total chunks in database: {collection.count()}")


if __name__ == "__main__":
    main()
