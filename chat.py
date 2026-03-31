"""
chat.py — Chat with your ingested documents using Claude + RAG.
Usage: python chat.py
"""

import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
N_RESULTS = 5


def get_context(query: str, collection) -> str:
    results = collection.query(query_texts=[query], n_results=N_RESULTS)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return ""

    parts = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc}")

    return "\n\n---\n\n".join(parts)


def build_user_message(question: str, context: str) -> str:
    if context:
        return (
            f"Use the following context from my documents to answer the question. "
            f"If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )
    return f"{question}\n\n(No relevant documents found in the knowledge base.)"


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY in a .env file (copy .env.example)")
        return

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        collection = db.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        print("No documents found. Run: python ingest.py <your_file.pdf>")
        return

    if collection.count() == 0:
        print("Database is empty. Run: python ingest.py <your_file.pdf>")
        return

    client = anthropic.Anthropic(api_key=api_key)
    history: list[dict] = []

    print(f"RAG Chatbot ready! ({collection.count()} chunks loaded)")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        context = get_context(question, collection)
        user_message = build_user_message(question, context)

        history.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=(
                "You are a helpful assistant that answers questions based on the user's documents. "
                "Be concise and cite the source document when possible."
            ),
            messages=history,
        )

        answer = response.content[0].text

        # Store plain question in history (not the context-injected version) to keep tokens low
        history[-1] = {"role": "user", "content": question}
        history.append({"role": "assistant", "content": answer})

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
