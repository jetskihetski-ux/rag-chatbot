# rag-chatbot

A local RAG (Retrieval-Augmented Generation) chatbot — ingest your PDFs and text files into a vector database, then chat with them using Claude.

## Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

## Usage

**1. Ingest documents:**

```bash
python ingest.py file1.pdf notes.txt
```

**2. Chat:**

```bash
python chat.py
```

## How it works

- `ingest.py` chunks documents and stores embeddings in a local ChromaDB database
- `chat.py` retrieves the most relevant chunks for each question and passes them as context to Claude

## Stack

- Claude (via Anthropic API) — LLM
- ChromaDB — local vector store
- pypdf — PDF parsing
