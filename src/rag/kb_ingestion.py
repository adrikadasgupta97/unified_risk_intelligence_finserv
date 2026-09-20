"""
Ingests knowledge base articles and FAQs into ChromaDB.
Run this once (or when KB updates) to build the vector store.
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import config


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(config.ROOT / "chroma_db"))


def get_collection(client: chromadb.PersistentClient = None):
    if client is None:
        client = get_chroma_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.rag.embedding_model
    )
    return client.get_or_create_collection(
        name=config.rag.collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_knowledge_base(kb_path: Path = None) -> int:
    if kb_path is None:
        kb_path = config.KB_DIR / "financial_kb.json"

    with open(kb_path, encoding="utf-8") as f:
        articles: list[dict] = json.load(f)

    collection = get_collection()

    # Skip articles already ingested
    existing_ids = set(collection.get()["ids"])
    new_articles = [a for a in articles if a["id"] not in existing_ids]

    if not new_articles:
        return 0

    documents = [f"{a['title']}\n\n{a['content']}" for a in new_articles]
    metadatas = [{"id": a["id"], "category": a["category"], "title": a["title"]} for a in new_articles]
    ids = [a["id"] for a in new_articles]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(new_articles)


if __name__ == "__main__":
    n = ingest_knowledge_base()
    print(f"Ingested {n} new articles into ChromaDB.")
