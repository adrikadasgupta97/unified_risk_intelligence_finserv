"""
Retrieves the top-k most relevant KB articles for a given complaint text.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import config
from src.rag.kb_ingestion import get_collection


@dataclass
class RetrievedDocument:
    doc_id: str
    title: str
    category: str
    content: str
    distance: float   # cosine distance; lower = more similar


def retrieve_relevant_articles(
    query: str,
    category_hint: str = None,
    top_k: int = None,
) -> list[RetrievedDocument]:
    if top_k is None:
        top_k = config.rag.top_k_retrieval

    collection = get_collection()

    where_filter = {"category": category_hint} if category_hint else None

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs: list[RetrievedDocument] = []
    if not results["ids"] or not results["ids"][0]:
        # Retry without category filter if no results found
        if where_filter:
            return retrieve_relevant_articles(query, category_hint=None, top_k=top_k)
        return docs

    for doc_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append(RetrievedDocument(
            doc_id=doc_id,
            title=metadata.get("title", ""),
            category=metadata.get("category", ""),
            content=document,
            distance=round(distance, 4),
        ))

    return docs
