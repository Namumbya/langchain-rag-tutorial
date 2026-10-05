"""Retriever utilities for 2-step RAG.

This module separates retrieval from generation so we can:
1. swap retrieval strategies without touching prompt/LLM code
2. evaluate retrieval quality independently of answer quality
3. keep query.py focused on prompt + generation
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.config import Settings, get_settings
from rag.ingest import open_vector_store


@dataclass
class RetrievedChunk:
    document: Document
    score: float | None
    display_source: str


def _display_source(doc: Document) -> str:
    title = str(doc.metadata.get("title") or "").strip()
    filename = str(doc.metadata.get("filename") or doc.metadata.get("source", "unknown"))
    if title:
        return f"{title} ({filename})"
    return filename


def retrieve_chunks(
    query: str,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks using the configured search strategy."""
    settings = settings or get_settings()
    db = open_vector_store(settings)

    if settings.retrieval_search_type == "similarity":
        raw = db.similarity_search_with_relevance_scores(query, k=settings.retrieval_k)
        return [
            RetrievedChunk(
                document=doc,
                score=float(score),
                display_source=_display_source(doc),
            )
            for doc, score in raw
            if float(score) >= settings.min_relevance_score
        ]

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retrieval_k, "fetch_k": settings.retrieval_fetch_k},
    )
    docs = retriever.invoke(query)
    return [
        RetrievedChunk(
            document=doc,
            score=None,
            display_source=_display_source(doc),
        )
        for doc in docs
    ]
