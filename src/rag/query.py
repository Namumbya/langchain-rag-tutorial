"""Retrieve relevant chunks and generate an answer with ChatOpenAI.

This is classic 2-step RAG:
  question → embed → similarity search → stuff context into prompt → LLM
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.config import Settings, get_settings
from rag.embeddings import describe_embeddings
from rag.ingest import open_vector_store

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the provided context. "
            "If the context is insufficient, say you don't know. "
            "Be concise and cite facts from the context when possible.",
        ),
        (
            "human",
            "Context:\n{context}\n\n---\n\nQuestion: {question}",
        ),
    ]
)


@dataclass
class RagResult:
    answer: str
    sources: list[str]
    scores: list[float]
    provider: str


def _format_context(pairs: list[tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(pairs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (score={score:.3f}, source={source})\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(question: str, settings: Settings | None = None) -> RagResult:
    settings = settings or get_settings()
    settings.require_openai()  # chat generation uses OpenAI

    db = open_vector_store(settings)
    # relevance_scores are normalized ~0..1 for cosine when embeddings are normalized
    raw = db.similarity_search_with_relevance_scores(question, k=settings.retrieval_k)
    pairs = [(doc, float(score)) for doc, score in raw if float(score) >= settings.min_relevance_score]

    if not pairs:
        return RagResult(
            answer="Unable to find matching results in the knowledge base.",
            sources=[],
            scores=[],
            provider=describe_embeddings(settings),
        )

    prompt = PROMPT.invoke(
        {
            "context": _format_context(pairs),
            "question": question,
        }
    )
    model = ChatOpenAI(model=settings.openai_chat_model, temperature=0)
    response = model.invoke(prompt)

    sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in pairs})

    return RagResult(
        answer=str(response.content),
        sources=sources,
        scores=[score for _, score in pairs],
        provider=describe_embeddings(settings),
    )
