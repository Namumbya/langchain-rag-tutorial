"""Retrieve relevant chunks and generate an answer with ChatOpenAI.

This is classic 2-step RAG:
  question → embed → similarity search → stuff context into prompt → LLM
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.config import Settings, get_settings
from rag.embeddings import describe_embeddings
from rag.retriever import RetrievedChunk, retrieve_chunks

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the provided context. "
            "If the context is insufficient, say you don't know. "
            "Be concise, and cite facts using the bracketed source numbers "
            "from the context, e.g. [1], [2].",
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
    scores: list[float | None]
    provider: str


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        score_text = "n/a" if chunk.score is None else f"{chunk.score:.3f}"
        parts.append(
            f"[{i}] (score={score_text}, source={chunk.display_source})\n"
            f"{chunk.document.page_content}"
        )
    return "\n\n".join(parts)


def ask(question: str, settings: Settings | None = None) -> RagResult:
    settings = settings or get_settings()
    settings.require_openai()  # chat generation uses OpenAI

    chunks = retrieve_chunks(question, settings)

    if not chunks:
        return RagResult(
            answer="Unable to find matching results in the knowledge base.",
            sources=[],
            scores=[],
            provider=describe_embeddings(settings),
        )

    prompt = PROMPT.invoke(
        {
            "context": _format_context(chunks),
            "question": question,
        }
    )
    model = ChatOpenAI(model=settings.openai_chat_model, temperature=0)
    response = model.invoke(prompt)

    sources = sorted({chunk.display_source for chunk in chunks})

    return RagResult(
        answer=str(response.content),
        sources=sources,
        scores=[chunk.score for chunk in chunks],
        provider=describe_embeddings(settings),
    )
