"""Embedding model factory: OpenAI (API) vs HuggingFace (local).

Teaching notes
--------------
Embeddings turn text into vectors so "similar meaning" ≈ "nearby in space".

* OpenAIEmbeddings  — remote API, usually stronger retrieval quality, costs money,
  needs OPENAI_API_KEY.
* HuggingFaceEmbeddings — downloads a sentence-transformers model once, then runs
  on your machine (CPU/GPU). Free, private, no API key for local models.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag.config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    """Return the embedding backend selected in settings."""
    if settings.embedding_provider == "openai":
        settings.require_openai()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        # Local sentence-transformers model; first run downloads weights.
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def describe_embeddings(settings: Settings) -> str:
    if settings.embedding_provider == "openai":
        return f"openai:{settings.openai_embedding_model}"
    return f"huggingface:{settings.hf_embedding_model}"
