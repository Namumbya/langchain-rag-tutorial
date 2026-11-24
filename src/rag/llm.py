"""Chat model factory: OpenAI (API) or Ollama (local).

Teaching notes
--------------
Retrieval can already run fully local with HuggingFace embeddings.
This module makes *generation* optional-local too, so the full RAG loop
can run with no paid API key when Ollama is installed.
"""

from __future__ import annotations

from typing import Any

from rag.config import Settings


def get_chat_model(settings: Settings) -> Any:
    """Return the chat model selected in settings."""
    if settings.llm_provider == "openai":
        settings.require_openai()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_chat_model, temperature=0)

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def describe_llm(settings: Settings) -> str:
    if settings.llm_provider == "openai":
        return f"openai:{settings.openai_chat_model}"
    return f"ollama:{settings.ollama_chat_model}"
