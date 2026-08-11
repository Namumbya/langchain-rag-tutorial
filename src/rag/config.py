"""Central configuration loaded from environment variables and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = repo root (parent of src/)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "knowledge_base"
CHROMA_ROOT = ROOT_DIR / "chroma_stores"

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for ingest / query / compare."""

    embedding_provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    openai_chat_model: str
    hf_embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    min_relevance_score: float
    data_dir: Path
    chroma_root: Path

    @property
    def chroma_path(self) -> Path:
        """One Chroma directory per embedding provider so indexes stay comparable."""
        return self.chroma_root / self.embedding_provider

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and set your key."
            )
        return self.openai_api_key


def get_settings(
    embedding_provider: str | None = None,
) -> Settings:
    provider = (embedding_provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")).lower()
    if provider not in {"openai", "huggingface"}:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER={provider!r}. Use 'openai' or 'huggingface'."
        )

    return Settings(
        embedding_provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.3")),
        data_dir=DATA_DIR,
        chroma_root=CHROMA_ROOT,
    )


# Default Wikipedia pages for the open knowledge base (CC BY-SA 4.0).
DEFAULT_WIKI_TITLES = [
    "Retrieval-augmented_generation",
    "Large_language_model",
    "Word_embedding",
    "Vector_database",
    "Semantic_search",
    "Transformer_(deep_learning_architecture)",
    "Prompt_engineering",
    "Hallucination_(artificial_intelligence)",
]
