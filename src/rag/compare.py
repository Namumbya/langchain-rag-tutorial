"""Compare OpenAI vs HuggingFace embeddings on the same phrases and queries.

What you learn
--------------
1. Vector length differs by model (e.g. MiniLM=384, OpenAI text-embedding-3-small=1536).
2. Cosine similarity ranks semantic closeness for a *single* embedding space.
   You cannot meaningfully cosine-compare an OpenAI vector to a HuggingFace vector —
   they live in different spaces. Compare *rankings* / retrieval results instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.embeddings import get_embeddings
from rag.ingest import open_vector_store


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must share the same dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class PairScore:
    left: str
    right: str
    similarity: float
    dimensions: int


def pairwise_similarity(
    pairs: list[tuple[str, str]],
    settings: Settings,
) -> list[PairScore]:
    embeddings = get_embeddings(settings)
    results: list[PairScore] = []
    for left, right in pairs:
        va = embeddings.embed_query(left)
        vb = embeddings.embed_query(right)
        results.append(
            PairScore(
                left=left,
                right=right,
                similarity=cosine_similarity(va, vb),
                dimensions=len(va),
            )
        )
    return results


def retrieval_preview(query: str, settings: Settings, k: int = 3) -> list[tuple[str, float, str]]:
    """Return (snippet, score, source) for top-k hits from an existing store."""
    db = open_vector_store(settings)
    hits = db.similarity_search_with_relevance_scores(query, k=k)
    preview: list[tuple[str, float, str]] = []
    for doc, score in hits:
        snippet = doc.page_content.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        preview.append((snippet, float(score), doc.metadata.get("source", "?")))
    return preview


DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("retrieval augmented generation", "fetch documents then generate an answer"),
    ("retrieval augmented generation", "banana bread recipe"),
    ("vector database", "store embeddings for similarity search"),
    ("AI hallucination", "language model inventing false facts"),
]


def run_comparison(query: str | None = None) -> None:
    """Print side-by-side metrics for both providers (OpenAI only if key present)."""
    query = query or "What is retrieval-augmented generation?"
    providers = ["huggingface"]
    openai_settings = get_settings("openai")
    key = openai_settings.openai_api_key or ""
    has_openai = bool(key) and "your-key" not in key.lower() and key.startswith("sk-")
    if has_openai:
        providers.insert(0, "openai")
    else:
        print("OPENAI_API_KEY not set - comparing HuggingFace only for pairwise scores.\n")

    print("=" * 72)
    print("1) Pairwise cosine similarity inside each embedding space")
    print("=" * 72)
    for provider in providers:
        settings = get_settings(provider)
        print(f"\n[{provider}] model settings loaded")
        for item in pairwise_similarity(DEFAULT_PAIRS, settings):
            print(
                f"  cos={item.similarity:.4f}  dim={item.dimensions}  "
                f"| {item.left!r}  vs  {item.right!r}"
            )

    print("\n" + "=" * 72)
    print(f"2) Retrieval preview for: {query!r}")
    print("=" * 72)
    for provider in providers:
        settings = get_settings(provider)
        try:
            hits = retrieval_preview(query, settings)
        except FileNotFoundError as exc:
            print(f"\n[{provider}] skipped - {exc}")
            continue
        print(f"\n[{provider}] top hits:")
        for i, (snippet, score, source) in enumerate(hits, start=1):
            print(f"  {i}. score={score:.3f}  {snippet}")
            print(f"     source={source}")
