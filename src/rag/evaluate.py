"""Retrieval evaluation for the tutorial knowledge base.

Reports hit@k and MRR so you can compare embedding providers / chunking
strategies with harder, paraphrased questions instead of title lookups.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.retriever import retrieve_chunks


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_source_contains: str


@dataclass(frozen=True)
class EvalResult:
    total: int
    hits_at_k: int
    mrr: float

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits_at_k / self.total


# Paraphrased / fact-oriented questions so evaluate can distinguish setups.
DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "How does fetching external documents before answering reduce made-up facts?",
        "retrieval-augmented_generation.md",
    ),
    EvalCase(
        "What limitation of trained language models makes them freeze outdated knowledge?",
        "large_language_model.md",
    ),
    EvalCase(
        "How do dense vectors place similar meanings close together in space?",
        "word_embedding.md",
    ),
    EvalCase(
        "What kind of database stores embeddings for nearest-neighbor lookup?",
        "vector_database.md",
    ),
    EvalCase(
        "How can search match meaning instead of exact keywords?",
        "semantic_search.md",
    ),
    EvalCase(
        "What neural network architecture uses self-attention for sequence modeling?",
        "transformer__deep_learning.md",
    ),
    EvalCase(
        "How can carefully worded instructions change a model's output quality?",
        "prompt_engineering.md",
    ),
    EvalCase(
        "Why do language models invent plausible but false citations?",
        "hallucination__artificial_intelligence.md",
    ),
    EvalCase(
        "What are the main stages of a RAG pipeline?",
        "retrieval-augmented_generation.md",
    ),
    EvalCase(
        "What is hybrid search that mixes keyword and vector results?",
        "retrieval-augmented_generation.md",
    ),
]


def _first_hit_rank(chunks_sources: list[str], expected: str) -> int | None:
    for index, source in enumerate(chunks_sources, start=1):
        if expected in source:
            return index
    return None


def evaluate_retrieval(
    settings: Settings | None = None,
    cases: list[EvalCase] | None = None,
) -> EvalResult:
    settings = settings or get_settings()
    cases = cases or DEFAULT_EVAL_CASES

    hits = 0
    reciprocal_ranks = 0.0
    for case in cases:
        chunks = retrieve_chunks(case.question, settings)
        sources = [chunk.display_source for chunk in chunks]
        rank = _first_hit_rank(sources, case.expected_source_contains)
        if rank is None:
            print(f"MISS | {case.question}")
            print("     top source: none" if not chunks else f"     top source: {sources[0]}")
            continue

        hits += 1
        reciprocal_ranks += 1.0 / rank
        print(f" HIT | rank={rank} | {case.question}")
        print(f"     top source: {sources[0]}")

    total = len(cases)
    mrr = reciprocal_ranks / total if total else 0.0
    return EvalResult(total=total, hits_at_k=hits, mrr=mrr)
