"""Small retrieval evaluation loop for the tutorial knowledge base.

The goal is not benchmark-grade rigor. It gives a quick signal for:
- whether retrieval hits the expected source document
- whether one embedding provider is outperforming another on this corpus
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

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits_at_k / self.total


DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase("What is retrieval-augmented generation?", "retrieval-augmented_generation.md"),
    EvalCase("What is semantic search?", "semantic_search.md"),
    EvalCase("What is a vector database?", "vector_database.md"),
    EvalCase("What is prompt engineering?", "prompt_engineering.md"),
    EvalCase("What are word embeddings?", "word_embedding.md"),
]


def evaluate_retrieval(
    settings: Settings | None = None,
    cases: list[EvalCase] | None = None,
) -> EvalResult:
    settings = settings or get_settings()
    cases = cases or DEFAULT_EVAL_CASES

    hits = 0
    for case in cases:
        chunks = retrieve_chunks(case.question, settings)
        matched = any(case.expected_source_contains in chunk.display_source for chunk in chunks)
        status = "HIT" if matched else "MISS"
        print(f"{status:>4} | {case.question}")
        if chunks:
            print(f"     top source: {chunks[0].display_source}")
        else:
            print("     top source: none")
        if matched:
            hits += 1

    return EvalResult(total=len(cases), hits_at_k=hits)
