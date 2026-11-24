from langchain_core.documents import Document

from rag.evaluate import EvalCase, _first_hit_rank, evaluate_retrieval
from rag.retriever import RetrievedChunk


def test_evaluate_retrieval_counts_hits_and_mrr(monkeypatch):
    def fake_retrieve_chunks(question, settings=None):
        if "vector" in question.lower():
            return [
                RetrievedChunk(
                    document=Document(page_content="x", metadata={}),
                    score=0.9,
                    display_source="Vector database (vector_database.md)",
                )
            ]
        # Expected doc is second → hit with rank 2
        return [
            RetrievedChunk(
                document=Document(page_content="x", metadata={}),
                score=0.8,
                display_source="Semantic search (semantic_search.md)",
            ),
            RetrievedChunk(
                document=Document(page_content="x", metadata={}),
                score=0.7,
                display_source="RAG (retrieval-augmented_generation.md)",
            ),
        ]

    monkeypatch.setattr("rag.evaluate.retrieve_chunks", fake_retrieve_chunks)

    cases = [
        EvalCase("What is a vector database?", "vector_database.md"),
        EvalCase("What is RAG?", "retrieval-augmented_generation.md"),
    ]
    result = evaluate_retrieval(cases=cases)

    assert result.total == 2
    assert result.hits_at_k == 2
    assert result.hit_rate == 1.0
    # ranks 1 and 2 → MRR = (1 + 0.5) / 2 = 0.75
    assert result.mrr == 0.75


def test_first_hit_rank_returns_none_when_missing():
    assert _first_hit_rank(["a.md", "b.md"], "c.md") is None
    assert _first_hit_rank(["a.md", "b.md"], "b.md") == 2
