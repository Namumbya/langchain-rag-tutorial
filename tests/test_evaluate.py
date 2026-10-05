from langchain_core.documents import Document

from rag.evaluate import EvalCase, evaluate_retrieval
from rag.retriever import RetrievedChunk


def test_evaluate_retrieval_counts_hits(monkeypatch):
    def fake_retrieve_chunks(question, settings=None):
        if "vector" in question.lower():
            return [
                RetrievedChunk(
                    document=Document(page_content="x", metadata={}),
                    score=0.9,
                    display_source="Vector database (vector_database.md)",
                )
            ]
        return [
            RetrievedChunk(
                document=Document(page_content="x", metadata={}),
                score=0.7,
                display_source="Semantic search (semantic_search.md)",
            )
        ]

    monkeypatch.setattr("rag.evaluate.retrieve_chunks", fake_retrieve_chunks)

    cases = [
        EvalCase("What is a vector database?", "vector_database.md"),
        EvalCase("What is RAG?", "retrieval-augmented_generation.md"),
    ]
    result = evaluate_retrieval(cases=cases)

    assert result.total == 2
    assert result.hits_at_k == 1
    assert result.hit_rate == 0.5
