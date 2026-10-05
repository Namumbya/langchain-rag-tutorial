from langchain_core.documents import Document

from rag.query import _format_context
from rag.retriever import RetrievedChunk


def test_format_context_includes_index_score_and_source():
    chunks = [
        RetrievedChunk(
            document=Document(page_content="Chunk one text", metadata={"source": "a.md"}),
            score=0.912345,
            display_source="Alpha (a.md)",
        ),
        RetrievedChunk(
            document=Document(page_content="Chunk two text", metadata={"source": "b.md"}),
            score=0.4,
            display_source="Beta (b.md)",
        ),
    ]

    formatted = _format_context(chunks)

    assert "[1] (score=0.912, source=Alpha (a.md))" in formatted
    assert "Chunk one text" in formatted
    assert "[2] (score=0.400, source=Beta (b.md))" in formatted
    assert "Chunk two text" in formatted


def test_format_context_defaults_missing_source_to_unknown():
    chunks = [
        RetrievedChunk(
            document=Document(page_content="No source here", metadata={}),
            score=None,
            display_source="unknown",
        )
    ]

    formatted = _format_context(chunks)

    assert "score=n/a" in formatted
    assert "source=unknown" in formatted
