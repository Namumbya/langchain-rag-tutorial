from langchain_core.documents import Document

from rag.query import _format_context


def test_format_context_includes_index_score_and_source():
    pairs = [
        (Document(page_content="Chunk one text", metadata={"source": "a.md"}), 0.912345),
        (Document(page_content="Chunk two text", metadata={"source": "b.md"}), 0.4),
    ]

    formatted = _format_context(pairs)

    assert "[1] (score=0.912, source=a.md)" in formatted
    assert "Chunk one text" in formatted
    assert "[2] (score=0.400, source=b.md)" in formatted
    assert "Chunk two text" in formatted


def test_format_context_defaults_missing_source_to_unknown():
    pairs = [(Document(page_content="No source here", metadata={}), 0.5)]

    formatted = _format_context(pairs)

    assert "source=unknown" in formatted
