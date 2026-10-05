from langchain_core.documents import Document

from rag.retriever import _display_source


def test_display_source_prefers_title_and_filename():
    doc = Document(
        page_content="x",
        metadata={"title": "Vector database", "filename": "vector_database.md"},
    )
    assert _display_source(doc) == "Vector database (vector_database.md)"


def test_display_source_falls_back_to_filename():
    doc = Document(page_content="x", metadata={"filename": "semantic_search.md"})
    assert _display_source(doc) == "semantic_search.md"
