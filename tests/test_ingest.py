import pytest
from langchain_core.documents import Document

from rag.ingest import load_documents, split_documents


def test_load_documents_reads_markdown_and_skips_attribution(tmp_path):
    (tmp_path / "a.md").write_text("Content A", encoding="utf-8")
    (tmp_path / "b.md").write_text("Content B", encoding="utf-8")
    (tmp_path / "ATTRIBUTION.md").write_text("should be skipped", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    documents = load_documents(tmp_path)

    contents = {doc.page_content for doc in documents}
    assert contents == {"Content A", "Content B"}
    assert all(doc.metadata["source"] for doc in documents)


def test_load_documents_missing_dir_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="No knowledge base"):
        load_documents(missing)


def test_load_documents_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No markdown documents"):
        load_documents(tmp_path)


def test_split_documents_respects_chunk_size():
    doc = Document(page_content="word " * 500, metadata={"source": "x"})

    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)
