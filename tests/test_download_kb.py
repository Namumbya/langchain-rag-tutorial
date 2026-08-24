from rag.download_kb import _slugify, write_article, write_attribution


def test_slugify_replaces_spaces_and_lowercases():
    assert _slugify("Retrieval-Augmented Generation") == "retrieval-augmented_generation"


def test_slugify_collapses_special_characters():
    assert (
        _slugify("Transformer (deep learning architecture)")
        == "transformer__deep_learning_architecture"
    )


def test_write_article_includes_source_and_license(tmp_path):
    path = write_article("Vector database", "A vector database stores embeddings.", tmp_path)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# Vector database" in content
    assert "https://en.wikipedia.org/wiki/Vector_database" in content
    assert "CC BY-SA 4.0" in content
    assert "A vector database stores embeddings." in content


def test_write_attribution_lists_all_titles(tmp_path):
    write_attribution(tmp_path, ["Vector database", "Semantic search"])

    content = (tmp_path / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "Vector database" in content
    assert "Semantic search" in content
    assert "Creative Commons Attribution-ShareAlike" in content
