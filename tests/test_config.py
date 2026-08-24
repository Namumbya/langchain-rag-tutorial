import pytest

from rag.config import get_settings


def test_default_provider_is_huggingface(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    settings = get_settings()
    assert settings.embedding_provider == "huggingface"


def test_explicit_provider_overrides_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    settings = get_settings("openai")
    assert settings.embedding_provider == "openai"


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER"):
        get_settings()


def test_require_openai_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = get_settings("huggingface")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing"):
        settings.require_openai()


def test_require_openai_returns_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    settings = get_settings("huggingface")
    assert settings.require_openai() == "sk-test123"


def test_chroma_path_is_per_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    settings = get_settings()
    assert settings.chroma_path.name == "openai"
    assert settings.chroma_path.parent == settings.chroma_root


def test_invalid_chunk_size_raises_friendly_error(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    with pytest.raises(ValueError, match="Invalid CHUNK_SIZE"):
        get_settings("huggingface")


def test_invalid_min_relevance_score_raises_friendly_error(monkeypatch):
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "not-a-float")
    with pytest.raises(ValueError, match="Invalid MIN_RELEVANCE_SCORE"):
        get_settings("huggingface")
