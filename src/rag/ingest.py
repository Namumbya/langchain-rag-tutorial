"""Ingest markdown documents into a Chroma vector store.

Pipeline
--------
1. Load documents from data/knowledge_base/*.md
2. Split into overlapping chunks (keeps context across boundaries)
3. Embed each chunk with the selected provider
4. Persist vectors under chroma_stores/<provider>/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings, get_settings
from rag.embeddings import describe_embeddings, get_embeddings


def load_documents(data_dir: Path) -> list[Document]:
    """Load local markdown files without heavy document-parser dependencies."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No knowledge base at {data_dir}. Run: python -m rag.cli download-kb"
        )

    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        if path.name.upper() == "ATTRIBUTION.MD":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "filename": path.name},
            )
        )

    if not documents:
        raise RuntimeError(f"No markdown documents found in {data_dir}")
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_vector_store(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    documents = load_documents(settings.data_dir)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(
        f"Loaded {len(documents)} docs -> {len(chunks)} chunks | "
        f"embeddings={describe_embeddings(settings)}"
    )

    if settings.chroma_path.exists():
        print(f"Rebuilding index: removing existing store at {settings.chroma_path}")
        shutil.rmtree(settings.chroma_path)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(settings)
    # persist_directory auto-saves; do NOT call .persist() on modern Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_path),
        collection_name=f"rag_{settings.embedding_provider}",
    )
    print(f"Saved vector store -> {settings.chroma_path}")
    return vector_store


def open_vector_store(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    if not settings.chroma_path.exists():
        raise FileNotFoundError(
            f"Missing vector store at {settings.chroma_path}. "
            f"Run ingest with --provider {settings.embedding_provider} first."
        )
    return Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embeddings(settings),
        collection_name=f"rag_{settings.embedding_provider}",
    )
