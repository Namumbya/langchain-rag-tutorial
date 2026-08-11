"""Command-line entry points for the RAG tutorial."""

from __future__ import annotations

import argparse
import sys

from rag.compare import run_comparison
from rag.config import get_settings
from rag.download_kb import download_knowledge_base
from rag.ingest import build_vector_store
from rag.query import ask


def download_kb(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download open Wikipedia articles into data/knowledge_base/"
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        help="Optional Wikipedia page titles (underscores or spaces)",
    )
    args = parser.parse_args(argv)
    download_knowledge_base(titles=args.titles)


def ingest(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chunk docs, embed them, and build a Chroma store"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "huggingface"],
        help="Override EMBEDDING_PROVIDER from .env",
    )
    args = parser.parse_args(argv)
    settings = get_settings(args.provider)
    build_vector_store(settings)


def query(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the vector store")
    parser.add_argument("question", type=str, help="Natural-language question")
    parser.add_argument(
        "--provider",
        choices=["openai", "huggingface"],
        help="Which embedding index to search (must already be ingested)",
    )
    args = parser.parse_args(argv)
    settings = get_settings(args.provider)
    result = ask(args.question, settings)
    print(f"Provider: {result.provider}")
    print(f"Scores:   {[round(s, 3) for s in result.scores]}")
    print(f"Sources:  {result.sources}")
    print("-" * 40)
    print(result.answer)


def compare(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare OpenAI vs HuggingFace embeddings / retrieval"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is retrieval-augmented generation?",
        help="Sample retrieval query",
    )
    args = parser.parse_args(argv)
    run_comparison(query=args.query)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Usage: python -m rag.cli <download-kb|ingest|query|compare> ...\n"
            "See README.md for the full learning path."
        )
        raise SystemExit(1)

    command, *rest = argv
    commands = {
        "download-kb": download_kb,
        "ingest": ingest,
        "query": query,
        "compare": compare,
    }
    if command not in commands:
        print(f"Unknown command: {command}")
        print("Available:", ", ".join(commands))
        raise SystemExit(1)
    commands[command](rest)


if __name__ == "__main__":
    main()
