"""Download open-source Wikipedia articles into data/knowledge_base/.

Source: English Wikipedia via the public MediaWiki API (no API key).
License: Creative Commons Attribution-ShareAlike (CC BY-SA).
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

from rag.config import DEFAULT_WIKI_TITLES, DATA_DIR

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "langchain-rag-tutorial/0.2 (educational; local RAG demo)"


def _slugify(title: str) -> str:
    slug = title.replace(" ", "_")
    slug = re.sub(r"[^\w\-]+", "_", slug, flags=re.UNICODE)
    return slug.strip("_").lower()


def fetch_wikipedia_extract(title: str) -> tuple[str, str]:
    """Return (canonical_title, plain_text_extract) for a Wikipedia page."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "true",
        "exsectionformat": "wiki",
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    response = requests.get(
        API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or "extract" not in page:
        raise ValueError(f"Wikipedia page not found: {title}")
    return page["title"], page["extract"]


def write_article(title: str, extract: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_slugify(title)}.md"
    url_title = title.replace(" ", "_")
    content = (
        f"# {title}\n\n"
        f"> Source: [Wikipedia - {title}](https://en.wikipedia.org/wiki/{url_title})\n"
        f"> License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)\n\n"
        f"{extract.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def write_attribution(output_dir: Path, titles: list[str]) -> None:
    lines = [
        "# Knowledge base attribution",
        "",
        "These documents were downloaded from English Wikipedia using the",
        "public MediaWiki API for educational use in this RAG tutorial.",
        "",
        "## License",
        "",
        "Wikipedia text is available under the",
        "[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)",
        "license. You must credit Wikipedia/authors and share alike if you remix.",
        "",
        "## Articles",
        "",
    ]
    for title in titles:
        wiki = title.replace(" ", "_")
        lines.append(f"- [{title}](https://en.wikipedia.org/wiki/{wiki})")
    lines.append("")
    (output_dir / "ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")


def download_knowledge_base(
    titles: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    titles = titles or DEFAULT_WIKI_TITLES
    output_dir = output_dir or DATA_DIR
    saved: list[Path] = []
    resolved_titles: list[str] = []

    for title in titles:
        canonical, extract = fetch_wikipedia_extract(title)
        path = write_article(canonical, extract, output_dir)
        saved.append(path)
        resolved_titles.append(canonical)
        print(f"Saved {path.name} ({len(extract)} chars)")

    write_attribution(output_dir, resolved_titles)
    print(f"Wrote attribution -> {output_dir / 'ATTRIBUTION.md'}")
    return saved
