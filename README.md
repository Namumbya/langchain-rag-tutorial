# LangChain RAG Tutorial (v0.3)

A small **Retrieval-Augmented Generation (RAG)** project that teaches the full loop:

**documents → chunks → embeddings → vector store → retrieve → LLM answer**

You can compare:

- **Embeddings:** OpenAI (API) vs HuggingFace / sentence-transformers (local)
- **Answers:** OpenAI chat vs **Ollama** (local, no API key)

Knowledge base: **English Wikipedia** articles (open, CC BY-SA) about RAG and related ML topics.

This stays classic **2-step RAG** (retrieve, then generate). It is meant to teach plain RAG clearly — not agentic workflows.

---

## What you will learn

| Concept | Where it lives |
|--------|----------------|
| Environment & secrets | `.env.example` → `.env` |
| Settings / paths | `src/rag/config.py` |
| Embedding backends | `src/rag/embeddings.py` |
| Chat backends | `src/rag/llm.py` |
| Download open docs | `src/rag/download_kb.py` |
| Chunk + index | `src/rag/ingest.py` |
| Retrieval layer | `src/rag/retriever.py` |
| Ask questions | `src/rag/query.py` |
| Compare embeddings | `src/rag/compare.py` |
| Evaluate retrieval | `src/rag/evaluate.py` |
| CLI | `python -m rag.cli ...` |

### RAG in one picture

```text
┌─────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────┐
│  Markdown   │ →  │  Split   │ →  │  Embed     │ →  │   Chroma    │
│  Wikipedia  │    │ + title  │    │  vectors   │    │  vector DB  │
└─────────────┘    └──────────┘    └────────────┘    └──────┬──────┘
                                                           │
     ┌────────────┐    ┌──────────┐    ┌────────────┐      │
     │   Answer   │ ←  │   LLM    │ ←  │  Prompt +  │ ←────┘
     │ + sources  │    │ OpenAI / │    │  top chunks│  MMR or similarity
     └────────────┘    │  Ollama  │    └────────────┘
                       └──────────┘
```

Current LangChain APIs used here: `langchain_chroma`, `langchain_text_splitters`,
`langchain_openai` / `langchain_huggingface` / `langchain_ollama`, and `.invoke()`
(not deprecated `.predict()`).

---

## Setup

### Option A — uv (recommended)

```bash
uv sync --extra dev
```

### Option B — pip + venv

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

`requirements.txt` mirrors `pyproject.toml` for plain pip users. Prefer editable install from `pyproject.toml`.

### Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `LLM_PROVIDER` | Optional (default `openai`) | `openai` or `ollama` |
| `OPENAI_API_KEY` | For OpenAI embeddings **or** OpenAI answers | https://platform.openai.com/api-keys |
| `EMBEDDING_PROVIDER` | Optional (default `huggingface`) | `huggingface` or `openai` |
| `OLLAMA_CHAT_MODEL` | Optional | default `llama3.2` |
| `OLLAMA_BASE_URL` | Optional | default `http://localhost:11434` |
| `RETRIEVAL_SEARCH_TYPE` | Optional | `mmr` (default) or `similarity` |
| `RETRIEVAL_K` | Optional | number of chunks returned |
| `RETRIEVAL_FETCH_K` | Optional | candidate pool for MMR |
| `MIN_RELEVANCE_SCORE` | Optional | score threshold for similarity mode |

**Fully free / local path**

1. `EMBEDDING_PROVIDER=huggingface`
2. `LLM_PROVIDER=ollama`
3. Install [Ollama](https://ollama.com), then: `ollama pull llama3.2`

No OpenAI key needed for that path.

---

## Run the tutorial (in order)

### Step A — Download the open knowledge base

```bash
python -m rag.cli download-kb
```

### Step B — Build a vector index (HuggingFace, free/local)

```bash
python -m rag.cli ingest --provider huggingface
```

Chunks are prefixed with the article title before embedding (helps paraphrased questions).

### Step C — (Optional) Build an OpenAI index for comparison

```bash
python -m rag.cli ingest --provider openai
```

### Step D — Compare embeddings

```bash
python -m rag.cli compare --query "What is retrieval-augmented generation?"
```

Do **not** cosine-compare an OpenAI vector to a HuggingFace vector. Different spaces. Compare rankings / hit rates instead.

### Step E — Ask a question

```bash
# OpenAI answers (needs OPENAI_API_KEY)
python -m rag.cli query "What problem does RAG solve?" --provider huggingface

# Or fully local answers with Ollama
# set LLM_PROVIDER=ollama in .env first
python -m rag.cli query "What problem does RAG solve?" --provider huggingface
```

The CLI prints **numbered sources in the same order as the prompt context**, so `[1]` / `[2]` in the answer match the Sources list.

### Step F — Evaluate retrieval quality

Harder paraphrased questions + **hit@k** and **MRR**:

```bash
python -m rag.cli evaluate --provider huggingface
```

```bash
python -m rag.cli evaluate --provider openai
```

Use this after changing chunking, embeddings, or retrieval settings.

---

## Testing & CI

```bash
pip install -e ".[dev]"   # or: uv sync --extra dev

pytest
ruff check src tests
mypy src
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on every push/PR to `main`.

---

## Project layout

```text
.
├── .env.example
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── data/knowledge_base/
├── chroma_stores/          # generated (gitignored)
├── src/rag/
│   ├── config.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── download_kb.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── query.py
│   ├── compare.py
│   ├── evaluate.py
│   └── cli.py
└── tests/
```

---

## Intentionally out of scope (for now)

To keep this a clear RAG tutorial:

- No Streamlit UI yet
- No hybrid BM25 search / re-ranker chapter yet
- No agent-with-tools chapter (that belongs in a separate agent project)

---

## Troubleshooting

**`OPENAI_API_KEY is missing`**  
Needed only when `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai`. For free local answers, set `LLM_PROVIDER=ollama`.

**Ollama connection errors**  
Confirm Ollama is running and the model is pulled: `ollama list`, `ollama pull llama3.2`.

**`No knowledge base` / `Missing vector store`**  
Run `download-kb`, then `ingest --provider ...`.

**First HuggingFace ingest is slow**  
Normal — model weights download once.

---

## License note on the knowledge base

Wikipedia article text is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See `data/knowledge_base/ATTRIBUTION.md` after download.
