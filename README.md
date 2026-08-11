# LangChain RAG Tutorial (v0.2)

A small, modern **Retrieval-Augmented Generation (RAG)** project that teaches the full loop:

**documents → chunks → embeddings → vector store → retrieve → LLM answer**

You can switch between **OpenAI embeddings** (API) and **HuggingFace / sentence-transformers** (local) and compare them.

Knowledge base: **English Wikipedia** articles (open, CC BY-SA) about RAG and related ML topics.

---

## What you will learn

| Concept | Where it lives |
|--------|----------------|
| Environment & secrets | `.env.example` → `.env` |
| Settings / paths | `src/rag/config.py` |
| Embedding backends | `src/rag/embeddings.py` |
| Download open docs | `src/rag/download_kb.py` |
| Chunk + index | `src/rag/ingest.py` |
| Ask questions | `src/rag/query.py` |
| Compare providers | `src/rag/compare.py` |
| CLI | `python -m rag.cli ...` |

### RAG in one picture

```text
┌─────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────┐
│  Markdown   │ →  │  Split   │ →  │  Embed     │ →  │   Chroma    │
│  Wikipedia  │    │  chunks  │    │  vectors   │    │  vector DB  │
└─────────────┘    └──────────┘    └────────────┘    └──────┬──────┘
                                                           │
     ┌────────────┐    ┌──────────┐    ┌────────────┐      │
     │   Answer   │ ←  │   LLM    │ ←  │  Prompt +  │ ←────┘
     │            │    │ (OpenAI) │    │  top chunks│  similarity search
     └────────────┘    └──────────┘    └────────────┘
```

Embeddings answer: *“which chunks mean something close to this question?”*  
The LLM answers: *“given those chunks, what should I say?”*

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. Install the package (editable)

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure environment variables

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env`:

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `OPENAI_API_KEY` | For OpenAI embeddings **and** for answering questions | https://platform.openai.com/api-keys |
| `EMBEDDING_PROVIDER` | Optional (default `huggingface`) | `huggingface` or `openai` |
| `OPENAI_EMBEDDING_MODEL` | Optional | default `text-embedding-3-small` |
| `HF_EMBEDDING_MODEL` | Optional | default `sentence-transformers/all-MiniLM-L6-v2` |
| `OPENAI_CHAT_MODEL` | Optional | default `gpt-4o-mini` |

**Keys you need:**

- **HuggingFace local embeddings:** no key. First ingest downloads the model weights.
- **OpenAI embeddings or chat answers:** `OPENAI_API_KEY`.

Optional LangSmith tracing vars are documented in `.env.example`.

---

## Run the tutorial (in order)

### Step A — Download the open knowledge base

```bash
python -m rag.cli download-kb
```

This pulls curated Wikipedia pages (RAG, LLMs, embeddings, vector DBs, …) into `data/knowledge_base/` and writes `ATTRIBUTION.md` (CC BY-SA).

Custom pages:

```bash
python -m rag.cli download-kb --titles "Chroma_(vector_database)" "LangChain"
```

### Step B — Build a vector index (HuggingFace, free/local)

```bash
python -m rag.cli ingest --provider huggingface
```

Creates `chroma_stores/huggingface/`.

### Step C — (Optional) Build an OpenAI index for comparison

Requires `OPENAI_API_KEY`:

```bash
python -m rag.cli ingest --provider openai
```

Creates `chroma_stores/openai/`.

Each provider gets its **own** Chroma folder because vectors from different models are **not** interchangeable.

### Step D — Compare embeddings

```bash
python -m rag.cli compare --query "What is retrieval-augmented generation?"
```

You will see:

1. **Pairwise cosine similarity** inside each model’s space (related phrases score high; unrelated score low).
2. **Retrieval previews** from each ingested store.

Important lesson: do **not** cosine-compare an OpenAI vector to a HuggingFace vector. Different spaces. Compare rankings / retrieval quality instead.

### Step E — Ask a question

Chat generation uses OpenAI (`OPENAI_API_KEY` required), even if retrieval used HuggingFace embeddings:

```bash
python -m rag.cli query "What problem does RAG solve?" --provider huggingface
```

```bash
python -m rag.cli query "What is a vector database?" --provider openai
```

---

## Project layout

```text
.
├── .env.example          # template for secrets / settings
├── pyproject.toml        # package + dependencies
├── requirements.txt      # pip-friendly mirror
├── README.md
├── data/
│   └── knowledge_base/   # Wikipedia markdown (open license)
├── chroma_stores/        # generated indexes (gitignored)
│   ├── huggingface/
│   └── openai/
└── src/
    └── rag/
        ├── config.py
        ├── embeddings.py
        ├── download_kb.py
        ├── ingest.py
        ├── query.py
        ├── compare.py
        └── cli.py
```

---

## OpenAI vs HuggingFace embeddings

| | OpenAI | HuggingFace (local) |
|--|--------|---------------------|
| Cost | Paid API | Free after download |
| Privacy | Text sent to API | Stays on your machine |
| Setup | `OPENAI_API_KEY` | Downloads model once |
| Default model | `text-embedding-3-small` (1536-d) | `all-MiniLM-L6-v2` (384-d) |
| Typical use | Strong baseline quality | Offline demos / private data |

For this tutorial, start with **HuggingFace** for ingest, then add OpenAI and run `compare`.

---

## Standard practices used here

- **Secrets in `.env`**, never committed (`.gitignore` includes `.env`)
- **`.env.example`** documents required keys without real secrets
- **Src layout** (`src/rag`) + editable install
- **One Chroma path per embedding provider**
- **Modern LangChain imports** (`langchain_chroma`, `langchain_text_splitters`, `invoke` not `predict`)
- **Lightweight markdown loading** (no heavy document parsers)
- **Attribution** for CC BY-SA Wikipedia content
- **CLI entrypoints** instead of loose scripts at the repo root

---

## Troubleshooting

**`OPENAI_API_KEY is missing`**  
Copy `.env.example` → `.env` and set a real key. Needed for OpenAI embeddings and for `query`.

**`No knowledge base`**  
Run `python -m rag.cli download-kb` first.

**`Missing vector store`**  
Run `python -m rag.cli ingest --provider ...` for that provider.

**First HuggingFace ingest is slow**  
Normal — `sentence-transformers` downloads model weights once.

**Windows + Chroma / onnxruntime issues**  
Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if wheel builds fail, then reinstall deps.

---

## License note on the knowledge base

Wikipedia article text is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See `data/knowledge_base/ATTRIBUTION.md` after download.
