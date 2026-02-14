# Policy RAG — README

A RAG (Retrieval-Augmented Generation) pipeline over company policy documents: ingest → embed → store in a vector DB → retrieve (with optional re-ranking) → generate answers with citations. Includes a web chat UI and a REST API.

---

## 1. Environment and Reproducibility

### Virtual environment

Create and activate a virtual environment (e.g. venv or conda):

```bash
# venv (recommended)
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# optional: conda
# conda create -n policy-rag python=3.11 && conda activate policy-rag
```

### Dependencies

- **requirements.txt** (pip): all dependencies with versions where needed for reproducibility.

```bash
pip install --upgrade pip
pip install -r requirements.txt
# for Streamlit UI:
pip install streamlit
```

- Optional: use **environment.yml** for conda (not included; `requirements.txt` is the source of truth).

### Fixed seeds

- **Ingestion:** Seeds are set for deterministic chunking and embedding order. Use `SEED` env or `--seed` (default `42`) in `src/ingest.py`. See `src/utils.py` (`set_seeds`) for Python, NumPy, and Torch.
- **Evaluation / sampling:** If you add evaluation or sampling scripts, set the same seed there for reproducibility.

---

## 2. Ingestion and Indexing

- **Parse & clean:** Supports **PDF, HTML, .md, .txt**. Parsing in `src/ingest.py` (`parse_pdf`, `parse_html`, `parse_markdown_or_txt`); cleaning in `src/utils.py` (`clean_text`).
- **Chunking:** By headings first, then token-style word windows with overlap. Optional LangChain `RecursiveCharacterTextSplitter` via `--use-langchain-splitter` (see `src/utils.py`: `chunk_text`, `chunk_text_with_langchain`).
- **Embeddings:** Free local model **sentence-transformers** (`all-MiniLM-L6-v2`) in `src/ingest.py`. No API key required.
- **Vector store:** **Chroma** (local, duckdb+parquet) by default. Persist directory: `./chroma_db`. You can swap to a cloud store (e.g. Pinecone) by changing the client in `src/ingest.py` and the retrieval code in `src/rag.py`.

### Build the vector DB

```bash
# from project root, with venv activated
python3 -u src/ingest.py --data-dir ./data --persist-dir ./chroma_db --chunk-size 200 --overlap 50

# optional: LangChain splitter
python3 -u src/ingest.py --data-dir ./data --persist-dir ./chroma_db --chunk-size 200 --overlap 50 --use-langchain-splitter
```

- Reset DB: `rm -rf ./chroma_db` then re-run ingest.
- **Seeds:** `SEED=42 python3 -u src/ingest.py ...` or `--seed 42` for reproducibility.

---

## 3. Retrieval and Generation (RAG)

- **Frameworks:** Implemented in `src/rag.py` (optional LangChain in `src/langchain_rag.py` if present). Retrieval, prompt building, and generation can be used with or without LangChain.
- **Top-k retrieval** with **optional re-ranking** (CrossEncoder): `src/rag.py` (`retrieve_top_k`, `rerank_with_crossencoder`).
- **Prompting:** Retrieved chunks and instructions are injected into the LLM context; model is instructed to cite sources (see `build_prompt` in `src/rag.py`).
- **Guardrails:**
  - Refuse answers outside the corpus: *"I can only answer questions about the provided policies. I don't have information on that."*
  - Output length limited via `max_tokens` (default 200).
  - Answers cite source doc IDs/titles and chunk index, e.g. `[source: filename, chunk_index]`.

### CLI

```bash
# RAG with re-ranking (no LLM generation; extractive + citations)
python3 src/rag.py --query "How do I request equipment?" --persist-dir ./chroma_db --re-rank

# With optional generation model
python3 src/rag.py --query "What is the remote work policy?" --persist-dir ./chroma_db --gen-model google/flan-t5-small --max-tokens 200
```

---

## 4. Web Application

You can use **Flask** or **Streamlit**.

### Flask (`src/app.py`)

- **`/`** — Web chat interface: text box for user input; submits to `/chat` and shows answer with source links and snippets.
- **`/chat`** — API endpoint: **POST** JSON `{"question": "..."}`. Returns JSON: `{"answer": "...", "sources": [{"source": "file.md", "chunk_index": 0, "snippet": "..."}], "refused": false}`. Optional body keys: `top_k`, `re_rank`, `max_tokens`, `gen_model`.
- **`/health`** — **GET** returns JSON status, e.g. `{"status": "ok", "db_ready": true, "data_files": 15}`.

Run Flask from project root:

```bash
source .venv/bin/activate
# Option A
export FLASK_APP=src.app:app && flask run --host 127.0.0.1 --port 5000

# Option B
python -c "from src.app import app; app.run(host='127.0.0.1', port=5000)"
```

Then open **http://127.0.0.1:5000** for the chat UI and **http://127.0.0.1:5000/health** for health. Source links on the chat page point to **/data/<filename>** (served from `data/`).

### Streamlit (`src/app_streamlit.py`)

- **`/`** — Full-page chat UI: text area, Top-K / Re-rank / model / max tokens in sidebar; answers with citations and snippets; “Open” to view full source document from `data/`.
- **Health:** When Streamlit runs, an embedded Flask server on port **8001** exposes **GET /health** (JSON).

Run Streamlit from project root:

```bash
source .venv/bin/activate
streamlit run src/app_streamlit.py
```

Open **http://localhost:8501**. Health: **http://127.0.0.1:8001/health**.

---

## Quick reference

| Item | Location |
|------|----------|
| Venv + deps | `.venv`, `requirements.txt` |
| Seeds | `src/utils.py` (`set_seeds`), `src/ingest.py` (`--seed`, `SEED`) |
| Parse/clean/chunk | `src/ingest.py`, `src/utils.py` |
| Embed + Chroma | `src/ingest.py` |
| Top-k + re-rank + prompt + guardrails | `src/rag.py` |
| Flask: `/`, `/chat`, `/health` | `src/app.py` |
| Streamlit UI + embedded health | `src/app_streamlit.py` |
| Policy documents | `data/*.md` |

---

## Testing

```bash
source .venv/bin/activate
pip install pytest
PYTHONPATH=. pytest tests/ -q
```

**CI:** A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR to `main`/`master`: installs dependencies, runs an import check (`from src.app import app`), and runs `pytest -q`.

---

## Troubleshooting

- **Python 3.13 / Chroma / Pydantic:** Use Python 3.11 and versions in `requirements.txt`.
- **Duplicate IDs on re-ingest:** Delete `./chroma_db` and re-run ingest, or add update/upsert logic in `src/ingest.py`.
- **“No RAG backend”:** Ensure `src/rag.py` (and optionally `src/langchain_rag.py`) and dependencies are installed; run from project root so `src` is importable.
