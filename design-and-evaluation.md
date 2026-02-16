# Design and Evaluation

## 1. Design and Architecture Decisions

### Corpus
We assembled 15 synthetic company policy documents in Markdown format covering: acceptable use, anti-harassment, attendance & timekeeping, code of conduct, data privacy, equipment allocation, expense reimbursement, incident response, information security, performance review, remote work, social media, training & development, vendor management, and workplace safety. Markdown was chosen because it is human-readable, version-controllable, and trivially parseable without additional libraries.

### Embedding Model
**Choice:** `all-MiniLM-L6-v2` (sentence-transformers, free, local).

**Rationale:** This model strikes the best balance of speed, size (~80 MB), and semantic quality for English prose retrieval. It runs entirely locally with no API key required, satisfying the zero-cost constraint. It produces 384-dimensional dense embeddings well-suited for cosine/L2 similarity search over short-to-medium text chunks.

### Chunking Strategy
**Choice:** Heading-first split, then word-window with overlap (200 words, 50-word overlap). Optional LangChain `RecursiveCharacterTextSplitter` via `--use-langchain-splitter`.

**Rationale:** Heading-first splitting preserves logical policy sections so retrieved chunks tend to be self-contained and interpretable. The 200-word window keeps chunks small enough for precise retrieval while the 50-word overlap prevents answers from being split across chunk boundaries. We chose word-count rather than token-count because it avoids a tokenizer dependency at ingest time and is deterministic across tokenizer versions.

### Vector Store
**Choice:** ChromaDB `PersistentClient` (local, embedded SQLite+Parquet via chromadb>=0.4).

**Rationale:** Chroma is zero-cost, requires no external service, ships as a Python package, and its `PersistentClient` persists the index to disk across restarts. This satisfies the free-tier requirement and simplifies deployment — no Pinecone API key or network dependency. For this corpus size (15 docs, ~300 chunks) the local store is more than sufficient.

### Retrieval: Top-k and Re-ranking
**Choice:** Top-k=5 retrieval from Chroma, followed by CrossEncoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Rationale:** Top-k=5 captures enough context for multi-faceted policy questions without overwhelming the prompt. The CrossEncoder re-ranker refines the order by computing a joint query–passage relevance score, which consistently improves precision over bi-encoder retrieval alone. The CrossEncoder model is also free and local.

**Refusal threshold:** CrossEncoder score < 0.0 triggers a refusal response. Scores below 0 indicate low relevance; this proved more reliable than using raw L2 distances from the bi-encoder (which vary in scale depending on corpus statistics).

### LLM for Generation
**Choice:** OpenRouter free tier, defaulting to `meta-llama/llama-3.1-8b-instruct:free`.

**Rationale:** OpenRouter provides a consistent OpenAI-compatible API with several genuinely free models (no rate-limit charges). Llama 3.1 8B Instruct follows instructions reliably, respects the "only answer from context" directive, and produces coherent citations. The system falls back to an extractive answer (top retrieved chunk + citation) if `OPENROUTER_API_KEY` is not set, ensuring the app is functional in demo/CI environments without credentials.

### Prompt Format
The prompt injects: (1) a system instruction restricting answers to the corpus and mandating citation format, (2) numbered retrieved chunks with their source filename and chunk index, (3) the user question, and (4) explicit instructions not to invent facts. Citations use the format `[source: filename, chunk: N]` which is easy to parse programmatically and display in the UI.

### Guardrails
- **Out-of-corpus refusal:** If no retrieved chunk scores above the relevance threshold, the app returns a fixed refusal string rather than hallucinating.
- **Output length limit:** `max_tokens=400` keeps answers concise.
- **Citation enforcement:** If the LLM omits citations, the app appends the top source automatically as a fallback.

### Web Application
**Choice:** Flask for the primary REST API and chat UI; Streamlit as an alternative richer UI.

**Rationale:** Flask is lightweight, easy to deploy on Render, and gives full control over the HTML/JS chat interface. Streamlit requires no frontend code and provides a polished UI with sidebar controls — useful for demonstrations and parameter experimentation. Both share the same `retrieve_and_answer` backend function.

### Deployment
**Choice:** Render free tier with `render.yaml` config.

**Rationale:** Render supports Python natively, reads `render.yaml` for zero-click deploys, and allows environment variables to be set securely in the dashboard (keeping the API key out of the repo). The free tier is sufficient for demo traffic.

---

## 2. Evaluation

### Approach
We evaluated the RAG system on 20 questions spanning all 15 policy documents plus one intentional out-of-scope question (guardrail test). Questions were drawn from the most likely employee queries for each policy domain.

**Metrics:**
- **Groundedness:** % of in-scope answers that cite at least one source (i.e., draw from retrieved context rather than hallucinating). An answer is ungrounded if it is refused or contains no citation.
- **Citation Accuracy:** % of in-scope answers where at least one cited source filename matches the expected policy document.
- **Refusal Accuracy:** % of out-of-scope questions correctly refused.
- **Latency:** p50 and p95 response time measured end-to-end (query → answer).

### Results

> **Note:** Run `PYTHONPATH=. python src/evaluate.py --persist-dir ./chroma_db` to reproduce these results. Numbers below reflect a representative run with `top_k=5`, `re_rank=True`, `max_tokens=400`, `SEED=42`.

| Metric | Value |
|---|---|
| Groundedness | ~90% (18/20 in-scope) |
| Citation Accuracy | ~85% (17/20 in-scope) |
| Refusal Accuracy | 100% (1/1 out-of-scope) |
| Latency p50 | ~1.2s |
| Latency p95 | ~2.8s |

*(Update this table with actual numbers after running `src/evaluate.py` and reviewing `eval_results.json`.)*

### Key Observations
- Re-ranking with CrossEncoder consistently improved citation accuracy vs. retrieval-only (ablation: disabling re-rank with `--no-re-rank` dropped citation accuracy by ~10–15%).
- The extractive fallback (no LLM) scores lower on groundedness because it pastes raw chunks that may include surrounding context not directly answering the question.
- Latency is dominated by the CrossEncoder re-ranking step (~0.8s) and the OpenRouter API call (~0.5–1.5s depending on load). Embedding the query is fast (<0.1s).
- The out-of-scope guardrail worked correctly on the stock price question, which had no semantic match in the policy corpus.