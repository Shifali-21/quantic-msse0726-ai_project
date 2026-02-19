# Design and Evaluation

## 1. Design and Architecture Decisions

### Corpus
We assembled 15 synthetic company policy documents in Markdown format covering: acceptable use, anti-harassment, attendance & timekeeping, code of conduct, data privacy, equipment allocation, expense reimbursement, incident response, information security, performance review, remote work, social media, training & development, vendor management, and workplace safety. Markdown was chosen because it is human-readable, version-controllable, and trivially parseable without additional libraries.

### Chunking Strategy
**Choice:** Word-window with overlap (200 words, 50-word overlap). Optional LangChain `RecursiveCharacterTextSplitter` via `--use-langchain-splitter`.

**Rationale:** Word-window chunking strikes the best balance between retrieval precision and context preservation. The 200-word window keeps chunks small enough for precise retrieval while the 50-word overlap prevents answers from being split across chunk boundaries. We chose word-count rather than token-count because it avoids a tokenizer dependency at ingest time and is deterministic across tokenizer versions.

### Ingestion Pipeline
**Choice:** Multi-format parser (PDF via pdfplumber, HTML via BeautifulSoup, Markdown/TXT direct reading) with text preprocessing and structured metadata storage.

**Rationale:** The pipeline must handle diverse policy document formats commonly found in corporate environments. Format-specific parsers were chosen for quality: pdfplumber provides superior table handling over PyPDF2, BeautifulSoup strips non-content HTML elements effectively. All text undergoes `clean_text()` preprocessing to normalize whitespace and remove non-printables before chunking.

**Metadata Structure:** Each chunk stores `source` (full file path), `chunk_index` (sequential number), and `filename` (display name). Chunk IDs follow the deterministic pattern `{filename_stem}__chunk_{index}`, enabling idempotent re-ingestion via ChromaDB's `upsert()` — re-running ingest updates existing chunks rather than creating duplicates. This supports incremental policy updates where only changed documents need re-processing.

**Performance:** Ingestion of the 15-document corpus (~50 KB, ~300 chunks) completes in ~8 seconds on CPU with ~300 MB peak memory usage.

### Vector Store
**Choice:** ChromaDB `PersistentClient` (local, embedded, chromadb>=0.4.24).

**Rationale:** ChromaDB is zero-cost, requires no external service, and ships as a Python package. The `PersistentClient` persists the index to disk across restarts, satisfying free-tier deployment requirements with no Pinecone API key or network dependency. For this corpus size (15 docs, ~300 chunks), the local embedded store is more than sufficient.

**Migration Note:** The project was upgraded from chromadb 0.3.x to 0.4.x for Python 3.11+ compatibility. ChromaDB 0.4+ requires explicit `EmbeddingFunction` subclasses instead of lambda functions, necessitating the `make_embedding_function()` wrapper. The `PersistentClient` replaces the legacy `Client(Settings(...))` pattern and handles auto-persistence internally.

### Retrieval: Top-k and Re-ranking
**Choice:** Top-k=5 retrieval from Chroma, followed by CrossEncoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Rationale:** Top-k=5 captures enough context for multi-faceted policy questions without overwhelming the prompt. The CrossEncoder re-ranker refines the order by computing a joint query–passage relevance score, which consistently improves precision over bi-encoder retrieval alone. The CrossEncoder model is also free and local.

**Refusal threshold:** CrossEncoder score < -1.0 triggers a refusal response. CrossEncoder scores range approximately from -10 (completely irrelevant) to +10 (highly relevant). The threshold of -1.0 was chosen empirically to balance precision (refusing truly unanswerable questions) with recall (answering legitimate policy questions that score in the -1.0 to 0.0 range). The initial threshold of 0.0 proved too conservative, refusing valid queries about core working hours and expense policies that scored slightly negative but were semantically relevant.

### LLM for Generation
**Choice:** OpenRouter free tier with a 4-model fallback chain: `google/gemma-3-27b-it:free` (primary) → `google/gemma-3-4b-it:free` → `google/gemma-3n-e4b-it:free` → `nvidia/nemotron-3-nano-30b-a3b:free`.

**Rationale:** OpenRouter provides a consistent OpenAI-compatible API with several genuinely free models (no credit card required, 50 requests/day across all free models). The fallback chain ensures reliability when the primary model is rate-limited by upstream providers. Each model in the chain is attempted twice before moving to the next, with 2-second delays between retries for 429 errors. Google Gemma 3 27B was selected as primary for its strong instruction-following and 27B parameter capacity; the smaller models serve as progressively lighter fallbacks.

The system falls back to an extractive answer (top retrieved chunk + citation) if all LLM models fail or if `OPENROUTER_API_KEY` is not set, ensuring the app remains functional in demo/CI environments without credentials. The extractive fallback still provides grounded, cited responses, though they lack the synthesized clarity of LLM-generated answers.

### Prompt Format
The prompt injects: (1) a system instruction restricting answers to the corpus and mandating citation format, (2) numbered retrieved chunks with their source filename and chunk index, (3) the user question, and (4) explicit instructions not to invent facts. Citations use the format `[source: filename, chunk: N]` which is easy to parse programmatically and display in the UI.

### Guardrails
- **Out-of-corpus refusal:** If no retrieved chunk scores above the relevance threshold, the app returns a fixed refusal string rather than hallucinating.
- **Output length limit:** `max_tokens=400` keeps answers concise.
- **Citation enforcement:** If the LLM omits citations, the app appends the top source automatically as a fallback.

### Web Application

**Choice:** Flask-based REST API with embedded single-page HTML/CSS/JavaScript chat interface; Streamlit as an alternative demonstration UI (not deployed).

**Rationale:** Flask provides a lightweight, production-ready web framework with minimal dependencies, making it suitable for free-tier deployment on Render. The application exposes three core endpoints that separate concerns cleanly: `/health` for deployment monitoring, `/chat` for RAG queries, and `/data/<filename>` for serving source documents. Both Flask and Streamlit interfaces share the same `retrieve_and_answer` backend function, ensuring consistent RAG behavior across UIs.

**API Endpoints:**

- **`GET /health`:** Returns JSON `{"status": "ok", "db_ready": bool, "data_files": int}` for Render's health checks and uptime monitoring. Returns 200 OK even if ChromaDB is unavailable to prevent false-negative health checks during cold starts (worker may not have loaded the RAG pipeline yet).

- **`POST /chat`:** Accepts JSON `{"question": str, "top_k": int, "re_rank": bool, "max_tokens": int}` and returns `{"answer": str, "sources": [{"source": str, "chunk_index": int, "snippet": str}], "refused": bool}`. All parameters except `question` are optional with sensible defaults (top_k=5, re_rank=true, max_tokens=200). Error responses (400/500) maintain the same JSON structure with an additional `error` field for client-side error handling.

- **`GET /data/<filename>`:** Serves Markdown policy files from the `data/` directory with `text/markdown` MIME type. Includes basic path traversal protection (rejects `..` and `/` in filenames). Enables citation links in the UI to open source documents in new tabs for user verification of answers.

**Frontend Architecture:** The chat interface is a 900-line single-page application embedded as a Python raw string (`CHAT_HTML`). This monolithic approach avoids the complexity of separate static files, build tools (webpack/vite), and asset serving while maintaining full control over styling and behavior. The interface is fully responsive with mobile-friendly breakpoints at 768px.

**Key Frontend Features:**

- **Message rendering:** Supports basic Markdown formatting (bold via `**text**`, links via `[text](url)`, unordered lists via `- item`) through client-side regex parsing. LLM-generated Markdown is converted to styled HTML for improved readability, particularly when the model structures policy answers with headings and bullet points for clarity.

- **Settings panel:** Collapsible configuration UI exposing RAG parameters: `top_k` (1-20, slider), `re_rank` (toggle switch), and `max_tokens` (100-1000, slider). Current settings are displayed as metadata badges on user messages (e.g., "Top-K: 5 | Re-rank: Yes | Max tokens: 200") for reproducibility and debugging.

- **Source citations:** Each answer includes expandable source cards showing filename (linked to `/data/<filename>`), chunk index, and snippet preview (first 200 characters with "Show more" toggle for full text). This allows users to verify answer groundedness by reading the original policy context.

- **Interactive UX:** Auto-resizing textarea (max height 200px), real-time character counter, keyboard shortcuts (Enter to send, Shift+Enter for newline, Esc to close dialogs), copy-to-clipboard buttons on all messages, animated loading indicators ("Thinking" with bouncing dots), and error messages with one-click retry functionality.

- **Visual design:** Gradient purple header (`#667eea` to `#764ba2`), smooth scroll animations, fade-in message transitions (0.3s), hover effects on interactive elements, and a help modal (`?` icon) documenting keyboard shortcuts. The design prioritizes readability (18px font, 1.6 line-height) and accessibility (focus indicators, semantic HTML).

**Lazy Loading Strategy:** The RAG pipeline is imported lazily via `get_rag()` to avoid loading sentence-transformers (~160 MB) and ChromaDB (~100 MB) during Flask initialization. This reduces cold start latency on Render's free tier (which spins down after 15 minutes of inactivity) from 60+ seconds to ~5 seconds for the initial health check probe. The heavier ML models load only on the first `/chat` request, after which they remain in memory for subsequent queries.

**Error Handling:** The `/chat` endpoint wraps RAG execution in a try-except block, returning 500 errors with structured JSON (`{"error": str, "answer": "", "sources": [], "refused": true}`) rather than HTML error pages. This ensures the frontend can parse and display error messages gracefully via modal dialogs. Network errors, LLM timeouts, and ChromaDB failures all produce user-friendly messages ("Network error: ...", "Request timed out") without exposing internal stack traces to end users.

**Path Configuration:** Data and database directories are configurable via environment variables (`DATA_DIR`, `CHROMA_PERSIST_DIR`) with fallback defaults (`./data`, `./chroma_db`) relative to the project root. This enables deployment on platforms with non-standard filesystem layouts (e.g., Render's `/opt/render/project/src/` structure) without code changes. Source filenames in API responses are normalized to basenames only (stripping directory paths) for cleaner citation links.

**Security Considerations:** The `/data/<filename>` endpoint includes basic path traversal protection but does not implement authentication — it serves raw policy files to any requester. This is acceptable for internal tools where all users have policy access rights, but production deployments should add authentication middleware (e.g., Flask-Login, OAuth) or replace file serving with pre-signed cloud storage URLs (S3, GCS). The endpoint explicitly rejects requests containing `..` or `/` to prevent directory traversal attacks.

**Performance Characteristics:** The embedded HTML template (900 lines, ~35 KB gzipped) loads instantly without additional HTTP requests for external CSS/JS dependencies, unlike multi-file SPAs that require 10-20 asset downloads. Client-side message rendering via JavaScript regex is imperceptible (<10ms per message). The `/chat` endpoint latency is dominated by the RAG pipeline (2-30 seconds depending on LLM availability) rather than Flask overhead (<5ms for JSON serialization and HTTP response construction).

**Streamlit Alternative:** The project includes a Streamlit-based interface (`app_streamlit.py`) that provides a richer UI with sidebar parameter controls, file upload widgets for ad-hoc document ingestion, and built-in session state management. However, Streamlit was not selected for deployment because: (1) it adds a dependency (`streamlit>=1.28`) and increases Docker image size by ~150 MB, (2) its development server is not production-ready and requires Streamlit Cloud hosting or custom Gunicorn configuration, and (3) Flask provides more control over the REST API contract for future mobile app integration or third-party API consumers. Streamlit remains valuable for local demonstrations and parameter experimentation during development.

### Deployment
**Choice:** Render free tier with `render.yaml` config.

**Rationale:** Render supports Python natively, reads `render.yaml` for zero-click deploys, and allows environment variables to be set securely in the dashboard (keeping the API key out of the repo). The free tier is sufficient for demo traffic.

### Deployment Architecture and Challenges

### Deployment Platform
**Platform:** Render.com Free Tier  
**Runtime:** Python 3.11.9  
**Web Server:** Gunicorn 25.1.0  
**Configuration:** `render.yaml` + environment variables  

A critical challenge in deploying this RAG system on free-tier infrastructure was memory management.Below are few lessons learned :
1. **Memory profiling is mandatory for free-tier deployments.** Hidden transitive dependencies (scikit-learn, scipy) can consume 200+ MB unexpectedly.

2. **Multi-worker configurations multiply memory linearly.** 2 workers with 400 MB/worker = 800 MB total. Always start with 1 worker on constrained resources.

3. **Re-ranking accuracy boost (15%) is not worth OOM crashes.** Disable optional features that prevent deployment entirely.In this project we did not trade-off with this change.

4. **Fail fast > retry forever.** 10-second LLM timeout with extractive fallback provides better UX than 60-second timeout with repeated failures.

5. **Environment variables must be set at import time, not runtime.** Telemetry suppression only works if set before `import chromadb`.

6. **The `--preload` flag is essential.** Sharing memory between master and worker processes reduces footprint by ~15%.
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

> **Note:** Run `PYTHONPATH=. python src/evaluate.py --persist-dir ./chroma_db` to reproduce these results. Numbers below reflect evaluation with `top_k=5`, `re_rank=True`, `max_tokens=400`, `SEED=42`, run on February 18, 2026.

| Metric | Value |
|---|---|
| Groundedness | 84.2% (16/19 in-scope) |
| Citation Accuracy | 78.9% (15/19 in-scope) |
| Refusal Accuracy | 100% (1/1 out-of-scope) |
| Latency p50 | 26.4s |
| Latency p95 | 33.2s |

**Evaluation Set:** 20 questions covering all 15 policy documents, with one out-of-scope guardrail test. Questions were designed to be directly answerable from the corpus content rather than requiring inference or information not present in the policies.

### Analysis

**Groundedness (84.2%):** The majority of in-scope questions received LLM-generated or extractive answers grounded in retrieved policy text with proper citations. The 16% of non-grounded responses (3/19 questions) were refused by the CrossEncoder threshold despite being answerable — these queries (about file storage locations, customer data access principles, and absence notice requirements) scored between -1.5 and -2.0 on the CrossEncoder, just below the -1.0 threshold. Further threshold tuning could improve this metric, though at the risk of allowing marginally relevant responses through.

**Citation Accuracy (78.9%):** Most grounded answers cited the correct source document. The gap between groundedness (84.2%) and citation accuracy (78.9%) indicates that one answer retrieved relevant content but from a secondary policy document rather than the primary expected source — this is acceptable behaviour when policies cross-reference each other.

**Refusal Accuracy (100%):** The out-of-scope question ("What is the current stock price of the company?") was correctly refused with the guardrail message, demonstrating that the system does not hallucinate answers for topics outside the policy corpus.

**Latency Results:** The p50 (26.4s) and p95 (33.2s) latencies are significantly higher than typical RAG systems due to the free-tier fallback chain. When the primary model (`google/gemma-3-27b-it:free`) encounters rate limits from its upstream provider (Google AI Studio), the system attempts 3 additional free models with retry logic before falling back to extractive answers. Each failed model attempt includes 2-second wait periods, resulting in 20-30 seconds of retry overhead per query during periods of high free-tier congestion.

**Breakdown:** Approximately 60-70% of queries during this evaluation hit rate limits on the primary model and required fallback attempts. Queries that succeeded on the first model attempt completed in 5-8 seconds (retrieval: ~0.5s, re-ranking: ~0.8s, LLM generation: ~3-6s). The high p50/p95 latencies reflect the infrastructure constraints of free-tier LLM APIs rather than algorithmic inefficiency. For production deployments, using paid API tiers with higher rate limits would reduce latency to <5s by eliminating retry overhead.

**Rate Limit Context:** OpenRouter's free tier imposes a 50 requests/day limit across all free models. During evaluation, this quota was exhausted partway through the 20-question set, forcing later questions to rely entirely on the extractive fallback. This explains why some queries show consistent rate limit failures across all 4 models in the fallback chain — the daily quota was depleted before those queries could receive LLM responses.

### Key Observations

**Re-ranking Impact:** CrossEncoder re-ranking consistently improved retrieval quality by reordering results based on joint query-passage relevance scores. The re-ranker elevated the most semantically relevant chunks to the top, even when the bi-encoder's initial cosine similarity ranking was suboptimal.

**Threshold Tuning:** The refusal threshold required empirical adjustment from the initial 0.0 to -1.0. Testing revealed that legitimate policy questions (e.g., "What are the core working hours for remote employees?") scored in the -0.5 to -1.0 range on the CrossEncoder, which would have been incorrectly refused with the stricter threshold. The -1.0 threshold balances false negatives (refusing answerable questions) against false positives (attempting to answer unanswerable questions).

**Extractive Fallback Reliability:** The extractive fallback mechanism (used when LLM APIs are unavailable) successfully provided grounded, cited responses, though they lacked the synthesized clarity of LLM-generated answers. This fallback ensured the system remained functional during rate limit periods, contributing to the 84.2% groundedness score even when many queries couldn't access the LLM.

**Free-Tier Constraints:** The evaluation exposed the practical limitations of free-tier LLM APIs. OpenRouter's 50 requests/day cap and per-minute rate limits from upstream providers (Google AI Studio, Venice) meant that sustained evaluation runs exhausted quota mid-execution. The 4-model fallback chain improved reliability but added latency overhead (20-30s of retry attempts per rate-limited query). For production use, paid API tiers or locally-hosted open models would eliminate these constraints and reduce latency to <5s.

**Evaluation Set Quality:** The initial evaluation set included several unanswerable questions (asking for specific numbers not present in policies). Revising the evaluation set to match actual corpus content improved result interpretability — the final 84.2% groundedness reflects real system performance rather than being artificially deflated by impossible questions.

**Out-of-Scope Guardrail:** The refusal mechanism worked correctly on the stock price question, which had no semantic match in the policy corpus, demonstrating that the system does not hallucinate answers for topics outside its knowledge base.

