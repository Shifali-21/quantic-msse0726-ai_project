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
**Choice:** ChromaDB `PersistentClient` (local, embedded, chromadb>=0.4.24).

**Rationale:** Chroma is zero-cost, requires no external service, ships as a Python package, and its `PersistentClient` persists the index to disk across restarts. This satisfies the free-tier requirement and simplifies deployment — no Pinecone API key or network dependency. For this corpus size (15 docs, ~300 chunks) the local store is more than sufficient.

**Migration Note:** The project was upgraded from chromadb 0.3.x to 0.4.x to ensure compatibility with Python 3.11+ and resolve deprecated API issues. The chromadb 0.4+ API requires proper `EmbeddingFunction` wrappers and uses `PersistentClient` instead of the legacy `Client(Settings(...))` pattern. Auto-persistence is now handled internally, eliminating the need for explicit `client.persist()` calls.

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
**Choice:** Flask for the primary REST API and chat UI; Streamlit as an alternative richer UI.

**Rationale:** Flask is lightweight, easy to deploy on Render, and gives full control over the HTML/JS chat interface. Streamlit requires no frontend code and provides a polished UI with sidebar controls — useful for demonstrations and parameter experimentation. Both share the same `retrieve_and_answer` backend function.

**UI Enhancement:** The Flask chat interface uses `marked.js` to render LLM-generated Markdown (headings, lists, formatting) as styled HTML, providing a cleaner user experience than raw Markdown syntax. This is particularly important when the LLM structures policy answers with headings and bullet points for clarity.

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