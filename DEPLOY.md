# Production Deployment (Render)

This guide covers the deployment of the Policy RAG app to [Render](https://render.com), where it is **publicly accessible** at a shareable URL.

---

## Live Deployment

| Item | Details |
|------|---------|
| **Platform** | Render (Web Service, free tier) |
| **Live URL** | https://quantic-msse0726-ai-project.onrender.com |
| **Chat UI** | https://quantic-msse0726-ai-project.onrender.com/ |
| **Chat API** | https://quantic-msse0726-ai-project.onrender.com/chat |
| **Health check** | https://quantic-msse0726-ai-project.onrender.com/health |

> **Note:** The Render free tier spins down after 15 minutes of inactivity. The first request after a period of inactivity may take 30–60 seconds while the server wakes up. This is expected behaviour on the free tier.

---

## Overview

- **Platform:** Render (Web Service).
- **Config:** `render.yaml` in the project root defines the build and start commands. Environment variables (e.g. API keys, DB paths) are set in the Render Dashboard — never committed to the repo.
- **LLM:** OpenRouter free tier using `google/gemma-3-27b-it:free` as the primary model, with a fallback chain of `google/gemma-3-4b-it:free`, `google/gemma-3n-e4b-it:free`, and `nvidia/nemotron-3-nano-30b-a3b:free`.
- **Vector DB:** ChromaDB `PersistentClient` built at deploy time by running `src/ingest.py` as part of the build command.
- **Web framework:** Flask served via Gunicorn in production.

---

## Environment Variables

Configure these in the **Render Dashboard** → your service → **Environment**. **Never commit API keys** to the repo.

| Variable | Required | Value Used | Description |
|----------|----------|------------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | Secret | API key from [OpenRouter](https://openrouter.ai). Used by `src/rag.py` for LLM generation. |
| `OPENROUTER_MODEL` | No | `google/gemma-3-27b-it:free` | Primary free model. Falls back to other free models if rate limited. |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Chroma DB path. Built at deploy time by ingest step. |
| `DATA_DIR` | No | `./data` | Path to policy markdown documents. |
| `SEED` | No | `42` | Reproducibility seed for ingest and evaluation. |

- **Local development:** Add `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` to a `.env` file in the project root. `python-dotenv` loads it automatically. `.env` is gitignored — never commit it.
- **Render:** In the Dashboard → **Environment**, add `OPENROUTER_API_KEY` as a **Secret**. All other variables have sensible defaults in `render.yaml`.

---

## Deploy Steps

### 1. Push repo to GitHub
Ensure the latest code including `render.yaml` is on the `main` branch.

### 2. Create a Web Service on Render
- Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**
- Connect the GitHub repository
- Render detects `render.yaml` automatically (Blueprint deploy)

### 3. Set the API key
In the service → **Environment** tab:
- Key: `OPENROUTER_API_KEY`
- Value: your OpenRouter API key
- Mark as **Secret**

### 4. Trigger the deploy
Render runs the following from `render.yaml`:

**Build command:**
```bash
pip install -r requirements.txt && python src/ingest.py --data-dir ./data --persist-dir ./chroma_db
```
Installs all dependencies and builds the Chroma vector DB from the 15 policy documents in `./data`.

**Start command:**
```bash
gunicorn src.app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```
Starts the Flask app via Gunicorn. `$PORT` is automatically assigned by Render.

### 5. Verify the deployment
Once the deploy shows green in the Render Dashboard, verify all three endpoints:

```
GET  https://quantic-msse0726-ai-project.onrender.com/health
→ {"status": "ok", "db_ready": true, "data_files": 15}

GET  https://quantic-msse0726-ai-project.onrender.com/
→ Chat UI loads in browser

POST https://quantic-msse0726-ai-project.onrender.com/chat
     {"question": "What is the remote work policy?"}
→ {"answer": "...", "sources": [...], "refused": false}
```

---

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:
- Installs dependencies
- Runs the Flask import check
- Runs `pytest tests/ -q`

On a successful merge to `main`, Render automatically redeploys the latest version via its GitHub integration.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Slow first response | Render free tier spin-down | Wait 30–60s for server to wake up |
| `db_ready: false` in health | Ingest didn't run at build time | Check Render build logs for ingest errors |
| LLM returning extractive fallback | `OPENROUTER_API_KEY` not set | Add key in Render Dashboard → Environment |
| 429 rate limit errors | Free model congestion | App automatically falls back to next model in chain |
| Build timeout | Heavy dependencies (torch, sentence-transformers) | Render free tier has a 15-min build limit; retry if it times out |