# Production deployment (Render)

This guide covers deploying the Policy RAG app to [Render](https://render.com) so it is **publicly accessible** at a shareable URL.

---

## Overview

- **Platform:** Render (web service).
- **Config:** `render.yaml` defines build and start commands; environment variables (e.g. API keys, DB paths) are set in the Render Dashboard.
- **Public URL:** After deploy, Render provides a URL like `https://policy-rag-xxxx.onrender.com` for the chat UI, API, and health check.

---

## Environment variables

Configure these in the **Render Dashboard** → your service → **Environment**. **Never commit API keys** to the repo.

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | **Yes** (for LLM answers) | API key from [OpenRouter](https://openrouter.ai). Used by `src/rag.py` for generation. |
| `OPENROUTER_MODEL` | No | Model ID (default: `meta-llama/llama-3.3-70b-instruct:free`). |
| `DATA_DIR` | No | Path to policy documents (default: `./data`). Override if you mount data elsewhere. |
| `CHROMA_PERSIST_DIR` | No | Chroma DB path (default: `./chroma_db`). Set if using a different path. |
| `SEED` | No | Seed for reproducibility (default: `42`). Used by ingest and evaluation. |

- **Local development:** Put `OPENROUTER_API_KEY` (and optional `OPENROUTER_MODEL`) in a `.env` file in the project root; `python-dotenv` loads it. `.env` is in `.gitignore`.
- **Render:** In the Dashboard → **Environment**, add `OPENROUTER_API_KEY` as a **Secret**. Other vars have defaults in `render.yaml`; override in the dashboard if needed.

---

## Deploy steps (Render)

1. **Push your repo to GitHub** (or connect Render to your Git provider).

2. **Create a Web Service on Render:**
   - [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
   - Connect the repository. Render can detect `render.yaml` (Blueprint) or you can configure the service manually.

3. **If using Blueprint:** Deploy from the Blueprint; it will use `render.yaml` for build/start. Env var **keys** can be defined there; **secret values** (e.g. `OPENROUTER_API_KEY`) must be set in the Dashboard.

4. **Set the API key:** In the service → **Environment**, add:
   - Key: `OPENROUTER_API_KEY`
   - Value: your OpenRouter API key (mark as **Secret**).

5. **Deploy:** Trigger a deploy (manual or on push).  
   - **Build:** `pip install -r requirements.txt` then `python src/ingest.py --data-dir ./data --persist-dir ./chroma_db` (creates the vector DB).  
   - **Start:** `gunicorn src.app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`.

6. **Public URL:** After a successful deploy, Render shows the service URL (e.g. `https://policy-rag-xxxx.onrender.com`). This URL is **public and shareable**.  
   - **`/`** — Chat UI  
   - **`/chat`** — POST API (questions → answers with citations)  
   - **`/health`** — JSON health check  

---

## Summary

- **Configure env vars** in the Render Dashboard (especially `OPENROUTER_API_KEY`).
- **Build** installs dependencies and runs ingest; **start** runs Gunicorn. The app reads `DATA_DIR` and `CHROMA_PERSIST_DIR` from the environment when set.
- The **shareable URL** is the Render service URL; no extra step is required for public access.
