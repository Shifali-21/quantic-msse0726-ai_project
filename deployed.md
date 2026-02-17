# Deployed Application

The Policy RAG application is publicly deployed on Render (free tier).

**Live URL:** https://quantic-msse0726-ai-project.onrender.com

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web chat UI |
| `/chat` | POST | API — send `{"question": "..."}`, returns answer with citations |
| `/health` | GET | Health check — returns `{"status": "ok", "db_ready": true}` |

> The Render free tier spins down after 15 minutes of inactivity. The first request after inactivity may take 30–60 seconds to respond while the server wakes up.