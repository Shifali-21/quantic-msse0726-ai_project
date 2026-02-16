"""
Streamlit chat UI for Policy RAG.
Run: streamlit run src/app_streamlit.py  (from project root)
Health endpoint available at http://127.0.0.1:8001/health
"""
import os
import sys
import pathlib
import streamlit as st
from flask import Flask, jsonify
from threading import Thread

# Ensure src modules are importable when run from project root
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA_DIR = str(ROOT / "data")
DB_DIR = str(ROOT / "chroma_db")

# Import RAG backend
try:
    from rag import retrieve_and_answer
    retrieve_backend = "rag"
except Exception as e:
    retrieve_backend = None
    _import_error = str(e)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Policy RAG Chat", layout="centered")
st.title("Policy Assistant")

if retrieve_backend is None:
    st.error(
        f"RAG backend unavailable: {_import_error}\n\n"
        "Ensure src/rag.py is present and dependencies are installed."
    )
    st.stop()

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    top_k = st.number_input("Top-K retrieved chunks", min_value=1, max_value=20, value=5, step=1)
    re_rank = st.checkbox("Re-rank with Cross-Encoder", value=True)
    max_tokens = st.number_input("Max answer tokens (approx)", min_value=50, max_value=1000, value=400, step=50)
    st.markdown("---")
    st.caption("LLM: OpenRouter (set `OPENROUTER_API_KEY` in `.env`). Falls back to extractive answer if key absent.")
    st.markdown("---")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()  # fixed: was st.experimental_rerun() which is removed in Streamlit >=1.37
    st.markdown("Run with: `streamlit run src/app_streamlit.py`")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Backend call ──────────────────────────────────────────────────────────────
def call_backend(query: str):
    res = retrieve_and_answer(
        query,
        persist_dir=DB_DIR,
        top_k=top_k,
        re_rank=re_rank,
        max_tokens=max_tokens,
    )
    answer = res.get("answer", "") if isinstance(res, dict) else str(res)
    sources = res.get("sources", []) if isinstance(res, dict) else []
    # Normalize sources to consistent dict shape
    norm_sources = []
    for s in sources:
        if isinstance(s, str):
            parts = s.split("::chunk_")
            norm_sources.append({
                "source": parts[0],
                "chunk_index": parts[1] if len(parts) > 1 else "n/a",
                "snippet": "",
            })
        elif isinstance(s, dict):
            norm_sources.append({
                "source": s.get("source", "unknown"),
                "chunk_index": s.get("chunk_index", "n/a"),
                "snippet": s.get("snippet", ""),
            })
    return answer, norm_sources


# ── Query form ────────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=False):
    q = st.text_area("Ask about company policies", value="", height=120,
                     placeholder="e.g. What is the remote work policy?")
    submitted = st.form_submit_button("Ask")
    if submitted and q.strip():
        with st.spinner("Retrieving and generating answer..."):
            try:
                answer, sources = call_backend(q.strip())
            except Exception as e:
                answer = f"Backend error: {e}"
                sources = []
        st.session_state.history.append({"q": q.strip(), "answer": answer, "sources": sources})


# ── Render conversation history ───────────────────────────────────────────────
for hist_idx, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item.get('answer', '')}")
    sources = item.get("sources") or []
    if sources:
        st.markdown("**Sources:**")
        for s_i, s in enumerate(sources):
            fname = s.get("source") or "unknown"
            chunk_idx = s.get("chunk_index", "n/a")
            snippet = s.get("snippet") or ""
            col1, col2 = st.columns([6, 1])
            with col1:
                st.write(f"- **{fname}** (chunk {chunk_idx})")
                if snippet:
                    st.caption(snippet)
            with col2:
                src_path = os.path.join(DATA_DIR, fname)
                key = f"open_{hist_idx}_{s_i}_{fname}"
                if os.path.exists(src_path):
                    if st.button("Open", key=key):
                        with open(src_path, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        st.code(content, language="markdown")
                else:
                    st.caption("File not found")
    st.markdown("---")


# ── Embedded health server (daemon thread) ────────────────────────────────────
_health_started = False

def _start_health_server(port: int = 8001):
    global _health_started
    if _health_started:
        return
    _health_started = True
    _app = Flask("health")

    @_app.route("/health", methods=["GET"])
    def health():
        try:
            files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
            db_exists = os.path.isdir(DB_DIR)
            return jsonify({"status": "ok", "data_files": len(files), "db_ready": db_exists})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    t = Thread(
        target=lambda: _app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()


_start_health_server(port=8001)