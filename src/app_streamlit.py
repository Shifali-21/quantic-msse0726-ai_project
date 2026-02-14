import os
import sys
import pathlib
import streamlit as st
from flask import Flask, jsonify
from threading import Thread

# ensure src modules importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA_DIR = str(ROOT / "data")
DB_DIR = str(ROOT / "chroma_db")

# Prefer LangChain RAG if available, else use built-in rag.retrieve_and_answer
retrieve_backend = None
try:
    from langchain_rag import retrieve_and_answer
    retrieve_backend = "langchain"
except Exception:
    try:
        from rag import retrieve_and_answer
        retrieve_backend = "rag"
    except Exception:
        retrieve_backend = None

st.set_page_config(page_title="Policy RAG Chat", layout="centered")

st.title("Policy Assistant (Streamlit)")

if retrieve_backend is None:
    st.error("No RAG backend available. Ensure src/langchain_rag.py or src/rag.py is present and dependencies installed.")
    st.stop()

with st.sidebar:
    st.header("Options")
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5, step=1)
    re_rank = st.checkbox("Re-rank (Cross-Encoder)", value=False)
    gen_model = st.text_input("Generation model (optional)", value="", help="e.g., google/flan-t5-small; leave empty for extractive fallback")
    max_tokens = st.number_input("Max tokens (approx)", min_value=50, max_value=1000, value=200, step=10)

if "history" not in st.session_state:
    st.session_state.history = []

def call_backend(query: str):
    # retrieve_and_answer (rag or langchain_rag) returns dict: answer, sources, refused
    gen = gen_model.strip() if gen_model else None
    if retrieve_backend == "langchain":
        res = retrieve_and_answer(query, persist_dir=DB_DIR, top_k=top_k, gen_model=(gen or "google/flan-t5-small"), max_tokens=max_tokens, device=-1)
    else:
        res = retrieve_and_answer(query, persist_dir=DB_DIR, top_k=top_k, re_rank=re_rank, gen_model=gen, max_tokens=max_tokens)
    answer = res.get("answer", "") if isinstance(res, dict) else str(res)
    sources = res.get("sources", []) if isinstance(res, dict) else []
    norm_sources = []
    for s in sources:
        if isinstance(s, str):
            parts = s.split("::chunk_")
            fn = parts[0]
            idx = parts[1] if len(parts) > 1 else "n/a"
            norm_sources.append({"source": fn, "chunk_index": idx, "snippet": ""})
        elif isinstance(s, dict):
            norm_sources.append({"source": s.get("source", "unknown"), "chunk_index": s.get("chunk_index", "n/a"), "snippet": s.get("snippet", "")})
    return answer, norm_sources

with st.form("query_form", clear_on_submit=False):
    q = st.text_area("Ask about policies", value="", height=120)
    submitted = st.form_submit_button("Ask")
    if submitted and q.strip():
        try:
            answer, sources = call_backend(q.strip())
        except Exception as e:
            answer = f"Backend error: {e}"
            sources = []
        st.session_state.history.append({"q": q.strip(), "answer": answer, "sources": sources})

# render history and last query handling
for idx, item in enumerate(reversed(st.session_state.history)):
    q_text = item["q"]
    ans_text = item.get("answer", "")
    st.markdown(f"**Q:** {q_text}")
    st.markdown(f"**A:** {ans_text}")
    sources = item.get("sources", []) or []
    if sources:
        st.markdown("**Sources:**")
        for s_i, s in enumerate(sources):
            fname = s.get("source") or "unknown"
            chunk_idx = s.get("chunk_index", "n/a")
            snippet = s.get("snippet", "") or ""
            col1, col2 = st.columns([6,1])
            with col1:
                st.write(f"- {fname} (chunk {chunk_idx})")
                if snippet:
                    st.caption(snippet)
            with col2:
                src_path = os.path.join(DATA_DIR, fname)
                key = f"open_{idx}_{s_i}_{fname}"
                if os.path.exists(src_path):
                    if st.button("Open", key=key):
                        with open(src_path, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        st.code(content, language="markdown")
                else:
                    st.write("No file")

st.sidebar.markdown("---")
if st.sidebar.button("Clear history"):
    st.session_state.history = []
    st.experimental_rerun()

st.sidebar.markdown("Run with: `streamlit run src/app_streamlit.py`")

DATA_DIR = str(ROOT / "data")

def _start_health_server(port: int = 8001):
    app = Flask("health")

    @app.route("/health", methods=["GET"])
    def health():
        try:
            files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
            return jsonify({"status": "ok", "num_files": len(files)})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    # run in daemon thread so Streamlit can exit normally
    t = Thread(target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False), daemon=True)
    t.start()

# start health endpoint once when Streamlit app starts
_start_health_server(port=8001)