"""
Flask app: Web chat UI at /, POST /chat API, GET /health.
Run: flask --app src.app:app run (from project root) or python -c "from src.app import app; app.run()"
"""
import os
import sys
import pathlib
from dotenv import load_dotenv
load_dotenv()  # Load .env if present, for config like OPENAI_API_KEY
import os
print("OPENROUTER_API_KEY loaded:", bool(os.getenv("OPENROUTER_API_KEY")))

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
# Allow override via env for production (e.g. Render)
DATA_DIR = os.getenv("DATA_DIR", str(ROOT / "data"))
DB_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "chroma_db"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Lazy import to avoid loading RAG at import time
_retrieve_and_answer = None


def get_rag():
    global _retrieve_and_answer
    if _retrieve_and_answer is None:
        try:
            from langchain_rag import retrieve_and_answer as lc_retrieve
            _retrieve_and_answer = lc_retrieve
        except Exception:
            from rag import retrieve_and_answer as rag_retrieve
            _retrieve_and_answer = rag_retrieve
    return _retrieve_and_answer


@app.route("/health", methods=["GET"])
def health():
    """Return simple JSON status for health checks."""
    try:
        db_exists = os.path.isdir(DB_DIR)
        num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]) if os.path.isdir(DATA_DIR) else 0
        return jsonify({
            "status": "ok",
            "db_ready": db_exists,
            "data_files": num_files,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint: POST JSON { "question": "..." }.
    Returns JSON { "answer", "sources": [ { "source", "chunk_index", "snippet" } ], "refused" }.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing or empty 'question'"}), 400
        top_k = int(data.get("top_k", 5))
        re_rank = bool(data.get("re_rank", True))
        max_tokens = int(data.get("max_tokens", 200))
        gen_model = (data.get("gen_model") or "").strip() or None

        retrieve_and_answer_fn = get_rag()
        res = retrieve_and_answer_fn(
            question,
            persist_dir=DB_DIR,
            top_k=top_k,
            re_rank=re_rank,
            gen_model=gen_model,
            max_tokens=max_tokens,
        )
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        refused = res.get("refused", False)
        # Normalize source filenames for links (basename only)
        for s in sources:
            if isinstance(s, dict) and "source" in s and isinstance(s["source"], str) and "/" in s["source"]:
                s["source"] = s["source"].split("/")[-1]
        return jsonify({"answer": answer, "sources": sources, "refused": refused})
    except Exception as e:
        return jsonify({"error": str(e), "answer": "", "sources": [], "refused": True}), 500


# Minimal HTML chat UI for GET /
CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Policy RAG Chat</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 1rem; }
    h1 { margin-top: 0; }
    #question { width: 100%; min-height: 80px; padding: 8px; margin-bottom: 8px; }
    button { padding: 8px 16px; cursor: pointer; }
    #answer { white-space: normal; margin: 1rem 0; padding: 1rem; background: #f5f5f5; border-radius: 8px; }
    #answer h1 { font-size: 1.3rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.3rem; }
    #answer h2 { font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.3rem; }
    #answer h3 { font-size: 1rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.3rem; }
    #answer ul, #answer ol { padding-left: 1.5rem; margin: 0.3rem 0; }
    #answer p { margin: 0.4rem 0; }
    .source { margin: 0.5rem 0; padding: 0.5rem; background: #eee; border-radius: 4px; font-size: 0.9rem; }
    .source a { color: #06c; }
    .snippet { color: #555; margin-top: 4px; }
    .error { color: #c00; }
  </style>
</head>
<body>
  <h1>Policy RAG Chat</h1>
  <p>Ask a question about company policies. Answers cite source documents.</p>
  <form id="form">
    <textarea id="question" placeholder="e.g. What is the remote work policy?" required></textarea>
    <br>
    <button type="submit">Ask</button>
  </form>
  <div id="answer"></div>
  <div id="sources"></div>

  <script>
    const form = document.getElementById('form');
    const questionEl = document.getElementById('question');
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const q = questionEl.value.trim();
      if (!q) return;
      answerEl.textContent = 'Loading...';
      sourcesEl.innerHTML = '';
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        if (!res.ok) {
          answerEl.innerHTML = '<span class="error">' + (data.error || res.statusText) + '</span>';
          return;
        }
        answerEl.innerHTML = marked.parse(data.answer || '(No answer)');
        const sources = data.sources || [];
        if (sources.length) {
          sources.forEach(s => {
            const d = document.createElement('div');
            d.className = 'source';
            const name = s.source || 'unknown';
            const chunk = s.chunk_index != null ? s.chunk_index : 'n/a';
            d.innerHTML = '<a href="/data/' + encodeURIComponent(name) + '" target="_blank">' + name + '</a> (chunk ' + chunk + ')' +
              (s.snippet ? '<div class="snippet">' + escapeHtml(s.snippet) + '</div>' : '');
            sourcesEl.appendChild(d);
          });
        }
      } catch (err) {
        answerEl.innerHTML = '<span class="error">' + escapeHtml(err.message) + '</span>';
      }
    });
    function escapeHtml(s) {
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Web chat interface: text box for user input, submits to /chat."""
    return CHAT_HTML


@app.route("/data/<filename>")
def serve_data_file(filename):
    """Serve a policy file from data/ for source links (e.g. /data/remote_work.md)."""
    if ".." in filename or "/" in filename:
        return "", 404
    return send_from_directory(DATA_DIR, filename, mimetype="text/markdown")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
