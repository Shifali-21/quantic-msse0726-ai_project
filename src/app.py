"""
Flask app: Web chat UI at /, POST /chat API, GET /health.
Run: flask --app src.app:app run (from project root) or python -c "from src.app import app; app.run()"
"""
import os
import sys
import pathlib
from dotenv import load_dotenv
load_dotenv()  # Load .env if present, for config like OPENAI_API_KEY

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
            from rag import retrieve_and_answer as lc_retrieve
            _retrieve_and_answer = lc_retrieve
        except Exception:
            from rag import retrieve_and_answer as rag_retrieve
            _retrieve_and_answer = rag_retrieve
    return _retrieve_and_answer


@app.route("/health", methods=["GET"])
def health():
    """Return 200 so Render marks service Live; include db_ready for debugging."""
    db_exists = os.path.isdir(DB_DIR)
    try:
        num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]) if os.path.isdir(DATA_DIR) else 0
    except Exception:
        num_files = 0
    return jsonify({
        "status": "ok",
        "db_ready": db_exists,
        "data_files": num_files,
    })


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
        return jsonify({
            "answer": answer,
            "sources": sources,
            "refused": refused
        })
    except Exception as e:
        return jsonify({"error": str(e), "answer": "", "sources": [], "refused": True}), 500


# Enhanced HTML chat UI with advanced UX features
CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Policy RAG Chat</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 1rem;
      color: #333;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: calc(100vh - 2rem);
    }
    .header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1.5rem 2rem;
      position: relative;
    }
    .header-content {
      text-align: center;
    }
    .header h1 {
      font-size: 1.8rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    .header p {
      opacity: 0.95;
      font-size: 0.95rem;
    }
    .header-actions {
      position: absolute;
      top: 1rem;
      right: 1rem;
      display: flex;
      gap: 0.5rem;
    }
    .btn-icon {
      background: rgba(255,255,255,0.2);
      border: none;
      color: white;
      width: 36px;
      height: 36px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s;
      font-size: 1.1rem;
    }
    .btn-icon:hover {
      background: rgba(255,255,255,0.3);
    }
    .chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      scroll-behavior: smooth;
    }
    .message {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      animation: fadeIn 0.3s ease-in;
      position: relative;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message.user {
      align-items: flex-end;
    }
    .message.assistant {
      align-items: flex-start;
    }
    .message-wrapper {
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
      max-width: 75%;
    }
    .message.user .message-wrapper {
      flex-direction: row-reverse;
    }
    .message-bubble {
      padding: 1rem 1.25rem;
      border-radius: 18px;
      word-wrap: break-word;
      line-height: 1.6;
      position: relative;
      flex: 1;
    }
    .message.user .message-bubble {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .message.assistant .message-bubble {
      background: #f1f3f5;
      color: #212529;
      border-bottom-left-radius: 4px;
    }
    .message.assistant .message-bubble.refused {
      background: #fff3cd;
      border-left: 3px solid #ffc107;
    }
    .message-bubble p {
      margin: 0.5rem 0;
    }
    .message-bubble p:first-child {
      margin-top: 0;
    }
    .message-bubble p:last-child {
      margin-bottom: 0;
    }
    .message-bubble ul, .message-bubble ol {
      margin: 0.5rem 0;
      padding-left: 1.5rem;
    }
    .message-bubble strong {
      font-weight: 600;
    }
    .message-bubble code {
      background: rgba(0,0,0,0.1);
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 0.9em;
    }
    .message-actions {
      opacity: 0;
      transition: opacity 0.2s;
      display: flex;
      gap: 0.25rem;
    }
    .message:hover .message-actions {
      opacity: 1;
    }
    .action-btn {
      background: rgba(0,0,0,0.05);
      border: none;
      width: 28px;
      height: 28px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      transition: background 0.2s;
    }
    .action-btn:hover {
      background: rgba(0,0,0,0.1);
    }
    .sources {
      margin-top: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;
    }
    .source {
      padding: 0.75rem;
      background: #e9ecef;
      border-radius: 8px;
      font-size: 0.875rem;
      border-left: 3px solid #667eea;
      cursor: pointer;
      transition: background 0.2s;
    }
    .source:hover {
      background: #dee2e6;
    }
    .source-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }
    .source-title {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .source a {
      color: #667eea;
      text-decoration: none;
      font-weight: 500;
    }
    .source a:hover {
      text-decoration: underline;
    }
    .source-badge {
      background: #667eea;
      color: white;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    .snippet {
      color: #6c757d;
      margin-top: 0.5rem;
      font-size: 0.85rem;
      line-height: 1.5;
      padding: 0.5rem;
      background: white;
      border-radius: 4px;
      max-height: 100px;
      overflow: hidden;
      transition: max-height 0.3s;
    }
    .snippet.expanded {
      max-height: 500px;
      overflow-y: auto;
    }
    .expand-btn {
      background: none;
      border: none;
      color: #667eea;
      cursor: pointer;
      font-size: 0.8rem;
      margin-top: 0.25rem;
      padding: 0.25rem 0;
    }
    .input-container {
      padding: 1.5rem;
      background: #f8f9fa;
      border-top: 1px solid #dee2e6;
    }
    .settings-toggle {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1rem;
      cursor: pointer;
      color: #6c757d;
      font-size: 0.875rem;
      user-select: none;
    }
    .settings-toggle:hover {
      color: #667eea;
    }
    .settings-panel {
      display: none;
      padding: 1rem;
      background: white;
      border-radius: 8px;
      margin-bottom: 1rem;
      border: 1px solid #dee2e6;
    }
    .settings-panel.active {
      display: block;
      animation: slideDown 0.2s ease-out;
    }
    @keyframes slideDown {
      from {
        opacity: 0;
        transform: translateY(-10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    .setting-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 0;
      border-bottom: 1px solid #f1f3f5;
    }
    .setting-item:last-child {
      border-bottom: none;
    }
    .setting-label {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      flex: 1;
    }
    .setting-label-text {
      font-weight: 500;
      color: #212529;
      font-size: 0.9rem;
    }
    .setting-label-desc {
      font-size: 0.8rem;
      color: #6c757d;
    }
    .setting-control {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .slider-wrapper {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      min-width: 150px;
    }
    .slider {
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #dee2e6;
      outline: none;
      -webkit-appearance: none;
    }
    .slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #667eea;
      cursor: pointer;
      transition: background 0.2s;
    }
    .slider::-webkit-slider-thumb:hover {
      background: #5568d3;
    }
    .slider::-moz-range-thumb {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #667eea;
      cursor: pointer;
      border: none;
      transition: background 0.2s;
    }
    .slider::-moz-range-thumb:hover {
      background: #5568d3;
    }
    .slider-value {
      min-width: 30px;
      text-align: right;
      font-weight: 600;
      color: #667eea;
      font-size: 0.9rem;
    }
    .toggle-switch {
      position: relative;
      width: 44px;
      height: 24px;
      background: #ccc;
      border-radius: 12px;
      cursor: pointer;
      transition: background 0.3s;
    }
    .toggle-switch.active {
      background: #667eea;
    }
    .toggle-switch::after {
      content: '';
      position: absolute;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: white;
      top: 3px;
      left: 3px;
      transition: transform 0.3s;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .toggle-switch.active::after {
      transform: translateX(20px);
    }
    .input-wrapper {
      display: flex;
      gap: 0.75rem;
      align-items: flex-end;
    }
    .textarea-wrapper {
      flex: 1;
      position: relative;
    }
    #question {
      width: 100%;
      min-height: 60px;
      max-height: 200px;
      padding: 0.875rem 1rem;
      padding-bottom: 1.5rem;
      border: 2px solid #dee2e6;
      border-radius: 12px;
      font-family: inherit;
      font-size: 1rem;
      resize: none;
      transition: border-color 0.2s, height 0.2s;
    }
    #question:focus {
      outline: none;
      border-color: #667eea;
    }
    .char-count {
      position: absolute;
      bottom: 0.5rem;
      right: 0.75rem;
      font-size: 0.75rem;
      color: #6c757d;
    }
    .input-actions {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    button {
      padding: 0.875rem 2rem;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      white-space: nowrap;
    }
    button:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    button:active:not(:disabled) {
      transform: translateY(0);
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .btn-secondary {
      background: #6c757d;
      padding: 0.5rem 1rem;
      font-size: 0.875rem;
    }
    .btn-secondary:hover:not(:disabled) {
      background: #5a6268;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .loading {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #6c757d;
      font-style: italic;
    }
    .loading-dots {
      display: inline-flex;
      gap: 0.25rem;
    }
    .loading-dots span {
      width: 6px;
      height: 6px;
      background: #667eea;
      border-radius: 50%;
      animation: bounce 1.4s infinite ease-in-out;
    }
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1); }
    }
    .error {
      color: #dc3545;
      background: #f8d7da;
      padding: 1rem;
      border-radius: 8px;
      border-left: 3px solid #dc3545;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }
    .error-message {
      flex: 1;
    }
    .retry-btn {
      background: #dc3545;
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.875rem;
    }
    .retry-btn:hover {
      background: #c82333;
    }
    .empty-state {
      text-align: center;
      color: #6c757d;
      padding: 3rem 1rem;
    }
    .empty-state svg {
      width: 64px;
      height: 64px;
      margin-bottom: 1rem;
      opacity: 0.5;
    }
    .toast {
      position: fixed;
      bottom: 2rem;
      left: 50%;
      transform: translateX(-50%);
      background: #28a745;
      color: white;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
      z-index: 1000;
      animation: slideUp 0.3s ease-out;
    }
    @keyframes slideUp {
      from {
        opacity: 0;
        transform: translateX(-50%) translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
      }
    }
    .modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.5);
      z-index: 2000;
      align-items: center;
      justify-content: center;
    }
    .modal.active {
      display: flex;
    }
    .modal-content {
      background: white;
      padding: 2rem;
      border-radius: 12px;
      max-width: 500px;
      width: 90%;
      max-height: 80vh;
      overflow-y: auto;
    }
    .modal-content h2 {
      margin-bottom: 1rem;
    }
    .modal-content ul {
      list-style: none;
      padding-left: 0;
    }
    .modal-content li {
      padding: 0.5rem 0;
      border-bottom: 1px solid #dee2e6;
    }
    .modal-content kbd {
      background: #f1f3f5;
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.9em;
    }
    @media (max-width: 768px) {
      .container {
        height: 100vh;
        border-radius: 0;
        margin: 0;
      }
      .header {
        padding: 1rem;
      }
      .header h1 {
        font-size: 1.5rem;
      }
      .header-actions {
        position: static;
        justify-content: center;
        margin-top: 1rem;
      }
      .message-wrapper {
        max-width: 85%;
      }
      .input-wrapper {
        flex-direction: column;
      }
      button {
        width: 100%;
      }
      .message-actions {
        opacity: 1;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-content">
        <h1>📋 Policy RAG Chat</h1>
        <p>Ask questions about company policies. Answers are sourced from official documents.</p>
      </div>
      <div class="header-actions">
        <button class="btn-icon" id="clearBtn" title="Clear chat" style="display: none;">🗑️</button>
        <button class="btn-icon" id="helpBtn" title="Keyboard shortcuts">❓</button>
      </div>
    </div>
    <div class="chat-container" id="chatContainer">
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <p>Start a conversation by asking a question below.</p>
        <p style="margin-top: 0.5rem; font-size: 0.875rem;">Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line</p>
      </div>
    </div>
    <div class="input-container">
      <div class="settings-toggle" id="settingsToggle">
        <span>⚙️</span>
        <span>Retrieval Settings</span>
        <span id="settingsIcon" style="margin-left: auto;">▼</span>
      </div>
      <div class="settings-panel" id="settingsPanel">
        <div class="setting-item">
          <div class="setting-label">
            <span class="setting-label-text">Top-K Retrieval</span>
            <span class="setting-label-desc">Number of document chunks to retrieve (1-20)</span>
          </div>
          <div class="setting-control">
            <div class="slider-wrapper">
              <input type="range" id="topKSlider" class="slider" min="1" max="20" value="5">
              <span class="slider-value" id="topKValue">5</span>
            </div>
          </div>
        </div>
        <div class="setting-item">
          <div class="setting-label">
            <span class="setting-label-text">Re-ranking</span>
            <span class="setting-label-desc">Use cross-encoder to re-rank results for better relevance</span>
          </div>
          <div class="setting-control">
            <div class="toggle-switch active" id="rerankToggle"></div>
          </div>
        </div>
        <div class="setting-item">
          <div class="setting-label">
            <span class="setting-label-text">Max Tokens</span>
            <span class="setting-label-desc">Maximum length of generated answer (100-1000)</span>
          </div>
          <div class="setting-control">
            <div class="slider-wrapper">
              <input type="range" id="maxTokensSlider" class="slider" min="100" max="1000" step="50" value="200">
              <span class="slider-value" id="maxTokensValue">200</span>
            </div>
          </div>
        </div>
      </div>
      <form id="form">
        <div class="input-wrapper">
          <div class="textarea-wrapper">
            <textarea id="question" placeholder="e.g. What is the remote work policy?" required></textarea>
            <span class="char-count" id="charCount">0</span>
            <div id="queryHint" style="display: none; font-size: 0.75rem; color: #667eea; margin-top: 0.25rem; font-style: italic;">
              💡 Tip: More specific questions work better (e.g., "social media policy" instead of "social")
            </div>
          </div>
          <div class="input-actions">
            <button type="submit" id="submitBtn">Send</button>
            <button type="button" class="btn-secondary" id="clearInputBtn" style="display: none;">Clear</button>
          </div>
        </div>
      </form>
    </div>
  </div>

  <div class="modal" id="helpModal">
    <div class="modal-content">
      <h2>Keyboard Shortcuts</h2>
      <ul>
        <li><kbd>Enter</kbd> - Send message</li>
        <li><kbd>Shift + Enter</kbd> - New line</li>
        <li><kbd>Esc</kbd> - Close this dialog</li>
      </ul>
      <button style="margin-top: 1rem; width: 100%;" onclick="document.getElementById('helpModal').classList.remove('active')">Close</button>
    </div>
  </div>

  <script>
    const form = document.getElementById('form');
    const questionEl = document.getElementById('question');
    const chatContainer = document.getElementById('chatContainer');
    const submitBtn = document.getElementById('submitBtn');
    const clearBtn = document.getElementById('clearBtn');
    const clearInputBtn = document.getElementById('clearInputBtn');
    const helpBtn = document.getElementById('helpBtn');
    const helpModal = document.getElementById('helpModal');
    const charCount = document.getElementById('charCount');
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsPanel = document.getElementById('settingsPanel');
    const settingsIcon = document.getElementById('settingsIcon');
    const topKSlider = document.getElementById('topKSlider');
    const topKValue = document.getElementById('topKValue');
    const rerankToggle = document.getElementById('rerankToggle');
    const maxTokensSlider = document.getElementById('maxTokensSlider');
    const maxTokensValue = document.getElementById('maxTokensValue');

    let firstMessage = true;
    let messageHistory = [];
    let settingsExpanded = false;
    let rerankEnabled = true;

    const queryHint = document.getElementById('queryHint');
    
    // Auto-resize textarea
    questionEl.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 200) + 'px';
      charCount.textContent = this.value.length;
      clearInputBtn.style.display = this.value.trim() ? 'block' : 'none';
      
      // Show hint for very short queries (1-2 words)
      const words = this.value.trim().split(/\s+/).filter(w => w.length > 0);
      queryHint.style.display = (words.length <= 2 && words.length > 0) ? 'block' : 'none';
    });

    // Clear input
    clearInputBtn.addEventListener('click', () => {
      questionEl.value = '';
      questionEl.style.height = 'auto';
      charCount.textContent = '0';
      clearInputBtn.style.display = 'none';
      questionEl.focus();
    });

    // Show/hide clear chat button
    function updateClearButton() {
      clearBtn.style.display = messageHistory.length > 0 ? 'block' : 'none';
    }

    // Clear chat
    clearBtn.addEventListener('click', () => {
      if (confirm('Clear all messages?')) {
        chatContainer.innerHTML = `
          <div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <p>Start a conversation by asking a question below.</p>
          </div>
        `;
        messageHistory = [];
        firstMessage = true;
        updateClearButton();
      }
    });

    // Help modal
    helpBtn.addEventListener('click', () => {
      helpModal.classList.add('active');
    });
    helpModal.addEventListener('click', (e) => {
      if (e.target === helpModal) {
        helpModal.classList.remove('active');
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        helpModal.classList.remove('active');
      }
    });

    // Simple markdown renderer
    function renderMarkdown(text) {
      let html = escapeHtml(text);
      // Bold
      html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      // Links
      html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
      // Lists
      html = html.replace(/^\- (.+)$/gm, '<li>$1</li>');
      html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
      // Line breaks
      html = html.replace(/\\n/g, '<br>');
      // Paragraphs
      const lines = html.split('\\n');
      return lines.map(line => line.trim() ? `<p>${line}</p>` : '').join('');
    }

    function addMessage(text, isUser, sources = [], refused = false, originalText = null, settings = null) {
      if (firstMessage) {
        chatContainer.innerHTML = '';
        firstMessage = false;
      }

      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

      const wrapper = document.createElement('div');
      wrapper.className = 'message-wrapper';

      const bubble = document.createElement('div');
      bubble.className = `message-bubble ${refused ? 'refused' : ''}`;
      if (isUser) {
        bubble.textContent = text;
        // Show settings badge for user messages
        if (settings) {
          const badge = document.createElement('div');
          badge.style.cssText = 'font-size: 0.7rem; opacity: 0.8; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.2);';
          badge.textContent = `Top-K: ${settings.top_k} | Re-rank: ${settings.re_rank ? 'Yes' : 'No'} | Max tokens: ${settings.max_tokens}`;
          bubble.appendChild(badge);
        }
      } else {
        bubble.innerHTML = renderMarkdown(text);
      }

      const actions = document.createElement('div');
      actions.className = 'message-actions';
      const copyBtn = document.createElement('button');
      copyBtn.className = 'action-btn';
      copyBtn.innerHTML = '📋';
      copyBtn.title = 'Copy to clipboard';
      copyBtn.onclick = () => {
        const textToCopy = originalText || text;
        navigator.clipboard.writeText(textToCopy).then(() => {
          showToast('Copied to clipboard!');
        });
      };
      actions.appendChild(copyBtn);
      wrapper.appendChild(bubble);
      wrapper.appendChild(actions);
      messageDiv.appendChild(wrapper);

      if (!isUser && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sources.forEach(s => {
          const sourceDiv = document.createElement('div');
          sourceDiv.className = 'source';
          const name = s.source || 'unknown';
          const chunk = s.chunk_index != null ? s.chunk_index : 'n/a';
          const snippet = s.snippet || '';
          const isLong = snippet.length > 200;
          sourceDiv.innerHTML = `
            <div class="source-header">
              <div class="source-title">
                <a href="/data/${encodeURIComponent(name)}" target="_blank">${escapeHtml(name)}</a>
                <span class="source-badge">Chunk ${chunk}</span>
              </div>
            </div>
            ${snippet ? `
              <div class="snippet ${isLong ? '' : 'expanded'}">${escapeHtml(isLong ? snippet.substring(0, 200) : snippet)}</div>
              ${isLong ? `<button class="expand-btn" onclick="this.previousElementSibling.classList.toggle('expanded'); this.textContent = this.previousElementSibling.classList.contains('expanded') ? 'Show less' : 'Show more';">Show more</button>` : ''}
            ` : ''}
          `;
          sourcesDiv.appendChild(sourceDiv);
        });
        messageDiv.appendChild(sourcesDiv);
      }

      chatContainer.appendChild(messageDiv);
      chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });

      messageHistory.push({ text, isUser, sources, refused, settings });
      updateClearButton();
    }

    function showLoading() {
      if (firstMessage) {
        chatContainer.innerHTML = '';
        firstMessage = false;
      }
      const loadingDiv = document.createElement('div');
      loadingDiv.className = 'message assistant';
      loadingDiv.id = 'loadingMessage';
      loadingDiv.innerHTML = `
        <div class="message-wrapper">
          <div class="message-bubble loading">
            <span>Thinking</span>
            <span class="loading-dots">
              <span></span><span></span><span></span>
            </span>
          </div>
        </div>
      `;
      chatContainer.appendChild(loadingDiv);
      chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    }

    function removeLoading() {
      const loading = document.getElementById('loadingMessage');
      if (loading) loading.remove();
    }

    function showError(message, retryFn = null) {
      removeLoading();
      const errorDiv = document.createElement('div');
      errorDiv.className = 'error';
      errorDiv.innerHTML = `
        <div class="error-message">${escapeHtml(message)}</div>
        ${retryFn ? `<button class="retry-btn" onclick="(${retryFn.toString()})()">Retry</button>` : ''}
      `;
      chatContainer.appendChild(errorDiv);
      chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    }

    function showToast(message) {
      const toast = document.createElement('div');
      toast.className = 'toast';
      toast.textContent = message;
      document.body.appendChild(toast);
      setTimeout(() => {
        toast.style.animation = 'slideUp 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
      }, 2000);
    }

    let lastQuestion = '';

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const q = questionEl.value.trim();
      if (!q) return;

      const currentSettings = {
        top_k: parseInt(topKSlider.value),
        re_rank: rerankEnabled,
        max_tokens: parseInt(maxTokensSlider.value)
      };

      lastQuestion = q;
      addMessage(q, true, [], false, null, currentSettings);
      questionEl.value = '';
      questionEl.style.height = 'auto';
      charCount.textContent = '0';
      clearInputBtn.style.display = 'none';
      submitBtn.disabled = true;
      showLoading();

      const retryFn = async () => {
        const errorDiv = document.querySelector('.error');
        if (errorDiv) errorDiv.remove();
        showLoading();
        submitBtn.disabled = true;
        await sendMessage(lastQuestion);
      };

      await sendMessage(q, retryFn);
    });

    async function sendMessage(q, retryFn = null) {
      try {
        const topK = parseInt(topKSlider.value);
        const reRank = rerankEnabled;
        const maxTokens = parseInt(maxTokensSlider.value);
        
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            question: q,
            top_k: topK,
            re_rank: reRank,
            max_tokens: maxTokens
          })
        });
        const data = await res.json();
        removeLoading();

        if (!res.ok) {
          showError(data.error || res.statusText, retryFn);
          submitBtn.disabled = false;
          return;
        }

        const answer = data.answer || '(No answer provided)';
        addMessage(answer, false, data.sources || [], data.refused || false, answer);
      } catch (err) {
        showError('Network error: ' + err.message, retryFn);
      } finally {
        submitBtn.disabled = false;
        questionEl.focus();
      }
    }

    // Enter to submit (Shift+Enter for new line)
    questionEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        if (!submitBtn.disabled) {
          form.dispatchEvent(new Event('submit'));
        }
      }
    });

    function escapeHtml(s) {
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }

    // Settings panel toggle
    settingsToggle.addEventListener('click', () => {
      settingsExpanded = !settingsExpanded;
      settingsPanel.classList.toggle('active', settingsExpanded);
      settingsIcon.textContent = settingsExpanded ? '▲' : '▼';
    });

    // Top-K slider
    topKSlider.addEventListener('input', (e) => {
      topKValue.textContent = e.target.value;
    });

    // Re-ranking toggle
    rerankToggle.addEventListener('click', () => {
      rerankEnabled = !rerankEnabled;
      rerankToggle.classList.toggle('active', rerankEnabled);
    });

    // Max tokens slider
    maxTokensSlider.addEventListener('input', (e) => {
      maxTokensValue.textContent = e.target.value;
    });

    // Focus textarea on load
    questionEl.focus();
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