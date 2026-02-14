import os
import re
import random
import numpy as np

def set_seeds(seed: int):
    """Set seeds for reproducibility (python, numpy, and torch if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not available or failed to import — still OK
        pass

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

def split_by_headings(text: str):
    h_re = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", flags=re.IGNORECASE | re.DOTALL)
    text = h_re.sub(lambda m: "\n# " + m.group(1) + "\n", text)
    parts = re.split(r'(?m)^(?=#\s)', text)
    return [p.strip() for p in parts if p and p.strip()]

def word_window_chunk(text: str, chunk_size_words: int = 500, overlap_words: int = 50):
    words = text.split()
    if len(words) <= chunk_size_words:
        return [" ".join(words)]
    chunks = []
    start = 0
    step = chunk_size_words - overlap_words
    while start < len(words):
        end = start + chunk_size_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks

def chunk_text(text: str, chunk_size_words: int = 500, overlap_words: int = 50):
    blocks = split_by_headings(text) or [text]
    out = []
    for b in blocks:
        b_clean = clean_text(b)
        if not b_clean:
            continue
        if len(b_clean.split()) <= chunk_size_words:
            out.append(b_clean)
        else:
            out.extend(word_window_chunk(b_clean, chunk_size_words, overlap_words))
    return out

def list_files(data_dir: str):
    exts = {".pdf", ".md", ".txt", ".html", ".htm"}
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(root, f)

def chunk_text_with_langchain(text: str, chunk_size_words: int = 500, overlap_words: int = 50):
    """
    Attempt to use LangChain RecursiveCharacterTextSplitter; fall back to header-first + word-window.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        avg_word_chars = 6
        chunk_chars = max(200, chunk_size_words * avg_word_chars)
        overlap_chars = max(50, overlap_words * avg_word_chars)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", " ", ""],
        )
        parts = splitter.split_text(text)
        return [clean_text(p) for p in parts if p and p.strip()]
    except Exception:
        # LangChain not installed or failed -> fallback
        return chunk_text(text, chunk_size_words=chunk_size_words, overlap_words=overlap_words)