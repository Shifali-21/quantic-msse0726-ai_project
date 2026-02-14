import argparse
import textwrap
import sys
import pathlib
from typing import List, Dict, Any

# ensure src is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from transformers import pipeline

MAX_DEFAULT_TOKENS = 200

def create_chroma_client(persist_dir: str):
    try:
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    except Exception:
        # fallback for chromadb versions with different config API
        return chromadb.Client()

def retrieve_top_k(query: str, persist_dir: str, k: int = 5):
    client = create_chroma_client(persist_dir)
    collection = client.get_collection(name="documents")
    # query_texts returns documents, metadatas and (optionally) distances
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0] if "distances" in res else None
    results = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        results.append({
            "text": d,
            "meta": m,
            "distance": distances[i] if distances is not None and i < len(distances) else None
        })
    return results

def rerank_with_crossencoder(query: str, results: List[Dict[str, Any]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        ce = CrossEncoder(model_name)
    except Exception:
        return results  # unable to load re-ranker
    pairs = [[query, r["text"]] for r in results]
    scores = ce.predict(pairs)  # higher = better
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [{"score": s, **r} for s, r in ranked]

def build_prompt(query: str, results: List[Dict[str, Any]], max_tokens: int):
    header = (
        "You are a concise assistant that ONLY answers questions about the company's policies.\n"
        "If the answer is not contained in the retrieved policy text, reply exactly: "
        "\"I can only answer questions about the provided policies. I don't have information on that.\"\n"
        f"Limit your answer to ~{max_tokens} tokens and always cite sources as [source: filename, chunk_index].\n\n"
    )
    retrieved_blocks = []
    for i, r in enumerate(results, 1):
        meta = r.get("meta", {})
        src = meta.get("source", "unknown")
        idx = meta.get("chunk_index", "n/a")
        snippet = r.get("text", "").strip()
        snippet = snippet if len(snippet) < 1200 else snippet[:1200] + " ..."
        retrieved_blocks.append(f"[{i}] source: {src} | chunk: {idx}\n{snippet}\n")
    docs_block = "\n".join(retrieved_blocks) if retrieved_blocks else "No retrieved documents."
    prompt = textwrap.dedent(f"""{header}
    QUESTION:
    {query}

    RETRIEVED:
    {docs_block}

    INSTRUCTIONS:
    - Answer only using the RETRIEVED text. If the answer cannot be found in the retrieved text, refuse as described.
    - Keep the answer concise and include inline citations like [source: filename, chunk_index].
    - Do not invent facts.
    - Max output ~{max_tokens} tokens.
    """)
    return prompt

def generate_with_model(prompt: str, model_name: str, max_tokens: int):
    try:
        gen = pipeline("text2text-generation", model=model_name, device=-1, tokenizer=model_name)
        out = gen(prompt, max_length=max_tokens, truncation=True, do_sample=False)
        return out[0].get("generated_text")
    except Exception:
        return None

def assemble_extractive_answer(results: List[Dict[str, Any]]):
    if not results:
        return "I can only answer questions about the provided policies. I don't have information on that."
    # Use top result and include short excerpt + citation
    top = results[0]
    meta = top.get("meta", {})
    src = meta.get("source", "unknown")
    idx = meta.get("chunk_index", "n/a")
    text = top.get("text", "").strip()
    excerpt = text if len(text) < 800 else text[:800] + "..."
    return f"{excerpt}\n\n[source: {src}, chunk_index: {idx}]\n\nNote: Answer is based solely on the provided policies."

def should_refuse(results: List[Dict[str, Any]], re_ranked: bool, threshold: float):
    # if no docs, refuse
    if not results:
        return True
    if re_ranked:
        top_score = results[0].get("score", None)
        # if CrossEncoder score exists, use it for thresholding
        if top_score is not None:
            return top_score < threshold
    # fallback: if first distance is present and large (or None), be conservative
    dist = results[0].get("distance", None)
    if dist is None:
        return False  # cannot judge -> don't refuse
    # assume smaller distance = more similar; use threshold as maximum acceptable distance
    return dist > threshold

def answer(query: str, persist_dir: str = "./chroma_db", top_k: int = 5, re_rank: bool = True, re_rank_model: str = None, gen_model: str = None, max_tokens: int = MAX_DEFAULT_TOKENS, refusal_threshold: float = 0.5):
    # retrieve
    results = retrieve_top_k(query, persist_dir, top_k)
    if re_rank:
        model_name = re_rank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        re_ranked = rerank_with_crossencoder(query, results, model_name)
        # re_ranked elements have 'score' key and original meta/text
        results = re_ranked
    # decide refuse
    if should_refuse(results, re_ranked=re_rank, threshold=refusal_threshold):
        return "I can only answer questions about the provided policies. I don't have information on that."
    # build prompt
    prompt = build_prompt(query, results, max_tokens)
    # try generation if requested
    if gen_model:
        gen = generate_with_model(prompt, gen_model, max_tokens)
        if gen:
            # ensure citations present; if not, append top sources
            if "[source:" not in gen:
                top_meta = results[0].get("meta", {})
                src = top_meta.get("source", "unknown")
                idx = top_meta.get("chunk_index", "n/a")
                gen = f"{gen}\n\n[source: {src}, chunk_index: {idx}]"
            return gen
    # fallback extractive conservative answer
    return assemble_extractive_answer(results)


def _results_to_sources(results: List[Dict[str, Any]], snippet_max_len: int = 400) -> List[Dict[str, Any]]:
    """Convert retrieval results to list of source dicts with source, chunk_index, snippet."""
    sources = []
    for r in results:
        meta = r.get("meta", {}) or {}
        src = meta.get("source", "unknown")
        if isinstance(src, str) and "/" in src:
            src = src.split("/")[-1]
        idx = meta.get("chunk_index", "n/a")
        text = (r.get("text") or "").strip()
        snippet = text if len(text) <= snippet_max_len else text[:snippet_max_len] + "..."
        sources.append({"source": src, "chunk_index": idx, "snippet": snippet})
    return sources


def retrieve_and_answer(
    query: str,
    persist_dir: str = "./chroma_db",
    top_k: int = 5,
    re_rank: bool = True,
    re_rank_model: str = None,
    gen_model: str = None,
    max_tokens: int = MAX_DEFAULT_TOKENS,
    refusal_threshold: float = 0.5,
    device: int = -1,
) -> Dict[str, Any]:
    """
    RAG query returning a dict with answer, sources (with snippets), and refused flag.
    Suitable for API and Streamlit. device is ignored (kept for LangChain-style API compatibility).
    """
    results = retrieve_top_k(query, persist_dir, top_k)
    re_ranked_list = results
    if re_rank:
        model_name = re_rank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        re_ranked_list = rerank_with_crossencoder(query, results, model_name)
    refused = should_refuse(re_ranked_list, re_ranked=re_rank, threshold=refusal_threshold)
    if refused:
        return {
            "answer": "I can only answer questions about the provided policies. I don't have information on that.",
            "sources": [],
            "refused": True,
        }
    ans = answer(
        query,
        persist_dir=persist_dir,
        top_k=top_k,
        re_rank=re_rank,
        re_rank_model=re_rank_model,
        gen_model=gen_model,
        max_tokens=max_tokens,
        refusal_threshold=refusal_threshold,
    )
    sources = _results_to_sources(re_ranked_list)
    return {"answer": ans, "sources": sources, "refused": False}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=False)
    p.add_argument("--persist-dir", type=str, default="./chroma_db")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-re-rank", dest="re_rank", action="store_false")
    p.add_argument("--re-rank-model", type=str, default=None)
    p.add_argument("--gen-model", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=MAX_DEFAULT_TOKENS)
    p.add_argument("--refusal-threshold", type=float, default=0.5, help="threshold for refusal; meaning depends on scorer (cross-encoder score or distance)")
    args = p.parse_args()

    q = args.query or input("Enter query: ").strip()
    out = answer(q, persist_dir=args.persist_dir, top_k=args.top_k, re_rank=args.re_rank, re_rank_model=args.re_rank_model, gen_model=args.gen_model, max_tokens=args.max_tokens, refusal_threshold=args.refusal_threshold)
    print("\n--- ANSWER ---\n")
    print(out)