"""
RAG pipeline: retrieve → (re-rank) → prompt → generate via OpenRouter LLM.
Falls back to extractive answer if no LLM key is configured.
"""
import os
import argparse
import textwrap
import sys
import pathlib
from typing import List, Dict, Any

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

MAX_DEFAULT_TOKENS = 400

# ---------------------------------------------------------------------------
# Embedding function wrapper (same as ingest.py — must match at query time)
# ---------------------------------------------------------------------------

def _get_embedding_function():
    from sentence_transformers import SentenceTransformer
    from chromadb import EmbeddingFunction
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class STEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input):  # noqa: A002
            return model.encode(input, show_progress_bar=False, convert_to_numpy=True).tolist()

    return STEmbeddingFunction()


# ---------------------------------------------------------------------------
# Chroma client — chromadb>=0.4 PersistentClient
# ---------------------------------------------------------------------------

def create_chroma_client(persist_dir: str):
    import chromadb
    return chromadb.PersistentClient(path=persist_dir)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(query: str, persist_dir: str, k: int = 5) -> List[Dict[str, Any]]:
    client = create_chroma_client(persist_dir)
    ef = _get_embedding_function()
    collection = client.get_collection(name="documents", embedding_function=ef)
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)
    results = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        results.append({
            "text": d,
            "meta": m,
            "distance": distances[i] if i < len(distances) else None,
        })
    return results


# ---------------------------------------------------------------------------
# Re-ranking with CrossEncoder
# ---------------------------------------------------------------------------

def rerank_with_crossencoder(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, Any]]:
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_name)
    except Exception:
        return results
    pairs = [[query, r["text"]] for r in results]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [{"score": float(s), **r} for s, r in ranked]


# ---------------------------------------------------------------------------
# Refusal logic
# ---------------------------------------------------------------------------

def should_refuse(results: List[Dict[str, Any]], re_ranked: bool, threshold: float) -> bool:
    if not results:
        return True
    if re_ranked:
        top_score = results[0].get("score", None)
        if top_score is not None:
            # CrossEncoder scores: higher = more relevant; typical range ~-10 to +10
            # A threshold of 0 is a reasonable "not relevant" cutoff
            return top_score < threshold
    # Distance fallback (L2 — lower = more similar)
    dist = results[0].get("distance", None)
    if dist is None:
        return False
    return dist > 1.5  # fixed sensible L2 distance cap; original 0.5 was too aggressive


# ---------------------------------------------------------------------------
# LLM generation via OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------

def generate_with_openrouter(prompt: str, max_tokens: int) -> str | None:
    """
    Call OpenRouter's free tier using the openai SDK.
    Set OPENROUTER_API_KEY in your .env or environment.
    Default model: meta-llama/llama-3.1-8b-instruct:free (free on OpenRouter).
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenRouter] generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, results: List[Dict[str, Any]], max_tokens: int) -> str:
    header = (
        "You are a helpful assistant that ONLY answers questions about the company's policies.\n"
        "If the answer is not contained in the retrieved policy text, reply exactly:\n"
        "\"I can only answer questions about the provided policies. "
        "I don't have information on that.\"\n"
        f"Limit your answer to approximately {max_tokens} tokens.\n"
        "Always cite sources inline as [source: <filename>, chunk: <chunk_index>].\n\n"
    )
    blocks = []
    for i, r in enumerate(results, 1):
        meta = r.get("meta", {}) or {}
        src = meta.get("filename") or meta.get("source", "unknown")
        if isinstance(src, str) and "/" in src:
            src = src.split("/")[-1]
        idx = meta.get("chunk_index", "n/a")
        snippet = (r.get("text") or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " ..."
        blocks.append(f"[{i}] source: {src} | chunk: {idx}\n{snippet}\n")
    docs_block = "\n".join(blocks) if blocks else "No retrieved documents."
    prompt = textwrap.dedent(f"""{header}
QUESTION:
{query}

RETRIEVED POLICY CONTEXT:
{docs_block}

INSTRUCTIONS:
- Answer ONLY using the RETRIEVED POLICY CONTEXT above.
- If the answer is not present, refuse as described above.
- Be concise and factual.
- Include inline citations like [source: filename, chunk: N] for every claim.
- Do not invent or assume any information not in the context.
""")
    return prompt


# ---------------------------------------------------------------------------
# Extractive fallback (no LLM)
# ---------------------------------------------------------------------------

def assemble_extractive_answer(results: List[Dict[str, Any]]) -> str:
    if not results:
        return (
            "I can only answer questions about the provided policies. "
            "I don't have information on that."
        )
    top = results[0]
    meta = top.get("meta", {}) or {}
    src = meta.get("filename") or meta.get("source", "unknown")
    if isinstance(src, str) and "/" in src:
        src = src.split("/")[-1]
    idx = meta.get("chunk_index", "n/a")
    text = (top.get("text") or "").strip()
    excerpt = text if len(text) <= 800 else text[:800] + "..."
    return (
        f"{excerpt}\n\n"
        f"[source: {src}, chunk: {idx}]\n\n"
        "Note: Answer is based solely on the provided policies."
    )


# ---------------------------------------------------------------------------
# Source formatter
# ---------------------------------------------------------------------------

def _results_to_sources(results: List[Dict[str, Any]], snippet_max_len: int = 400) -> List[Dict[str, Any]]:
    sources = []
    for r in results:
        meta = r.get("meta", {}) or {}
        src = meta.get("filename") or meta.get("source", "unknown")
        if isinstance(src, str) and "/" in src:
            src = src.split("/")[-1]
        idx = meta.get("chunk_index", "n/a")
        text = (r.get("text") or "").strip()
        snippet = text if len(text) <= snippet_max_len else text[:snippet_max_len] + "..."
        sources.append({"source": src, "chunk_index": idx, "snippet": snippet})
    return sources


# ---------------------------------------------------------------------------
# Public API: retrieve_and_answer
# ---------------------------------------------------------------------------

def retrieve_and_answer(
    query: str,
    persist_dir: str = "./chroma_db",
    top_k: int = 5,
    re_rank: bool = True,
    re_rank_model: str = None,
    gen_model: str = None,   # kept for API compat; OpenRouter model set via env
    max_tokens: int = MAX_DEFAULT_TOKENS,
    refusal_threshold: float = 0.0,
    device: int = -1,        # unused; kept for API compat
) -> Dict[str, Any]:
    """
    Full RAG pipeline. Returns {answer, sources, refused}.
    LLM generation uses OpenRouter if OPENROUTER_API_KEY is set.
    Falls back to extractive answer otherwise.
    """
    results = retrieve_top_k(query, persist_dir, top_k)
    re_ranked_list = results
    if re_rank:
        model_name = re_rank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        re_ranked_list = rerank_with_crossencoder(query, results, model_name)

    refused = should_refuse(re_ranked_list, re_ranked=re_rank, threshold=refusal_threshold)
    if refused:
        return {
            "answer": (
                "I can only answer questions about the provided policies. "
                "I don't have information on that."
            ),
            "sources": [],
            "refused": True,
        }

    prompt = build_prompt(query, re_ranked_list, max_tokens)

    # Try OpenRouter LLM first
    answer = generate_with_openrouter(prompt, max_tokens)

    # Ensure citations are present if LLM omitted them
    if answer and "[source:" not in answer and re_ranked_list:
        top_meta = re_ranked_list[0].get("meta", {}) or {}
        src = top_meta.get("filename") or top_meta.get("source", "unknown")
        if isinstance(src, str) and "/" in src:
            src = src.split("/")[-1]
        idx = top_meta.get("chunk_index", "n/a")
        answer = f"{answer}\n\n[source: {src}, chunk: {idx}]"

    # Fallback to extractive if no LLM answer
    if not answer:
        answer = assemble_extractive_answer(re_ranked_list)

    sources = _results_to_sources(re_ranked_list)
    return {"answer": answer, "sources": sources, "refused": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=False)
    p.add_argument("--persist-dir", type=str, default="./chroma_db")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-re-rank", dest="re_rank", action="store_false")
    p.add_argument("--max-tokens", type=int, default=MAX_DEFAULT_TOKENS)
    p.add_argument("--refusal-threshold", type=float, default=0.0)
    args = p.parse_args()

    q = args.query or input("Enter query: ").strip()
    out = retrieve_and_answer(
        q,
        persist_dir=args.persist_dir,
        top_k=args.top_k,
        re_rank=args.re_rank,
        max_tokens=args.max_tokens,
        refusal_threshold=args.refusal_threshold,
    )
    print("\n--- ANSWER ---\n")
    print(out["answer"])
    print("\n--- SOURCES ---")
    for s in out["sources"]:
        print(f"  {s['source']} (chunk {s['chunk_index']}): {s['snippet'][:100]}...")