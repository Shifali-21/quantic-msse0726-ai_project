import os
import argparse
from pathlib import Path
import traceback

from utils import set_seeds, clean_text, chunk_text, list_files, chunk_text_with_langchain

def parse_pdf(path: str) -> str:
    import pdfplumber
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

def parse_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()
    return soup.get_text(separator="\n")

def parse_markdown_or_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in {".html", ".htm"}:
        return parse_html(path)
    if ext in {".md", ".txt"}:
        return parse_markdown_or_txt(path)
    return ""


# chromadb>=0.4 requires an EmbeddingFunction subclass, not a plain lambda
def make_embedding_function(model):
    """Wrap a SentenceTransformer model as a chromadb EmbeddingFunction."""
    from chromadb import EmbeddingFunction, Embeddings
    class STEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input):  # noqa: A002
            vecs = model.encode(input, show_progress_bar=False, convert_to_numpy=True)
            return vecs.tolist()
    return STEmbeddingFunction()

def main(args):
    try:
        print("INGEST START", args)
        seed = int(os.getenv("SEED", args.seed))
        print("SEED:", seed)
        set_seeds(seed)

        print("Locating files in:", args.data_dir)
        files = list(list_files(args.data_dir))
        print(f"Found {len(files)} files")
        for f in files:
            print(" -", f)

        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")

        import chromadb
        # chromadb>=0.4: use PersistentClient — no more Settings(chroma_db_impl=...)
        client = chromadb.PersistentClient(path=args.persist_dir)
        print("Chroma PersistentClient initialized. persist_dir:", args.persist_dir)

        ef = make_embedding_function(model)
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"source": "local"},
            embedding_function=ef,
        )

        ids, metadatas, documents = [], [], []

        for path in files:
            print("Parsing:", path)
            txt = parse_file(path)
            txt = clean_text(txt)
            print("  parsed length:", len(txt))
            if not txt:
                print("  -> empty after parse/clean; skipping")
                continue
            if args.use_langchain_splitter:
                chunks = chunk_text_with_langchain(txt, chunk_size_words=args.chunk_size, overlap_words=args.overlap)
            else:
                chunks = chunk_text(txt, chunk_size_words=args.chunk_size, overlap_words=args.overlap)
            print(f"  -> {len(chunks)} chunks")
            for i, c in enumerate(chunks):
                ids.append(f"{Path(path).stem}__chunk_{i}")
                metadatas.append({"source": str(path), "chunk_index": i, "filename": Path(path).name})
                documents.append(c)

        if documents:
            print("Adding", len(documents), "chunks to collection...")
            # chromadb>=0.4 upsert avoids duplicate ID errors on re-ingest
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"Persisted {len(documents)} chunks to {args.persist_dir}")
        else:
            print("No documents found/parsed.")
    except Exception as e:
        print("ERROR during ingest:", e)
        traceback.print_exc()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--persist-dir", type=str, default="./chroma_db")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--use-langchain-splitter", action="store_true",
                   help="Use LangChain RecursiveCharacterTextSplitter")
    p.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = p.parse_args()
    main(args)