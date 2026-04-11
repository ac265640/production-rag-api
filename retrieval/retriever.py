# retrieval/retriever.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

# load embedding model once at import time
model = SentenceTransformer(settings.EMBEDDING_MODEL)


def retrieve(query: str, k: int = None):
    """
    Naive FAISS retrieval.
    Loads index + chunk store from disk, embeds query, returns top-k results.
    Signature matches HyDERetriever.retrieve() and MultiQueryRetriever.retrieve()
    so routes.py can call all three the same way:
        contexts, meta = retrieve(question, top_k)
    """
    if k is None:
        k = settings.TOP_K_RETRIEVE

    # ── Load persisted FAISS index ──────────────────────────────────────────
    if not __import__("os").path.exists(settings.FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{settings.FAISS_INDEX_PATH}'. "
            "Run POST /ingest first."
        )

    index = faiss.read_index(settings.FAISS_INDEX_PATH)

    # ── Load chunk metadata store ────────────────────────────────────────────
    with open(settings.CHUNKS_STORE_PATH, "rb") as f:
        chunks = pickle.load(f)

    # ── Embed query ───────────────────────────────────────────────────────────
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    # ── FAISS search ──────────────────────────────────────────────────────────
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_meta = []

    for idx in indices[0]:
        if idx == -1:          # FAISS returns -1 for padding when k > index size
            continue
        chunk = chunks[idx]
        retrieved_chunks.append(chunk["text"])
        retrieved_meta.append({
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "chunk_id":    chunk["chunk_id"],
        })

    return retrieved_chunks, retrieved_meta