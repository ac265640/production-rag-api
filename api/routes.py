# api/routes.py

import os
import json
import pickle

from fastapi import APIRouter
from config import settings
from sentence_transformers import CrossEncoder

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import build_index

from retrieval.retriever import retrieve as naive_retrieve
from retrieval.hyde_retriever import HyDERetriever
from retrieval.multi_query_retriever import MultiQueryRetriever

from generation.generator import generate

# ✅ import ONLY from models (clean architecture)
from api.models import IngestRequest, QueryRequest

# Cross-encoder loaded once at startup (shared across requests)
_reranker = CrossEncoder(settings.RERANKER_MODEL)

router = APIRouter()


# ---------------- INGEST ----------------
@router.post("/ingest")
def ingest(req: IngestRequest):
    docs = load_pdf(req.file_path)
    chunks = chunk_documents(docs)
    build_index(chunks)

    return {
        "status": "indexed",
        "chunks": len(chunks)
    }


# ---------------- QUERY ----------------
@router.post("/query")
def query(req: QueryRequest):

    top_k = req.top_k or settings.TOP_K_RETRIEVE

    # ✅ per-request mode override; falls back to global config
    mode = req.retrieval_mode or settings.RETRIEVAL_MODE

    if mode == "hyde":
        retriever = HyDERetriever()
        contexts, meta = retriever.retrieve(req.question, top_k)

    elif mode == "multi":
        retriever = MultiQueryRetriever()
        contexts, meta = retriever.retrieve(req.question, top_k)

    elif mode == "multi_rerank":
        retriever = MultiQueryRetriever()
        contexts, meta = retriever.retrieve(req.question, top_k * 2)

        # ✅ Rerank keeping context + meta aligned, then trim to top_k
        pairs = [[req.question, c] for c in contexts]
        scores = _reranker.predict(pairs)
        ranked = sorted(zip(contexts, meta, scores), key=lambda x: x[2], reverse=True)
        contexts = [r[0] for r in ranked[:top_k]]
        meta     = [r[1] for r in ranked[:top_k]]

    else:
        # naive retrieval
        contexts, meta = naive_retrieve(req.question, top_k)

    # ✅ ALWAYS generate answer (fixed bug)
    answer = generate(req.question, contexts)

    # ✅ Enrich each source meta with its chunk text for the Streamlit UI
    enriched_sources = []
    for text, m in zip(contexts, meta):
        enriched_sources.append({**m, "text": text})

    return {
        "answer": answer,
        "sources": enriched_sources
    }


# ---------------- HEALTH ----------------
@router.get("/health")
def health():
    return {"status": "ok"}


# ---------------- DELETE INDEX ----------------
@router.delete("/index")
def delete_index():
    if os.path.exists(settings.FAISS_INDEX_PATH):
        os.remove(settings.FAISS_INDEX_PATH)

    if os.path.exists(settings.CHUNKS_STORE_PATH):
        os.remove(settings.CHUNKS_STORE_PATH)

    return {"status": "deleted"}


# ---------------- SOURCES ----------------
@router.get("/sources")
def list_sources():
    """
    Returns unique source PDF filenames currently in the FAISS index.
    Used by the Streamlit UI source filter dropdown.
    """
    if not os.path.exists(settings.CHUNKS_STORE_PATH):
        return {"sources": []}

    with open(settings.CHUNKS_STORE_PATH, "rb") as f:
        chunks = pickle.load(f)

    unique_sources = sorted(set(c["source_file"] for c in chunks))
    return {"sources": unique_sources}


# ---------------- RESULTS ----------------
@router.get("/results")
def get_results():
    """
    Returns cached RAGAS evaluation scores from experiments/results.json.
    Used by the Streamlit UI RAGAS panel.
    """
    if not os.path.exists(settings.EVAL_RESULTS_PATH):
        return {"results": {}}

    try:
        with open(settings.EVAL_RESULTS_PATH, "r") as f:
            data = json.load(f)
        return {"results": data}
    except (json.JSONDecodeError, IOError):
        return {"results": {}}