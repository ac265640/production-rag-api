# api/routes.py

import faiss
import pickle
from fastapi import APIRouter
from config import settings
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import build_index
from retrieval.retriever import retrieve as naive_retrieve
from retrieval.hyde_retriever import HyDERetriever
from generation.generator import generate
from api.models import IngestRequest, QueryRequest
from retrieval.multi_query_retriever import MultiQueryRetriever
from retrieval.reranker import rerank
router = APIRouter()


# api/routes.py

from pydantic import BaseModel

class IngestRequest(BaseModel):
    file_path: str


@router.post("/ingest")
def ingest(req: IngestRequest):
    docs = load_pdf(req.file_path)
    chunks = chunk_documents(docs)
    build_index(chunks)

    return {"status": "indexed", "chunks": len(chunks)}


@router.post("/query")
def query(req: QueryRequest):

    if settings.RETRIEVAL_MODE == "hyde":
        retriever = HyDERetriever()
        contexts, meta = retriever.retrieve(req.question, req.top_k)

    elif settings.RETRIEVAL_MODE == "multi":
        retriever = MultiQueryRetriever()
        contexts, meta = retriever.retrieve(req.question, req.top_k)

    elif settings.RETRIEVAL_MODE == "multi_rerank":
        retriever = MultiQueryRetriever()
        contexts, meta = retriever.retrieve(req.question, req.top_k * 2)

        contexts = rerank(req.question, contexts)[:req.top_k]

    else:
    
        contexts, meta = naive_retrieve(req.question, req.top_k)

        answer = generate(req.question, contexts)

    return {
        "answer": answer,
        "sources": meta
    }


@router.get("/health")
def health():
    return {"status": "ok"}


@router.delete("/index")
def delete_index():
    import os
    if os.path.exists(settings.FAISS_INDEX_PATH):
        os.remove(settings.FAISS_INDEX_PATH)
    if os.path.exists(settings.CHUNKS_STORE_PATH):
        os.remove(settings.CHUNKS_STORE_PATH)

    return {"status": "deleted"}