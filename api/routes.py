# api/routes.py

import faiss
import pickle
from fastapi import APIRouter
from config import settings
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import build_index
from retrieval.retriever import retrieve
from generation.generator import generate

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
def query(req: dict):
    index = faiss.read_index(settings.FAISS_INDEX_PATH)

    with open(settings.CHUNKS_STORE_PATH, "rb") as f:
        chunks = pickle.load(f)

    contexts, meta = retrieve(req["question"], index, chunks)

    answer = generate(req["question"], contexts)

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