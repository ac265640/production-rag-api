# api/models.py

from typing import Optional, List
from pydantic import BaseModel


class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    retrieval_mode: Optional[str] = None  # per-request override; falls back to settings.RETRIEVAL_MODE


class QueryResponse(BaseModel):
    answer: str
    sources: List