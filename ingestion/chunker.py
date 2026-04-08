# ingestion/chunker.py

from abc import ABC, abstractmethod
from typing import List, Dict
from config import settings


# -------- BASE CLASS --------
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass


# -------- FIXED SIZE --------
class FixedSizeChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + settings.CHUNK_SIZE
            chunks.append(text[start:end])
            start += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP

        return chunks


# -------- RECURSIVE --------
class RecursiveChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        max_size = settings.CHUNK_SIZE

        if len(text) <= max_size:
            return [text]

        if "\n\n" in text:
            parts = text.split("\n\n")
        elif ". " in text:
            parts = text.split(". ")
        else:
            parts = text.split(" ")

        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) < max_size:
                current += part + " "
            else:
                chunks.append(current.strip())
                current = part + " "

        if current:
            chunks.append(current.strip())

        return chunks


# -------- DOCUMENT LEVEL --------
def chunk_documents(docs: List[Dict]) -> List[Dict]:
    # choose strategy dynamically
    if settings.CHUNKING_STRATEGY == "fixed":
        chunker = FixedSizeChunker()
    else:
        chunker = RecursiveChunker()

    all_chunks = []

    for doc in docs:
        chunks = chunker.chunk(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source_file": doc["source"],
                "page_number": doc["page"],
                "chunk_id": f"{doc['source']}_{doc['page']}_{i}"
            })

    return all_chunks