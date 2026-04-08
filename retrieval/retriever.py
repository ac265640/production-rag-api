# retrieval/retriever.py

import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

# load embedding model once
model = SentenceTransformer(settings.EMBEDDING_MODEL)


def retrieve(query, index, chunks, k=None):
    """
    Retrieves top-k relevant chunks from FAISS index
    """

    if k is None:
        k = settings.TOP_K_RETRIEVE

    # embed query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    # search FAISS
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_meta = []

    for idx in indices[0]:
        chunk = chunks[idx]

        retrieved_chunks.append(chunk["text"])
        retrieved_meta.append({
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"]
        })

    return retrieved_chunks, retrieved_meta