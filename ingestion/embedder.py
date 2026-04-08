# ingestion/embedder.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings


model = SentenceTransformer(settings.EMBEDDING_MODEL)


def build_index(chunks):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # save index
    faiss.write_index(index, settings.FAISS_INDEX_PATH)

    # save chunk metadata
    with open(settings.CHUNKS_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)