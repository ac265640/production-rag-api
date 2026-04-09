from typing import List
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import settings
import requests


model = SentenceTransformer(settings.EMBEDDING_MODEL)


class HyDERetriever:

    def generate_hypothetical_doc(self, query: str) -> str:
        prompt = f"""
You are an expert assistant.

Given the query, generate a detailed answer.
The answer does NOT need to be factually correct,
but must contain relevant domain-specific terms.

Query: {query}

Answer:
"""

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": settings.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        res = requests.post(url, headers=headers, json=data)

        try:
            return res.json()["choices"][0]["message"]["content"]
        except:
            return query  # fallback safety

    def retrieve(self, query: str, k=None):
        if k is None:
            k = settings.TOP_K_RETRIEVE

        # Load index + chunks
        index = faiss.read_index(settings.FAISS_INDEX_PATH)

        with open(settings.CHUNKS_STORE_PATH, "rb") as f:
            chunks = pickle.load(f)

        # 🔥 Step 1: Generate hypothetical doc
        hypo_doc = self.generate_hypothetical_doc(query)

        # 🔥 Step 2: Embed hypothetical doc
        query_embedding = model.encode([hypo_doc])
        query_embedding = np.array(query_embedding)

        # 🔥 Step 3: Search
        distances, indices = index.search(query_embedding, k)

        results = []
        meta = []

        for idx in indices[0]:
            chunk = chunks[idx]

            results.append(chunk["text"])
            meta.append({
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"]
            })

        return results, meta