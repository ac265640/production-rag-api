"""
retrieval/multi_query_retriever.py

Multi-Query retriever with Reciprocal Rank Fusion (RRF).
Uses a plain synchronous loop — asyncio.get_event_loop() + run_until_complete()
crashes inside FastAPI's already-running event loop.
FAISS search is sub-millisecond so there is no practical benefit to async here.
"""

import re
import json
import faiss
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from config import settings

# Embedding model loaded once at import time
model = SentenceTransformer(settings.EMBEDDING_MODEL)


class MultiQueryRetriever:

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Generate query variants via Groq LLM
    # ──────────────────────────────────────────────────────────────────────────
    def generate_queries(self, query: str) -> list:
        prompt = f"""You are an expert search query generator.

Generate {settings.MULTI_QUERY_COUNT} DISTINCT search queries for the following question.
Each query must explore a different angle or rephrase.

Respond with ONLY valid JSON — no markdown, no code fences, no explanation.
Format: {{"queries": ["q1", "q2", "q3"]}}

Question: {query}
"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": settings.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }

        try:
            res = requests.post(url, headers=headers, json=data, timeout=30)
            raw = res.json()["choices"][0]["message"]["content"]

            # Strip markdown code fences if the LLM adds them
            raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()

            parsed = json.loads(raw)
            queries = parsed.get("queries", [])
            if queries and isinstance(queries, list):
                print(f"\n=== MULTI QUERY DEBUG ===")
                print(f"Original : {query}")
                print(f"Variants : {queries}")
                print(f"=========================\n")
                return queries
        except Exception as e:
            print(f"[MultiQueryRetriever] generate_queries fallback ({e}), using original query.")

        return [query]  # safe fallback

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Retrieve for a single query (synchronous)
    # ──────────────────────────────────────────────────────────────────────────
    def _retrieve_single(self, query: str, index, chunks: list, k: int) -> list:
        """Returns list of (chunk_id, rank) tuples."""
        query_embedding = np.array(model.encode([query]), dtype="float32")
        distances, indices = index.search(query_embedding, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append((chunks[idx]["chunk_id"], rank))
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Reciprocal Rank Fusion
    # ──────────────────────────────────────────────────────────────────────────
    def rrf_merge(self, results_per_query: list) -> list:
        """Merges multiple ranked lists via RRF. Returns ordered chunk_ids."""
        scores = {}
        for results in results_per_query:
            for rank, (doc_id, _) in enumerate(results):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + settings.RRF_K)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Full pipeline  (sync — no asyncio)
    # ──────────────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = None):
        if k is None:
            k = settings.TOP_K_RETRIEVE

        # Load FAISS index + chunk store
        index = faiss.read_index(settings.FAISS_INDEX_PATH)
        with open(settings.CHUNKS_STORE_PATH, "rb") as f:
            chunks = pickle.load(f)

        # Generate query variants
        queries = self.generate_queries(query)

        # Retrieve per-query rankings (synchronous loop — safe inside FastAPI)
        results_per_query = [
            self._retrieve_single(q, index, chunks, k)
            for q in queries
        ]

        # Merge with RRF
        merged_ids = self.rrf_merge(results_per_query)

        # Map IDs → text + metadata
        id_to_chunk = {c["chunk_id"]: c for c in chunks}

        final_chunks = []
        meta = []
        for doc_id in merged_ids[:k]:
            chunk = id_to_chunk.get(doc_id)
            if chunk is None:
                continue
            final_chunks.append(chunk["text"])
            meta.append({
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_id":    chunk["chunk_id"],
            })

        return final_chunks, meta