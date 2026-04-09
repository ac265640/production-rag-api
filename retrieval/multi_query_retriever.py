import json
import asyncio
import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
from config import settings

model = SentenceTransformer(settings.EMBEDDING_MODEL)


class MultiQueryRetriever:

    # ----------------------------
    # 1. Generate Queries (LLM)
    # ----------------------------
    def generate_queries(self, query: str):
        prompt = f"""
You are an expert search query generator.

Generate {settings.MULTI_QUERY_COUNT} DISTINCT queries.
Each must explore a different angle.

Return JSON:
{{"queries": ["q1", "q2", "q3"]}}

Query: {query}
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
            return json.loads(res.json()["choices"][0]["message"]["content"])["queries"]
        except:
            return [query]

    # ----------------------------
    # 2. Single Retrieval
    # ----------------------------
    def _retrieve_single(self, query, index, chunks, k):
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding)

        distances, indices = index.search(query_embedding, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            chunk = chunks[idx]
            results.append((chunk["chunk_id"], rank))

        return results

    # ----------------------------
    # 3. Async Retrieval
    # ----------------------------
    async def retrieve_all(self, queries, index, chunks, k):
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(None, self._retrieve_single, q, index, chunks, k)
            for q in queries
        ]

        return await asyncio.gather(*tasks)

    # ----------------------------
    # 4. RRF Merge
    # ----------------------------
    def rrf_merge(self, results_per_query):
        scores = {}

        for results in results_per_query:
            for rank, (doc_id, _) in enumerate(results):
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += 1 / (rank + settings.RRF_K)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in sorted_docs]

    # ----------------------------
    # 5. Full Pipeline
    # ----------------------------
    def retrieve(self, query, k=None):
        if k is None:
            k = settings.TOP_K_RETRIEVE

        index = faiss.read_index(settings.FAISS_INDEX_PATH)

        with open(settings.CHUNKS_STORE_PATH, "rb") as f:
            chunks = pickle.load(f)

        queries = self.generate_queries(query)

        print("\n=== MULTI QUERY DEBUG ===")
        print("Original:", query)
        print("Variants:", queries)
        print("=========================\n")

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.retrieve_all(queries, index, chunks, k)
        )

        merged_ids = self.rrf_merge(results)

        # map IDs → text
        id_to_chunk = {c["chunk_id"]: c for c in chunks}

        final_chunks = []
        meta = []

        for doc_id in merged_ids[:k]:
            chunk = id_to_chunk[doc_id]
            final_chunks.append(chunk["text"])
            meta.append({
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"]
            })

        return final_chunks, meta