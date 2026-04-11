# Interview Prep — 3-Minute RAG System Explanation

## The Structured Answer (Architecture → Retrieval → Evaluation → Production)

---

### ⏱ Minute 1 — Architecture (What it is and why)

> "I built a production-grade RAG system using FastAPI, FAISS, and Groq's LLaMA 3.1 8B.
> The core idea is simple: instead of relying on the LLM's parametric memory, we
> retrieve relevant document chunks at query time and force the model to answer only
> from that retrieved context — using a grounded prompt that says 'If the answer is
> not present, say: Not in context.'
>
> The ingestion pipeline uses PyPDF to load documents, splits them into fixed-size
> chunks of 500 tokens with 100-token overlap, embeds them using sentence-transformers
> MiniLM, and stores the vectors in a FAISS flat index on disk alongside a pickle
> store that maps FAISS integer IDs back to raw chunk text, page numbers, and source
> filenames."

**Key point to emphasize:**  
The chunking + embedding happens once at ingest time. The FAISS search at query time is sub-millisecond — the bottleneck is always the LLM call, not the retrieval.

---

### ⏱ Minute 2 — Retrieval (The technically interesting part)

> "I implemented four retrieval strategies, each with different quality/latency tradeoffs:
>
> **Naive:** Direct cosine similarity between query embedding and chunk embeddings. Fast,
> interpretable, but fails when the query vocabulary doesn't match the document vocabulary.
>
> **HyDE:** Instead of embedding the query, we first ask the LLM to generate a hypothetical
> answer — something plausible but not necessarily correct — and embed that instead. This
> bridges vocabulary mismatch in technical corpora.
>
> **Multi-Query:** The LLM generates 3 query variants exploring different angles. We retrieve
> top-K chunks per variant, then merge using Reciprocal Rank Fusion — summing 1/(rank + 60)
> per document across all lists. Score-free, no normalization needed, robust to outliers.
>
> **Hybrid:** We run Multi-Query first, take the top 2K candidates, then run a cross-encoder
> reranker — which reads query and document jointly with full attention — and keep only the
> top K. This gives the highest RAGAS precision but adds ~500ms of latency."

**Key point to emphasize:**  
The retrieval mode is a per-request parameter, so you can A/B test strategies in production without restarting the server.

---

### ⏱ Minute 3 — Evaluation & Production (How you know it works)

> "I used RAGAS for automated evaluation — it's model-based, so you don't need human
> annotators. The four metrics are: faithfulness (are all claims in the answer grounded
> in context?), answer relevancy (does the answer address the question?), context precision
> (are the retrieved chunks actually relevant?), and context recall (did we retrieve all
> relevant chunks?).
>
> Across my evaluation dataset, Hybrid retrieval outperformed Naive by roughly 16 points
> on faithfulness — the reranker eliminates false positives that dense retrieval promotes.
>
> For production readiness: the system is containerized with Docker Compose — the FastAPI
> backend and Streamlit UI are separate services, with the FAISS index persisted on a
> mounted volume so it survives container restarts. Health checks ensure the frontend
> only starts after the backend is ready. The one known production risk I'd address next
> is the /ingest endpoint accepting a raw file path — that's a path traversal vulnerability
> that should be replaced with authenticated upload + server-side path validation."

**Key point to emphasize:**  
Knowing your system's failure modes is as important as its strengths. Mentioning the path traversal issue shows production engineering maturity.

---

## Anticipated Follow-Up Questions

| Question | Sharp Answer |
|---|---|
| **"Why FAISS-flat and not HNSW?"** | Flat gives exact search (no approximation error); at the scale of hundreds of PDFs, exact search is fast enough. HNSW makes sense at millions of vectors where ANN is necessary. |
| **"Why not just use LangChain?"** | I built this from scratch to understand every component — embedding, storage, retrieval, merging, reranking. LangChain is a valid production choice but abstracts away the details that matter for debugging retrieval quality. |
| **"What would you change at 10× scale?"** | Replace FAISS on-disk with a hosted vector DB (Pinecone or Weaviate) for concurrent writes, replace synchronous Groq calls with async, add per-query latency tracing with OpenTelemetry. |
| **"How do you prevent the LLM from hallucinating?"** | Grounded prompt template that instructs the model to say "Not in context" if the answer isn't present. Faithfulness measured by RAGAS confirms whether it actually complied. |
| **"How would you handle multi-document questions?"** | The Multi-Query + RRF approach already helps — different query variants tend to pull from different documents. For explicit multi-hop reasoning, I'd add a query decomposition step similar to IRCoT. |
| **"What's the biggest risk in this system in production?"** | The embedding model is a singleton loaded at import time — it blocks the Python process during model loading (~2s). Under cold starts or restarts, the first request pays that cost. Fix: lazy loading with a warmup endpoint. |

---

## Closing Statement (if asked "tell me about yourself")

> "I care about the gap between RAG demos and production RAG. Anyone can wire together
> a vector store and an LLM in 20 lines. What I focused on was: how do you measure
> retrieval quality systematically, how do you compare strategies rigorously, and how
> do you build something that doesn't break when the FAISS index is missing or the
> API is down. Those are the engineering problems that actually matter."
