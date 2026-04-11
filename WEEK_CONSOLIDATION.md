# Week Consolidation — Production RAG API
# Days 1–6 Rapid-Fire Q&A + Mistakes + Senior Evaluation Criteria

---

## Part 1 — 12 Rapid-Fire Questions (Days 1–6)

| # | Question | One-Line Answer |
|---|---|---|
| 1 | What is RAG and why is it used? | RAG grounds LLM answers in external documents at inference time, eliminating hallucination on knowledge it wasn't trained on. |
| 2 | What is the role of FAISS in this system? | FAISS stores dense vector embeddings and executes approximate nearest-neighbor search in sub-millisecond time to find relevant chunks. |
| 3 | What is chunk overlap and why does it matter? | Overlap (100 tokens) ensures sentences split across chunk boundaries are still captured during retrieval, preventing context loss at edges. |
| 4 | What is HyDE retrieval? | HyDE generates a hypothetical answer to the query first, embeds that answer, and searches the index — bridging vocabulary gaps in technical corpora. |
| 5 | What is Reciprocal Rank Fusion (RRF)? | RRF merges multiple ranked lists by summing `1/(rank + k)` scores per document — no score normalization needed, robust to outlier rankings. |
| 6 | What does a cross-encoder reranker do vs a bi-encoder? | A cross-encoder reads query + document jointly (full attention), producing more accurate relevance scores; a bi-encoder encodes them separately and uses cosine distance. |
| 7 | What is faithfulness in RAGAS? | Faithfulness measures whether every claim in the generated answer is supported by the retrieved context — detects hallucination. |
| 8 | What is answer relevancy in RAGAS? | Answer relevancy scores how well the answer addresses the original question, regardless of factual grounding. |
| 9 | What does context precision measure? | The fraction of retrieved chunks that are actually relevant to the ground truth — measures retrieval precision, not recall. |
| 10 | Why use Groq instead of OpenAI for generation? | Groq's LPU inference hardware delivers ~500 tok/s vs ~60 tok/s for GPT-4, making it production-viable for low-latency RAG pipelines. |
| 11 | Why is RETRIEVAL_MODE a per-request override rather than only a global config? | Per-request mode enables live A/B testing across retrieval strategies without server restarts, critical for production experimentation. |
| 12 | What is the purpose of `data/chunks.pkl` alongside `data/faiss.index`? | FAISS stores only dense vectors with integer IDs; the pkl maps those IDs back to raw chunk text, page numbers, and source filenames needed in API responses. |

---

## Part 2 — 3 Common RAG Mistakes

### ❌ Mistake 1: Retrieving too few or too many chunks (wrong Top-K)
**What goes wrong:** Too few chunks → incomplete context → the LLM hallucinates to fill gaps. Too many chunks → noise overwhelms signal → context_precision collapses.  
**Fix:** Evaluate context_precision and context_recall across K=3,5,8,10. The crossover point where precision drops faster than recall improves is your optimal K. Use reranking to recover precision at higher K.

### ❌ Mistake 2: Not separating retrieval quality from generation quality
**What goes wrong:** Engineers attribute bad answers to the LLM when the retriever is returning wrong chunks. Low faithfulness + low context_recall = retrieval problem, not generation.  
**Fix:** Log retrieved chunks with every response. Evaluate context_precision and context_recall independently. Fix the retriever before tuning prompts.

### ❌ Mistake 3: Using the same embedding model for indexing and evaluation
**What goes wrong:** If your embedding model (e.g. MiniLM) was used to build the index AND is implicitly used by RAGAS's `answer_relevancy` metric, you get artificially inflated scores — the evaluator favors its own embeddings.  
**Fix:** Use a different, stronger embedding model for RAGAS evaluation than the one used for retrieval (e.g. `text-embedding-3-small` for eval vs. `all-MiniLM-L6-v2` for retrieval).

---

## Part 3 — What Senior Engineers Actually Evaluate

| Dimension | What It Really Means |
|---|---|
| **Retrieval Precision at K** | Not just "does it find relevant docs" but "does the top-1 chunk contribute to the answer" — seniors look at per-chunk attribution, not just end-to-end scores |
| **Latency vs. Quality Tradeoff** | HyDE and reranking add 300–800ms; seniors want to see the latency budget quantified and justified for the production SLA |
| **Failure mode handling** | What happens when FAISS index is missing? When Groq times out? When the chunk store is corrupted? Every uncaught exception is a production incident |
| **Eval dataset leakage** | Seniors check whether the ground truth questions overlap with the exact phrasing in indexed documents — if they do, all RAGAS scores are artificially high |
| **Chunking strategy fit** | Fixed-size chunking is a simplification; seniors ask "have you tested semantic chunking, and does your chunk size align with your model's context window and typical answer density?" |
| **Index persistence strategy** | Saving to local disk is fine for prototypes; seniors ask "what happens when the pod restarts — is the index rebuilt from source-of-truth PDFs or cached?" |
| **Observability** | No logging, no tracing, no latency histograms = not production-ready; seniors expect structured logs with question, mode, retrieved_sources, latency, and answer per request |
| **Security** | The `/ingest` endpoint takes a `file_path` string — a senior will immediately flag this as a path traversal vulnerability in production |
