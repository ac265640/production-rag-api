from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = None

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    CHUNKING_STRATEGY: str = "fixed"

    TOP_K_RETRIEVE: int = 5

    RETRIEVAL_MODE: str = "multi"
    MULTI_QUERY_COUNT: int = 3
    RRF_K: int = 60

    USE_RERANKER: bool = False
    RERANK_TOP_K: int = 5

    FAISS_INDEX_PATH: str = "data/faiss.index"
    CHUNKS_STORE_PATH: str = "data/chunks.pkl"

    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.0

    EVAL_DATASET_PATH: str = "evaluation/ground_truth.json"
    EVAL_RESULTS_PATH: str = "experiments/results.json"

    EVAL_METRICS: List[str] = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    class Config:
        env_file = ".env"

settings = Settings()