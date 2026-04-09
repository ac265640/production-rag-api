# config.py

from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # -------- API KEYS --------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # -------- MODELS --------
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # -------- CHUNKING --------
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    CHUNKING_STRATEGY: str = "fixed"  # fixed | recursive

    # -------- RETRIEVAL --------
    TOP_K_RETRIEVE: int = 5

# 🔥 DAY 4 UPDATED RETRIEVAL CONFIG
    RETRIEVAL_MODE: str = "multi"  
    # options: "naive", "hyde", "multi", "multi_rerank"

    MULTI_QUERY_COUNT: int = 3   # number of query variants
    RRF_K: int = 60              # RRF smoothing constant

    USE_RERANKER: bool = False   # toggle reranker
    RERANK_TOP_K: int = 5        # final results after reranking
    
    
    

    # -------- STORAGE --------
    FAISS_INDEX_PATH: str = "data/faiss.index"
    CHUNKS_STORE_PATH: str = "data/chunks.pkl"

    # -------- LLM --------
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.0

    class Config:
        env_file = ".env"


settings = Settings()