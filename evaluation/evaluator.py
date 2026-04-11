# evaluation/evaluator.py
#
# Uses ragas==0.2.6 (Python 3.9 compatible).
# Wires Groq (via langchain-groq) as the LLM backend so we don't need OpenAI.

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

from config import settings


def _get_ragas_llm():
    """Returns a RAGAS-compatible Groq LLM wrapper."""
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.LLM_MODEL,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    """Returns a RAGAS-compatible HuggingFace embedding wrapper."""
    hf_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(hf_embeddings)


def run_ragas(dataset_dict: list) -> dict:
    """
    Runs RAGAS evaluation on a list of dicts with keys:
        question, answer, contexts (list[str]), ground_truth
    Returns a plain dict of metric -> score.
    """
    dataset = Dataset.from_list(dataset_dict)
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,   # don't abort the full run on a single row error
    )

    # Convert EvaluationResult to a plain serialisable dict
    df = result.to_pandas()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    output = {}
    
    import math
    for m in metrics:
        if m in df.columns:
            val = float(df[m].mean())
            output[m] = 0.0 if math.isnan(val) else round(val, 4)
            
    return output