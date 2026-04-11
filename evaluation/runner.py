from evaluation.dataset import load_ground_truth
from evaluation.evaluator import run_ragas
from config import settings

from retrieval.retriever import retrieve as naive_retrieve
from retrieval.multi_query_retriever import MultiQueryRetriever
from retrieval.hyde_retriever import HyDERetriever
from retrieval.reranker import rerank

from generation.generator import generate

import json


def run_pipeline(mode):

    data = load_ground_truth()
    results = []

    for item in data:
        q = item["question"]

        # -------- RETRIEVAL --------
        if mode == "hyde":
            retriever = HyDERetriever()
            contexts, _ = retriever.retrieve(q)

        elif mode == "multi":
            retriever = MultiQueryRetriever()
            contexts, _ = retriever.retrieve(q)

        elif mode == "multi_rerank":
            retriever = MultiQueryRetriever()
            contexts, _ = retriever.retrieve(q, k=10)
            contexts = rerank(q, contexts)[:5]

        else:
            contexts, _ = naive_retrieve(q)

        # -------- GENERATION --------
        answer = generate(q, contexts)

        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

    return results


def evaluate_mode(mode):
    dataset = run_pipeline(mode)
    score = run_ragas(dataset)

    return score


def run_all_modes():
    modes = ["naive", "hyde", "multi", "multi_rerank"]

    final_results = {}

    for mode in modes:
        print(f"Running evaluation for: {mode}")
        score = evaluate_mode(mode)

        final_results[mode] = score

    with open(settings.EVAL_RESULTS_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    return final_results