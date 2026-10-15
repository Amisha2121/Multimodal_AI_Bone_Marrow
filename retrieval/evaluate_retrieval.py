"""
retrieval/evaluate_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Benchmark & Quantitative Evaluation for Pediatric Leukemia RAG.

Evaluates:
  1. Dense-only (FAISS cosine similarity)
  2. Sparse-only (BM25 lexical matching)
  3. Hybrid RRF (Reciprocal Rank Fusion)

Metrics:
  - Recall@1, Recall@3, Recall@5
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (MAP@5)

Outputs:
  data/processed/retrieval_index/retrieval_benchmark_results.json
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.retrieve import HybridRetriever

RESULTS_FILE = REPO_ROOT / "data" / "processed" / "retrieval_index" / "retrieval_benchmark_results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")

# Ground truth test cases representing core clinical pediatric leukemia questions
BENCHMARK_CASES = [
    {
        "query_id": "Q01_BALL_CRITERIA",
        "query": "What are the diagnostic blast criteria and cytomorphological blast features for pediatric B-ALL?",
        "relevant_chunks": ["WHO2022_BALL_001"],
    },
    {
        "query_id": "Q02_HAEMATOGONES",
        "query": "How to differentiate benign haematogones from leukemic lymphoblasts in children?",
        "relevant_chunks": ["WHO2022_HAEMATOGONES_009"],
    },
    {
        "query_id": "Q03_BURKITT",
        "query": "Burkitt leukemia starry sky cytomorphology lipid vacuoles and MYC translocation",
        "relevant_chunks": ["WHO2022_BURKITT_003"],
    },
    {
        "query_id": "Q04_APL_EMERGENCY",
        "query": "Auer rods faggot cells acute promyelocytic leukemia APL PML-RARA ATRA emergency treatment",
        "relevant_chunks": ["WHO2022_AML_SUB_005"],
    },
    {
        "query_id": "Q05_CML_CHRONIC",
        "query": "CML chronic phase myelocyte bulge high ME ratio and blast percentage cutoff under 10%",
        "relevant_chunks": ["WHO2022_CML_CP_007"],
    },
    {
        "query_id": "Q06_TALL",
        "query": "T-ALL cytomorphology pleomorphic blasts convoluted nuclei and mediastinal mass",
        "relevant_chunks": ["WHO2022_TALL_002"],
    },
    {
        "query_id": "Q07_REACTIVE",
        "query": "Reactive granulocytic hyperplasia infection and normal pediatric bone marrow cellularity",
        "relevant_chunks": ["WHO2022_REACTIVE_010"],
    },
    {
        "query_id": "Q08_SAMPLING_BIAS",
        "query": "Monolayer ROI sampling bias vs clinical aspirate differential count limitation",
        "relevant_chunks": ["DISCLAIMER_MONOLAYER_012"],
    },
    {
        "query_id": "Q09_MPAL",
        "query": "Mixed phenotype acute leukemia MPAL diagnostic lineage criteria biphenotypic",
        "relevant_chunks": ["WHO2022_MPAL_011"],
    },
    {
        "query_id": "Q10_CML_ACCELERATED",
        "query": "CML accelerated phase blast cutoff 10 to 19 percent persistent basophilia",
        "relevant_chunks": ["WHO2022_CML_ADV_008"],
    },
]


def calculate_metrics(ranked_cids: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Calculates R@1, R@3, R@5, MRR, and AP@5 for a ranked candidate list."""
    gt_set = set(ground_truth)
    r1 = 1.0 if any(cid in gt_set for cid in ranked_cids[:1]) else 0.0
    r3 = 1.0 if any(cid in gt_set for cid in ranked_cids[:3]) else 0.0
    r5 = 1.0 if any(cid in gt_set for cid in ranked_cids[:5]) else 0.0

    # Reciprocal Rank
    rr = 0.0
    for rank, cid in enumerate(ranked_cids, 1):
        if cid in gt_set:
            rr = 1.0 / rank
            break

    # Average Precision @ 5
    hits = 0
    precisions = []
    for rank, cid in enumerate(ranked_cids[:5], 1):
        if cid in gt_set:
            hits += 1
            precisions.append(hits / rank)
    ap = (sum(precisions) / len(gt_set)) if precisions else 0.0

    return {"r1": r1, "r3": r3, "r5": r5, "rr": rr, "ap": ap}


def run_benchmark():
    logger.info("Initializing HybridRetriever for benchmark...")
    retriever = HybridRetriever()

    methods = ["dense", "sparse", "hybrid"]
    accumulators = {m: {"r1": [], "r3": [], "r5": [], "rr": [], "ap": []} for m in methods}
    detailed_results = []

    print("\n" + "=" * 78)
    print("  PEDIATRIC LEUKEMIA RAG RETRIEVAL BENCHMARK EVALUATION")
    print("=" * 78)

    for case in BENCHMARK_CASES:
        qid = case["query_id"]
        qtext = case["query"]
        gt = case["relevant_chunks"]

        dense_res = [c["chunk_id"] for c in retriever.dense_search(qtext, top_k=5)]
        sparse_res = [c["chunk_id"] for c in retriever.sparse_search(qtext, top_k=5)]
        hybrid_res = [c["chunk_id"] for c in retriever.retrieve(qtext, top_k=5)]

        case_entry = {"query_id": qid, "query": qtext, "ground_truth": gt, "rankings": {}}

        for m_name, ranked in [("dense", dense_res), ("sparse", sparse_res), ("hybrid", hybrid_res)]:
            m_scores = calculate_metrics(ranked, gt)
            for k, val in m_scores.items():
                accumulators[m_name][k].append(val)
            case_entry["rankings"][m_name] = {"retrieved": ranked, "metrics": m_scores}

        detailed_results.append(case_entry)

    # Summary table
    print(f"\n{'Method':<12} | {'Recall@1':>9} | {'Recall@3':>9} | {'Recall@5':>9} | {'MRR':>9} | {'MAP@5':>9}")
    print("-" * 65)

    summary_metrics = {}
    for m in methods:
        r1_avg = sum(accumulators[m]["r1"]) / len(BENCHMARK_CASES)
        r3_avg = sum(accumulators[m]["r3"]) / len(BENCHMARK_CASES)
        r5_avg = sum(accumulators[m]["r5"]) / len(BENCHMARK_CASES)
        mrr_avg = sum(accumulators[m]["rr"]) / len(BENCHMARK_CASES)
        map_avg = sum(accumulators[m]["ap"]) / len(BENCHMARK_CASES)

        summary_metrics[m] = {
            "recall@1": round(r1_avg, 4),
            "recall@3": round(r3_avg, 4),
            "recall@5": round(r5_avg, 4),
            "mrr": round(mrr_avg, 4),
            "map@5": round(map_avg, 4),
        }

        print(f"{m.upper():<12} | {r1_avg:>9.4f} | {r3_avg:>9.4f} | {r5_avg:>9.4f} | {mrr_avg:>9.4f} | {map_avg:>9.4f}")

    print("=" * 65)

    # Save artifact
    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_queries": len(BENCHMARK_CASES),
        "summary": summary_metrics,
        "details": detailed_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Retrieval benchmark results saved -> {RESULTS_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run_benchmark()
