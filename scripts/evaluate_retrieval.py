"""
scripts/evaluate_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Evaluate Retrieval System Performance with Quantitative Metrics.

Usage:
    python scripts/evaluate_retrieval.py --test-queries data/retrieval_test_queries.json

This script:
1. Loads test queries with ground truth relevance
2. Runs retrieval (dense-only, sparse-only, hybrid)
3. Computes metrics (Precision@K, Recall@K, MRR, MAP, NDCG)
4. Performs ablation study
5. Optimizes RRF weights via grid search
6. Exports results to JSON/markdown

Author: AI Assistant
Date: November 8, 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.retrieve import HybridRetriever
from retrieval.eval_metrics import (
    QueryRelevance,
    RetrievalResult,
    EvaluationMetrics,
    evaluate_retrieval_system,
    format_evaluation_report,
    compare_retrieval_systems,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate_retrieval")

OUTPUT_DIR = REPO_ROOT / "results" / "retrieval_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_test_queries(filepath: Path) -> List[QueryRelevance]:
    """Load test queries with ground truth relevance."""
    logger.info(f"Loading test queries from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        query = QueryRelevance(
            query_id=item["query_id"],
            query_text=item["query_text"],
            relevant_chunk_ids=set(item["relevant_chunk_ids"]),
            graded_relevance=item.get("graded_relevance", {})
        )
        queries.append(query)
    
    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def run_retrieval(
    retriever: HybridRetriever,
    queries: List[QueryRelevance],
    method: str = "hybrid",
    top_k: int = 10,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[RetrievalResult]:
    """
    Run retrieval for all queries using specified method.
    
    Args:
        retriever: HybridRetriever instance
        queries: List of test queries
        method: "dense", "sparse", or "hybrid"
        top_k: Number of results to retrieve
        dense_weight: Weight for dense retrieval in RRF
        sparse_weight: Weight for sparse retrieval in RRF
    
    Returns:
        List of RetrievalResult objects
    """
    logger.info(f"Running {method} retrieval for {len(queries)} queries (top-{top_k})...")
    
    results = []
    
    for query in queries:
        if method == "dense":
            retrieved_chunks = retriever.dense_search(query.query_text, top_k=top_k)
        elif method == "sparse":
            retrieved_chunks = retriever.sparse_search(query.query_text, top_k=top_k)
        elif method == "hybrid":
            retrieved_chunks = retriever.retrieve(
                query.query_text,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        scores = [chunk.get("hybrid_rrf_score", chunk.get("dense_score", chunk.get("sparse_score", 0.0))) 
                  for chunk in retrieved_chunks]
        
        results.append(RetrievalResult(
            query_id=query.query_id,
            retrieved_chunk_ids=retrieved_ids,
            scores=scores
        ))
    
    return results


def optimize_rrf_weights(
    retriever: HybridRetriever,
    queries: List[QueryRelevance],
    weight_range: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> Tuple[float, float, EvaluationMetrics]:
    """
    Grid search over RRF weights to find optimal combination.
    
    Args:
        retriever: HybridRetriever instance
        queries: List of test queries
        weight_range: List of weight values to try
    
    Returns:
        (best_dense_weight, best_sparse_weight, best_metrics)
    """
    logger.info("Optimizing RRF weights via grid search...")
    
    best_f1_at_3 = 0.0
    best_weights = (0.6, 0.4)
    best_metrics = None
    
    for dense_w in weight_range:
        sparse_w = 1.0 - dense_w
        
        # Run retrieval with these weights
        results = run_retrieval(
            retriever, queries,
            method="hybrid",
            top_k=10,
            dense_weight=dense_w,
            sparse_weight=sparse_w
        )
        
        # Evaluate
        metrics = evaluate_retrieval_system(results, queries, k_values=[1, 3, 5, 10])
        
        # Use F1@3 as optimization target
        if metrics.f1_at_3 > best_f1_at_3:
            best_f1_at_3 = metrics.f1_at_3
            best_weights = (dense_w, sparse_w)
            best_metrics = metrics
        
        logger.info(f"  Dense={dense_w:.1f}, Sparse={sparse_w:.1f}: F1@3={metrics.f1_at_3:.3f}, P@3={metrics.precision_at_3:.3f}, R@3={metrics.recall_at_3:.3f}")
    
    logger.info(f"Best weights: Dense={best_weights[0]:.1f}, Sparse={best_weights[1]:.1f} (F1@3={best_f1_at_3:.3f})")
    
    return best_weights[0], best_weights[1], best_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retrieval System")
    parser.add_argument(
        "--test-queries",
        type=str,
        default=str(REPO_ROOT / "data" / "retrieval_test_queries.json"),
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Run grid search to optimize RRF weights"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test queries
    queries = load_test_queries(Path(args.test_queries))
    
    # Initialize retriever
    logger.info("Initializing HybridRetriever...")
    retriever = HybridRetriever()
    
    # === Ablation Study: Dense vs Sparse vs Hybrid ===
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY: Dense vs Sparse vs Hybrid")
    logger.info("="*70)
    
    ablation_results = {}
    
    # 1. Dense-only
    logger.info("\n[1/3] Evaluating Dense-only (FAISS)...")
    dense_results = run_retrieval(retriever, queries, method="dense", top_k=args.top_k)
    dense_metrics = evaluate_retrieval_system(dense_results, queries, k_values=[1, 3, 5, 10])
    ablation_results["Dense-only (FAISS)"] = dense_metrics
    
    # 2. Sparse-only
    logger.info("\n[2/3] Evaluating Sparse-only (BM25)...")
    sparse_results = run_retrieval(retriever, queries, method="sparse", top_k=args.top_k)
    sparse_metrics = evaluate_retrieval_system(sparse_results, queries, k_values=[1, 3, 5, 10])
    ablation_results["Sparse-only (BM25)"] = sparse_metrics
    
    # 3. Hybrid (default weights)
    logger.info("\n[3/3] Evaluating Hybrid (RRF, default 0.6/0.4)...")
    hybrid_results = run_retrieval(retriever, queries, method="hybrid", top_k=args.top_k, dense_weight=0.6, sparse_weight=0.4)
    hybrid_metrics = evaluate_retrieval_system(hybrid_results, queries, k_values=[1, 3, 5, 10])
    ablation_results["Hybrid (0.6/0.4)"] = hybrid_metrics
    
    # === Weight Optimization (optional) ===
    if args.optimize_weights:
        logger.info("\n" + "="*70)
        logger.info("RRF WEIGHT OPTIMIZATION")
        logger.info("="*70)
        
        best_dense_w, best_sparse_w, optimized_metrics = optimize_rrf_weights(retriever, queries)
        ablation_results[f"Hybrid (optimized {best_dense_w:.1f}/{best_sparse_w:.1f})"] = optimized_metrics
    
    # === Generate Reports ===
    logger.info("\n" + "="*70)
    logger.info("GENERATING REPORTS")
    logger.info("="*70)
    
    # Comparison report
    comparison_report = compare_retrieval_systems(ablation_results)
    comparison_path = output_dir / "retrieval_comparison.md"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    logger.info(f"Comparison report saved to {comparison_path}")
    
    # Individual reports
    for system_name, metrics in ablation_results.items():
        report = format_evaluation_report(metrics, system_name=system_name)
        safe_name = system_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        report_path = output_dir / f"report_{safe_name}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Individual report saved to {report_path}")
    
    # Export raw metrics to JSON
    metrics_json = {
        system_name: {
            "precision@1": m.precision_at_1,
            "precision@3": m.precision_at_3,
            "precision@5": m.precision_at_5,
            "recall@1": m.recall_at_1,
            "recall@3": m.recall_at_3,
            "recall@5": m.recall_at_5,
            "f1@3": m.f1_at_3,
            "mrr": m.mrr,
            "map": m.map_score,
            "ndcg@3": m.ndcg_at_3,
            "ndcg@5": m.ndcg_at_5,
            "hit_rate@3": m.hit_rate_at_3,
        }
        for system_name, m in ablation_results.items()
    }
    
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Metrics JSON saved to {json_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("="*70)
    print(comparison_report)
    
    logger.info("\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
