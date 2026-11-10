"""
retrieval/eval_metrics.py
─────────────────────────────────────────────────────────────────────────────
Quantitative Retrieval Evaluation Metrics for RAG Systems.

Literature Foundation:
    - Thakur et al. (2021): "BEIR: A Heterogeneous Benchmark for Zero-shot 
      Evaluation of Information Retrieval Models"
      NeurIPS 2021. https://arxiv.org/abs/2104.08663
      
    - Voorhees & Harman (2005): "TREC: Experiment and Evaluation in Information
      Retrieval" MIT Press.
      
    - Järvelin & Kekäläinen (2002): "Cumulated gain-based evaluation of IR techniques"
      ACM TOIS 20(4):422-446.

Implemented Metrics:
    1. Precision@K: What fraction of retrieved items are relevant?
    2. Recall@K: What fraction of relevant items are retrieved?
    3. F1@K: Harmonic mean of precision and recall
    4. MRR (Mean Reciprocal Rank): Position of first relevant item
    5. MAP (Mean Average Precision): Average precision across queries
    6. NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
    7. Hit Rate@K: Did we retrieve at least one relevant item?

Clinical Rationale:
    Real clinical RAG systems must retrieve the CORRECT guideline passages
    (not just similar text). These metrics quantify retrieval quality and
    enable ablation studies (dense vs sparse vs hybrid).

Author: AI Assistant
Date: November 8, 2026
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("eval_metrics")


@dataclass
class QueryRelevance:
    """
    Ground truth relevance judgments for a single query.
    
    Attributes:
        query_id: Unique identifier for the query
        query_text: The query string
        relevant_chunk_ids: Set of chunk IDs that are relevant (binary relevance)
        graded_relevance: Optional dict mapping chunk_id -> relevance score (0-3)
            0 = not relevant
            1 = marginally relevant
            2 = relevant
            3 = highly relevant
    """
    query_id: str
    query_text: str
    relevant_chunk_ids: Set[str]
    graded_relevance: Dict[str, int] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Retrieved results for a single query.
    
    Attributes:
        query_id: Matches QueryRelevance.query_id
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (rank 1, 2, ...)
        scores: Optional scores for each retrieved chunk
    """
    query_id: str
    retrieved_chunk_ids: List[str]
    scores: List[float] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """
    Aggregated evaluation metrics across all queries.
    
    All metrics are averaged over the query set.
    """
    # Precision, Recall, F1 at various K
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    
    f1_at_1: float = 0.0
    f1_at_3: float = 0.0
    f1_at_5: float = 0.0
    f1_at_10: float = 0.0
    
    # Ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_3: float = 0.0  # Normalized Discounted Cumulative Gain
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    
    # Hit rate (binary: did we get at least one relevant?)
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    
    # Aggregate statistics
    num_queries: int = 0
    avg_num_relevant: float = 0.0  # Average number of relevant docs per query
    avg_num_retrieved: float = 0.0  # Average number retrieved per query


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K: Fraction of top-K retrieved items that are relevant.
    
    Formula: P@K = |relevant ∩ retrieved[:K]| / K
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        relevant: Set of relevant item IDs
        k: Cutoff rank
    
    Returns:
        Precision score in [0, 1]
    """
    if k == 0 or len(retrieved) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    num_relevant_in_top_k = sum(1 for item_id in top_k if item_id in relevant)
    
    return num_relevant_in_top_k / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K: Fraction of relevant items that appear in top-K.
    
    Formula: R@K = |relevant ∩ retrieved[:K]| / |relevant|
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        relevant: Set of relevant item IDs
        k: Cutoff rank
    
    Returns:
        Recall score in [0, 1]
    """
    if len(relevant) == 0:
        return 0.0
    
    top_k = set(retrieved[:k])
    num_relevant_in_top_k = len(relevant & top_k)
    
    return num_relevant_in_top_k / len(relevant)


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    F1@K: Harmonic mean of Precision@K and Recall@K.
    
    Formula: F1 = 2 * (P * R) / (P + R)
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        relevant: Set of relevant item IDs
        k: Cutoff rank
    
    Returns:
        F1 score in [0, 1]
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciprocal Rank: 1 / (position of first relevant item).
    
    If no relevant item is retrieved, RR = 0.
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        relevant: Set of relevant item IDs
    
    Returns:
        Reciprocal rank in [0, 1]
    """
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / rank
    
    return 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision: Mean of precision values at each relevant item position.
    
    Formula: AP = (1/|relevant|) * Σ [P(k) * rel(k)]
    where rel(k) = 1 if item at rank k is relevant, 0 otherwise
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        relevant: Set of relevant item IDs
    
    Returns:
        Average precision in [0, 1]
    """
    if len(relevant) == 0:
        return 0.0
    
    num_relevant_seen = 0
    sum_precisions = 0.0
    
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            num_relevant_seen += 1
            precision_at_rank = num_relevant_seen / rank
            sum_precisions += precision_at_rank
    
    return sum_precisions / len(relevant)


def dcg_at_k(retrieved: List[str], graded_relevance: Dict[str, int], k: int) -> float:
    """
    Discounted Cumulative Gain@K: Weighted sum of relevance scores with position discount.
    
    Formula: DCG@K = Σ[i=1 to K] (rel_i / log2(i + 1))
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        graded_relevance: Dict mapping item_id -> relevance score (0-3)
        k: Cutoff rank
    
    Returns:
        DCG score
    """
    dcg = 0.0
    
    for rank, item_id in enumerate(retrieved[:k], start=1):
        rel = graded_relevance.get(item_id, 0)
        dcg += rel / np.log2(rank + 1)
    
    return dcg


def ndcg_at_k(retrieved: List[str], graded_relevance: Dict[str, int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain@K: DCG normalized by ideal DCG.
    
    IDCG = DCG of perfect ranking (all relevant items ranked by relevance score).
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        graded_relevance: Dict mapping item_id -> relevance score (0-3)
        k: Cutoff rank
    
    Returns:
        NDCG score in [0, 1]
    """
    dcg = dcg_at_k(retrieved, graded_relevance, k)
    
    # Compute ideal DCG (perfect ranking)
    ideal_ranking = sorted(graded_relevance.keys(), key=lambda x: graded_relevance[x], reverse=True)
    idcg = dcg_at_k(ideal_ranking, graded_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Hit Rate@K: Binary indicator whether at least one relevant item is in top-K.
    
    Args:
        retrieved: Ordered list of retrieved item IDs
        relevant: Set of relevant item IDs
        k: Cutoff rank
    
    Returns:
        1.0 if hit, 0.0 if miss
    """
    top_k = set(retrieved[:k])
    return 1.0 if len(relevant & top_k) > 0 else 0.0


def evaluate_single_query(
    retrieval_result: RetrievalResult,
    query_relevance: QueryRelevance,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Evaluate retrieval performance for a single query.
    
    Args:
        retrieval_result: Retrieved items for the query
        query_relevance: Ground truth relevance for the query
        k_values: List of K values to evaluate
    
    Returns:
        Dict of metric_name -> score
    """
    retrieved = retrieval_result.retrieved_chunk_ids
    relevant = query_relevance.relevant_chunk_ids
    graded = query_relevance.graded_relevance if query_relevance.graded_relevance else {}
    
    metrics = {}
    
    # Precision, Recall, F1 at various K
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"f1@{k}"] = f1_at_k(retrieved, relevant, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
    
    # Ranking metrics
    metrics["reciprocal_rank"] = reciprocal_rank(retrieved, relevant)
    metrics["average_precision"] = average_precision(retrieved, relevant)
    
    # NDCG (if graded relevance available)
    if graded:
        for k in [3, 5, 10]:
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, graded, k)
    
    return metrics


def evaluate_retrieval_system(
    retrieval_results: List[RetrievalResult],
    query_relevances: List[QueryRelevance],
    k_values: List[int] = [1, 3, 5, 10],
) -> EvaluationMetrics:
    """
    Evaluate retrieval system across multiple queries.
    
    Args:
        retrieval_results: Retrieved results for all queries
        query_relevances: Ground truth relevance for all queries
        k_values: List of K values to evaluate
    
    Returns:
        EvaluationMetrics with aggregated scores
    """
    # Match queries by ID
    relevance_by_id = {qr.query_id: qr for qr in query_relevances}
    results_by_id = {rr.query_id: rr for rr in retrieval_results}
    
    # Compute per-query metrics
    all_metrics = []
    for query_id in relevance_by_id.keys():
        if query_id not in results_by_id:
            logger.warning(f"Query {query_id} has no retrieval results, skipping")
            continue
        
        query_metrics = evaluate_single_query(
            results_by_id[query_id],
            relevance_by_id[query_id],
            k_values=k_values
        )
        all_metrics.append(query_metrics)
    
    if len(all_metrics) == 0:
        logger.error("No queries evaluated!")
        return EvaluationMetrics()
    
    # Aggregate metrics (mean across queries)
    agg = EvaluationMetrics()
    agg.num_queries = len(all_metrics)
    
    # Average metrics
    agg.precision_at_1 = np.mean([m.get("precision@1", 0.0) for m in all_metrics])
    agg.precision_at_3 = np.mean([m.get("precision@3", 0.0) for m in all_metrics])
    agg.precision_at_5 = np.mean([m.get("precision@5", 0.0) for m in all_metrics])
    agg.precision_at_10 = np.mean([m.get("precision@10", 0.0) for m in all_metrics])
    
    agg.recall_at_1 = np.mean([m.get("recall@1", 0.0) for m in all_metrics])
    agg.recall_at_3 = np.mean([m.get("recall@3", 0.0) for m in all_metrics])
    agg.recall_at_5 = np.mean([m.get("recall@5", 0.0) for m in all_metrics])
    agg.recall_at_10 = np.mean([m.get("recall@10", 0.0) for m in all_metrics])
    
    agg.f1_at_1 = np.mean([m.get("f1@1", 0.0) for m in all_metrics])
    agg.f1_at_3 = np.mean([m.get("f1@3", 0.0) for m in all_metrics])
    agg.f1_at_5 = np.mean([m.get("f1@5", 0.0) for m in all_metrics])
    agg.f1_at_10 = np.mean([m.get("f1@10", 0.0) for m in all_metrics])
    
    agg.hit_rate_at_1 = np.mean([m.get("hit_rate@1", 0.0) for m in all_metrics])
    agg.hit_rate_at_3 = np.mean([m.get("hit_rate@3", 0.0) for m in all_metrics])
    agg.hit_rate_at_5 = np.mean([m.get("hit_rate@5", 0.0) for m in all_metrics])
    agg.hit_rate_at_10 = np.mean([m.get("hit_rate@10", 0.0) for m in all_metrics])
    
    agg.mrr = np.mean([m.get("reciprocal_rank", 0.0) for m in all_metrics])
    agg.map_score = np.mean([m.get("average_precision", 0.0) for m in all_metrics])
    
    agg.ndcg_at_3 = np.mean([m.get("ndcg@3", 0.0) for m in all_metrics if "ndcg@3" in m])
    agg.ndcg_at_5 = np.mean([m.get("ndcg@5", 0.0) for m in all_metrics if "ndcg@5" in m])
    agg.ndcg_at_10 = np.mean([m.get("ndcg@10", 0.0) for m in all_metrics if "ndcg@10" in m])
    
    # Statistics
    agg.avg_num_relevant = np.mean([len(relevance_by_id[qid].relevant_chunk_ids) 
                                    for qid in relevance_by_id.keys()])
    agg.avg_num_retrieved = np.mean([len(results_by_id[qid].retrieved_chunk_ids) 
                                     for qid in results_by_id.keys() 
                                     if qid in results_by_id])
    
    return agg


def format_evaluation_report(metrics: EvaluationMetrics, system_name: str = "Retrieval System") -> str:
    """
    Format evaluation metrics as human-readable report.
    
    Args:
        metrics: Aggregated evaluation metrics
        system_name: Name of the retrieval system being evaluated
    
    Returns:
        Formatted markdown report
    """
    report = f"""
# Retrieval Evaluation Report: {system_name}

**Queries Evaluated**: {metrics.num_queries}  
**Avg Relevant per Query**: {metrics.avg_num_relevant:.2f}  
**Avg Retrieved per Query**: {metrics.avg_num_retrieved:.2f}

---

## Precision, Recall, F1

| Metric | @1 | @3 | @5 | @10 |
|--------|----|----|----|----|
| **Precision** | {metrics.precision_at_1:.3f} | {metrics.precision_at_3:.3f} | {metrics.precision_at_5:.3f} | {metrics.precision_at_10:.3f} |
| **Recall** | {metrics.recall_at_1:.3f} | {metrics.recall_at_3:.3f} | {metrics.recall_at_5:.3f} | {metrics.recall_at_10:.3f} |
| **F1** | {metrics.f1_at_1:.3f} | {metrics.f1_at_3:.3f} | {metrics.f1_at_5:.3f} | {metrics.f1_at_10:.3f} |

---

## Ranking Metrics

| Metric | Score |
|--------|-------|
| **MRR** (Mean Reciprocal Rank) | {metrics.mrr:.3f} |
| **MAP** (Mean Average Precision) | {metrics.map_score:.3f} |
| **NDCG@3** | {metrics.ndcg_at_3:.3f} |
| **NDCG@5** | {metrics.ndcg_at_5:.3f} |
| **NDCG@10** | {metrics.ndcg_at_10:.3f} |

---

## Hit Rate

| Metric | @1 | @3 | @5 | @10 |
|--------|----|----|----|----|
| **Hit Rate** | {metrics.hit_rate_at_1:.3f} | {metrics.hit_rate_at_3:.3f} | {metrics.hit_rate_at_5:.3f} | {metrics.hit_rate_at_10:.3f} |

---

## Clinical Interpretation

"""
    
    # Interpret Precision@3 (our main metric since we retrieve top-4)
    p3 = metrics.precision_at_3
    if p3 >= 0.80:
        report += "✅ **HIGH PRECISION**: Most retrieved guidelines are relevant. Excellent retrieval quality.\n"
    elif p3 >= 0.60:
        report += "⚠️ **MODERATE PRECISION**: Some irrelevant guidelines retrieved. Consider improving query formulation or weights.\n"
    else:
        report += "❌ **LOW PRECISION**: Many irrelevant guidelines retrieved. Retrieval needs improvement.\n"
    
    # Interpret Recall@3
    r3 = metrics.recall_at_3
    if r3 >= 0.70:
        report += "✅ **HIGH RECALL**: Most relevant guidelines are retrieved. Good coverage.\n"
    elif r3 >= 0.50:
        report += "⚠️ **MODERATE RECALL**: Some relevant guidelines missed. Consider expanding top-K or improving embedding model.\n"
    else:
        report += "❌ **LOW RECALL**: Many relevant guidelines missed. Retrieval may be too narrow.\n"
    
    # Interpret MRR
    mrr = metrics.mrr
    if mrr >= 0.80:
        report += "✅ **EXCELLENT RANKING**: Relevant guidelines consistently appear in top positions.\n"
    elif mrr >= 0.60:
        report += "⚠️ **GOOD RANKING**: Relevant guidelines appear fairly early, but could be better.\n"
    else:
        report += "❌ **POOR RANKING**: Relevant guidelines often buried. Improve ranking algorithm.\n"
    
    report += "\n---\n\n"
    report += "**Literature**: Thakur et al. (2021) BEIR, Voorhees & Harman (2005) TREC\n"
    
    return report


def compare_retrieval_systems(
    results_dict: Dict[str, EvaluationMetrics],
) -> str:
    """
    Compare multiple retrieval systems side-by-side.
    
    Args:
        results_dict: Dict mapping system_name -> EvaluationMetrics
    
    Returns:
        Comparative markdown table
    """
    if len(results_dict) == 0:
        return "No systems to compare."
    
    report = "# Retrieval System Comparison\n\n"
    report += "## Precision@3, Recall@3, MRR\n\n"
    report += "| System | P@3 | R@3 | MRR | MAP | F1@3 |\n"
    report += "|--------|-----|-----|-----|-----|------|\n"
    
    for name, metrics in results_dict.items():
        report += f"| {name} | {metrics.precision_at_3:.3f} | {metrics.recall_at_3:.3f} | "
        report += f"{metrics.mrr:.3f} | {metrics.map_score:.3f} | {metrics.f1_at_3:.3f} |\n"
    
    # Identify best system for each metric
    best_p3 = max(results_dict.items(), key=lambda x: x[1].precision_at_3)
    best_r3 = max(results_dict.items(), key=lambda x: x[1].recall_at_3)
    best_mrr = max(results_dict.items(), key=lambda x: x[1].mrr)
    
    report += f"\n**Best Precision@3**: {best_p3[0]} ({best_p3[1].precision_at_3:.3f})\n"
    report += f"**Best Recall@3**: {best_r3[0]} ({best_r3[1].recall_at_3:.3f})\n"
    report += f"**Best MRR**: {best_mrr[0]} ({best_mrr[1].mrr:.3f})\n"
    
    return report
