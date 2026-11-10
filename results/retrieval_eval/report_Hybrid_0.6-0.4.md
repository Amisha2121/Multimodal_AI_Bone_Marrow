
# Retrieval Evaluation Report: Hybrid (0.6/0.4)

**Queries Evaluated**: 20  
**Avg Relevant per Query**: 2.05  
**Avg Retrieved per Query**: 10.00

---

## Precision, Recall, F1

| Metric | @1 | @3 | @5 | @10 |
|--------|----|----|----|----|
| **Precision** | 0.000 | 0.000 | 0.010 | 0.005 |
| **Recall** | 0.000 | 0.000 | 0.025 | 0.025 |
| **F1** | 0.000 | 0.000 | 0.014 | 0.008 |

---

## Ranking Metrics

| Metric | Score |
|--------|-------|
| **MRR** (Mean Reciprocal Rank) | 0.010 |
| **MAP** (Mean Average Precision) | 0.005 |
| **NDCG@3** | 0.000 |
| **NDCG@5** | 0.014 |
| **NDCG@10** | 0.014 |

---

## Hit Rate

| Metric | @1 | @3 | @5 | @10 |
|--------|----|----|----|----|
| **Hit Rate** | 0.000 | 0.000 | 0.050 | 0.050 |

---

## Clinical Interpretation

❌ **LOW PRECISION**: Many irrelevant guidelines retrieved. Retrieval needs improvement.
❌ **LOW RECALL**: Many relevant guidelines missed. Retrieval may be too narrow.
❌ **POOR RANKING**: Relevant guidelines often buried. Improve ranking algorithm.

---

**Literature**: Thakur et al. (2021) BEIR, Voorhees & Harman (2005) TREC
