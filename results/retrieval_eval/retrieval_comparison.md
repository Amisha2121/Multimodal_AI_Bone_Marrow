# Retrieval System Comparison

## Precision@3, Recall@3, MRR

| System | P@3 | R@3 | MRR | MAP | F1@3 |
|--------|-----|-----|-----|-----|------|
| Dense-only (FAISS) | 0.017 | 0.025 | 0.050 | 0.025 | 0.020 |
| Sparse-only (BM25) | 0.017 | 0.025 | 0.050 | 0.025 | 0.020 |
| Hybrid (0.6/0.4) | 0.000 | 0.000 | 0.010 | 0.005 | 0.000 |

**Best Precision@3**: Dense-only (FAISS) (0.017)
**Best Recall@3**: Dense-only (FAISS) (0.025)
**Best MRR**: Dense-only (FAISS) (0.050)
