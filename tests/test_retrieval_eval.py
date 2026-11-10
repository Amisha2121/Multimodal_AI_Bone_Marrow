"""
Unit tests for retrieval evaluation metrics.

Tests all metrics: Precision@K, Recall@K, F1@K, MRR, MAP, NDCG, Hit Rate
"""

import pytest
import numpy as np
from retrieval.eval_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    reciprocal_rank,
    average_precision,
    ndcg_at_k,
    hit_rate_at_k,
    evaluate_single_query,
    evaluate_retrieval_system,
    QueryRelevance,
    RetrievalResult,
    EvaluationMetrics
)


class TestPrecisionAtK:
    """Test precision@K metric."""
    
    def test_perfect_precision(self):
        """All retrieved docs are relevant."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc2', 'doc3', 'doc4'}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_zero_precision(self):
        """No retrieved docs are relevant."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc4', 'doc5'}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0
    
    def test_partial_precision(self):
        """Some retrieved docs are relevant."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc1', 'doc3'}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5
    
    def test_k_smaller_than_retrieved(self):
        """K is smaller than number of retrieved docs."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc1', 'doc2'}
        assert precision_at_k(retrieved, relevant, k=2) == 1.0
    
    def test_empty_relevant(self):
        """No relevant docs."""
        retrieved = ['doc1', 'doc2']
        relevant = set()
        assert precision_at_k(retrieved, relevant, k=2) == 0.0
    
    def test_empty_retrieved(self):
        """No retrieved docs."""
        retrieved = []
        relevant = {'doc1', 'doc2'}
        assert precision_at_k(retrieved, relevant, k=5) == 0.0


class TestRecallAtK:
    """Test recall@K metric."""
    
    def test_perfect_recall(self):
        """All relevant docs are retrieved."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc1', 'doc2'}
        assert recall_at_k(retrieved, relevant, k=4) == 1.0
    
    def test_zero_recall(self):
        """No relevant docs are retrieved."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc3', 'doc4'}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0
    
    def test_partial_recall(self):
        """Some relevant docs are retrieved."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc4', 'doc5', 'doc6'}
        assert recall_at_k(retrieved, relevant, k=3) == 0.25
    
    def test_empty_relevant(self):
        """No relevant docs - recall is 0."""
        retrieved = ['doc1', 'doc2']
        relevant = set()
        assert recall_at_k(retrieved, relevant, k=2) == 0.0
    
    def test_empty_retrieved(self):
        """No retrieved docs."""
        retrieved = []
        relevant = {'doc1', 'doc2'}
        assert recall_at_k(retrieved, relevant, k=5) == 0.0


class TestF1AtK:
    """Test F1@K metric."""
    
    def test_perfect_f1(self):
        """Perfect precision and recall."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc1', 'doc2'}
        assert f1_at_k(retrieved, relevant, k=2) == 1.0
    
    def test_zero_f1(self):
        """Zero precision and recall."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc3', 'doc4'}
        assert f1_at_k(retrieved, relevant, k=2) == 0.0
    
    def test_harmonic_mean(self):
        """F1 is harmonic mean of P and R."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc4'}
        
        p = precision_at_k(retrieved, relevant, k=3)  # 1/3
        r = recall_at_k(retrieved, relevant, k=3)     # 1/2
        f1 = f1_at_k(retrieved, relevant, k=3)
        
        expected_f1 = 2 * (p * r) / (p + r)
        assert abs(f1 - expected_f1) < 1e-6
    
    def test_empty_relevant(self):
        """No relevant docs."""
        retrieved = ['doc1', 'doc2']
        relevant = set()
        assert f1_at_k(retrieved, relevant, k=2) == 0.0


class TestReciprocalRank:
    """Test Reciprocal Rank metric."""
    
    def test_first_position(self):
        """First retrieved doc is relevant."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc4'}
        assert reciprocal_rank(retrieved, relevant) == 1.0
    
    def test_second_position(self):
        """Second retrieved doc is relevant."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc2', 'doc4'}
        assert reciprocal_rank(retrieved, relevant) == 0.5
    
    def test_third_position(self):
        """Third retrieved doc is relevant."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc3', 'doc4'}
        assert abs(reciprocal_rank(retrieved, relevant) - 1/3) < 1e-6
    
    def test_no_relevant_retrieved(self):
        """No relevant docs retrieved."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc3', 'doc4'}
        assert reciprocal_rank(retrieved, relevant) == 0.0
    
    def test_empty_retrieved(self):
        """No docs retrieved."""
        retrieved = []
        relevant = {'doc1', 'doc2'}
        assert reciprocal_rank(retrieved, relevant) == 0.0


class TestAveragePrecision:
    """Test Average Precision (AP) metric."""
    
    def test_perfect_ap(self):
        """All relevant docs at top positions."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc2'}
        # Precision at positions: P@1=1.0, P@2=1.0
        # AP = (1.0 + 1.0) / 2 = 1.0
        assert average_precision(retrieved, relevant) == 1.0
    
    def test_interleaved_ap(self):
        """Relevant docs interleaved with irrelevant."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc1', 'doc3'}
        # Relevant at positions 1 and 3
        # P@1 = 1/1 = 1.0
        # P@3 = 2/3 = 0.667
        # AP = (1.0 + 0.667) / 2 = 0.833
        ap = average_precision(retrieved, relevant)
        assert abs(ap - 0.833) < 0.01
    
    def test_zero_ap(self):
        """No relevant docs retrieved."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc3', 'doc4'}
        assert average_precision(retrieved, relevant) == 0.0
    
    def test_late_relevant(self):
        """Relevant docs at end."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc4'}
        # P@4 = 1/4 = 0.25
        # AP = 0.25 / 1 = 0.25
        assert average_precision(retrieved, relevant) == 0.25
    
    def test_empty_retrieved(self):
        """No docs retrieved."""
        retrieved = []
        relevant = {'doc1', 'doc2'}
        assert average_precision(retrieved, relevant) == 0.0


class TestNDCGAtK:
    """Test NDCG@K metric."""
    
    def test_perfect_ndcg(self):
        """Perfect ranking."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevance = {'doc1': 3, 'doc2': 2, 'doc3': 1}
        assert ndcg_at_k(retrieved, relevance, k=3) == 1.0
    
    def test_reversed_ranking(self):
        """Worst possible ranking."""
        retrieved = ['doc3', 'doc2', 'doc1']
        relevance = {'doc1': 3, 'doc2': 2, 'doc3': 1}
        ndcg = ndcg_at_k(retrieved, relevance, k=3)
        # NDCG should be < 1.0 for non-optimal ranking
        assert 0 < ndcg < 1.0
    
    def test_binary_relevance(self):
        """Binary relevance (0/1)."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevance = {'doc1': 1, 'doc2': 1, 'doc3': 0, 'doc4': 0}
        ndcg = ndcg_at_k(retrieved, relevance, k=4)
        assert ndcg == 1.0  # Relevant docs at top
    
    def test_empty_relevance(self):
        """No relevant docs."""
        retrieved = ['doc1', 'doc2']
        relevance = {}
        assert ndcg_at_k(retrieved, relevance, k=2) == 0.0
    
    def test_k_parameter(self):
        """NDCG respects K parameter."""
        retrieved = ['doc3', 'doc1', 'doc2']  # Bad order
        relevance = {'doc1': 3, 'doc2': 2, 'doc3': 1}
        
        ndcg_1 = ndcg_at_k(retrieved, relevance, k=1)
        ndcg_3 = ndcg_at_k(retrieved, relevance, k=3)
        
        # Both should be < 1.0 (non-optimal)
        assert 0 < ndcg_1 < 1.0
        assert 0 < ndcg_3 < 1.0


class TestHitRateAtK:
    """Test Hit Rate@K metric."""
    
    def test_hit(self):
        """At least one relevant doc retrieved."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc2', 'doc5'}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_miss(self):
        """No relevant docs retrieved."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc3', 'doc4'}
        assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
    
    def test_first_position_hit(self):
        """Relevant doc at first position."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1'}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_last_position_hit(self):
        """Relevant doc at last position."""
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc3'}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_beyond_k(self):
        """Relevant doc beyond K."""
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = {'doc4'}
        assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0


class TestEvaluationAPI:
    """Test high-level evaluation API."""
    
    def test_evaluate_single_query_binary(self):
        """Evaluate single query with binary relevance."""
        retrieval_result = RetrievalResult(
            query_id='q1',
            retrieved_chunk_ids=['doc1', 'doc2', 'doc3']
        )
        
        query_relevance = QueryRelevance(
            query_id='q1',
            query_text='test query',
            relevant_chunk_ids={'doc1', 'doc3'}
        )
        
        metrics = evaluate_single_query(
            retrieval_result,
            query_relevance,
            k_values=[3]
        )
        
        assert 'precision@3' in metrics
        assert 'recall@3' in metrics
        assert 'f1@3' in metrics
        assert 'reciprocal_rank' in metrics
        assert 'average_precision' in metrics
        assert 'hit_rate@3' in metrics
        
        # Check values
        assert metrics['precision@3'] == 2/3  # 2 relevant out of 3 retrieved
        assert metrics['recall@3'] == 1.0  # Both relevant items retrieved
        assert metrics['reciprocal_rank'] == 1.0  # First doc is relevant
        assert metrics['hit_rate@3'] == 1.0
    
    def test_evaluate_single_query_graded(self):
        """Evaluate single query with graded relevance."""
        retrieval_result = RetrievalResult(
            query_id='q1',
            retrieved_chunk_ids=['doc1', 'doc2', 'doc3']
        )
        
        query_relevance = QueryRelevance(
            query_id='q1',
            query_text='test query',
            relevant_chunk_ids={'doc1', 'doc2', 'doc3'},
            graded_relevance={'doc1': 3, 'doc2': 2, 'doc3': 1}
        )
        
        metrics = evaluate_single_query(
            retrieval_result,
            query_relevance,
            k_values=[3]
        )
        
        assert 'ndcg@3' in metrics
        assert metrics['ndcg@3'] == 1.0  # Perfect ranking
    
    def test_evaluate_retrieval_system(self):
        """Evaluate system across multiple queries."""
        retrieval_results = [
            RetrievalResult('q1', ['doc1', 'doc2']),
            RetrievalResult('q2', ['doc3', 'doc4'])
        ]
        
        query_relevances = [
            QueryRelevance('q1', 'test query 1', {'doc1', 'doc2'}),  # Perfect
            QueryRelevance('q2', 'test query 2', {'doc3'})  # Partial
        ]
        
        metrics = evaluate_retrieval_system(
            retrieval_results,
            query_relevances,
            k_values=[2]
        )
        
        assert isinstance(metrics, EvaluationMetrics)
        assert hasattr(metrics, 'precision_at_1')
        assert hasattr(metrics, 'recall_at_1')
        assert hasattr(metrics, 'mrr')
        
        # Note: k_values=[2] doesn't populate the fixed fields (1,3,5,10)
        # So we just check structure is correct
        assert metrics.num_queries == 2
        assert metrics.mrr == 1.0  # Both queries have relevant doc at position 1
    
    def test_empty_evaluation(self):
        """Evaluate with no results."""
        metrics = evaluate_retrieval_system([], [], k_values=[3])
        
        assert isinstance(metrics, EvaluationMetrics)
        # Should have 0 or NaN values
        assert metrics.num_queries == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_k_larger_than_retrieved(self):
        """K is larger than number of retrieved docs."""
        retrieved = ['doc1', 'doc2']
        relevant = {'doc1', 'doc2', 'doc3'}
        
        # Should still work, using available docs
        p = precision_at_k(retrieved, relevant, k=5)
        r = recall_at_k(retrieved, relevant, k=5)
        
        # P@5 = 2 / 5 = 0.4 (divides by k, not len(retrieved))
        assert p == 0.4
        # R@5 = 2 / 3 = 0.667 (2 out of 3 relevant docs retrieved)
        assert abs(r - 2/3) < 1e-6
    
    def test_very_large_k(self):
        """Very large K value."""
        retrieved = ['doc1']
        relevant = {'doc1'}
        
        p = precision_at_k(retrieved, relevant, k=1000)
        r = recall_at_k(retrieved, relevant, k=1000)
        h = hit_rate_at_k(retrieved, relevant, k=1000)
        
        # P@1000 = 1 / 1000 = 0.001 (penalizes retrieving too few)
        assert p == 0.001
        # R@1000 = 1 / 1 = 1.0 (all relevant docs retrieved)
        assert r == 1.0
        # Hit@1000 = 1.0 (at least one relevant doc retrieved)
        assert h == 1.0
    
    def test_duplicate_retrieved_docs(self):
        """Duplicate docs in retrieved list."""
        retrieved = ['doc1', 'doc1', 'doc2']
        relevant = {'doc1', 'doc2'}
        
        # Should handle duplicates
        p = precision_at_k(retrieved, relevant, k=3)
        assert p == 1.0  # All are relevant (even duplicates)
    
    def test_relevance_scores_zero(self):
        """Relevance score of 0."""
        retrieved = ['doc1', 'doc2']
        relevance = {'doc1': 0, 'doc2': 0}
        
        ndcg = ndcg_at_k(retrieved, relevance, k=2)
        assert ndcg == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
