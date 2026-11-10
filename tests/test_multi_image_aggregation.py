"""
tests/test_multi_image_aggregation.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for multi-image aggregation module (Improvement #3).

Literature:
- Campanella et al. (2019) Nature Medicine 25:1301-1309
  "Clinical-grade computational pathology using weakly supervised deep learning"

Test coverage:
1. Single-image handling (graceful degradation)
2. Multi-image consistency scoring
3. Outlier detection (MAD-based)
4. Confidence interval computation (t-distribution)
5. Clinical recommendation logic
6. Edge cases (zero cells, extreme variance)
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np
from typing import List, Dict

# Use the actual implementation's data structures
from pipeline.multi_image_aggregation import (
    ImageLevelStatistics,
    AggregatedStatistics,
    aggregate_multi_image_predictions,
    compute_confidence_interval,
    detect_outliers,
    compute_consistency_score,
    get_aggregation_recommendation,
    format_aggregation_report,
)


def create_mock_predictions(
    image_id: str, 
    num_cells: int, 
    blast_fraction: float,
    mean_confidence: float = 0.85
) -> List[Dict]:
    """Create mock cell predictions for testing."""
    predictions = []
    num_blasts = int(num_cells * blast_fraction)
    
    for i in range(num_cells):
        is_blast = i < num_blasts
        predictions.append({
            "cell_id": f"{image_id}_CELL-{i:03d}",
            "image_id": image_id,
            "predicted_class": "Blast" if is_blast else "Granulocyte (Mature/Band)",
            "confidence": mean_confidence + np.random.uniform(-0.05, 0.05),
        })
    
    return predictions


class TestMultiImageAggregation(unittest.TestCase):
    """Test suite for multi-image aggregation functionality."""
    
    def test_single_image_graceful_degradation(self):
        """Test that single-image cases return appropriate statistics."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.25)
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        self.assertEqual(result.total_images, 1)
        self.assertAlmostEqual(result.blast_fraction_mean, 0.25, places=2)
        self.assertEqual(len(result.outlier_images), 0)
        self.assertGreater(result.consistency_score, 0)
        self.assertIn("Single image", ' '.join(result.consistency_warnings))
    
    def test_multi_image_consistent_case(self):
        """Test aggregation with consistent blast fractions across images."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.24),
            "IMAGE-02": create_mock_predictions("IMAGE-02", 95, blast_fraction=0.26),
            "IMAGE-03": create_mock_predictions("IMAGE-03", 110, blast_fraction=0.25),
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        self.assertEqual(result.total_images, 3)
        self.assertAlmostEqual(result.blast_fraction_mean, 0.25, delta=0.02)
        self.assertLess(result.blast_fraction_std, 0.05)  # Low variance
        self.assertTrue(result.is_consistent)
        self.assertEqual(len(result.outlier_images), 0)
        self.assertGreater(result.consistency_score, 0.90)  # High consistency
    
    def test_multi_image_inconsistent_case(self):
        """Test detection of inconsistent blast fractions (outliers)."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.15),
            "IMAGE-02": create_mock_predictions("IMAGE-02", 95, blast_fraction=0.18),
            "IMAGE-03": create_mock_predictions("IMAGE-03", 110, blast_fraction=0.65),  # Outlier
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        self.assertEqual(result.total_images, 3)
        self.assertFalse(result.is_consistent)  # Should detect inconsistency
        self.assertGreater(len(result.outlier_images), 0)
        self.assertIn("IMAGE-03", result.outlier_images)
        self.assertLess(result.consistency_score, 0.70)  # Low consistency
    
    def test_confidence_interval_computation(self):
        """Test t-distribution confidence interval (α=0.05)."""
        values = [0.20, 0.22, 0.24, 0.21, 0.23]
        ci_lower, ci_upper = compute_confidence_interval(values, confidence_level=0.95)
        
        mean = np.mean(values)
        self.assertLess(ci_lower, mean)
        self.assertGreater(ci_upper, mean)
        self.assertGreater(ci_upper - ci_lower, 0)  # Non-zero interval
    
    def test_outlier_detection_mad(self):
        """Test MAD-based outlier detection (threshold=2.5σ)."""
        values = [0.20, 0.21, 0.22, 0.19, 0.80]  # Last is outlier
        outlier_indices = detect_outliers(values, threshold=2.5)
        
        self.assertEqual(len(outlier_indices), 1)
        self.assertIn(4, outlier_indices)  # Index of 0.80
    
    def test_consistency_score(self):
        """Test consistency scoring (1.0 - coefficient of variation)."""
        # Low variance = high consistency
        consistent_values = [0.20, 0.21, 0.20, 0.21]
        consistent_score = compute_consistency_score(consistent_values)
        self.assertGreater(consistent_score, 0.90)
        
        # High variance = low consistency
        inconsistent_values = [0.10, 0.50, 0.20, 0.80]
        inconsistent_score = compute_consistency_score(inconsistent_values)
        self.assertLess(inconsistent_score, 0.60)
    
    def test_zero_cells_all_images(self):
        """Test graceful handling when all images have zero cells."""
        predictions = {
            "IMAGE-01": [],
            "IMAGE-02": [],
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        self.assertEqual(result.total_images, 2)
        self.assertEqual(result.blast_fraction_mean, 0.0)
        self.assertEqual(result.total_cells, 0)
        # When all images have zero cells, consistency_score is 1.0 (perfect consistency in zeros)
        self.assertEqual(result.consistency_score, 1.0)
    
    def test_zero_cells_some_images(self):
        """Test handling when some images have zero cells."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.25),
            "IMAGE-02": [],  # Zero cells
            "IMAGE-03": create_mock_predictions("IMAGE-03", 80, blast_fraction=0.28),
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        # Should compute statistics from all images (including zero-cell ones)
        self.assertEqual(result.total_images, 3)
        self.assertGreater(result.total_cells, 0)
        # Zero-cell images might be flagged as having low cell count
        self.assertGreater(len(result.low_cell_count_images), 0)
    
    def test_extreme_variance(self):
        """Test handling of extreme blast fraction variance."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.05),
            "IMAGE-02": create_mock_predictions("IMAGE-02", 100, blast_fraction=0.50),
            "IMAGE-03": create_mock_predictions("IMAGE-03", 100, blast_fraction=0.95),
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        self.assertFalse(result.is_consistent)
        self.assertGreater(result.blast_fraction_std, 0.20)  # High variance
        self.assertLess(result.consistency_score, 0.50)
    
    def test_clinical_recommendation_logic(self):
        """Test clinical recommendation thresholds via get_aggregation_recommendation."""
        # Test with a mock AggregatedStatistics object
        from pipeline.multi_image_aggregation import AggregatedStatistics
        
        # Create test case with high consistency
        agg = AggregatedStatistics(
            total_images=3,
            total_cells=300,
            blast_fraction_mean=0.25,
            blast_fraction_std=0.02,
            consistency_score=0.92,
            is_consistent=True,
            outlier_images=[],
        )
        
        recommendations = get_aggregation_recommendation(agg)
        self.assertIsInstance(recommendations, dict)
        # Check for actual keys returned by the function
        self.assertIn("confidence_level", recommendations)
        self.assertIn("diagnostic_certainty", recommendations)
        self.assertEqual(recommendations["confidence_level"], "high")
    
    def test_format_aggregation_report(self):
        """Test report formatting for UI display."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.24),
            "IMAGE-02": create_mock_predictions("IMAGE-02", 95, blast_fraction=0.26),
            "IMAGE-03": create_mock_predictions("IMAGE-03", 110, blast_fraction=0.25),
        }
        
        result = aggregate_multi_image_predictions(predictions)
        report = format_aggregation_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn("Multi-Image Aggregation", report)
        self.assertIn("Blast Fraction", report)
        # Report should contain the blast percentage (around 24-26%)
        self.assertTrue(any(pct in report for pct in ["24", "25", "26"]))
    
    def test_confidence_interval_with_two_images(self):
        """Test CI computation with minimum viable sample (n=2)."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 100, blast_fraction=0.20),
            "IMAGE-02": create_mock_predictions("IMAGE-02", 100, blast_fraction=0.30),
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        # With n=2, t-distribution has wide CI
        self.assertLess(result.blast_fraction_ci_lower, result.blast_fraction_mean)
        self.assertGreater(result.blast_fraction_ci_upper, result.blast_fraction_mean)
        ci_width = result.blast_fraction_ci_upper - result.blast_fraction_ci_lower
        self.assertGreater(ci_width, 0.05)  # Wide CI for n=2
    
    def test_aggregation_with_different_cell_counts(self):
        """Test aggregation when images have vastly different cell counts."""
        predictions = {
            "IMAGE-01": create_mock_predictions("IMAGE-01", 15, blast_fraction=0.25),   # Very low count (<20)
            "IMAGE-02": create_mock_predictions("IMAGE-02", 200, blast_fraction=0.24),  # High count
            "IMAGE-03": create_mock_predictions("IMAGE-03", 100, blast_fraction=0.26),  # Medium
        }
        
        result = aggregate_multi_image_predictions(predictions)
        
        # Should handle different sample sizes gracefully
        self.assertEqual(result.total_images, 3)
        self.assertEqual(result.total_cells, 315)
        self.assertAlmostEqual(result.blast_fraction_mean, 0.25, delta=0.02)
        # Low count image should be flagged (default threshold is 20 cells)
        self.assertIn("IMAGE-01", result.low_cell_count_images)


def run_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultiImageAggregation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("MULTI-IMAGE AGGREGATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All multi-image aggregation tests PASSED")
    else:
        print("\n❌ Some tests FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
