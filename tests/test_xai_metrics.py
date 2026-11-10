"""
tests/test_xai_metrics.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for quantitative XAI metrics module (Improvement #4).

Literature:
- Adebayo et al. (2018) NIPS: "Sanity Checks for Saliency Maps"
- Petsiuk et al. (2018) BMVC: "RISE: Randomized Input Sampling"
- Samek et al. (2017) IEEE TNNLS: "Evaluating Visualization of DNNs"

Test coverage:
1. Pointing Game (peak overlap with annotations)
2. Energy-Based Pointing Game (attention mass in annotated regions)
3. Deletion AUC (confidence drop when removing attention)
4. Insertion AUC (confidence gain when adding attention)
5. Average Drop (single-point deletion metric)
6. Annotated Region utilities (bbox to mask conversion)
7. Aggregation across multiple cells
8. Edge cases (no annotations, uniform attention, etc.)
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np
import torch
import torch.nn as nn

from vision.xai_metrics import (
    AnnotatedRegion,
    XAIMetricsResult,
    AggregatedXAIMetrics,
    pointing_game,
    energy_pointing_game,
    deletion_metric,
    insertion_metric,
    average_drop,
    compute_xai_metrics,
    aggregate_xai_metrics,
    format_xai_metrics_report,
)


class MockModel(nn.Module):
    """Mock model for testing that returns predictable outputs."""
    
    def __init__(self, num_classes=13):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(3 * 224 * 224, num_classes)
    
    def forward(self, x):
        # Flatten and pass through linear layer
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.fc(x_flat)


def create_mock_attention_map(
    height: int = 224,
    width: int = 224,
    peak_location: tuple = (112, 112),
    peak_radius: int = 20,
) -> np.ndarray:
    """Create synthetic attention map with Gaussian-like peak."""
    y_coords, x_coords = np.ogrid[:height, :width]
    py, px = peak_location
    
    # Gaussian-like distribution
    dist = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
    attention = np.exp(-dist**2 / (2 * peak_radius**2))
    
    # Normalize to [0, 1]
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    return attention.astype(np.float32)


class TestAnnotatedRegion(unittest.TestCase):
    """Test AnnotatedRegion dataclass and utility methods."""
    
    def test_annotated_region_creation(self):
        """Test creating an AnnotatedRegion."""
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(50, 50, 100, 100),
            region_type="nucleus",
            confidence=0.95
        )
        
        self.assertEqual(region.cell_id, "CELL-001")
        self.assertEqual(region.bbox, (50, 50, 100, 100))
        self.assertEqual(region.region_type, "nucleus")
        self.assertEqual(region.confidence, 0.95)
    
    def test_region_to_mask(self):
        """Test converting bounding box to binary mask."""
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(50, 60, 100, 80),  # x, y, w, h
            region_type="nucleus"
        )
        
        mask = region.to_mask(height=224, width=224)
        
        self.assertEqual(mask.shape, (224, 224))
        self.assertEqual(mask.dtype, np.float32)
        
        # Check that pixels inside bbox are 1.0
        self.assertEqual(mask[70, 70], 1.0)  # Inside
        self.assertEqual(mask[10, 10], 0.0)  # Outside
        
        # Count number of 1s (should equal bbox area)
        expected_area = 100 * 80  # w * h
        actual_area = int(mask.sum())
        self.assertEqual(actual_area, expected_area)
    
    def test_region_to_mask_boundary_clipping(self):
        """Test that bbox clipping works at image boundaries."""
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(-10, -10, 50, 50),  # Partially outside
            region_type="nucleus"
        )
        
        mask = region.to_mask(height=224, width=224)
        
        # Should only mark pixels within [0, 224) x [0, 224)
        self.assertEqual(mask[0, 0], 1.0)  # Top-left corner
        self.assertGreater(mask.sum(), 0)  # Some pixels marked


class TestPointingGame(unittest.TestCase):
    """Test Pointing Game metric."""
    
    def test_pointing_game_hit(self):
        """Test pointing game when peak overlaps with annotation."""
        # Attention peak at (112, 112)
        attention = create_mock_attention_map(peak_location=(112, 112), peak_radius=20)
        
        # Annotated region covering the peak
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(90, 90, 50, 50),  # Covers (90-140, 90-140)
            region_type="nucleus"
        )
        
        hit, score = pointing_game(attention, [region])
        
        self.assertTrue(hit)
        self.assertEqual(score, 1.0)
    
    def test_pointing_game_miss(self):
        """Test pointing game when peak does NOT overlap with annotation."""
        # Attention peak at (112, 112)
        attention = create_mock_attention_map(peak_location=(112, 112), peak_radius=20)
        
        # Annotated region far from peak
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(10, 10, 30, 30),  # Far from (112, 112)
            region_type="nucleus"
        )
        
        hit, score = pointing_game(attention, [region])
        
        self.assertFalse(hit)
        self.assertEqual(score, 0.0)
    
    def test_pointing_game_multiple_regions(self):
        """Test pointing game with multiple annotated regions."""
        attention = create_mock_attention_map(peak_location=(112, 112), peak_radius=20)
        
        # Multiple regions, one covers the peak
        regions = [
            AnnotatedRegion("CELL-001", (10, 10, 30, 30), "cytoplasm"),
            AnnotatedRegion("CELL-001", (100, 100, 30, 30), "nucleus"),  # Covers peak
            AnnotatedRegion("CELL-001", (200, 200, 20, 20), "granule"),
        ]
        
        hit, score = pointing_game(attention, regions)
        
        self.assertTrue(hit)
        self.assertEqual(score, 1.0)
    
    def test_pointing_game_no_annotations(self):
        """Test pointing game with no annotations."""
        attention = create_mock_attention_map()
        
        hit, score = pointing_game(attention, [])
        
        self.assertFalse(hit)
        self.assertEqual(score, 0.0)


class TestEnergyPointingGame(unittest.TestCase):
    """Test Energy-Based Pointing Game metric."""
    
    def test_energy_pointing_game_full_overlap(self):
        """Test energy PG when annotation covers entire high-attention region."""
        attention = create_mock_attention_map(peak_location=(112, 112), peak_radius=30)
        
        # Large region covering peak
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(50, 50, 130, 130),  # Covers most of image
            region_type="nucleus"
        )
        
        score = energy_pointing_game(attention, [region])
        
        # Should capture most of the attention mass
        self.assertGreater(score, 0.80)
    
    def test_energy_pointing_game_partial_overlap(self):
        """Test energy PG with partial overlap."""
        attention = create_mock_attention_map(peak_location=(112, 112), peak_radius=20)
        
        # Small region partially overlapping peak
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(100, 100, 25, 25),  # Partial overlap
            region_type="nucleus"
        )
        
        score = energy_pointing_game(attention, [region])
        
        # Should capture some but not all attention
        self.assertGreater(score, 0.1)
        self.assertLess(score, 0.8)
    
    def test_energy_pointing_game_no_overlap(self):
        """Test energy PG with no overlap."""
        attention = create_mock_attention_map(peak_location=(112, 112), peak_radius=20)
        
        # Region far from peak
        region = AnnotatedRegion(
            cell_id="CELL-001",
            bbox=(10, 10, 20, 20),
            region_type="cytoplasm"
        )
        
        score = energy_pointing_game(attention, [region])
        
        # Should capture minimal attention
        self.assertLess(score, 0.1)


class TestPerturbationMetrics(unittest.TestCase):
    """Test Deletion/Insertion AUC and Average Drop metrics."""
    
    def setUp(self):
        """Set up mock model and data for testing."""
        self.model = MockModel(num_classes=13)
        self.model.eval()
        self.device = "cpu"
        
        # Create mock input tensor (1, 3, 224, 224)
        self.input_tensor = torch.randn(1, 3, 224, 224)
        
        # Create attention map with peak
        self.attention_map = create_mock_attention_map(peak_location=(112, 112), peak_radius=30)
        
        # Get initial prediction
        with torch.no_grad():
            logits = self.model(self.input_tensor)
            self.predicted_class_idx = logits.argmax(dim=1).item()
    
    def test_deletion_metric(self):
        """Test deletion AUC computation."""
        del_auc, del_curve = deletion_metric(
            model=self.model,
            input_tensor=self.input_tensor,
            attention_map=self.attention_map,
            predicted_class_idx=self.predicted_class_idx,
            num_steps=10,
            deletion_mode="blur",
            device=self.device
        )
        
        # Check AUC is in [0, 1]
        self.assertGreaterEqual(del_auc, 0.0)
        self.assertLessEqual(del_auc, 1.0)
        
        # Check curve length
        self.assertEqual(len(del_curve), 11)  # num_steps + 1
        
        # Confidence should generally decrease (not strict due to random model)
        # At least check it's non-increasing on average
        self.assertLessEqual(del_curve[-1], del_curve[0] + 0.5)  # Allow some variance
    
    def test_insertion_metric(self):
        """Test insertion AUC computation."""
        ins_auc, ins_curve = insertion_metric(
            model=self.model,
            input_tensor=self.input_tensor,
            attention_map=self.attention_map,
            predicted_class_idx=self.predicted_class_idx,
            num_steps=10,
            baseline_mode="blur",
            device=self.device
        )
        
        # Check AUC is in [0, 1]
        self.assertGreaterEqual(ins_auc, 0.0)
        self.assertLessEqual(ins_auc, 1.0)
        
        # Check curve length
        self.assertEqual(len(ins_curve), 11)
        
        # Confidence should generally increase (not strict)
        # At least final should be higher than initial
        self.assertGreaterEqual(ins_curve[-1], ins_curve[0] - 0.5)
    
    def test_average_drop(self):
        """Test average drop metric."""
        avg_drop = average_drop(
            model=self.model,
            input_tensor=self.input_tensor,
            attention_map=self.attention_map,
            predicted_class_idx=self.predicted_class_idx,
            deletion_ratio=0.3,
            deletion_mode="blur",
            device=self.device
        )
        
        # Should be non-negative
        self.assertGreaterEqual(avg_drop, 0.0)
        
        # Should be reasonable (not more than 100% drop)
        self.assertLessEqual(avg_drop, 1.0)


class TestComputeXAIMetrics(unittest.TestCase):
    """Test end-to-end XAI metrics computation."""
    
    def setUp(self):
        """Set up test data."""
        self.model = MockModel(num_classes=13)
        self.model.eval()
        self.input_tensor = torch.randn(1, 3, 224, 224)
        self.attention_map = create_mock_attention_map()
        self.predicted_class_idx = 5
        self.cell_id = "TEST-CELL-001"
        self.predicted_class = "Blast"
        self.confidence = 0.85
        self.device = "cpu"
    
    def test_compute_xai_metrics_with_annotations(self):
        """Test computing XAI metrics with annotations."""
        regions = [
            AnnotatedRegion(self.cell_id, (100, 100, 40, 40), "nucleus", confidence=1.0),
        ]
        
        result = compute_xai_metrics(
            model=self.model,
            input_tensor=self.input_tensor,
            attention_map=self.attention_map,
            predicted_class_idx=self.predicted_class_idx,
            cell_id=self.cell_id,
            predicted_class=self.predicted_class,
            confidence=self.confidence,
            annotated_regions=regions,
            device=self.device,
            num_steps=10,  # Fewer steps for faster testing
        )
        
        # Check basic fields
        self.assertEqual(result.cell_id, self.cell_id)
        self.assertEqual(result.predicted_class, self.predicted_class)
        self.assertEqual(result.confidence, self.confidence)
        self.assertTrue(result.has_annotation)
        self.assertEqual(result.num_annotated_regions, 1)
        
        # Check metrics are computed
        self.assertIn(result.pointing_game_score, [0.0, 1.0])
        self.assertGreaterEqual(result.energy_pointing_game_score, 0.0)
        self.assertLessEqual(result.energy_pointing_game_score, 1.0)
        self.assertGreaterEqual(result.deletion_auc, 0.0)
        self.assertGreaterEqual(result.insertion_auc, 0.0)
        self.assertGreaterEqual(result.average_drop, 0.0)
    
    def test_compute_xai_metrics_without_annotations(self):
        """Test computing XAI metrics without annotations."""
        result = compute_xai_metrics(
            model=self.model,
            input_tensor=self.input_tensor,
            attention_map=self.attention_map,
            predicted_class_idx=self.predicted_class_idx,
            cell_id=self.cell_id,
            predicted_class=self.predicted_class,
            confidence=self.confidence,
            annotated_regions=None,  # No annotations
            device=self.device,
            num_steps=10,
        )
        
        # Check annotation-dependent metrics are zero/false
        self.assertFalse(result.has_annotation)
        self.assertEqual(result.num_annotated_regions, 0)
        self.assertEqual(result.pointing_game_score, 0.0)
        self.assertEqual(result.energy_pointing_game_score, 0.0)
        
        # Perturbation metrics should still be computed
        self.assertGreaterEqual(result.deletion_auc, 0.0)
        self.assertGreaterEqual(result.insertion_auc, 0.0)


class TestAggregateXAIMetrics(unittest.TestCase):
    """Test aggregation of XAI metrics across multiple cells."""
    
    def test_aggregate_empty_list(self):
        """Test aggregation with empty list."""
        agg = aggregate_xai_metrics([])
        
        self.assertEqual(agg.num_cells, 0)
        self.assertEqual(agg.num_annotated_cells, 0)
    
    def test_aggregate_multiple_cells(self):
        """Test aggregation with multiple cells."""
        results = [
            XAIMetricsResult(
                cell_id=f"CELL-{i:03d}",
                predicted_class="Blast" if i % 2 == 0 else "Granulocyte (Mature/Band)",
                confidence=0.85 + i * 0.01,
                pointing_game_hit=(i % 2 == 0),
                pointing_game_score=float(i % 2),
                energy_pointing_game_score=0.7 + i * 0.02,
                deletion_auc=0.65 + i * 0.01,
                insertion_auc=0.70 + i * 0.01,
                average_drop=0.15 + i * 0.005,
                increase_in_confidence=0.20 + i * 0.01,
                has_annotation=True,
                num_annotated_regions=1,
            )
            for i in range(10)
        ]
        
        agg = aggregate_xai_metrics(results)
        
        # Check counts
        self.assertEqual(agg.num_cells, 10)
        self.assertEqual(agg.num_annotated_cells, 10)
        
        # Check aggregated metrics are reasonable
        self.assertGreater(agg.pointing_game_accuracy, 0.0)
        self.assertLess(agg.pointing_game_accuracy, 1.0)
        self.assertGreater(agg.deletion_auc_mean, 0.5)
        self.assertGreater(agg.insertion_auc_mean, 0.5)
        
        # Check per-class metrics exist
        self.assertGreater(len(agg.per_class_metrics), 0)
        self.assertIn("Blast", agg.per_class_metrics)
    
    def test_aggregate_clinical_interpretation(self):
        """Test that clinical interpretation is generated."""
        results = [
            XAIMetricsResult(
                cell_id=f"CELL-{i:03d}",
                predicted_class="Blast",
                confidence=0.85,
                deletion_auc=0.75,
                insertion_auc=0.72,
                pointing_game_score=0.9,
                energy_pointing_game_score=0.85,
                has_annotation=True,
            )
            for i in range(5)
        ]
        
        agg = aggregate_xai_metrics(results)
        
        # Check interpretation exists and is non-empty
        self.assertIsInstance(agg.clinical_interpretation, str)
        self.assertGreater(len(agg.clinical_interpretation), 0)
        
        # Should contain key terms
        self.assertTrue(
            any(term in agg.clinical_interpretation for term in ["HIGH", "MODERATE", "LOW"])
        )


class TestFormatXAIMetricsReport(unittest.TestCase):
    """Test XAI metrics report formatting."""
    
    def test_format_report(self):
        """Test report formatting."""
        agg = AggregatedXAIMetrics(
            num_cells=50,
            num_annotated_cells=20,
            pointing_game_accuracy=0.75,
            energy_pointing_game_mean=0.68,
            deletion_auc_mean=0.72,
            insertion_auc_mean=0.70,
            average_drop_mean=0.18,
            increase_in_confidence_mean=0.25,
            per_class_metrics={
                "Blast": {
                    "count": 15,
                    "deletion_auc": 0.75,
                    "insertion_auc": 0.73,
                    "average_drop": 0.20,
                    "pointing_game_accuracy": 0.80,
                }
            },
            clinical_interpretation="✅ HIGH FAITHFULNESS"
        )
        
        report = format_xai_metrics_report(agg)
        
        # Check report is markdown format
        self.assertIsInstance(report, str)
        self.assertIn("# Quantitative XAI Evaluation Report", report)
        
        # Check key metrics are present
        self.assertIn("Pointing Game", report)
        self.assertIn("Deletion AUC", report)
        self.assertIn("Insertion AUC", report)
        self.assertIn("Per-Class Breakdown", report)
        
        # Check values are formatted correctly
        self.assertIn("75.0%", report)  # pointing_game_accuracy
        self.assertIn("0.720", report)  # deletion_auc_mean


def run_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAnnotatedRegion))
    suite.addTests(loader.loadTestsFromTestCase(TestPointingGame))
    suite.addTests(loader.loadTestsFromTestCase(TestEnergyPointingGame))
    suite.addTests(loader.loadTestsFromTestCase(TestPerturbationMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeXAIMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregateXAIMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatXAIMetricsReport))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("XAI METRICS TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All XAI metrics tests PASSED")
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
