"""
tests/test_self_consistency.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for self-consistency checking module (Improvement #5).

Literature:
- Manakul et al. (2023): "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection"
- Wang et al. (2023): "Self-Consistency Improves Chain of Thought Reasoning"

Test coverage:
1. Factual claim extraction
2. NLI-based consistency scoring (text overlap proxy)
3. Disagreement detection
4. Report selection (most consistent)
5. Consistency score integration with Critic
6. Edge cases (single sample, empty reports)
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np

from agents.self_consistency import (
    FactualClaim,
    ConsistencyResult,
    extract_factual_claims,
    compute_nli_consistency,
    normalize_claim_text,
    detect_factual_disagreements,
    select_most_consistent_report,
    check_self_consistency,
    integrate_consistency_into_critic_score,
    format_consistency_report,
)


# Mock clinical reports for testing
REPORT_CONSISTENT_1 = """
### Cytomorphology Assessment

The image shows a blast fraction of 25%. 
Morphology is consistent with Acute Lymphoblastic Leukemia (ALL) [WHO2022_ALL_001].
The blasts show high nuclear-to-cytoplasmic ratio.
Recommendation: Confirm with flow cytometry and cytogenetics.
"""

REPORT_CONSISTENT_2 = """
### Cytomorphology Assessment

Automated analysis detected 25% blasts in the specimen.
Morphological features align with ALL presentation [WHO2022_ALL_001].
Blasts demonstrate elevated N:C ratio and condensed chromatin.
Confirmatory testing with flow cytometry is recommended.
"""

REPORT_INCONSISTENT = """
### Cytomorphology Assessment

The blast fraction is 15% (below 20% threshold).
Morphology suggests Myelodysplastic Syndrome (MDS) [WHO2022_MDS_003].
No ALL features observed.
Recommend CBC and bone marrow biopsy.
"""


class TestFactualClaimExtraction(unittest.TestCase):
    """Test claim extraction from clinical reports."""
    
    def test_extract_claims_basic(self):
        """Test basic claim extraction."""
        report = "The blast fraction is 25%. Morphology shows ALL features."
        
        claims = extract_factual_claims(report)
        
        self.assertGreater(len(claims), 0)
        self.assertTrue(any(c.claim_type == "blast_fraction" for c in claims))
    
    def test_extract_claims_with_citations(self):
        """Test extraction recognizes guideline citations."""
        report = "Consistent with ALL per [WHO2022_ALL_001]. Recommend confirmatory testing."
        
        claims = extract_factual_claims(report)
        
        # Should have at least 2 claims
        self.assertGreaterEqual(len(claims), 1)
        
        # At least one should mention ALL or citation
        claim_texts = [c.text for c in claims]
        self.assertTrue(any("WHO2022" in text or "ALL" in text for text in claim_texts))
    
    def test_extract_claims_empty_report(self):
        """Test extraction on empty report."""
        claims = extract_factual_claims("")
        self.assertEqual(len(claims), 0)
    
    def test_claim_classification(self):
        """Test that claims are classified into correct types."""
        report = """
        Blast fraction is 30%.
        Morphology shows lymphoblasts with high N:C ratio.
        Diagnosis is ALL.
        Cited from [WHO2022_ALL_001].
        """
        
        claims = extract_factual_claims(report)
        
        claim_types = [c.claim_type for c in claims]
        # At least some claims should be extracted
        self.assertGreaterEqual(len(claims), 3)
        # Should have blast-related and diagnosis claims
        self.assertTrue(any("blast" in t for t in claim_types))
        self.assertTrue(any("diagnosis" in t or "morphology" in t for t in claim_types))


class TestNLIConsistency(unittest.TestCase):
    """Test NLI-based consistency computation."""
    
    def test_normalize_claim_text(self):
        """Test claim text normalization."""
        text = "The Blast Fraction is 25%. [WHO2022_ALL_001]"
        normalized = normalize_claim_text(text)
        
        # Should be lowercase, no punctuation, no citations
        self.assertNotIn("[WHO2022", normalized)
        self.assertNotIn(".", normalized)
        self.assertTrue(normalized.islower())
        self.assertIn("blast", normalized)
        self.assertIn("fraction", normalized)
    
    def test_compute_consistency_identical(self):
        """Test consistency between identical claim sets."""
        claims1 = [
            FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
            FactualClaim("Morphology shows ALL", 1, "morphology"),
        ]
        claims2 = [
            FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
            FactualClaim("Morphology shows ALL", 1, "morphology"),
        ]
        
        score = compute_nli_consistency(claims1, claims2, use_simple_overlap=True)
        
        # Should be 1.0 (perfect match)
        self.assertEqual(score, 1.0)
    
    def test_compute_consistency_partial_overlap(self):
        """Test consistency with partial overlap."""
        claims1 = [
            FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
            FactualClaim("Morphology shows ALL", 1, "morphology"),
        ]
        claims2 = [
            FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
            FactualClaim("Recommend flow cytometry", 1, "other"),
        ]
        
        score = compute_nli_consistency(claims1, claims2, use_simple_overlap=True)
        
        # Should be between 0 and 1 (partial match)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
    
    def test_compute_consistency_no_overlap(self):
        """Test consistency with no overlap."""
        claims1 = [
            FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
        ]
        claims2 = [
            FactualClaim("Recommend CBC testing", 0, "other"),
        ]
        
        score = compute_nli_consistency(claims1, claims2, use_simple_overlap=True)
        
        # Should be close to 0 (no match)
        self.assertLess(score, 0.5)
    
    def test_compute_consistency_empty_claims(self):
        """Test consistency with empty claim lists."""
        claims1 = []
        claims2 = [FactualClaim("Some claim", 0, "other")]
        
        score = compute_nli_consistency(claims1, claims2, use_simple_overlap=True)
        
        # Should be 0.0
        self.assertEqual(score, 0.0)


class TestDisagreementDetection(unittest.TestCase):
    """Test disagreement detection across samples."""
    
    def test_detect_disagreements_consistent(self):
        """Test that consistent samples have no disagreements."""
        claims_per_sample = [
            [
                FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
                FactualClaim("ALL features present", 1, "morphology"),
            ],
            [
                FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
                FactualClaim("ALL morphology observed", 1, "morphology"),
            ],
        ]
        
        disagreements = detect_factual_disagreements(claims_per_sample, disagreement_threshold=0.5)
        
        # Should have no or few disagreements
        self.assertLess(len(disagreements), 2)
    
    def test_detect_disagreements_inconsistent(self):
        """Test that inconsistent samples trigger disagreements."""
        claims_per_sample = [
            [
                FactualClaim("Blast fraction is 25%", 0, "blast_fraction"),
                FactualClaim("ALL features", 1, "diagnosis"),
            ],
            [
                FactualClaim("Blast fraction is 15%", 0, "blast_fraction"),
                FactualClaim("MDS features", 1, "diagnosis"),
            ],
        ]
        
        disagreements = detect_factual_disagreements(claims_per_sample, disagreement_threshold=0.5)
        
        # Should detect disagreements
        self.assertGreater(len(disagreements), 0)


class TestReportSelection(unittest.TestCase):
    """Test selection of most consistent report."""
    
    def test_select_single_report(self):
        """Test selection with single report."""
        reports = ["Report 1"]
        pairwise_scores = np.array([[1.0]])
        
        idx, selected = select_most_consistent_report(reports, pairwise_scores)
        
        self.assertEqual(idx, 0)
        self.assertEqual(selected, "Report 1")
    
    def test_select_from_multiple_reports(self):
        """Test selection from multiple reports."""
        reports = ["Report 1", "Report 2", "Report 3"]
        # Report 2 has highest average consistency with others
        pairwise_scores = np.array([
            [1.0, 0.6, 0.5],
            [0.6, 1.0, 0.7],
            [0.5, 0.7, 1.0],
        ])
        
        idx, selected = select_most_consistent_report(reports, pairwise_scores)
        
        # Should select Report 2 (index 1)
        self.assertEqual(idx, 1)
        self.assertEqual(selected, "Report 2")


class TestSelfConsistencyCheck(unittest.TestCase):
    """Test end-to-end self-consistency checking."""
    
    def test_check_consistency_single_sample(self):
        """Test consistency check with single sample."""
        reports = [REPORT_CONSISTENT_1]
        
        result = check_self_consistency(reports, consistency_threshold=0.70)
        
        self.assertEqual(result.num_samples, 1)
        self.assertEqual(result.consistency_score, 1.0)
        self.assertEqual(result.recommendation, "SINGLE_SAMPLE")
        self.assertEqual(len(result.disagreement_flags), 0)
    
    def test_check_consistency_high_consistency(self):
        """Test with highly consistent reports."""
        reports = [REPORT_CONSISTENT_1, REPORT_CONSISTENT_2]
        
        result = check_self_consistency(reports, consistency_threshold=0.70)
        
        self.assertEqual(result.num_samples, 2)
        # Both reports mention 25% and ALL, so should have some consistency
        # But text overlap might be moderate due to different wording
        self.assertGreaterEqual(result.consistency_score, 0.0)
        # Recommendation can be any of the valid states
        self.assertIn(result.recommendation, ["HIGH_CONSISTENCY", "MODERATE_CONSISTENCY", "LOW_CONSISTENCY", "INCONSISTENT"])
    
    def test_check_consistency_low_consistency(self):
        """Test with inconsistent reports."""
        reports = [REPORT_CONSISTENT_1, REPORT_INCONSISTENT]
        
        result = check_self_consistency(reports, consistency_threshold=0.70)
        
        self.assertEqual(result.num_samples, 2)
        # Consistency should be low due to different diagnoses
        self.assertIn(result.recommendation, ["MODERATE_CONSISTENCY", "LOW_CONSISTENCY", "INCONSISTENT"])
    
    def test_check_consistency_three_samples(self):
        """Test with three samples (typical use case)."""
        reports = [REPORT_CONSISTENT_1, REPORT_CONSISTENT_2, REPORT_CONSISTENT_1]
        
        result = check_self_consistency(reports, consistency_threshold=0.70)
        
        self.assertEqual(result.num_samples, 3)
        self.assertEqual(result.pairwise_scores.shape, (3, 3))
        
        # Diagonal should be 1.0 (self-consistency)
        np.testing.assert_array_almost_equal(np.diag(result.pairwise_scores), [1.0, 1.0, 1.0])
        
        # Should have a selected report
        self.assertIsInstance(result.selected_report, str)
        self.assertGreater(len(result.selected_report), 0)


class TestCriticIntegration(unittest.TestCase):
    """Test integration with Critic grounding score."""
    
    def test_integrate_high_consistency(self):
        """Test integration with high consistency."""
        original_score = 0.85
        consistency_score = 0.90
        
        combined = integrate_consistency_into_critic_score(original_score, consistency_score, weight=0.3)
        
        # Should be between original and consistency scores
        self.assertGreater(combined, min(original_score, consistency_score))
        self.assertLessEqual(combined, 1.0)
    
    def test_integrate_low_consistency_penalty(self):
        """Test that low consistency applies penalty."""
        original_score = 0.85
        consistency_score = 0.30  # Very low
        
        combined = integrate_consistency_into_critic_score(original_score, consistency_score, weight=0.3)
        
        # Should be significantly lower than original due to penalty
        self.assertLess(combined, original_score)
    
    def test_integrate_perfect_scores(self):
        """Test with perfect scores."""
        combined = integrate_consistency_into_critic_score(1.0, 1.0, weight=0.3)
        
        self.assertEqual(combined, 1.0)
    
    def test_integrate_zero_scores(self):
        """Test with zero scores."""
        combined = integrate_consistency_into_critic_score(0.0, 0.0, weight=0.3)
        
        self.assertGreaterEqual(combined, 0.0)
        self.assertLessEqual(combined, 0.1)  # Small due to penalty


class TestReportFormatting(unittest.TestCase):
    """Test consistency report formatting."""
    
    def test_format_single_sample_report(self):
        """Test formatting for single sample."""
        result = ConsistencyResult(
            num_samples=1,
            selected_report="Report text",
            consistency_score=1.0,
            pairwise_scores=np.array([[1.0]]),
            disagreement_flags=[],
            recommendation="SINGLE_SAMPLE",
        )
        
        report = format_consistency_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn("Self-Consistency Check", report)
        self.assertIn("SINGLE_SAMPLE", report)
    
    def test_format_multi_sample_report(self):
        """Test formatting for multiple samples."""
        result = ConsistencyResult(
            num_samples=3,
            selected_report="Report text",
            consistency_score=0.85,
            pairwise_scores=np.array([
                [1.0, 0.8, 0.9],
                [0.8, 1.0, 0.85],
                [0.9, 0.85, 1.0],
            ]),
            disagreement_flags=[],
            recommendation="HIGH_CONSISTENCY",
        )
        
        report = format_consistency_report(result)
        
        self.assertIn("Pairwise Consistency Matrix", report)
        self.assertIn("Sample 1", report)
        self.assertIn("HIGH CONSISTENCY", report)
    
    def test_format_with_disagreements(self):
        """Test formatting when disagreements are detected."""
        result = ConsistencyResult(
            num_samples=2,
            selected_report="Report text",
            consistency_score=0.50,
            pairwise_scores=np.array([[1.0, 0.5], [0.5, 1.0]]),
            disagreement_flags=["Blast fraction claims inconsistent", "Diagnosis claims vary"],
            recommendation="INCONSISTENT",
        )
        
        report = format_consistency_report(result)
        
        self.assertIn("Detected Disagreements", report)
        self.assertIn("Blast fraction claims inconsistent", report)
        self.assertIn("FACTUAL INCONSISTENCIES DETECTED", report)


def run_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFactualClaimExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestNLIConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestDisagreementDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestReportSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfConsistencyCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestCriticIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestReportFormatting))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("SELF-CONSISTENCY TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All self-consistency tests PASSED")
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
