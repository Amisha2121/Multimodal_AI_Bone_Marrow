"""
Unit tests for active learning pipeline.

Tests uncertainty quantification, priority scoring, queue management,
and feedback collection.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from pipeline.active_learning import (
    ActiveLearningPipeline,
    ActiveLearningConfig,
    CaseUncertainty,
    UncertaintyStrategy,
    PriorityReason,
    format_review_queue_report
)


class TestUncertaintyComputation:
    """Test uncertainty quantification."""
    
    def test_compute_basic_uncertainty(self):
        """Compute uncertainty for simple case."""
        pipeline = ActiveLearningPipeline()
        
        # Confident prediction: [0.9, 0.05, 0.03, 0.02]
        probs = np.array([0.9, 0.05, 0.03, 0.02])
        
        uncertainty = pipeline.compute_uncertainty(
            case_id="CASE-001",
            predicted_class="AML",
            class_probabilities=probs
        )
        
        assert uncertainty.case_id == "CASE-001"
        assert uncertainty.predicted_class == "AML"
        assert uncertainty.confidence == 0.9
        assert uncertainty.least_confident_score == pytest.approx(0.1, abs=0.01)
        assert uncertainty.margin == pytest.approx(0.85, abs=0.01)
        assert uncertainty.entropy > 0  # Some uncertainty
        assert uncertainty.combined_uncertainty < 0.5  # Low overall
    
    def test_compute_uncertain_prediction(self):
        """Compute uncertainty for uncertain case."""
        pipeline = ActiveLearningPipeline()
        
        # Uncertain prediction: [0.4, 0.35, 0.15, 0.1]
        probs = np.array([0.4, 0.35, 0.15, 0.1])
        
        uncertainty = pipeline.compute_uncertainty(
            case_id="CASE-002",
            predicted_class="ALL",
            class_probabilities=probs
        )
        
        assert uncertainty.confidence == 0.4
        assert uncertainty.least_confident_score == 0.6  # High
        assert uncertainty.margin == pytest.approx(0.05, abs=0.01)  # Small
        assert uncertainty.entropy > 1.0  # High entropy
        assert uncertainty.combined_uncertainty > 0.4  # High overall
    
    def test_compute_with_grounding(self):
        """Compute uncertainty with grounding score."""
        pipeline = ActiveLearningPipeline()
        
        probs = np.array([0.8, 0.1, 0.06, 0.04])
        
        # High confidence, low grounding → uncertain
        uncertainty = pipeline.compute_uncertainty(
            case_id="CASE-003",
            predicted_class="AML",
            class_probabilities=probs,
            grounding_score=0.3  # Low grounding
        )
        
        assert uncertainty.grounding_score == 0.3
        assert uncertainty.combined_uncertainty > 0.3  # Elevated by low grounding
    
    def test_compute_with_consistency(self):
        """Compute uncertainty with multi-image consistency."""
        pipeline = ActiveLearningPipeline()
        
        probs = np.array([0.85, 0.08, 0.04, 0.03])
        
        # High confidence, low consistency → uncertain
        uncertainty = pipeline.compute_uncertainty(
            case_id="CASE-004",
            predicted_class="ALL",
            class_probabilities=probs,
            consistency_score=0.4  # Low consistency
        )
        
        assert uncertainty.consistency_score == 0.4
        assert uncertainty.combined_uncertainty > 0.25  # Elevated by inconsistency


class TestPriorityScoring:
    """Test priority scoring logic."""
    
    def test_high_uncertainty_prioritized(self):
        """High uncertainty cases get prioritized."""
        # Use lower threshold for testing
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(config=config)
        
        # Uncertain case
        probs = np.array([0.4, 0.35, 0.15, 0.1])
        uncertainty = pipeline.compute_uncertainty(
            "CASE-001", "AML", probs
        )
        
        priority, reasons = pipeline.prioritize_case(uncertainty)
        
        assert priority > 0
        assert PriorityReason.HIGH_UNCERTAINTY in reasons
    
    def test_low_grounding_prioritized(self):
        """Low grounding score gets flagged."""
        pipeline = ActiveLearningPipeline()
        
        # High confidence, low grounding
        probs = np.array([0.9, 0.05, 0.03, 0.02])
        uncertainty = pipeline.compute_uncertainty(
            "CASE-002", "ALL", probs,
            grounding_score=0.4  # Below threshold (0.6)
        )
        
        priority, reasons = pipeline.prioritize_case(uncertainty)
        
        assert priority > 0
        assert PriorityReason.LOW_GROUNDING in reasons
    
    def test_inconsistent_multi_image_prioritized(self):
        """Inconsistent multi-image results get flagged."""
        pipeline = ActiveLearningPipeline()
        
        probs = np.array([0.85, 0.08, 0.04, 0.03])
        uncertainty = pipeline.compute_uncertainty(
            "CASE-003", "AML", probs,
            consistency_score=0.5  # Below threshold (0.7)
        )
        
        priority, reasons = pipeline.prioritize_case(uncertainty)
        
        assert priority > 0
        assert PriorityReason.INCONSISTENT_MULTI_IMAGE in reasons
    
    def test_rare_diagnosis_prioritized(self):
        """Rare diagnoses get prioritized."""
        pipeline = ActiveLearningPipeline()
        
        probs = np.array([0.7, 0.2, 0.06, 0.04])
        uncertainty = pipeline.compute_uncertainty(
            "CASE-004", "BPDCN", probs
        )
        
        priority, reasons = pipeline.prioritize_case(
            uncertainty,
            metadata={"predicted_diagnosis": "BPDCN"}
        )
        
        assert priority > 0
        assert PriorityReason.RARE_DIAGNOSIS in reasons
    
    def test_high_blast_prioritized(self):
        """High blast fraction gets prioritized."""
        pipeline = ActiveLearningPipeline()
        
        probs = np.array([0.75, 0.15, 0.06, 0.04])
        uncertainty = pipeline.compute_uncertainty(
            "CASE-005", "AML", probs
        )
        
        priority, reasons = pipeline.prioritize_case(
            uncertainty,
            metadata={"blast_fraction": 0.45}  # >30%
        )
        
        assert priority > 0
        assert PriorityReason.HIGH_CLINICAL_IMPACT in reasons
    
    def test_low_uncertainty_not_prioritized(self):
        """Low uncertainty cases don't get prioritized."""
        pipeline = ActiveLearningPipeline()
        
        # Very confident, high grounding, consistent
        probs = np.array([0.95, 0.03, 0.01, 0.01])
        uncertainty = pipeline.compute_uncertainty(
            "CASE-006", "AML", probs,
            grounding_score=0.9,
            consistency_score=0.95
        )
        
        priority, reasons = pipeline.prioritize_case(uncertainty)
        
        assert len(reasons) == 0  # No reasons to prioritize


class TestQueueManagement:
    """Test review queue management."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "queue.json"
        self.feedback_path = Path(self.temp_dir) / "feedback.json"
    
    def test_add_to_queue(self):
        """Add case to review queue."""
        # Use lower threshold for testing
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Uncertain case
        probs = np.array([0.45, 0.40, 0.10, 0.05])
        
        added = pipeline.add_to_queue(
            case_id="CASE-001",
            predicted_class="AML",
            class_probabilities=probs
        )
        
        assert added is True
        assert len(pipeline.queue) == 1
        assert pipeline.queue[0].case_id == "CASE-001"
    
    def test_confident_case_not_added(self):
        """Confident case not added to queue."""
        pipeline = ActiveLearningPipeline(
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Very confident
        probs = np.array([0.95, 0.03, 0.01, 0.01])
        
        added = pipeline.add_to_queue(
            case_id="CASE-002",
            predicted_class="ALL",
            class_probabilities=probs,
            grounding_score=0.9,
            consistency_score=0.95
        )
        
        assert added is False
        assert len(pipeline.queue) == 0
    
    def test_queue_sorting(self):
        """Queue sorted by priority."""
        # Use lower threshold
        config = ActiveLearningConfig(uncertainty_threshold=0.3)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add 3 cases with different priorities
        # Case 1: High uncertainty
        pipeline.add_to_queue(
            "CASE-001", "AML",
            np.array([0.4, 0.35, 0.15, 0.1])
        )
        
        # Case 2: Low grounding
        pipeline.add_to_queue(
            "CASE-002", "ALL",
            np.array([0.8, 0.1, 0.06, 0.04]),
            grounding_score=0.3
        )
        
        # Case 3: Moderate uncertainty
        pipeline.add_to_queue(
            "CASE-003", "AML",
            np.array([0.6, 0.25, 0.1, 0.05])
        )
        
        # Queue should be sorted by priority
        assert len(pipeline.queue) == 3
        assert pipeline.queue[0].priority_score >= pipeline.queue[1].priority_score
        assert pipeline.queue[1].priority_score >= pipeline.queue[2].priority_score
    
    def test_queue_size_limit(self):
        """Queue respects size limit."""
        config = ActiveLearningConfig(queue_size=5, uncertainty_threshold=0.3)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add 10 uncertain cases
        for i in range(10):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.4 + i*0.01, 0.35, 0.15, 0.1])
            )
        
        # Should keep only top 5
        assert len(pipeline.queue) == 5
    
    def test_get_review_queue(self):
        """Get review queue with filtering."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add 5 cases
        for i in range(5):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.45 + i*0.01, 0.35, 0.12, 0.08])
            )
        
        # Mark 2 as reviewed
        pipeline.queue[0].reviewed = True
        pipeline.queue[1].reviewed = True
        
        # Get unreviewed only
        unreviewed = pipeline.get_review_queue(unreviewed_only=True)
        assert len(unreviewed) == 3
        
        # Get top-2
        top2 = pipeline.get_review_queue(top_k=2, unreviewed_only=False)
        assert len(top2) == 2


class TestFeedbackCollection:
    """Test expert feedback collection."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "queue.json"
        self.feedback_path = Path(self.temp_dir) / "feedback.json"
    
    def test_submit_review(self):
        """Submit expert review."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add case
        pipeline.add_to_queue(
            "CASE-001", "AML",
            np.array([0.45, 0.40, 0.10, 0.05])
        )
        
        # Submit review
        success = pipeline.submit_review(
            case_id="CASE-001",
            expert_label="ALL",  # Correction
            expert_feedback="Morphology suggests lymphoid lineage"
        )
        
        assert success is True
        assert pipeline.queue[0].reviewed is True
        assert pipeline.queue[0].expert_label == "ALL"
        assert len(pipeline.feedback) == 1
    
    def test_submit_review_case_not_found(self):
        """Submit review for non-existent case."""
        pipeline = ActiveLearningPipeline(
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        success = pipeline.submit_review(
            case_id="CASE-999",
            expert_label="AML"
        )
        
        assert success is False
    
    def test_get_corrections(self):
        """Get cases where expert corrected prediction."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add and review 3 cases
        for i, (pred, expert) in enumerate([
            ("AML", "ALL"),  # Correction
            ("ALL", "ALL"),  # Agreement
            ("AML", "MDS")   # Correction
        ]):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", pred,
                np.array([0.45, 0.35, 0.12, 0.08])
            )
            pipeline.submit_review(f"CASE-{i:03d}", expert)
        
        corrections = pipeline.get_corrections()
        
        assert len(corrections) == 2  # 2 corrections
        assert corrections[0]["case_id"] == "CASE-000"
        assert corrections[0]["predicted"] == "AML"
        assert corrections[0]["expert_label"] == "ALL"
    
    def test_retraining_recommendation(self):
        """Check retraining recommendation logic."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add 5 corrections
        for i in range(5):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.45, 0.35, 0.12, 0.08])
            )
            pipeline.submit_review(f"CASE-{i:03d}", "ALL")  # All corrections
        
        # Not enough corrections yet
        should_retrain, num, _ = pipeline.get_retraining_candidates(min_corrections=10)
        assert should_retrain is False
        assert num == 5
        
        # Add 5 more
        for i in range(5, 10):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.45, 0.35, 0.12, 0.08])
            )
            pipeline.submit_review(f"CASE-{i:03d}", "ALL")
        
        # Now enough
        should_retrain, num, corrections = pipeline.get_retraining_candidates(min_corrections=10)
        assert should_retrain is True
        assert num == 10
        assert len(corrections) == 10


class TestPersistence:
    """Test queue and feedback persistence."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "queue.json"
        self.feedback_path = Path(self.temp_dir) / "feedback.json"
    
    def test_save_and_load_queue(self):
        """Save and load queue from disk."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        
        # Create pipeline and add cases
        pipeline1 = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        for i in range(3):
            pipeline1.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.45, 0.35, 0.12, 0.08])
            )
        
        # Create new pipeline (should load existing queue)
        pipeline2 = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        assert len(pipeline2.queue) == 3
        assert pipeline2.queue[0].case_id == "CASE-000"
    
    def test_save_and_load_feedback(self):
        """Save and load feedback from disk."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        
        # Create pipeline and submit reviews
        pipeline1 = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        pipeline1.add_to_queue("CASE-001", "AML", np.array([0.45, 0.35, 0.12, 0.08]))
        pipeline1.submit_review("CASE-001", "ALL", "Correction")
        
        # Create new pipeline (should load existing feedback)
        pipeline2 = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        assert len(pipeline2.feedback) == 1
        assert pipeline2.feedback[0]["expert_label"] == "ALL"


class TestStatistics:
    """Test statistics generation."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "queue.json"
        self.feedback_path = Path(self.temp_dir) / "feedback.json"
    
    def test_get_statistics(self):
        """Get pipeline statistics."""
        config = ActiveLearningConfig(uncertainty_threshold=0.4)
        pipeline = ActiveLearningPipeline(
            config=config,
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add 5 cases, review 3
        for i in range(5):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.45, 0.35, 0.12, 0.08])
            )
        
        for i in range(3):
            expert_label = "ALL" if i < 2 else "AML"  # 2 corrections
            pipeline.submit_review(f"CASE-{i:03d}", expert_label)
        
        stats = pipeline.get_statistics()
        
        assert stats["queue"]["total"] == 5
        assert stats["queue"]["reviewed"] == 3
        assert stats["queue"]["unreviewed"] == 2
        assert stats["feedback"]["total"] == 3
        assert stats["feedback"]["corrections"] == 2
        assert stats["feedback"]["correction_rate"] == pytest.approx(2/3)


class TestReporting:
    """Test report formatting."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "queue.json"
        self.feedback_path = Path(self.temp_dir) / "feedback.json"
    
    def test_format_report(self):
        """Format review queue report."""
        pipeline = ActiveLearningPipeline(
            queue_path=self.queue_path,
            feedback_path=self.feedback_path
        )
        
        # Add 5 cases
        for i in range(5):
            pipeline.add_to_queue(
                f"CASE-{i:03d}", "AML",
                np.array([0.45, 0.35, 0.12, 0.08]),
                grounding_score=0.4  # Low grounding
            )
        
        report = format_review_queue_report(pipeline, top_k=3)
        
        assert "Active Learning Review Queue" in report
        assert "Statistics" in report
        assert "Top 3 Priority Cases" in report
        assert "CASE-000" in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
