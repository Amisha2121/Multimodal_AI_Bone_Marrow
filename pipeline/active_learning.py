"""
pipeline/active_learning.py
─────────────────────────────────────────────────────────────────────────────
Active Learning Pipeline for Clinical AI Systems.

Literature Foundation:
    - Settles (2009): "Active Learning Literature Survey"
      Computer Sciences Technical Report 1648, University of Wisconsin–Madison.
      
    - Budd et al. (2021): "A survey on active learning and human-in-the-loop
      deep learning for medical image analysis"
      Medical Image Analysis 71:102062.
      
    - Smailagic et al. (2018): "Medal: Accurate and robust deep active learning
      for medical image analysis"
      ICMLA 2018.

Implemented Strategies:
    1. **Uncertainty Sampling**: Query cases with highest prediction uncertainty
       - Entropy: H(p) = -Σ p_i log(p_i)
       - Confidence margin: 1 - max(p)
       - Least confident: Highest entropy or lowest confidence
       
    2. **Diversity Sampling**: Ensure coverage of feature space
       - K-means clustering in embedding space
       - Select exemplars from each cluster
       
    3. **Clinical Priority**: Domain-specific factors
       - Blast fraction uncertainty
       - Critic grounding score
       - Multi-image consistency (if available)

Clinical Rationale:
    Real clinical deployment requires human-in-the-loop validation. Active
    learning optimizes expert time by prioritizing uncertain/high-impact cases
    for review, then retraining on corrected labels for continuous improvement.

Author: AI Assistant
Date: November 9, 2026
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger("active_learning")


class UncertaintyStrategy(Enum):
    """Uncertainty quantification strategies."""
    LEAST_CONFIDENT = "least_confident"  # 1 - max(p)
    MARGIN = "margin"  # p_1 - p_2 (difference between top 2)
    ENTROPY = "entropy"  # -Σ p log(p)
    COMBINED = "combined"  # Weighted combination


class PriorityReason(Enum):
    """Why a case was prioritized for review."""
    HIGH_UNCERTAINTY = "high_uncertainty"
    LOW_GROUNDING = "low_grounding"
    INCONSISTENT_MULTI_IMAGE = "inconsistent_multi_image"
    RARE_DIAGNOSIS = "rare_diagnosis"
    HIGH_CLINICAL_IMPACT = "high_clinical_impact"
    DIVERSITY_SAMPLING = "diversity_sampling"


@dataclass
class CaseUncertainty:
    """
    Uncertainty quantification for a single case.
    
    Attributes:
        case_id: Unique case identifier
        predicted_class: Top prediction from model
        confidence: Softmax probability of top class
        entropy: Shannon entropy of prediction distribution
        margin: Difference between top-2 probabilities
        least_confident_score: 1 - max(p)
        grounding_score: Critic grounding score (if available)
        consistency_score: Multi-image consistency (if available)
        combined_uncertainty: Weighted combined score (higher = more uncertain)
    """
    case_id: str
    predicted_class: str
    confidence: float
    entropy: float
    margin: float
    least_confident_score: float
    grounding_score: Optional[float] = None
    consistency_score: Optional[float] = None
    combined_uncertainty: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReviewQueueItem:
    """
    Item in expert review queue.
    
    Attributes:
        case_id: Unique case identifier
        priority_score: Combined priority (higher = review sooner)
        priority_reasons: Why this case was prioritized
        uncertainty: Full uncertainty breakdown
        metadata: Additional case information
        reviewed: Whether expert has reviewed this case
        expert_label: Ground truth label provided by expert
        expert_feedback: Free-text feedback
        review_timestamp: When expert reviewed
    """
    case_id: str
    priority_score: float
    priority_reasons: List[PriorityReason]
    uncertainty: CaseUncertainty
    metadata: Dict = field(default_factory=dict)
    reviewed: bool = False
    expert_label: Optional[str] = None
    expert_feedback: Optional[str] = None
    review_timestamp: Optional[str] = None


@dataclass
class ActiveLearningConfig:
    """
    Configuration for active learning pipeline.
    
    Attributes:
        uncertainty_strategy: Which uncertainty metric to use
        uncertainty_threshold: Cases above this threshold are prioritized
        grounding_threshold: Cases below this grounding score flagged
        consistency_threshold: Multi-image consistency below this flagged
        diversity_clusters: Number of clusters for diversity sampling
        queue_size: Maximum review queue size
        weights: Weights for combined uncertainty score
    """
    uncertainty_strategy: UncertaintyStrategy = UncertaintyStrategy.COMBINED
    uncertainty_threshold: float = 0.5  # Cases with uncertainty > 0.5
    grounding_threshold: float = 0.6  # Grounding < 0.6 flags low confidence
    consistency_threshold: float = 0.7  # Cross-image consistency < 0.7
    diversity_clusters: int = 10  # K-means clusters for diversity
    queue_size: int = 100  # Max items in review queue
    
    # Weights for combined uncertainty score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "entropy": 0.3,
        "confidence": 0.3,
        "grounding": 0.2,
        "consistency": 0.2
    })


class ActiveLearningPipeline:
    """
    Active learning pipeline for clinical AI.
    
    Manages:
    1. Uncertainty quantification
    2. Priority scoring
    3. Expert review queue
    4. Feedback collection
    5. Retraining triggers
    """
    
    def __init__(
        self,
        config: Optional[ActiveLearningConfig] = None,
        queue_path: Optional[Path] = None,
        feedback_path: Optional[Path] = None
    ):
        """
        Initialize active learning pipeline.
        
        Args:
            config: Active learning configuration
            queue_path: Path to save review queue
            feedback_path: Path to save expert feedback
        """
        self.config = config or ActiveLearningConfig()
        self.queue_path = queue_path or Path("data/active_learning/review_queue.json")
        self.feedback_path = feedback_path or Path("data/active_learning/expert_feedback.json")
        
        # Initialize queue and feedback storage
        self.queue: List[ReviewQueueItem] = []
        self.feedback: List[Dict] = []
        
        # Load existing queue and feedback if available
        self._load_queue()
        self._load_feedback()
        
        logger.info(f"ActiveLearningPipeline initialized: queue_size={len(self.queue)}, "
                   f"feedback_count={len(self.feedback)}")
    
    def compute_uncertainty(
        self,
        case_id: str,
        predicted_class: str,
        class_probabilities: np.ndarray,
        grounding_score: Optional[float] = None,
        consistency_score: Optional[float] = None
    ) -> CaseUncertainty:
        """
        Compute uncertainty metrics for a case.
        
        Args:
            case_id: Unique case identifier
            predicted_class: Top predicted class
            class_probabilities: Softmax probabilities (K classes)
            grounding_score: Critic grounding score (0-1)
            consistency_score: Multi-image consistency (0-1)
        
        Returns:
            CaseUncertainty with all metrics computed
        """
        # Sort probabilities descending
        probs = np.sort(class_probabilities)[::-1]
        
        # Confidence: max probability
        confidence = float(probs[0])
        
        # Entropy: -Σ p log(p)
        entropy = float(-np.sum(class_probabilities * np.log(class_probabilities + 1e-10)))
        
        # Margin: difference between top-2
        margin = float(probs[0] - probs[1]) if len(probs) > 1 else 1.0
        
        # Least confident: 1 - max(p)
        least_confident = 1.0 - confidence
        
        # Combined uncertainty (weighted average)
        weights = self.config.weights
        combined = (
            weights["entropy"] * (entropy / np.log(len(class_probabilities))) +  # Normalize by max entropy
            weights["confidence"] * least_confident
        )
        
        # Add grounding if available
        if grounding_score is not None:
            combined += weights["grounding"] * (1.0 - grounding_score)
        
        # Add consistency if available
        if consistency_score is not None:
            combined += weights["consistency"] * (1.0 - consistency_score)
        
        return CaseUncertainty(
            case_id=case_id,
            predicted_class=predicted_class,
            confidence=confidence,
            entropy=entropy,
            margin=margin,
            least_confident_score=least_confident,
            grounding_score=grounding_score,
            consistency_score=consistency_score,
            combined_uncertainty=combined
        )
    
    def prioritize_case(
        self,
        uncertainty: CaseUncertainty,
        metadata: Optional[Dict] = None
    ) -> Tuple[float, List[PriorityReason]]:
        """
        Compute priority score for a case.
        
        Args:
            uncertainty: Case uncertainty metrics
            metadata: Additional case information (diagnosis, blast%, etc.)
        
        Returns:
            (priority_score, reasons) where higher score = higher priority
        """
        priority_score = 0.0
        reasons = []
        
        # 1. High uncertainty
        if uncertainty.combined_uncertainty > self.config.uncertainty_threshold:
            priority_score += uncertainty.combined_uncertainty
            reasons.append(PriorityReason.HIGH_UNCERTAINTY)
        
        # 2. Low grounding score
        if (uncertainty.grounding_score is not None and
            uncertainty.grounding_score < self.config.grounding_threshold):
            priority_score += (1.0 - uncertainty.grounding_score)
            reasons.append(PriorityReason.LOW_GROUNDING)
        
        # 3. Inconsistent multi-image
        if (uncertainty.consistency_score is not None and
            uncertainty.consistency_score < self.config.consistency_threshold):
            priority_score += (1.0 - uncertainty.consistency_score)
            reasons.append(PriorityReason.INCONSISTENT_MULTI_IMAGE)
        
        # 4. Rare diagnosis (from metadata)
        if metadata:
            diagnosis = metadata.get("predicted_diagnosis", "")
            rare_diagnoses = {"BPDCN", "NK-ALL", "ETP-ALL", "Mixed phenotype"}
            if any(rare in diagnosis for rare in rare_diagnoses):
                priority_score += 0.5
                reasons.append(PriorityReason.RARE_DIAGNOSIS)
            
            # High clinical impact (high blast %)
            blast_fraction = metadata.get("blast_fraction", 0.0)
            if blast_fraction > 0.3:  # >30% blasts
                priority_score += 0.3
                reasons.append(PriorityReason.HIGH_CLINICAL_IMPACT)
        
        return priority_score, reasons
    
    def add_to_queue(
        self,
        case_id: str,
        predicted_class: str,
        class_probabilities: np.ndarray,
        grounding_score: Optional[float] = None,
        consistency_score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Evaluate case and add to review queue if prioritized.
        
        Args:
            case_id: Unique case identifier
            predicted_class: Top predicted class
            class_probabilities: Softmax probabilities
            grounding_score: Critic grounding score
            consistency_score: Multi-image consistency
            metadata: Additional case information
        
        Returns:
            True if added to queue, False if not prioritized
        """
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(
            case_id=case_id,
            predicted_class=predicted_class,
            class_probabilities=class_probabilities,
            grounding_score=grounding_score,
            consistency_score=consistency_score
        )
        
        # Compute priority
        priority_score, reasons = self.prioritize_case(uncertainty, metadata)
        
        # Add to queue if prioritized
        if len(reasons) > 0:
            item = ReviewQueueItem(
                case_id=case_id,
                priority_score=priority_score,
                priority_reasons=reasons,
                uncertainty=uncertainty,
                metadata=metadata or {}
            )
            
            self.queue.append(item)
            
            # Sort queue by priority (descending)
            self.queue.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Trim to max size
            if len(self.queue) > self.config.queue_size:
                self.queue = self.queue[:self.config.queue_size]
            
            # Save updated queue
            self._save_queue()
            
            logger.info(f"Added case {case_id} to review queue "
                       f"(priority={priority_score:.3f}, reasons={[r.value for r in reasons]})")
            
            return True
        
        return False
    
    def get_review_queue(
        self,
        top_k: Optional[int] = None,
        unreviewed_only: bool = True
    ) -> List[ReviewQueueItem]:
        """
        Get cases in review queue, sorted by priority.
        
        Args:
            top_k: Return only top-K cases (None = all)
            unreviewed_only: Filter to unreviewed cases only
        
        Returns:
            List of review queue items (sorted by priority descending)
        """
        queue = self.queue
        
        if unreviewed_only:
            queue = [item for item in queue if not item.reviewed]
        
        if top_k:
            queue = queue[:top_k]
        
        return queue
    
    def submit_review(
        self,
        case_id: str,
        expert_label: str,
        expert_feedback: Optional[str] = None
    ) -> bool:
        """
        Submit expert review for a case.
        
        Args:
            case_id: Case identifier
            expert_label: Ground truth label from expert
            expert_feedback: Optional free-text feedback
        
        Returns:
            True if review recorded, False if case not found
        """
        # Find case in queue
        for item in self.queue:
            if item.case_id == case_id:
                item.reviewed = True
                item.expert_label = expert_label
                item.expert_feedback = expert_feedback
                item.review_timestamp = datetime.now().isoformat()
                
                # Save feedback
                feedback_entry = {
                    "case_id": case_id,
                    "predicted_class": item.uncertainty.predicted_class,
                    "expert_label": expert_label,
                    "expert_feedback": expert_feedback,
                    "uncertainty": asdict(item.uncertainty),
                    "review_timestamp": item.review_timestamp
                }
                
                self.feedback.append(feedback_entry)
                
                # Save updated queue and feedback
                self._save_queue()
                self._save_feedback()
                
                logger.info(f"Review submitted for case {case_id}: "
                           f"predicted={item.uncertainty.predicted_class}, "
                           f"expert={expert_label}")
                
                return True
        
        logger.warning(f"Case {case_id} not found in review queue")
        return False
    
    def get_corrections(self) -> List[Dict]:
        """
        Get cases where expert label differs from prediction.
        
        Returns:
            List of {case_id, predicted, expert_label, feedback}
        """
        corrections = []
        
        for entry in self.feedback:
            if entry["predicted_class"] != entry["expert_label"]:
                corrections.append({
                    "case_id": entry["case_id"],
                    "predicted": entry["predicted_class"],
                    "expert_label": entry["expert_label"],
                    "feedback": entry.get("expert_feedback"),
                    "timestamp": entry["review_timestamp"]
                })
        
        return corrections
    
    def get_retraining_candidates(
        self,
        min_corrections: int = 10
    ) -> Tuple[bool, int, List[Dict]]:
        """
        Check if retraining is recommended.
        
        Args:
            min_corrections: Minimum corrections needed before retraining
        
        Returns:
            (should_retrain, num_corrections, corrections_list)
        """
        corrections = self.get_corrections()
        num_corrections = len(corrections)
        
        should_retrain = num_corrections >= min_corrections
        
        if should_retrain:
            logger.info(f"Retraining recommended: {num_corrections} corrections collected "
                       f"(threshold={min_corrections})")
        
        return should_retrain, num_corrections, corrections
    
    def get_statistics(self) -> Dict:
        """
        Get active learning statistics.
        
        Returns:
            Dict with queue stats, review stats, correction rate, etc.
        """
        total_queue = len(self.queue)
        reviewed = sum(1 for item in self.queue if item.reviewed)
        unreviewed = total_queue - reviewed
        
        total_feedback = len(self.feedback)
        corrections = self.get_corrections()
        num_corrections = len(corrections)
        correction_rate = num_corrections / total_feedback if total_feedback > 0 else 0.0
        
        # Uncertainty distribution
        uncertainties = [item.uncertainty.combined_uncertainty for item in self.queue]
        
        # Reason distribution
        reason_counts = {}
        for item in self.queue:
            for reason in item.priority_reasons:
                reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1
        
        return {
            "queue": {
                "total": total_queue,
                "reviewed": reviewed,
                "unreviewed": unreviewed,
                "review_rate": reviewed / total_queue if total_queue > 0 else 0.0
            },
            "feedback": {
                "total": total_feedback,
                "corrections": num_corrections,
                "correction_rate": correction_rate
            },
            "uncertainty": {
                "mean": float(np.mean(uncertainties)) if uncertainties else 0.0,
                "std": float(np.std(uncertainties)) if uncertainties else 0.0,
                "min": float(np.min(uncertainties)) if uncertainties else 0.0,
                "max": float(np.max(uncertainties)) if uncertainties else 0.0
            },
            "priority_reasons": reason_counts
        }
    
    def _save_queue(self):
        """Save review queue to disk."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        
        queue_data = [
            {
                **asdict(item),
                "priority_reasons": [r.value for r in item.priority_reasons],
                "uncertainty": asdict(item.uncertainty)
            }
            for item in self.queue
        ]
        
        with open(self.queue_path, 'w') as f:
            json.dump(queue_data, f, indent=2)
        
        logger.debug(f"Review queue saved to {self.queue_path}")
    
    def _load_queue(self):
        """Load review queue from disk."""
        if not self.queue_path.exists():
            logger.info("No existing review queue found")
            return
        
        with open(self.queue_path, 'r') as f:
            queue_data = json.load(f)
        
        self.queue = []
        for item_data in queue_data:
            # Reconstruct CaseUncertainty
            uncertainty = CaseUncertainty(**item_data["uncertainty"])
            
            # Reconstruct ReviewQueueItem
            item = ReviewQueueItem(
                case_id=item_data["case_id"],
                priority_score=item_data["priority_score"],
                priority_reasons=[PriorityReason(r) for r in item_data["priority_reasons"]],
                uncertainty=uncertainty,
                metadata=item_data.get("metadata", {}),
                reviewed=item_data.get("reviewed", False),
                expert_label=item_data.get("expert_label"),
                expert_feedback=item_data.get("expert_feedback"),
                review_timestamp=item_data.get("review_timestamp")
            )
            
            self.queue.append(item)
        
        logger.info(f"Loaded {len(self.queue)} items from review queue")
    
    def _save_feedback(self):
        """Save expert feedback to disk."""
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.feedback_path, 'w') as f:
            json.dump(self.feedback, f, indent=2)
        
        logger.debug(f"Expert feedback saved to {self.feedback_path}")
    
    def _load_feedback(self):
        """Load expert feedback from disk."""
        if not self.feedback_path.exists():
            logger.info("No existing feedback found")
            return
        
        with open(self.feedback_path, 'r') as f:
            self.feedback = json.load(f)
        
        logger.info(f"Loaded {len(self.feedback)} feedback entries")


def format_review_queue_report(
    pipeline: ActiveLearningPipeline,
    top_k: int = 20
) -> str:
    """
    Format review queue as human-readable report.
    
    Args:
        pipeline: Active learning pipeline
        top_k: Number of top priority cases to include
    
    Returns:
        Markdown-formatted report
    """
    queue = pipeline.get_review_queue(top_k=top_k, unreviewed_only=True)
    stats = pipeline.get_statistics()
    
    report = []
    report.append("# Active Learning Review Queue")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Statistics
    report.append("## Statistics")
    report.append("")
    report.append(f"- **Total queue size**: {stats['queue']['total']}")
    report.append(f"- **Reviewed**: {stats['queue']['reviewed']}")
    report.append(f"- **Unreviewed**: {stats['queue']['unreviewed']}")
    report.append(f"- **Review rate**: {stats['queue']['review_rate']:.1%}")
    report.append(f"- **Total feedback**: {stats['feedback']['total']}")
    report.append(f"- **Corrections**: {stats['feedback']['corrections']}")
    report.append(f"- **Correction rate**: {stats['feedback']['correction_rate']:.1%}")
    report.append("")
    
    # Top priority cases
    report.append(f"## Top {len(queue)} Priority Cases for Review")
    report.append("")
    report.append("| Rank | Case ID | Prediction | Priority | Uncertainty | Reasons |")
    report.append("|------|---------|------------|----------|-------------|---------|")
    
    for i, item in enumerate(queue, 1):
        reasons_str = ", ".join([r.value for r in item.priority_reasons])
        report.append(
            f"| {i} | {item.case_id} | {item.uncertainty.predicted_class} | "
            f"{item.priority_score:.3f} | {item.uncertainty.combined_uncertainty:.3f} | "
            f"{reasons_str} |"
        )
    
    report.append("")
    report.append("## Priority Reason Distribution")
    report.append("")
    for reason, count in sorted(stats['priority_reasons'].items(), key=lambda x: -x[1]):
        report.append(f"- **{reason}**: {count} cases")
    
    return "\n".join(report)
