"""
pipeline/multi_image_aggregation.py
─────────────────────────────────────────────────────────────────────────────
Multi-Image Aggregation Module for Bone Marrow Cytomorphology Analysis.

Clinical Rationale:
    Real pathologists examine MULTIPLE fields of view, not single images.
    Standard manual differential counts 500 cells across many fields.
    Multi-image aggregation:
    - Increases robustness (outlier rejection)
    - Provides confidence intervals (uncertainty quantification)
    - Checks cross-image consistency (QC metric)
    - Matches clinical workflow

Literature Foundation:
    Section 10.2 of LITERATURE_SURVEY_CONTEXT.md - Future Work
    
Author: AI Assistant
Date: November 8, 2026
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from vision.taxonomy import MORPH_CLASSES

logger = logging.getLogger("multi_image_aggregation")


@dataclass
class ImageLevelStatistics:
    """
    Statistics for a single image.
    """
    image_id: str
    total_cells: int
    blast_fraction: float
    morphology_counts: Dict[str, int]
    morphology_fractions: Dict[str, float]
    predominant_category: str
    mean_confidence: float


@dataclass
class AggregatedStatistics:
    """
    Aggregated statistics across multiple images.
    """
    # Basic counts
    total_images: int
    total_cells: int
    cells_per_image: List[int] = field(default_factory=list)
    
    # Blast fraction statistics
    blast_fraction_mean: float = 0.0
    blast_fraction_std: float = 0.0
    blast_fraction_min: float = 0.0
    blast_fraction_max: float = 0.0
    blast_fraction_ci_lower: float = 0.0  # 95% CI
    blast_fraction_ci_upper: float = 0.0  # 95% CI
    blast_fraction_cv: float = 0.0  # Coefficient of variation
    
    # Morphology aggregation
    morphology_counts_total: Dict[str, int] = field(default_factory=dict)
    morphology_fractions_mean: Dict[str, float] = field(default_factory=dict)
    morphology_fractions_std: Dict[str, float] = field(default_factory=dict)
    
    # Predominant category
    predominant_category: str = ""
    predominant_category_agreement: float = 0.0  # % images agreeing on predominant
    
    # Confidence metrics
    mean_confidence_overall: float = 0.0
    mean_confidence_per_image: List[float] = field(default_factory=list)
    
    # Cross-image consistency
    consistency_score: float = 0.0  # 0.0-1.0 (low-high consistency)
    is_consistent: bool = True
    consistency_warnings: List[str] = field(default_factory=list)
    
    # Quality flags
    low_cell_count_images: List[str] = field(default_factory=list)  # <50 cells
    outlier_images: List[str] = field(default_factory=list)  # Blast fraction outliers


def compute_image_statistics(
    image_id: str,
    cell_predictions: List,  # Can be dicts or CellInferenceResult objects
) -> ImageLevelStatistics:
    """
    Compute statistics for a single image.
    
    Args:
        image_id: Image identifier
        cell_predictions: List of cell predictions (dict or CellInferenceResult)
    
    Returns:
        ImageLevelStatistics object
    """
    total_cells = len(cell_predictions)
    
    if total_cells == 0:
        return ImageLevelStatistics(
            image_id=image_id,
            total_cells=0,
            blast_fraction=0.0,
            morphology_counts={c: 0 for c in MORPH_CLASSES},
            morphology_fractions={c: 0.0 for c in MORPH_CLASSES},
            predominant_category="Unknown",
            mean_confidence=0.0,
        )
    
    # Count morphology categories
    counts = {c: 0 for c in MORPH_CLASSES}
    confidences = []
    
    for pred in cell_predictions:
        # Handle both dict and dataclass formats
        if isinstance(pred, dict):
            cat = pred.get("predicted_class", "Unknown")
            conf = pred.get("confidence", 0.0)
        else:
            # CellInferenceResult object
            cat = getattr(pred, "predicted_class", "Unknown")
            conf = getattr(pred, "confidence", 0.0)
        
        if cat in counts:
            counts[cat] += 1
        confidences.append(conf)
    
    # Compute fractions
    fractions = {c: count / total_cells for c, count in counts.items()}
    blast_fraction = counts.get("Blast", 0) / total_cells
    
    # Predominant category (most common)
    predominant = max(counts.items(), key=lambda x: x[1])[0]
    
    # Mean confidence
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    
    return ImageLevelStatistics(
        image_id=image_id,
        total_cells=total_cells,
        blast_fraction=blast_fraction,
        morphology_counts=counts,
        morphology_fractions=fractions,
        predominant_category=predominant,
        mean_confidence=mean_conf,
    )


def compute_confidence_interval(
    values: List[float],
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values.
    
    Args:
        values: List of values (e.g., blast fractions from multiple images)
        confidence_level: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        # Can't compute CI with <2 samples, return mean ± 0
        mean_val = np.mean(values) if values else 0.0
        return (mean_val, mean_val)
    
    # Use t-distribution for small samples
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of mean
    
    # t-critical value
    df = len(values) - 1
    t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
    
    margin = t_crit * std_err
    
    return (float(max(0.0, mean - margin)), float(min(1.0, mean + margin)))


def detect_outliers(
    values: List[float],
    threshold: float = 2.0,
) -> List[int]:
    """
    Detect outliers using modified Z-score (MAD-based).
    
    Args:
        values: List of values
        threshold: Z-score threshold (default: 2.0 for moderate outliers)
    
    Returns:
        List of indices that are outliers
    """
    if len(values) < 3:
        return []  # Can't detect outliers with <3 samples
    
    values_arr = np.array(values)
    median = np.median(values_arr)
    mad = np.median(np.abs(values_arr - median))
    
    if mad == 0:
        # No variability, use standard deviation
        std = np.std(values_arr)
        if std == 0:
            return []
        modified_z_scores = np.abs(values_arr - np.mean(values_arr)) / std
    else:
        # Modified Z-score: 0.6745 * (x - median) / MAD
        modified_z_scores = 0.6745 * np.abs(values_arr - median) / mad
    
    outlier_indices = np.where(modified_z_scores > threshold)[0].tolist()
    return outlier_indices


def compute_consistency_score(
    blast_fractions: List[float],
) -> float:
    """
    Compute cross-image consistency score (0.0-1.0).
    
    High consistency: blast fractions are similar across images
    Low consistency: blast fractions vary widely (potential sampling issue or mixed pathology)
    
    Args:
        blast_fractions: List of blast fractions from multiple images
    
    Returns:
        Consistency score (0.0 = inconsistent, 1.0 = highly consistent)
    """
    if len(blast_fractions) < 2:
        return 1.0  # Single image = perfectly consistent with itself
    
    # Use coefficient of variation (CV = std / mean)
    mean_bf = np.mean(blast_fractions)
    std_bf = np.std(blast_fractions)
    
    if mean_bf == 0:
        # All zeros = consistent
        if std_bf == 0:
            return 1.0
        else:
            # Some non-zero, some zero = inconsistent
            return 0.0
    
    cv = std_bf / mean_bf
    
    # Map CV to consistency score:
    # CV = 0.0 → score = 1.0 (perfect consistency)
    # CV = 0.5 → score = 0.5 (moderate consistency)
    # CV >= 1.0 → score = 0.0 (poor consistency)
    
    score = max(0.0, 1.0 - cv)
    return float(score)


def aggregate_multi_image_predictions(
    per_image_predictions: Dict[str, List[dict]],
    min_cells_per_image: int = 20,
    consistency_threshold: float = 0.60,
) -> AggregatedStatistics:
    """
    Aggregate cell predictions across multiple images.
    
    Args:
        per_image_predictions: Dict mapping image_id -> list of cell prediction dicts
        min_cells_per_image: Minimum cells required per image (flag low-count images)
        consistency_threshold: Consistency score threshold (flag if below)
    
    Returns:
        AggregatedStatistics object
    """
    # Compute per-image statistics
    image_stats: List[ImageLevelStatistics] = []
    for image_id, preds in per_image_predictions.items():
        stats_obj = compute_image_statistics(image_id, preds)
        image_stats.append(stats_obj)
    
    n_images = len(image_stats)
    
    if n_images == 0:
        # No images
        return AggregatedStatistics(
            total_images=0,
            total_cells=0,
            consistency_score=0.0,
            is_consistent=False,
            consistency_warnings=["No images provided"],
        )
    
    # Total cell counts
    total_cells = sum(img.total_cells for img in image_stats)
    cells_per_image = [img.total_cells for img in image_stats]
    
    # Blast fraction statistics
    blast_fractions = [img.blast_fraction for img in image_stats]
    bf_mean = float(np.mean(blast_fractions))
    bf_std = float(np.std(blast_fractions))
    bf_min = float(np.min(blast_fractions))
    bf_max = float(np.max(blast_fractions))
    
    # Confidence interval (95%)
    bf_ci_lower, bf_ci_upper = compute_confidence_interval(blast_fractions)
    
    # Coefficient of variation
    bf_cv = bf_std / bf_mean if bf_mean > 0 else 0.0
    
    # Morphology aggregation (weighted by cell count)
    morphology_counts_total = {c: 0 for c in MORPH_CLASSES}
    for img in image_stats:
        for c, count in img.morphology_counts.items():
            morphology_counts_total[c] += count
    
    # Mean fractions across images (unweighted average of per-image fractions)
    morphology_fractions_mean = {c: 0.0 for c in MORPH_CLASSES}
    morphology_fractions_std = {c: 0.0 for c in MORPH_CLASSES}
    
    for c in MORPH_CLASSES:
        fractions = [img.morphology_fractions[c] for img in image_stats]
        morphology_fractions_mean[c] = float(np.mean(fractions))
        morphology_fractions_std[c] = float(np.std(fractions))
    
    # Predominant category (most common across images)
    predominant_categories = [img.predominant_category for img in image_stats]
    predominant_overall = max(set(predominant_categories), key=predominant_categories.count)
    predominant_agreement = predominant_categories.count(predominant_overall) / n_images
    
    # Confidence metrics
    mean_confs = [img.mean_confidence for img in image_stats]
    mean_confidence_overall = float(np.mean(mean_confs))
    
    # Cross-image consistency
    consistency_score = compute_consistency_score(blast_fractions)
    is_consistent = consistency_score >= consistency_threshold
    
    # Quality flags
    low_cell_count_images = [
        img.image_id for img in image_stats 
        if img.total_cells < min_cells_per_image
    ]
    
    # Detect outlier images (blast fraction outliers)
    outlier_indices = detect_outliers(blast_fractions, threshold=2.0)
    outlier_images = [image_stats[i].image_id for i in outlier_indices]
    
    # Consistency warnings
    warnings = []
    if not is_consistent:
        warnings.append(
            f"Low cross-image consistency (score: {consistency_score:.2f}, CV: {bf_cv:.2f}). "
            f"Blast fractions vary widely: {bf_min:.1%} - {bf_max:.1%}."
        )
    
    if outlier_images:
        warnings.append(
            f"Outlier images detected: {', '.join(outlier_images)}. "
            f"Consider excluding or investigating these images."
        )
    
    if low_cell_count_images:
        warnings.append(
            f"Low cell count in images: {', '.join(low_cell_count_images)} "
            f"(<{min_cells_per_image} cells). May reduce reliability."
        )
    
    if n_images == 1:
        warnings.append(
            "Single image analysis. Multi-field examination recommended for robust assessment."
        )
    
    return AggregatedStatistics(
        total_images=n_images,
        total_cells=total_cells,
        cells_per_image=cells_per_image,
        blast_fraction_mean=bf_mean,
        blast_fraction_std=bf_std,
        blast_fraction_min=bf_min,
        blast_fraction_max=bf_max,
        blast_fraction_ci_lower=bf_ci_lower,
        blast_fraction_ci_upper=bf_ci_upper,
        blast_fraction_cv=bf_cv,
        morphology_counts_total=morphology_counts_total,
        morphology_fractions_mean=morphology_fractions_mean,
        morphology_fractions_std=morphology_fractions_std,
        predominant_category=predominant_overall,
        predominant_category_agreement=predominant_agreement,
        mean_confidence_overall=mean_confidence_overall,
        mean_confidence_per_image=mean_confs,
        consistency_score=consistency_score,
        is_consistent=is_consistent,
        consistency_warnings=warnings,
        low_cell_count_images=low_cell_count_images,
        outlier_images=outlier_images,
    )


def format_aggregation_report(agg: AggregatedStatistics) -> str:
    """
    Format aggregated statistics as a human-readable report.
    
    Args:
        agg: AggregatedStatistics object
    
    Returns:
        Formatted markdown report
    """
    report_lines = []
    
    report_lines.append("## Multi-Image Aggregation Summary")
    report_lines.append("")
    report_lines.append(f"**Images Analyzed**: {agg.total_images}")
    report_lines.append(f"**Total Cells**: {agg.total_cells}")
    report_lines.append(f"**Cells per Image**: {min(agg.cells_per_image)} - {max(agg.cells_per_image)} "
                       f"(mean: {np.mean(agg.cells_per_image):.0f})")
    report_lines.append("")
    
    report_lines.append("### Blast Fraction")
    report_lines.append(f"- **Mean**: {agg.blast_fraction_mean:.1%}")
    report_lines.append(f"- **95% CI**: [{agg.blast_fraction_ci_lower:.1%}, {agg.blast_fraction_ci_upper:.1%}]")
    report_lines.append(f"- **Range**: {agg.blast_fraction_min:.1%} - {agg.blast_fraction_max:.1%}")
    report_lines.append(f"- **Std Dev**: {agg.blast_fraction_std:.1%}")
    report_lines.append(f"- **CV**: {agg.blast_fraction_cv:.2f}")
    report_lines.append("")
    
    report_lines.append("### Cross-Image Consistency")
    report_lines.append(f"- **Consistency Score**: {agg.consistency_score:.2f} / 1.00")
    report_lines.append(f"- **Status**: {'Consistent' if agg.is_consistent else 'Inconsistent'}")
    report_lines.append("")
    
    if agg.consistency_warnings:
        report_lines.append("### Quality Flags")
        for warning in agg.consistency_warnings:
            report_lines.append(f"- [Warning] {warning}")
        report_lines.append("")
    
    report_lines.append("### Predominant Morphology")
    report_lines.append(f"- **Category**: {agg.predominant_category}")
    report_lines.append(f"- **Agreement**: {agg.predominant_category_agreement:.0%} of images")
    report_lines.append("")
    
    report_lines.append("### Mean Confidence")
    report_lines.append(f"- **Overall**: {agg.mean_confidence_overall:.1%}")
    report_lines.append("")
    
    return "\n".join(report_lines)


def get_aggregation_recommendation(agg: AggregatedStatistics) -> Dict[str, any]:
    """
    Generate clinical recommendations based on aggregation results.
    
    Args:
        agg: AggregatedStatistics object
    
    Returns:
        Dict with recommendation details
    """
    recommendations = []
    confidence_level = "high"
    
    # Check consistency
    if not agg.is_consistent:
        recommendations.append(
            "Cross-image variability detected. Consider: (1) examining additional fields, "
            "(2) reviewing for sampling bias, or (3) consulting with experienced pathologist."
        )
        confidence_level = "moderate"
    
    # Check outliers
    if agg.outlier_images:
        recommendations.append(
            f"Outlier images identified ({', '.join(agg.outlier_images)}). "
            f"Review these images for technical artifacts or true heterogeneous pathology."
        )
        confidence_level = "moderate"
    
    # Check low cell counts
    if agg.low_cell_count_images:
        recommendations.append(
            f"Some images have low cell counts ({', '.join(agg.low_cell_count_images)}). "
            f"Additional fields recommended for robust statistics."
        )
    
    # Check single image
    if agg.total_images == 1:
        recommendations.append(
            "Single-image analysis. Multi-field examination strongly recommended "
            "for accurate blast enumeration (standard: 500-cell differential)."
        )
        confidence_level = "low"
    
    # Check wide confidence interval
    ci_width = agg.blast_fraction_ci_upper - agg.blast_fraction_ci_lower
    if ci_width > 0.10:  # >10% width
        recommendations.append(
            f"Wide confidence interval (±{ci_width/2:.1%}). "
            f"Additional images would narrow uncertainty."
        )
        confidence_level = "moderate"
    
    # Check if near diagnostic threshold
    bf_mean = agg.blast_fraction_mean
    ci_lower = agg.blast_fraction_ci_lower
    ci_upper = agg.blast_fraction_ci_upper
    
    if ci_lower < 0.20 < ci_upper:
        recommendations.append(
            "Blast fraction confidence interval crosses 20% threshold (diagnostic cutoff for acute leukemia). "
            "Consider: (1) examining additional fields, (2) flow cytometry, (3) ancillary testing."
        )
        confidence_level = "low"
    
    if not recommendations:
        recommendations.append(
            "Consistent multi-image analysis. Results are robust."
        )
        confidence_level = "high"
    
    return {
        "confidence_level": confidence_level,
        "recommendations": recommendations,
        "should_review": len(agg.consistency_warnings) > 0,
        "diagnostic_certainty": "high" if confidence_level == "high" and agg.is_consistent else "moderate",
    }


# Example usage
if __name__ == "__main__":
    # Simulate 3 images with cell predictions
    per_image_preds = {
        "IMAGE-01": [
            {"predicted_class": "Blast", "confidence": 0.92},
            {"predicted_class": "Blast", "confidence": 0.88},
            {"predicted_class": "Granulocyte (Mature/Band)", "confidence": 0.95},
            {"predicted_class": "Erythroid Precursor", "confidence": 0.90},
            {"predicted_class": "Blast", "confidence": 0.85},
        ] * 20,  # 100 cells total
        "IMAGE-02": [
            {"predicted_class": "Blast", "confidence": 0.90},
            {"predicted_class": "Blast", "confidence": 0.87},
            {"predicted_class": "Granulocyte (Mature/Band)", "confidence": 0.93},
            {"predicted_class": "Lymphoid Series", "confidence": 0.91},
            {"predicted_class": "Blast", "confidence": 0.89},
        ] * 18,  # 90 cells total
        "IMAGE-03": [
            {"predicted_class": "Blast", "confidence": 0.91},
            {"predicted_class": "Blast", "confidence": 0.86},
            {"predicted_class": "Granulocyte (Mature/Band)", "confidence": 0.94},
            {"predicted_class": "Monocyte", "confidence": 0.88},
            {"predicted_class": "Blast", "confidence": 0.90},
        ] * 22,  # 110 cells total
    }
    
    agg = aggregate_multi_image_predictions(per_image_preds)
    
    print(format_aggregation_report(agg))
    print("\n" + "=" * 70)
    
    rec = get_aggregation_recommendation(agg)
    print(f"\nConfidence Level: {rec['confidence_level'].upper()}")
    print(f"Diagnostic Certainty: {rec['diagnostic_certainty'].upper()}")
    print("\nRecommendations:")
    for r in rec['recommendations']:
        print(f"  - {r}")
