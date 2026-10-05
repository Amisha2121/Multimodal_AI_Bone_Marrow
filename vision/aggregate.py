"""
vision/aggregate.py
─────────────────────────────────────────────────────────────────────────────
Patient-Level Aggregation Module.

Aggregates N individual cell-crop predictions for a given patient/slide
into a cell-differential vector and computes the annotated-cell blast
fraction under the 8-class BoneMarrowWSI-PediatricLeukemia taxonomy.

NOTE: The "annotated-cell blast fraction" is NOT equivalent to a clinical
blast percentage.  The consensus annotations are not guaranteed to be a
complete, representative bone-marrow differential.  Do not make clinical
claims from this value.
"""

from typing import Dict, List, Optional, Tuple, Union
import math

# All taxonomy constants now live in vision.taxonomy.
# This module re-exports CELL_CLASSES for backward compatibility.
from vision.taxonomy import (
    MORPH_CLASSES,
    MORPH_TO_IDX,
    IDX_TO_MORPH,
    BLAST_CATEGORIES,
    EXCLUDED_CATEGORIES,
)

# Backward-compatibility alias (used by legacy imports from vision.aggregate)
CELL_CLASSES = MORPH_CLASSES


def wilson_score_interval(count: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    Robust for small counts and extreme proportions near 0 or 1.
    """
    if n == 0:
        return (0.0, 0.0)

    z = 1.95996  # 95% confidence standard normal quantile
    if confidence == 0.99:
        z = 2.57583
    elif confidence == 0.90:
        z = 1.64485

    p = count / n
    denominator = 1.0 + (z**2) / n
    centre_adjusted_probability = p + (z**2) / (2 * n)
    adjusted_std = math.sqrt((p * (1 - p) / n) + ((z**2) / (4 * (n**2))))

    lower_raw = (centre_adjusted_probability - z * adjusted_std) / denominator
    upper_raw = (centre_adjusted_probability + z * adjusted_std) / denominator

    lower = 0.0 if count == 0 else max(0.0, float(lower_raw))
    upper = 1.0 if count == n else min(1.0, float(upper_raw))

    return (lower, upper)


def aggregate_patient_crops(
    predicted_classes: List[Union[str, int]],
    class_names: Optional[List[str]] = None,
    min_crops_threshold: int = 50,
    confidence_level: float = 0.95,
) -> Dict[str, Union[Dict, float, int, bool, str]]:
    """
    Aggregates per-crop predictions into a patient-level differential vector.

    Args:
        predicted_classes: List of class strings (e.g. 'Blast') or integer indices.
        class_names: List of all class names if integer indices are provided.
            Defaults to MORPH_CLASSES (8-class real taxonomy).
        min_crops_threshold: Minimum recommended crops for reliable differential.
        confidence_level: Confidence level for Wilson intervals (default 0.95).

    Returns:
        Structured dictionary containing:
          - 'total_crops': Total number of analyzed cells
          - 'is_sufficient_sample': Boolean flag (N >= min_crops_threshold)
          - 'differential': Dict mapping class -> fractional percentage [0.0 - 1.0]
          - 'confidence_intervals': Dict mapping class -> [lower_ci, upper_ci]
          - 'counts': Dict mapping class -> integer count
          - 'annotated_cell_blast_fraction': Fraction of annotated cells in blast categories
          - 'blast_fraction_ci': [lower_ci, upper_ci] for blast fraction
          - 'blast_fraction_note': Disclaimer note about clinical interpretation
          - 'warning': Warning message if sample count is low
    """
    if class_names is None:
        class_names = MORPH_CLASSES

    n_crops = len(predicted_classes)
    warning = None
    is_sufficient = n_crops >= min_crops_threshold
    if not is_sufficient:
        warning = (
            f"Low cell count warning: Analyzed {n_crops} cells, which is below the recommended "
            f"minimum of {min_crops_threshold} cells for a reliable bone marrow differential."
        )

    # Count occurrences
    counts = {c: 0 for c in class_names}
    for item in predicted_classes:
        if isinstance(item, int):
            c_name = IDX_TO_MORPH.get(item, class_names[-1])
        else:
            c_name = str(item)
        if c_name in counts:
            counts[c_name] += 1
        else:
            # Map unknown predictions to "Other Morphological" if present, else last class
            fallback = "Other Morphological" if "Other Morphological" in counts else class_names[-1]
            counts[fallback] = counts.get(fallback, 0) + 1

    # Fractional differential and Wilson confidence intervals
    differential = {}
    intervals = {}
    for c, cnt in counts.items():
        fraction = (cnt / n_crops) if n_crops > 0 else 0.0
        differential[c] = round(fraction, 4)
        intervals[c] = [round(x, 4) for x in wilson_score_interval(cnt, n_crops, confidence_level)]

    # Annotated-cell blast fraction (NOT clinical blast %)
    total_blasts = sum(counts.get(b, 0) for b in BLAST_CATEGORIES)
    blast_fraction = round((total_blasts / n_crops) if n_crops > 0 else 0.0, 4)
    blast_ci = [round(x, 4) for x in wilson_score_interval(total_blasts, n_crops, confidence_level)]

    return {
        "total_crops": n_crops,
        "is_sufficient_sample": is_sufficient,
        "differential": differential,
        "confidence_intervals": intervals,
        "counts": counts,
        "annotated_cell_blast_fraction": blast_fraction,
        "blast_fraction_ci": blast_ci,
        "blast_fraction_note": (
            "RESEARCH NOTE: This value represents the fraction of ANNOTATED cells classified "
            "as blast-type by the model.  It is NOT a clinical blast percentage.  The consensus "
            "annotations are not a complete representative differential of the bone marrow slide."
        ),
        "warning": warning,
    }
