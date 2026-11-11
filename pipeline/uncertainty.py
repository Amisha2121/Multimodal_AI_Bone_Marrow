# -*- coding: utf-8 -*-
"""
Uncertainty and Statistical Limits for Bone Marrow Cytomorphology.
Provides Wilson score confidence intervals, confidence categorisation bands,
and reviewer attention flags.
"""

import math
from typing import Optional, Tuple


def wilson_interval(
    k: Optional[int],
    n: Optional[int],
    z: float = 1.96
) -> Optional[Tuple[float, float]]:
    """
    Compute the two-sided Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes / positive events (e.g. blast count).
        n: Total trials / sample size (e.g. total cells).
        z: Normal distribution critical value (default 1.96 for 95% CI).

    Returns:
        Tuple of (low, high) proportions rounded to 4 decimals, or None if inputs invalid.
    """
    if k is None or n is None or n <= 0 or k < 0 or k > n:
        return None

    p = k / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p + (z2 / (2.0 * n))) / denom
    
    radicand = (p * (1.0 - p) / n) + (z2 / (4.0 * (n * n)))
    if radicand < 0.0:
        radicand = 0.0
    spread = (z / denom) * math.sqrt(radicand)

    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return (round(low, 4), round(high, 4))


def confidence_band(conf: Optional[float]) -> Optional[str]:
    """
    Categorise a confidence score into a qualitative band.

    Args:
        conf: Model probability score between 0.0 and 1.0.

    Returns:
        "high" (>=0.70), "medium" (0.40 - 0.69), "low" (<0.40), or None if missing.
    """
    if conf is None:
        return None
    try:
        f = float(conf)
        if math.isnan(f) or math.isinf(f):
            return None
        if f >= 0.70:
            return "high"
        if f >= 0.40:
            return "medium"
        return "low"
    except (TypeError, ValueError):
        return None


def needs_review(conf: Optional[float]) -> bool:
    """
    Determine whether a cell prediction falls below the high-reliability threshold (<0.50).

    Args:
        conf: Model probability score between 0.0 and 1.0.

    Returns:
        True if confidence is below 0.50, False otherwise (or if missing).
    """
    if conf is None:
        return False
    try:
        f = float(conf)
        if math.isnan(f) or math.isinf(f):
            return False
        return f < 0.50
    except (TypeError, ValueError):
        return False


def build_confidence_and_limits_section(summary: dict) -> dict:
    """
    Generate deterministic 'Confidence and limitations' report section.
    Computed purely from mathematical statistics without invoking an LLM.
    """
    total_cells = summary.get("total_cells", 0)
    blast_count = summary.get("blast_count", 0)
    blast_fraction = summary.get("blast_fraction", 0.0)

    ci = wilson_interval(blast_count, total_cells, z=1.96)
    ci_str = f"{ci[0]*100:.1f}%–{ci[1]*100:.1f}%" if ci else "N/A"

    mean_conf = summary.get("overall_mean_confidence")
    mean_conf_str = f"{mean_conf*100:.1f}%" if mean_conf is not None else "N/A"

    # Count bands and cells under 50%
    all_confs = summary.get("all_confidences", [])
    if all_confs:
        high_cnt = sum(1 for c in all_confs if c >= 0.70)
        med_cnt = sum(1 for c in all_confs if 0.40 <= c < 0.70)
        low_cnt = sum(1 for c in all_confs if c < 0.40)
        under_50_count = sum(1 for c in all_confs if c < 0.50)
        band_str = f"High (≥70%): {high_cnt}, Medium (40–69%): {med_cnt}, Low (<40%): {low_cnt}"
        under_50_str = f"{under_50_count} ({under_50_count / len(all_confs) * 100:.1f}%)"
    else:
        band_str = "N/A"
        under_50_str = "N/A"

    content_lines = [
        f"- **Overall Mean Confidence:** {mean_conf_str}",
        f"- **Cells Under 50% Confidence (Needs Review):** {under_50_str}",
        f"- **Cells by Confidence Band:** {band_str}",
        f"- **Blast Count & Proportion:** {blast_count} cells ({blast_fraction*100:.1f}%, 95% CI: {ci_str}, n={total_cells})",
        f"- **Total Evaluated Nucleated Cells:** {total_cells}",
    ]

    if total_cells < 200:
        content_lines.append(
            "- **Sample Size Caution:** Low cell count (<200 cells). Standard manual differentials evaluate "
            "200–500 cells across multiple high-power monolayer fields. Interpret proportions with caution."
        )

    content_lines.append(
        "\n*Blast classification is the least reliable class in published automated systems "
        "(blast accuracy of 61.8% was reported for Morphogo in a 2025 pediatric study).*"
    )

    return {
        "title": "Confidence and Limitations",
        "content": "\n".join(content_lines),
        "cell_images": [],
    }
