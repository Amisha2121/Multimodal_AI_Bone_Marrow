# -*- coding: utf-8 -*-
"""
Single Source of Truth for Case Morphology Summary.
Computes authoritative per-class counts, percentages, blast fraction, and confidence metrics.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from vision.taxonomy import MORPH_CLASSES

logger = logging.getLogger("morphology_summary")


def build_morphology_summary(
    case_or_analysis_id: str,
    case_dict: Optional[Dict[str, Any]] = None,
    use_reviewer_labels: bool = False,
) -> Dict[str, Any]:
    """
    Authoritative single-source-of-truth function for case morphology statistics.

    Args:
        case_or_analysis_id: Case ID (e.g. 'CASE-F09AFAC4') or Analysis ID (e.g. 'adb52ad345f1').
        case_dict: Optional pre-loaded case dictionary. If None, loaded from SQLite / in-memory.
        use_reviewer_labels: When True, use reviewer_label if present on a cell.

    Returns:
        Dictionary with:
          - case_id: canonical case ID string
          - analysis_id: analysis hex ID string
          - total_cells: integer total count
          - cells_detected: integer count
          - cells_classified: integer count
          - counts: Dict[str, int] covering all 8 MORPH_CLASSES (including zeros)
          - percentages: Dict[str, float] per class
          - blast_count: int
          - blast_fraction: float (0.0 to 1.0)
          - blast_pct: float (alias for blast_fraction)
          - mean_confidences: Dict[str, Optional[float]] mean confidence per class
          - overall_mean_confidence: Optional[float]
          - sorted_categories: List[Tuple[str, int]] sorted descending by count
          - predominant_categories: List[str] categories with count > 0
    """
    case = case_dict

    # 1. Resolve case dictionary if not provided
    if case is None:
        try:
            from app.db import get_case_from_db
            case = get_case_from_db(case_or_analysis_id)
        except Exception as e:
            logger.debug(f"Could not load case from SQLite: {e}")

    if case is None:
        # Check in-memory analyses from flask_app if available
        try:
            from app.flask_app import analyses
            case = analyses.get(case_or_analysis_id)
            if not case:
                for aid, entry in analyses.items():
                    if entry.get("case_id") == case_or_analysis_id:
                        case = entry
                        break
        except Exception:
            pass

    if case is None:
        case = {}

    canonical_case_id = (
        case.get("case_id") or case.get("id") or str(case_or_analysis_id)
    )
    analysis_id = case.get("id") or case.get("analysis_id") or str(case_or_analysis_id)

    # 2. Extract cells list
    raw_cells = case.get("cell_images")
    if not raw_cells:
        raw_result = case.get("analysis")
        if raw_result is not None:
            raw_cells = getattr(raw_result, "cell_predictions", None)

    # Initialize all 8 classes to zero
    counts: Dict[str, int] = {c: 0 for c in MORPH_CLASSES}
    confidences: Dict[str, List[float]] = {c: [] for c in MORPH_CLASSES}

    if raw_cells:
        for c in raw_cells:
            # Extract class name and confidence
            if isinstance(c, dict):
                ai_cls = c.get("class") or c.get("predicted_class") or "Other Morphological"
                rev_cls = c.get("reviewer_label") if use_reviewer_labels else None
                cls_name = rev_cls if rev_cls else ai_cls
                conf = c.get("confidence")
            else:
                ai_cls = getattr(c, "predicted_class", "Other Morphological")
                rev_cls = getattr(c, "reviewer_label", None) if use_reviewer_labels else None
                cls_name = rev_cls if rev_cls else ai_cls
                conf = getattr(c, "confidence", None)

            # Map to canonical morphology category
            if cls_name not in counts:
                counts[cls_name] = 0
                confidences[cls_name] = []

            counts[cls_name] += 1
            if conf is not None:
                try:
                    conf_val = float(conf)
                    if not (np.isnan(conf_val) or np.isinf(conf_val)):
                        confidences[cls_name].append(conf_val)
                except (TypeError, ValueError):
                    pass

        total_cells = len(raw_cells)
    else:
        # Fall back to existing morphology dictionary if cell_images missing
        existing_morph = case.get("morphology") or case.get("morphology_counts") or {}
        if isinstance(existing_morph, list):
            # Formatted list of {type, count, percentage}
            for item in existing_morph:
                if isinstance(item, dict):
                    t = item.get("type")
                    cnt = item.get("count", 0)
                    if t:
                        counts[t] = int(cnt)
        elif isinstance(existing_morph, dict):
            for t, cnt in existing_morph.items():
                counts[t] = int(cnt)

        total_cells = sum(counts.values()) or case.get("cells_count", 0)

    # 3. Compute derived metrics
    percentages: Dict[str, float] = {}
    for c, cnt in counts.items():
        percentages[c] = round(cnt / total_cells, 4) if total_cells > 0 else 0.0

    blast_count = counts.get("Blast", 0)
    blast_fraction = round(blast_count / total_cells, 4) if total_cells > 0 else 0.0

    mean_confidences: Dict[str, Optional[float]] = {}
    all_confs: List[float] = []
    for c, clist in confidences.items():
        if clist:
            mean_confidences[c] = round(float(np.mean(clist)), 4)
            all_confs.extend(clist)
        else:
            mean_confidences[c] = None

    overall_mean_confidence = round(float(np.mean(all_confs)), 4) if all_confs else None

    sorted_categories = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    predominant_categories = [c[0] for c in sorted_categories if c[1] > 0]

    return {
        "case_id": canonical_case_id,
        "analysis_id": analysis_id,
        "total_cells": total_cells,
        "cells_detected": total_cells,
        "cells_classified": total_cells,
        "counts": counts,
        "percentages": percentages,
        "blast_count": blast_count,
        "blast_fraction": blast_fraction,
        "blast_pct": blast_fraction,
        "mean_confidences": mean_confidences,
        "overall_mean_confidence": overall_mean_confidence,
        "all_confidences": all_confs,
        "sorted_categories": sorted_categories,
        "predominant_categories": predominant_categories,
    }

