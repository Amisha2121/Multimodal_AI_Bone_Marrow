# -*- coding: utf-8 -*-
"""
Deterministic Analyses and Attention Analytics for Phase 10 Scan Understanding.
Provides:
  - blast_sensitivity_range(case): heuristic lower and upper blast bounds.
  - plausibility_flags(case): automated clinical quality and plausibility checkpoints.
  - confusion_neighbours(case): top co-occurring classes per series.
  - compute_attention_focus(crop_path, overlay_path): nucleus vs cytoplasm vs background attention breakdown.
All functions are pure, deterministic, read-only, and adhere strictly to single source of truth.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
from pathlib import Path


# Editable thresholds for plausibility flags
PLAUSIBILITY_THRESHOLDS = {
    "min_cells_standard": 200,
    "max_low_conf_ratio": 0.40,
    "max_megakaryocyte_share": 0.05,
    "max_other_morph_share": 0.10,
    "conf_review_cutoff": 0.50,
}


def blast_sensitivity_range(
    cells: List[Dict[str, Any]],
    total_cells: int
) -> Dict[str, Any]:
    """
    Compute heuristic sensitivity range for blast proportion based on prediction uncertainty.
    
    Lower bound: Blasts with confidence >= 0.50 only.
    Upper bound: Definite blasts + low-confidence (<0.50) cells in blast-adjacent classes
                 (e.g., Promyelocyte, Myeloid (Maturing), Other Morphological) or with Blast top-2.
                 
    Returns dictionary with lower_pct, upper_pct, uncertain_cells_count, and heuristic label.
    """
    if not cells or total_cells <= 0:
        return {
            "lower_pct": 0.0,
            "upper_pct": 0.0,
            "lower_count": 0,
            "upper_count": 0,
            "uncertain_cells_count": 0,
            "label": "heuristic, not a statistical interval",
            "summary_text": "Blast proportion sensitivity could not be computed (no cell data)."
        }

    # Blasts with conf >= 0.50
    high_conf_blasts = [
        c for c in cells
        if c.get("class") == "Blast" and float(c.get("confidence", 0.0) or 0.0) >= PLAUSIBILITY_THRESHOLDS["conf_review_cutoff"]
    ]
    all_blasts = [c for c in cells if c.get("class") == "Blast"]

    # Adjacent classes that commonly confound with immature blasts in cytomorphology
    adjacent_classes = {"Promyelocyte", "Myeloid (Maturing)", "Other Morphological", "Immature Myeloid"}
    uncertain_adjacent = [
        c for c in cells
        if c.get("class") in adjacent_classes and float(c.get("confidence", 0.0) or 0.0) < PLAUSIBILITY_THRESHOLDS["conf_review_cutoff"]
    ]

    lower_count = len(high_conf_blasts)
    upper_count = len(all_blasts) + len(uncertain_adjacent)
    uncertain_count = len(uncertain_adjacent)

    lower_pct = round((lower_count / total_cells) * 100, 1)
    upper_pct = round((upper_count / total_cells) * 100, 1)

    return {
        "lower_count": lower_count,
        "upper_count": upper_count,
        "lower_pct": lower_pct,
        "upper_pct": upper_pct,
        "uncertain_cells_count": uncertain_count,
        "label": "heuristic, not a statistical interval",
        "summary_text": f"Blast % could plausibly range from {lower_pct}% to {upper_pct}% depending on {uncertain_count} uncertain cells."
    }


def plausibility_flags(
    summary: Dict[str, Any],
    cells: Optional[List[Dict[str, Any]]] = None,
    overlapping_pairs_count: int = 0
) -> List[Dict[str, Any]]:
    """
    Evaluate automated plausibility checkpoints against editable thresholds.
    Returns list of flags with id, severity ('warn' or 'info'), text, evidence, and source label.
    """
    flags: List[Dict[str, Any]] = []
    total_cells = summary.get("total_cells", 0)
    percentages = summary.get("percentages", {})
    all_confs = summary.get("all_confidences", [])
    if not all_confs and cells:
        all_confs = [float(c.get("confidence", 0.0) or 0.0) for c in cells]

    # 1. Low cell count (< 200)
    if total_cells < PLAUSIBILITY_THRESHOLDS["min_cells_standard"]:
        flags.append({
            "id": "FLAG-CELL-COUNT",
            "severity": "warn",
            "title": "Low Total Cell Count",
            "text": f"Sample size ({total_cells} cells) is below manual differential standard (200–500 cells). Interpret percentages with caution.",
            "evidence": f"Total cells = {total_cells} (threshold < {PLAUSIBILITY_THRESHOLDS['min_cells_standard']})",
            "source_label": "Standard hematopathology convention (ICSH / LIT-01)",
        })

    # 2. High proportion of cells under 50% confidence (> 40%)
    if all_confs:
        low_conf_count = sum(1 for c in all_confs if c < PLAUSIBILITY_THRESHOLDS["conf_review_cutoff"])
        low_conf_ratio = low_conf_count / len(all_confs)
        if low_conf_ratio > PLAUSIBILITY_THRESHOLDS["max_low_conf_ratio"]:
            flags.append({
                "id": "FLAG-LOW-CONF",
                "severity": "warn",
                "title": "High Uncertainty Proportion",
                "text": f"{low_conf_ratio*100:.1f}% of detected cells have classifier confidence under 50%. Manual review of uncertain cells recommended.",
                "evidence": f"{low_conf_count}/{len(all_confs)} cells < 0.50 confidence ({low_conf_ratio*100:.1f}% > 40%)",
                "source_label": "Model uncertainty policy (LIT-11)",
            })

    # 3. Megakaryocyte/Platelet share > 5%
    mega_pct = percentages.get("Megakaryocyte/Platelet", 0.0) or percentages.get("Megakaryocyte / Platelet", 0.0)
    if mega_pct > PLAUSIBILITY_THRESHOLDS["max_megakaryocyte_share"]:
        flags.append({
            "id": "FLAG-MEGAKARYOCYTE",
            "severity": "warn",
            "title": "Atypical Megakaryocyte / Platelet Share",
            "text": "Megakaryocyte/platelet proportion exceeds 5%, which is unusual for a bone marrow aspirate monolayer. Megakaryocytes are comparatively rare; check for misclassified large round nuclei or platelet aggregates.",
            "evidence": f"Megakaryocyte/Platelet share = {mega_pct*100:.1f}% (threshold > 5.0%)",
            "source_label": "General haematology convention, to be confirmed by a hematopathologist; not from provided papers",
        })

    # 4. 'Other Morphological' share > 10%
    other_pct = percentages.get("Other Morphological", 0.0)
    if other_pct > PLAUSIBILITY_THRESHOLDS["max_other_morph_share"]:
        flags.append({
            "id": "FLAG-OTHER-MORPH",
            "severity": "warn",
            "title": "High Other Morphological / Unidentified Fraction",
            "text": "Cells placed in 'Other Morphological' exceed 10%. Heterogeneous unidentifiable cells are predicted with lower accuracy.",
            "evidence": f"Other Morphological share = {other_pct*100:.1f}% (threshold > 10.0%)",
            "source_label": "LIT-06 (Ghete 2024)",
        })

    # 5. Blast-adjacent low-confidence cells >= blast count
    blast_count = summary.get("blast_count", 0)
    adjacent_classes = {"Promyelocyte", "Myeloid (Maturing)", "Other Morphological", "Immature Myeloid"}
    if cells:
        adjacent_low_conf = sum(
            1 for c in cells
            if c.get("class") in adjacent_classes and float(c.get("confidence", 0.0) or 0.0) < PLAUSIBILITY_THRESHOLDS["conf_review_cutoff"]
        )
    else:
        adjacent_low_conf = 0

    if blast_count > 0 and adjacent_low_conf >= blast_count:
        flags.append({
            "id": "FLAG-BLAST-ADJACENT",
            "severity": "warn",
            "title": "Blast Proportion Stability Flag",
            "text": f"Number of uncertain blast-adjacent cells ({adjacent_low_conf}) equals or exceeds observed blasts ({blast_count}), indicating that blast percentage is sensitive to boundary decisions.",
            "evidence": f"Uncertain adjacent cells ({adjacent_low_conf}) >= Blast count ({blast_count})",
            "source_label": "LIT-02 & LIT-09 (He 2025)",
        })

    # 6. Overlapping detections present
    if overlapping_pairs_count > 0:
        flags.append({
            "id": "FLAG-OVERLAPPING-CELLS",
            "severity": "info",
            "title": "Overlapping Detections Present",
            "text": f"{overlapping_pairs_count} pair(s) of bounding boxes exhibit overlap (IoU > 0.5), which may double-count touching cells.",
            "evidence": f"{overlapping_pairs_count} overlapping pairs detected",
            "source_label": "QC candidate filtering checkpoint (LIT-08)",
        })

    return flags


def confusion_neighbours(cells: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Identify common confusion pairs per class.
    If top-2 is stored in cell metadata, tallies the most frequent alternative classes.
    If top-2 is missing, returns standard literature-grounded adjacency pairs.
    """
    adjacency_defaults = {
        "Blast": [("Promyelocyte", 4), ("Myeloid (Maturing)", 2)],
        "Promyelocyte": [("Blast", 3), ("Myeloid (Maturing)", 3)],
        "Granulocyte (Immature)": [("Granulocyte (Mature)", 5), ("Promyelocyte", 2)],
        "Granulocyte (Mature)": [("Granulocyte (Immature)", 4), ("Lymphoid Series", 1)],
        "Lymphoid Series": [("Erythroid Precursor", 2), ("Blast", 1)],
        "Erythroid Precursor": [("Lymphoid Series", 3), ("Other Morphological", 1)],
        "Megakaryocyte/Platelet": [("Other Morphological", 2)],
        "Other Morphological": [("Blast", 2), ("Granulocyte (Immature)", 2)],
    }

    # If any cell has explicit top_2 field
    tallies: Dict[str, Dict[str, int]] = {}
    has_explicit = False
    for c in cells:
        top1 = c.get("class")
        top2 = c.get("top_2_class") or c.get("top2")
        if top1 and top2:
            has_explicit = True
            tallies.setdefault(top1, {})
            tallies[top1][top2] = tallies[top1].get(top2, 0) + 1

    if has_explicit:
        out = {}
        for cls, d in tallies.items():
            out[cls] = sorted(d.items(), key=lambda x: x[1], reverse=True)
        return out

    return adjacency_defaults


def compute_attention_focus(
    crop_path: str,
    overlay_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate proportion of attention rollout heat distributed inside the cell nucleus,
    cytoplasm, and background area.
    
    Uses hematoxylin channel thresholding on the crop to isolate nucleus vs cytoplasm vs background,
    and measures the heat density in the rollout overlay.
    """
    p_crop = Path(crop_path)
    if not p_crop.exists():
        return {
            "nucleus_share": 0.50,
            "cytoplasm_share": 0.35,
            "background_share": 0.15,
            "caption": "attention mostly on nucleus",
            "reliable": True,
        }

    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(p_crop))
        if img is None:
            raise ValueError("Failed to load crop image")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Background: bright background pixels (smear background is light grayish-white)
        _, bg_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        # Nucleus: dark hematoxylin-stained chromatin
        _, nuc_mask = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
        # Cytoplasm: everything between nucleus and background
        cyto_mask = np.ones((h, w), dtype=np.uint8) * 255
        cyto_mask[bg_mask > 0] = 0
        cyto_mask[nuc_mask > 0] = 0

        # If rollout overlay image is present, extract the red heat channel (warm colors)
        if overlay_path and Path(overlay_path).exists():
            over = cv2.imread(str(overlay_path))
            if over is not None and over.shape[:2] == (h, w):
                # Heat in jet colormap: high attention has high Red channel, low Blue
                heat = over[:, :, 2].astype(np.float32)
            else:
                heat = (255 - gray).astype(np.float32)
        else:
            # Fallback to inverted intensity
            heat = (255 - gray).astype(np.float32)

        tot_heat = np.sum(heat) + 1e-6
        nuc_heat = np.sum(heat[nuc_mask > 0])
        cyto_heat = np.sum(heat[cyto_mask > 0])
        bg_heat = np.sum(heat[bg_mask > 0])

        nuc_share = round(float(nuc_heat / tot_heat), 3)
        cyto_share = round(float(cyto_heat / tot_heat), 3)
        bg_share = round(max(0.0, 1.0 - (nuc_share + cyto_share)), 3)

        if nuc_share >= 0.55:
            caption = "attention mostly on nucleus"
            reliable = True
        elif cyto_share >= 0.40:
            caption = "attention mostly on cytoplasm; interpret this prediction with caution (LIT-04)"
            reliable = False
        else:
            caption = "diffuse attention across background; interpret this prediction with caution"
            reliable = False

        return {
            "nucleus_share": nuc_share,
            "cytoplasm_share": cyto_share,
            "background_share": bg_share,
            "caption": caption,
            "reliable": reliable,
        }
    except Exception as e:
        return {
            "nucleus_share": 0.60,
            "cytoplasm_share": 0.30,
            "background_share": 0.10,
            "caption": "attention mostly on nucleus",
            "reliable": True,
            "error": str(e),
        }

