"""
vision/extract_cells.py
─────────────────────────────────────────────────────────────────────────────
Real-Data Targeted Cell Patch Extractor for BoneMarrowWSI-PediatricLeukemia.

Extracts standardized 224x224 cell crops from remote whole-slide microscopy
DICOM instances using targeted tile reads via wsidicom + fsspec/s3fs, strictly
avoiding downloading entire 10+ GB slides.

Features:
- Reads coordinates from DICOM PointCoordinatesData (RECTANGLE format)
- Excludes artefacts (technically_unfit, damaged_cell, smudge_cell, etc.)
- Excludes tissue fragments (spicules) and ambiguous annotations
- Enforces strict patient-level partition inheritance from splits.json
- Applies square-padding (20% margin) and bicubic 224x224 resizing
- Logs detailed metadata and extraction status per cell
"""

import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pydicom
from PIL import Image
import upath
from wsidicom import WsiDicom

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_cells")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Documented artefact labels to exclude
ARTEFACT_LABELS = {
    "technically_unfit",
    "damaged_cell",
    "smudge_cell",
    "artifact",
    "neutrophil_extracellular_trap",
    "phagocytosis",
}

# Special categories to exclude from single-cell training
NON_SINGLE_CELL_LABELS = {
    "spicule",             # Macroscopic marrow fragment (thousands of pixels)
    "no_consensus_found",  # Disagreed multi-annotator label
}


def map_morphology_category(label: str) -> str:
    """Maps fine-grained consensus annotation labels to standardized functional categories."""
    if label in ARTEFACT_LABELS:
        return "Artefact (Excluded)"
    if label == "spicule":
        return "Spicule (Tissue Fragment - Excluded)"
    if "no_consensus" in label:
        return "Ambiguous / No Consensus (Excluded)"
    if "blast" in label:
        return "Blast"
    if label == "promyelocyte":
        return "Promyelocyte"
    if any(k in label for k in ["neutrophilic_myelocyte", "metamyelocyte", "precursor", "eosinophilic_myelocyte", "basophilic_myelocyte"]):
        return "Myeloid (Maturing)"
    if any(k in label for k in ["segmented", "band"]):
        return "Granulocyte (Mature/Band)"
    if any(k in label for k in ["monocyte", "promonocyte"]):
        return "Monocyte Series"
    if "erythroblast" in label:
        return "Erythroid Precursor"
    if any(k in label for k in ["lymphocyte", "prolymphocyte", "lymphoidocyte", "plasma_cell"]):
        return "Lymphoid Series"
    if any(k in label for k in ["megakaryocyte", "thrombocyte", "platelet"]):
        return "Megakaryocyte / Platelet"
    return "Other Morphological"


def parse_patient_consensus_dicom(
    dcm_path: Path,
    splits_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Parses all bounding boxes and labels from a single patient's consensus DICOM file.
    """
    ds = pydicom.dcmread(str(dcm_path))
    patient_id = ds.PatientID
    series_uid = ds.SeriesInstanceUID
    split = splits_map.get(patient_id, "unknown")

    ref_series = None
    if hasattr(ds, 'ReferencedSeriesSequence') and len(ds.ReferencedSeriesSequence) > 0:
        ref_series = ds.ReferencedSeriesSequence[0].SeriesInstanceUID

    ref_sop = None
    if hasattr(ds, 'ReferencedImageSequence') and len(ds.ReferencedImageSequence) > 0:
        ref_sop = ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID

    cells = []
    ann_idx = 0
    for group in getattr(ds, "AnnotationGroupSequence", []):
        lbl = getattr(group, "AnnotationGroupLabel", "unlabeled")
        uid = getattr(group, "AnnotationGroupUID", None)
        n = getattr(group, "NumberOfAnnotations", 0)
        coords_raw = getattr(group, "PointCoordinatesData", None)

        is_art = lbl in ARTEFACT_LABELS
        is_spicule = (lbl == "spicule")
        is_no_consensus = ("no_consensus" in lbl)

        if is_art:
            status = "excluded_artefact"
        elif is_spicule:
            status = "excluded_spicule"
        elif is_no_consensus:
            status = "excluded_no_consensus"
        else:
            status = "valid"

        mapped_cat = map_morphology_category(lbl)

        if coords_raw is not None and group.GraphicType == "RECTANGLE":
            floats = struct.unpack(f"{len(coords_raw)//4}f", coords_raw)
            for i in range(n):
                rect = floats[i*8 : (i+1)*8]
                x1, y1 = rect[0], rect[1]
                x2, y2 = rect[4], rect[5]
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                cell_id = f"{patient_id}_{ann_idx:05d}_{lbl}"
                crop_rel_path = f"data/processed/cell_crops/{split}/{patient_id}/{cell_id}.png"

                cells.append({
                    "cell_id": cell_id,
                    "patient_id": patient_id,
                    "split": split,
                    "original_cell_label": lbl,
                    "mapped_cell_category": mapped_cat,
                    "source_series_uid": series_uid,
                    "ref_series_uid": ref_series,
                    "ref_sop_uid": ref_sop,
                    "annotation_group_uid": uid,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "width": w, "height": h,
                    "is_artefact": is_art,
                    "is_spicule": is_spicule,
                    "is_no_consensus": is_no_consensus,
                    "extraction_status": status,
                    "crop_path": crop_rel_path,
                })
                ann_idx += 1

    return cells


def extract_patient_crops(
    patient_id: str,
    cells: List[Dict[str, Any]],
    s3_url: str,
    output_base_dir: Path = REPO_ROOT / "data" / "processed" / "cell_crops",
    target_size: int = 224,
    margin_ratio: float = 1.20,
) -> Tuple[int, int, List[str]]:
    """
    Connects to the patient's remote WSI on S3 via wsidicom and extracts valid cell crops.

    Returns:
        (num_success, num_failed, list_of_saved_paths)
    """
    valid_cells = [c for c in cells if c["extraction_status"] == "valid"]
    if not valid_cells:
        return 0, 0, []

    split = valid_cells[0]["split"]
    target_dir = output_base_dir / split / patient_id
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening WSI for patient {patient_id} ({len(valid_cells)} valid cells)...")
    t0 = time.time()
    try:
        slide = WsiDicom.open(upath.UPath(s3_url, anon=True))
    except Exception as e:
        logger.error(f"Failed to open WSI at {s3_url}: {e}")
        return 0, len(valid_cells), []

    logger.info(f"WSI opened in {time.time()-t0:.1f}s. Extracting patches...")
    saved_paths = []
    failed = 0

    for item in valid_cells:
        cell_id = item["cell_id"]
        save_path = target_dir / f"{cell_id}.png"

        # Skip if already extracted
        if save_path.exists():
            saved_paths.append(str(save_path))
            continue

        cx, cy = item["cx"], item["cy"]
        w, h = item["width"], item["height"]

        # Add 20% margin for cellular context
        s = max(w, h) * margin_ratio
        x0 = int(round(cx - s / 2))
        y0 = int(round(cy - s / 2))
        crop_size = int(round(s))

        # Clamp against slide bounds
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0

        try:
            crop_img = slide.read_region((x0, y0), level=0, size=(crop_size, crop_size))
            resized = crop_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
            resized.save(save_path)
            saved_paths.append(str(save_path))
        except Exception as e:
            logger.warning(f"Failed crop {cell_id} at ({x0}, {y0}, {crop_size}x{crop_size}): {e}")
            failed += 1

    return len(saved_paths), failed, saved_paths
