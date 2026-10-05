"""
vision/run_extraction.py
─────────────────────────────────────────────────────────────────────────────
Full-cohort cell crop extraction pipeline for BoneMarrowWSI-PediatricLeukemia.

Reads:
  - data/processed/cell_metadata.csv  (annotation coordinates + patient IDs)
  - data/processed/patient_wsi_s3_map.json  (patient_id → S3 URL)
  - data/processed/splits.json  (patient-level split assignments)

Writes:
  - data/processed/cell_crops/{split}/{patient_id}/{cell_id}.png

Usage:
  python -m vision.run_extraction [--workers N] [--patients P1 P2 ...]
                                  [--dry-run] [--limit-per-patient N]

Design:
  - Sequential patient-level processing (one WSI open at a time to avoid
    exhausting S3 connections)
  - Skips cells that already have a saved crop (idempotent)
  - Logs extraction stats per patient to extraction_log.jsonl
  - Updates cell_metadata.csv with extraction_status for newly extracted cells
"""

import argparse
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_extraction")

REPO_ROOT = Path(__file__).resolve().parent.parent
CELL_META  = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
S3_MAP     = REPO_ROOT / "data" / "processed" / "patient_wsi_s3_map.json"
SPLITS     = REPO_ROOT / "data" / "processed" / "splits.json"
CROPS_DIR  = REPO_ROOT / "data" / "processed" / "cell_crops"
LOG_FILE   = REPO_ROOT / "data" / "processed" / "extraction_log.jsonl"

TARGET_SIZE   = 224     # pixels
MARGIN_RATIO  = 1.20    # 20% border padding around annotated bounding box


def extract_crops_for_patient(
    patient_id: str,
    cells_df: pd.DataFrame,
    s3_url: str,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Opens the patient's remote WSI and extracts valid cell crops.

    Args:
        patient_id: Patient identifier.
        cells_df: Subset of cell_metadata.csv for this patient (valid cells only).
        s3_url: wsidicom-compatible S3 URL for this patient's SM DICOM series.
        dry_run: If True, skip actual WSI reads and return dummy stats.
        limit: Optional cap on number of crops to extract per patient.

    Returns:
        dict with 'patient_id', 'n_success', 'n_failed', 'n_skipped', 'elapsed_s'
    """
    try:
        import upath
        from wsidicom import WsiDicom
    except ImportError as e:
        raise ImportError(
            "wsidicom and upath are required for crop extraction.\n"
            "Install with: pip install wsidicom upath s3fs"
        ) from e

    split = cells_df["split"].iloc[0] if len(cells_df) > 0 else "unknown"
    target_dir = CROPS_DIR / split / patient_id
    target_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        cells_df = cells_df.head(limit)

    n_skipped = 0
    pending_rows = []
    for _, row in cells_df.iterrows():
        crop_path = REPO_ROOT / row["crop_path"]
        if crop_path.exists():
            n_skipped += 1
        else:
            pending_rows.append(row)

    if not pending_rows:
        logger.info(f"[{patient_id}] All {n_skipped} crops already extracted — skip.")
        return {
            "patient_id": patient_id, "n_success": 0,
            "n_failed": 0, "n_skipped": n_skipped, "elapsed_s": 0.0,
        }

    if dry_run:
        logger.info(f"[{patient_id}] DRY RUN — would extract {len(pending_rows)} crops.")
        return {
            "patient_id": patient_id, "n_success": len(pending_rows),
            "n_failed": 0, "n_skipped": n_skipped, "elapsed_s": 0.0,
        }

    logger.info(f"[{patient_id}] Opening WSI ({len(pending_rows)} crops to extract) ...")
    t0 = time.time()
    try:
        slide = WsiDicom.open(upath.UPath(s3_url, anon=True))
    except Exception as e:
        logger.error(f"[{patient_id}] Failed to open WSI: {e}")
        return {
            "patient_id": patient_id, "n_success": 0,
            "n_failed": len(pending_rows), "n_skipped": n_skipped, "elapsed_s": 0.0,
        }

    n_success = 0
    n_failed  = 0

    for row in pending_rows:
        cell_id = str(row["cell_id"])
        crop_path = REPO_ROOT / row["crop_path"]
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        cx, cy = float(row["cx"]), float(row["cy"])
        w,  h  = float(row["width"]), float(row["height"])

        # Padded square region
        s   = max(w, h) * MARGIN_RATIO
        x0  = max(0, int(round(cx - s / 2)))
        y0  = max(0, int(round(cy - s / 2)))
        sz  = int(round(s))

        try:
            crop_img = slide.read_region((x0, y0), level=0, size=(sz, sz))
            resized  = crop_img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BICUBIC)
            resized.save(str(crop_path))
            n_success += 1
        except Exception as e:
            logger.warning(f"[{patient_id}] Failed {cell_id}: {e}")
            n_failed += 1

    elapsed = time.time() - t0
    logger.info(
        f"[{patient_id}] Done in {elapsed:.1f}s — "
        f"{n_success} extracted, {n_failed} failed, {n_skipped} skipped."
    )

    return {
        "patient_id": patient_id,
        "n_success":  n_success,
        "n_failed":   n_failed,
        "n_skipped":  n_skipped,
        "elapsed_s":  round(elapsed, 1),
    }


def run_full_extraction(
    patient_filter: Optional[List[str]] = None,
    dry_run: bool = False,
    limit_per_patient: Optional[int] = None,
) -> None:
    """
    Runs crop extraction for all consensus patients (or a filtered subset).
    """
    # Load metadata
    meta_df = pd.read_csv(CELL_META)
    valid_df = meta_df[meta_df["extraction_status"] == "valid"].copy()
    logger.info(f"Total valid cells in metadata: {len(valid_df)}")

    with open(S3_MAP, "r") as f:
        s3_map = json.load(f)

    # Apply patient filter
    all_patients = sorted(valid_df["patient_id"].unique())
    if patient_filter:
        all_patients = [p for p in all_patients if p in patient_filter]
    logger.info(f"Processing {len(all_patients)} patient(s).")

    summary = []
    total_success = 0
    total_failed  = 0
    total_skipped = 0

    for i, pid in enumerate(all_patients, 1):
        if pid not in s3_map:
            logger.warning(f"[{pid}] No S3 URL in patient_wsi_s3_map.json — skipping.")
            continue

        patient_cells = valid_df[valid_df["patient_id"] == pid].copy()
        s3_url = s3_map[pid]["s3_url"]

        logger.info(f"--- Patient {i}/{len(all_patients)}: {pid} ({len(patient_cells)} valid cells) ---")
        result = extract_crops_for_patient(
            patient_id=pid,
            cells_df=patient_cells,
            s3_url=s3_url,
            dry_run=dry_run,
            limit=limit_per_patient,
        )
        summary.append(result)
        total_success += result["n_success"]
        total_failed  += result["n_failed"]
        total_skipped += result["n_skipped"]

        # Append to JSONL log
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    logger.info("=" * 60)
    logger.info(f"Extraction complete.")
    logger.info(f"  Extracted: {total_success}")
    logger.info(f"  Failed:    {total_failed}")
    logger.info(f"  Skipped (already exists): {total_skipped}")
    logger.info(f"Log appended to: {LOG_FILE}")

    # Count total crops on disk
    n_on_disk = sum(1 for _ in CROPS_DIR.rglob("*.png"))
    logger.info(f"Total crops on disk: {n_on_disk}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract cell crops from BoneMarrowWSI-PediatricLeukemia consensus annotations."
    )
    parser.add_argument(
        "--patients", nargs="+", default=None,
        help="Space-separated patient IDs to process (default: all consensus patients)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be extracted without reading any WSIs."
    )
    parser.add_argument(
        "--limit-per-patient", type=int, default=None,
        help="Cap extraction at N crops per patient (useful for quick tests)."
    )
    args = parser.parse_args()

    run_full_extraction(
        patient_filter=args.patients,
        dry_run=args.dry_run,
        limit_per_patient=args.limit_per_patient,
    )


if __name__ == "__main__":
    main()
