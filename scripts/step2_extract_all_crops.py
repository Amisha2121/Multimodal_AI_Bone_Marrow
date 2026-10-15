"""
scripts/step2_extract_all_crops.py
─────────────────────────────────────────────────────────────────────────────
STEP 2: Re-extract ALL 20,852 real cell crops from IDC S3 DICOM sources.

Reads:
  data/processed/cell_metadata.csv     — annotation coordinates + split labels
  data/processed/patient_wsi_s3_map.json — patient_id → IDC S3 URL
  data/processed/splits.json           — patient-level split assignments

Writes:
  data/processed/cell_crops/{split}/{patient_id}/{cell_id}.png  (224×224 RGB)
  data/processed/extraction_log.jsonl  (per-patient stats, appended)

Design:
  - Streams DICOM tiles from IDC open-data S3 (anonymous, no credentials needed)
  - Does NOT download full multi-GB slides
  - Idempotent: skips crops that already exist
  - Logs per-patient successes/failures
  - Zero modification of metadata, labels, splits, or coordinates

Usage:
  python scripts/step2_extract_all_crops.py
  python scripts/step2_extract_all_crops.py --dry-run
  python scripts/step2_extract_all_crops.py --patients <ID1> <ID2> ...
  python scripts/step2_extract_all_crops.py --workers 4

Prerequisites:
  pip install wsidicom upath s3fs tqdm
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step2_extract")

CELL_META   = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
S3_MAP      = REPO_ROOT / "data" / "processed" / "patient_wsi_s3_map.json"
CROPS_DIR   = REPO_ROOT / "data" / "processed" / "cell_crops"
LOG_FILE    = REPO_ROOT / "data" / "processed" / "extraction_log.jsonl"

TARGET_SIZE  = 224
MARGIN_RATIO = 1.20

# Categories to EXCLUDE (artefacts, no-consensus, spicules)
EXCLUDED_STATUSES = {"excluded_artefact", "excluded_spicule", "excluded_no_consensus"}


def extract_patient(
    patient_id: str,
    cells_df: pd.DataFrame,
    s3_url: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Stream crop tiles for one patient from IDC S3."""
    split = cells_df["split"].iloc[0]
    target_dir = CROPS_DIR / split / patient_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Identify pending (missing) crops
    pending = []
    n_skipped = 0
    for _, row in cells_df.iterrows():
        crop_path = REPO_ROOT / row["crop_path"]
        if crop_path.exists():
            n_skipped += 1
        else:
            pending.append(row)

    if not pending:
        logger.info(f"[{patient_id}] All {n_skipped} crops already exist — skip.")
        return {"patient_id": patient_id, "n_success": 0,
                "n_failed": 0, "n_skipped": n_skipped, "elapsed_s": 0.0}

    if dry_run:
        logger.info(f"[{patient_id}] DRY RUN — would extract {len(pending)} crops.")
        return {"patient_id": patient_id, "n_success": len(pending),
                "n_failed": 0, "n_skipped": n_skipped, "elapsed_s": 0.0}

    # Import S3-streaming libraries
    try:
        import upath
        from wsidicom import WsiDicom
    except ImportError as e:
        raise ImportError(
            "wsidicom and upath are required.\n"
            "Install: pip install wsidicom upath s3fs"
        ) from e

    logger.info(f"[{patient_id}] Opening WSI from S3 ({len(pending)} crops pending)…")
    t0 = time.time()

    try:
        slide = WsiDicom.open(upath.UPath(s3_url, anon=True))
    except Exception as e:
        logger.error(f"[{patient_id}] Failed to open WSI: {e}")
        return {"patient_id": patient_id, "n_success": 0,
                "n_failed": len(pending), "n_skipped": n_skipped, "elapsed_s": 0.0}

    n_success = 0
    n_failed = 0

    for row in tqdm(pending, desc=patient_id, leave=False):
        cell_id   = str(row["cell_id"])
        crop_path = REPO_ROOT / row["crop_path"]
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        cx, cy = float(row["cx"]), float(row["cy"])
        w,  h  = float(row["width"]), float(row["height"])
        s   = max(w, h) * MARGIN_RATIO
        x0  = max(0, int(round(cx - s / 2)))
        y0  = max(0, int(round(cy - s / 2)))
        sz  = int(round(s))

        try:
            crop_img = slide.read_region((x0, y0), level=0, size=(sz, sz))
            if crop_img.mode != "RGB":
                crop_img = crop_img.convert("RGB")
            resized = crop_img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BICUBIC)
            resized.save(str(crop_path), format="PNG")
            n_success += 1
        except Exception as e:
            logger.warning(f"[{patient_id}] Failed {cell_id}: {e}")
            n_failed += 1

    try:
        slide.close()
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(
        f"[{patient_id}] Done {elapsed:.1f}s — "
        f"{n_success} extracted, {n_failed} failed, {n_skipped} skipped."
    )
    return {"patient_id": patient_id, "n_success": n_success,
            "n_failed": n_failed, "n_skipped": n_skipped, "elapsed_s": round(elapsed, 1)}


def run_extraction(
    patient_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    logger.info("=" * 70)
    logger.info("STEP 2 — Re-extract all real cell crops from IDC S3")
    logger.info("=" * 70)

    # Load metadata
    meta_df = pd.read_csv(CELL_META)
    valid_df = meta_df[meta_df["extraction_status"] == "valid"].copy()
    logger.info(f"Valid cells in metadata: {len(valid_df):,}")

    with open(S3_MAP) as f:
        s3_map = json.load(f)

    # All patients with valid cells
    all_patients = sorted(valid_df["patient_id"].unique())
    if patient_filter:
        all_patients = [p for p in all_patients if p in patient_filter]
    logger.info(f"Patients to process: {len(all_patients)}")

    total_success = total_failed = total_skipped = 0
    results = []

    for i, pid in enumerate(all_patients, 1):
        if pid not in s3_map:
            logger.warning(f"[{pid}] Not in S3 map — skipping.")
            continue
        patient_cells = valid_df[valid_df["patient_id"] == pid].copy()
        s3_url = s3_map[pid]["s3_url"]
        logger.info(
            f"--- Patient {i}/{len(all_patients)}: {pid} "
            f"({len(patient_cells)} valid cells, split={patient_cells['split'].iloc[0]}) ---"
        )
        result = extract_patient(pid, patient_cells, s3_url, dry_run=dry_run)
        results.append(result)
        total_success += result["n_success"]
        total_failed  += result["n_failed"]
        total_skipped += result["n_skipped"]
        # Append to extraction log
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    logger.info("=" * 70)
    logger.info(f"EXTRACTION SUMMARY")
    logger.info(f"  Extracted:  {total_success:,}")
    logger.info(f"  Failed:     {total_failed:,}")
    logger.info(f"  Skipped:    {total_skipped:,}")
    n_on_disk = sum(1 for _ in CROPS_DIR.rglob("*.png")) if CROPS_DIR.exists() else 0
    logger.info(f"  Total PNGs on disk: {n_on_disk:,}  (expected: 20,852)")

    if total_failed > 0:
        logger.warning(
            f"{total_failed} crops failed. Run step3_verify_crops.py to get the "
            "full failure list before proceeding to training."
        )

    logger.info("Run step3_verify_crops.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-extract all 20,852 cell crops from IDC S3 DICOM sources."
    )
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Specific patient IDs to process (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without reading any WSIs.")
    args = parser.parse_args()
    run_extraction(patient_filter=args.patients, dry_run=args.dry_run)
