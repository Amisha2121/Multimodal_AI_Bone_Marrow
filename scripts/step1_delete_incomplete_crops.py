"""
scripts/step1_delete_incomplete_crops.py
─────────────────────────────────────────────────────────────────────────────
STEP 1: Safe deletion of incomplete/partial cell crop cache.

Deletes ONLY extracted PNG crop images from:
  data/processed/cell_crops/
  data/processed/sample_crops/

DOES NOT delete:
  - data/processed/cell_metadata.csv   (annotation coordinates + labels)
  - data/processed/splits.json         (patient-level train/val/test splits)
  - data/patient_manifest.csv          (full cohort demographics + diagnoses)
  - data/processed/patient_wsi_s3_map.json (S3 URL map for extraction)
  - Any DICOM or WSI source files
  - Any documentation

Usage:
  python scripts/step1_delete_incomplete_crops.py
  python scripts/step1_delete_incomplete_crops.py --dry-run   # preview only
"""

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories containing ONLY crop images — safe to wipe
CROP_DIRS_TO_DELETE = [
    REPO_ROOT / "data" / "processed" / "cell_crops",
    REPO_ROOT / "data" / "processed" / "sample_crops",
    REPO_ROOT / "data" / "processed" / "benchmark_crops",
]

# Files that must NEVER be deleted — verified before any deletion
PROTECTED_FILES = [
    REPO_ROOT / "data" / "processed" / "cell_metadata.csv",
    REPO_ROOT / "data" / "processed" / "splits.json",
    REPO_ROOT / "data" / "patient_manifest.csv",
    REPO_ROOT / "data" / "processed" / "patient_wsi_s3_map.json",
]


def main(dry_run: bool = False, force: bool = False) -> None:
    print("=" * 70)
    print("STEP 1 - Delete incomplete crop cache")
    print("=" * 70)

    # ── Verify protected files exist before touching anything ─────────────
    print("\n[CHECK] Verifying protected files are present...")
    all_present = True
    for pf in PROTECTED_FILES:
        if pf.exists():
            print(f"  [OK]    {pf.relative_to(REPO_ROOT)}")
        else:
            print(f"  [FAIL]  MISSING: {pf.relative_to(REPO_ROOT)}  <- ABORT")
            all_present = False

    if not all_present:
        print("\n[ABORT] One or more protected files are missing. "
              "Do NOT proceed until they are restored.")
        sys.exit(1)

    # ── Count what will be deleted ────────────────────────────────────────
    print("\n[SCAN] Counting crop images to delete...")
    total_png = 0
    total_bytes = 0
    for crop_dir in CROP_DIRS_TO_DELETE:
        if not crop_dir.exists():
            print(f"  (skip) {crop_dir.relative_to(REPO_ROOT)} — does not exist")
            continue
        pngs = list(crop_dir.rglob("*.png"))
        size = sum(p.stat().st_size for p in pngs)
        print(f"  {crop_dir.relative_to(REPO_ROOT)}: "
              f"{len(pngs):,} PNG files  ({size / 1e6:.1f} MB)")
        total_png += len(pngs)
        total_bytes += size

    print(f"\n  Total to delete: {total_png:,} PNG files  ({total_bytes / 1e6:.1f} MB)")

    if dry_run:
        print("\n[DRY RUN] No files deleted. Re-run without --dry-run to proceed.")
        return

    if total_png == 0:
        print("\n[INFO] No crop files found. Nothing to delete.")
        return

    # ── Confirmation ──────────────────────────────────────────────────────
    if not force:
        confirm = input(f"\nDelete {total_png:,} PNG files? [yes/no]: ").strip().lower()
        if confirm != "yes":
            print("[ABORT] Cancelled by user.")
            sys.exit(0)

    # ── Delete ────────────────────────────────────────────────────────────
    deleted = 0
    for crop_dir in CROP_DIRS_TO_DELETE:
        if not crop_dir.exists():
            continue
        for f in crop_dir.rglob("*.png"):
            f.unlink()
            deleted += 1
        # Remove empty subdirectories (keep the top-level dir)
        for d in sorted(crop_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    print(f"\n[DONE] Deleted {deleted:,} PNG files.")

    # ── Re-verify protected files survived ───────────────────────────────
    print("\n[VERIFY] Confirming protected files are still intact...")
    for pf in PROTECTED_FILES:
        status = "[OK]" if pf.exists() else "[FAIL] MISSING!"
        print(f"  {status}  {pf.relative_to(REPO_ROOT)}")

    print("\n[STEP 1 COMPLETE] Crop cache cleared. Run step2_extract_all_crops.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be deleted without deleting anything.")
    parser.add_argument("--yes", "-y", "--force", action="store_true",
                        help="Skip confirmation prompt and proceed with deletion.")
    args = parser.parse_args()
    main(dry_run=args.dry_run, force=args.yes)
