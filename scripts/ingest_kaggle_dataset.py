"""
scripts/ingest_kaggle_dataset.py
-----------------------------------------------------------------------------
Ingest the Kaggle Bone Marrow Cell Classification dataset (Matek et al.)
into the existing pipeline format.
ZIP: archive (2).zip  (bone_marrow_cell_dataset/ + abbreviations.csv)
"""

import argparse
import csv
import logging
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ingest_kaggle")

# Kaggle 3-letter code -> MORPH_CLASS (None = excluded from training)
KAGGLE_TO_MORPH: Dict[str, Optional[str]] = {
    "BLA": "Blast",
    "FGC": "Blast",                       # Faggott cell
    "HAC": "Blast",                       # Hairy cell
    "LYI": "Lymphoid Series",
    "LYT": "Lymphoid Series",
    "PLM": "Lymphoid Series",
    "MON": "Monocyte Series",
    "NGB": "Granulocyte (Mature/Band)",
    "NGS": "Granulocyte (Mature/Band)",
    "BAS": "Granulocyte (Mature/Band)",
    "EOS": "Granulocyte (Mature/Band)",
    "ABE": "Granulocyte (Mature/Band)",
    "MMZ": "Myeloid (Maturing)",
    "MYB": "Myeloid (Maturing)",
    "PMO": "Promyelocyte",
    "EBO": "Other Morphological",
    "PEB": "Other Morphological",
    "OTH": "Other Morphological",
    "ART": None,  # Artefact - excluded
    "KSC": None,  # Smudge cell - excluded
    "NIF": None,  # Not identifiable - excluded
}


def extract_zip(zip_path: Path, out_dir: Path, dry_run: bool) -> Path:
    dataset_dir = out_dir / "bone_marrow_cell_dataset"
    if dataset_dir.exists():
        logger.info(f"Already extracted: {dataset_dir} (skipping unzip)")
        return dataset_dir
    logger.info(f"Extracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB) -> {out_dir}")
    logger.info("This may take 10-20 minutes ...")
    if dry_run:
        logger.info("[DRY RUN] Would extract zip here.")
        return dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith("bone_marrow_cell_dataset/")]
        total = len(members)
        for i, member in enumerate(members):
            zf.extract(member, out_dir)
            if i % 20000 == 0:
                logger.info(f"  Extracted {i:,}/{total:,} files ...")
    logger.info(f"Extraction complete -> {dataset_dir}")
    return dataset_dir


def collect_images(dataset_dir: Path, max_per_class: int) -> Dict[str, List[Path]]:
    images: Dict[str, List[Path]] = {}
    if not dataset_dir.exists():
        logger.info(f"[DRY RUN] Dataset dir does not exist yet: {dataset_dir}")
        logger.info("[DRY RUN] Simulating class list from KAGGLE_TO_MORPH mapping.")
        for code in KAGGLE_TO_MORPH:
            images[code] = []  # empty — just shows the mapping
        return images
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        code = class_dir.name.upper()
        if code not in KAGGLE_TO_MORPH:
            logger.warning(f"Unknown class code '{code}' - skipping.")
            continue
        paths: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(class_dir.rglob(ext))
        random.shuffle(paths)
        images[code] = paths[:max_per_class]
        logger.info(f"  {code}: {len(paths):,} images found, using {len(images[code]):,}")
    return images


def split_and_copy(
    images: Dict[str, List[Path]],
    crops_dir: Path,
    train_frac: float,
    val_frac: float,
    dry_run: bool,
) -> List[dict]:
    rows: List[dict] = []
    for code, paths in images.items():
        morph_class = KAGGLE_TO_MORPH.get(code)
        if morph_class is None:
            continue
        n = len(paths)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        split_map = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        patient_id = f"KAG_{code}"
        for split_name, split_paths in split_map.items():
            dest_dir = crops_dir / split_name / patient_id
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_paths:
                cell_id = f"{code}_{src_path.stem}"
                dest_path = dest_dir / f"{cell_id}.jpg"
                if not dry_run and not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                rows.append({
                    "cell_id": cell_id,
                    "patient_id": patient_id,
                    "split": split_name,
                    "crop_path": str(dest_path),
                    "class_code": morph_class,
                    "kaggle_code": code,
                    "extraction_status": "valid",
                    "source": "kaggle_matek",
                    "cohort": "matek_bone_marrow",
                })
    return rows


def append_to_cell_metadata(new_rows: List[dict], meta_path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info(f"[DRY RUN] Would append {len(new_rows):,} rows to {meta_path}")
        return
    existing_ids: set = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_ids.add(row.get("cell_id", ""))
    to_add = [r for r in new_rows if r["cell_id"] not in existing_ids]
    if not to_add:
        logger.info("All Kaggle cells already in cell_metadata.csv.")
        return
    fieldnames = ["cell_id", "patient_id", "split", "crop_path", "class_code",
                  "kaggle_code", "extraction_status", "source", "cohort"]
    write_header = not meta_path.exists()
    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(to_add)
    logger.info(f"Appended {len(to_add):,} new rows to {meta_path}")


def write_manifest(rows: List[dict], manifest_path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info(f"[DRY RUN] Would write manifest -> {manifest_path}")
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Manifest -> {manifest_path} ({len(rows):,} cells)")


def print_summary(rows: List[dict]) -> None:
    split_counts = Counter(r["split"] for r in rows)
    class_counts = Counter(r["class_code"] for r in rows)
    print("\n" + "=" * 60)
    print("KAGGLE INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total cells (after exclusions): {len(rows):,}")
    print("\nSplit distribution:")
    for split in ["train", "val", "test"]:
        print(f"  {split:<8}: {split_counts[split]:>7,}")
    print("\nMorphology class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<35}: {cnt:>7,}")
    print("=" * 60)
    print("\nNext: python scripts/step4_train_cell_classifier.py\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle bone marrow dataset.")
    parser.add_argument("--zip", default=r"C:\Users\AMISHA\Downloads\archive (2).zip")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "kaggle" / "raw"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        logger.error(f"ZIP not found: {zip_path}")
        sys.exit(1)

    random.seed(args.seed)
    crops_dir = REPO_ROOT / "data" / "processed" / "cell_crops"
    meta_path = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
    manifest_path = REPO_ROOT / "data" / "kaggle" / "kaggle_manifest.csv"

    dataset_dir = extract_zip(zip_path, Path(args.out_dir), args.dry_run)
    logger.info("Collecting images ...")
    images = collect_images(dataset_dir, args.max_per_class)
    if args.dry_run:
        logger.info("[DRY RUN] Class mapping preview:")
        for code, morph in KAGGLE_TO_MORPH.items():
            status = morph if morph else "EXCLUDED"
            logger.info(f"  {code} -> {status}")
        logger.info("[DRY RUN] Complete. Remove --dry-run to run for real.")
        return
    logger.info("Copying and splitting crops ...")
    rows = split_and_copy(images, crops_dir, args.train_frac, args.val_frac, args.dry_run)
    append_to_cell_metadata(rows, meta_path, args.dry_run)
    write_manifest(rows, manifest_path, args.dry_run)
    print_summary(rows)


if __name__ == "__main__":
    main()
