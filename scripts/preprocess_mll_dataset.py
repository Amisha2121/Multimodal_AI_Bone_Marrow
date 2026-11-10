"""
Preprocess MLL-Helmholtz-Fraunhofer Dataset for Multi-Dataset Training

This script converts the MLL dataset (21 classes) to our taxonomy (8 classes)
and creates the required metadata CSV, splits JSON, and cell crops directory.

Usage:
    python scripts/preprocess_mll_dataset.py --input data/external/mll_helmholtz/organized
    
    # With custom output:
    python scripts/preprocess_mll_dataset.py \
        --input data/external/mll_helmholtz/organized \
        --output data/external/mll_processed
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Missing required packages. Install with:")
    print("  pip install pandas pillow tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_mll")

# ══════════════════════════════════════════════════════════════════════════════
# Label Mapping: MLL 21 classes → Our 8 classes
# ══════════════════════════════════════════════════════════════════════════════

LABEL_MAPPING = {
    # ── Blasts ────────────────────────────────────────────────────────────
    "BLA": "blast",  # Blast cells
    
    # ── Mature Neutrophils ────────────────────────────────────────────────
    "NGS": "mature_neutrophil",  # Segmented neutrophil
    "NGB": "mature_neutrophil",  # Band neutrophil
    
    # ── Immature Granulocytes ─────────────────────────────────────────────
    "MYB": "immature_granulocyte",  # Myelocyte
    "MMZ": "immature_granulocyte",  # Metamyelocyte
    "PMO": "immature_granulocyte",  # Promyelocyte
    
    # ── Lymphocytes ───────────────────────────────────────────────────────
    "LYT": "lymphocyte",  # Lymphocyte
    "LYI": "lymphocyte",  # Immature lymphocyte
    "PLM": "lymphocyte",  # Plasma cell (lymphoid lineage)
    
    # ── Erythroblasts ─────────────────────────────────────────────────────
    "EBO": "erythroblast",  # Erythroblast
    "PEB": "erythroblast",  # Proerythroblast
    
    # ── Monocytes ─────────────────────────────────────────────────────────
    "MON": "monocyte",  # Monocyte
    
    # ── Eosinophils ───────────────────────────────────────────────────────
    "EOS": "eosinophil",  # Eosinophil
    "ABE": "eosinophil",  # Abnormal eosinophil
    
    # ── Basophils ─────────────────────────────────────────────────────────
    "BAS": "basophil",  # Basophil
    
    # ── EXCLUDE (not in our taxonomy) ─────────────────────────────────────
    "ART": "EXCLUDE",  # Artifact
    "KSC": "EXCLUDE",  # Smudge cell
    "NIF": "EXCLUDE",  # Not identifiable
    "OTH": "EXCLUDE",  # Other cell
    "FGC": "EXCLUDE",  # Faggott cell (rare, AML M3-specific)
    "HAC": "EXCLUDE",  # Hairy cell (rare, specific leukemia)
}

# MLL uses patient IDs embedded in filenames - we'll extract them
# Format: {CLASS}_{NUMBER}.jpg (e.g., BLA_00001.jpg)
# We'll group by patient using the file number ranges


def extract_patient_id_from_filename(filename: str, class_code: str) -> str:
    """
    Extract pseudo-patient ID from MLL filename.
    
    MLL doesn't provide explicit patient IDs, so we'll create synthetic ones
    by grouping images into pseudo-patients based on sequential numbering.
    
    Strategy: Group every 50-200 cells into a "patient" to ensure proper splits.
    """
    # Extract number from filename (e.g., BLA_00001.jpg -> 1)
    number_str = filename.replace(f"{class_code}_", "").replace(".jpg", "").replace(".png", "")
    try:
        number = int(number_str)
    except ValueError:
        # Fallback: hash the filename
        return f"MLL_PATIENT_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
    
    # Group into pseudo-patients (100 cells per patient)
    # This ensures good split separation while keeping reasonable patient sizes
    patient_batch = number // 100
    return f"MLL_PATIENT_{class_code}_{patient_batch:04d}"


def collect_image_paths(input_dir: Path) -> Dict[str, List[Path]]:
    """Collect all image paths organized by MLL class."""
    logger.info(f"Scanning for images in: {input_dir}")
    
    images_by_class = defaultdict(list)
    
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_code = class_dir.name
        if class_code not in LABEL_MAPPING:
            logger.warning(f"Unknown class: {class_code}, skipping")
            continue
        
        # Recursively find all JPG/PNG files
        image_files = list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.png"))
        images_by_class[class_code].extend(image_files)
        
        if image_files:
            logger.info(f"  {class_code}: {len(image_files)} images")
    
    return dict(images_by_class)


def create_metadata_entries(
    images_by_class: Dict[str, List[Path]],
    output_crops_dir: Path,
    resize_to: int = 224,
) -> List[Dict]:
    """
    Create metadata entries and copy/resize images to crops directory.
    """
    logger.info("Creating metadata and processing images...")
    
    metadata_entries = []
    
    # Create output crops directory structure
    output_crops_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = sum(len(imgs) for imgs in images_by_class.values())
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for mll_class, image_paths in images_by_class.items():
            mapped_class = LABEL_MAPPING[mll_class]
            
            if mapped_class == "EXCLUDE":
                logger.info(f"  Excluding {len(image_paths)} {mll_class} images")
                pbar.update(len(image_paths))
                continue
            
            # Create class-specific subdirectory
            class_output_dir = output_crops_dir / mapped_class
            class_output_dir.mkdir(exist_ok=True)
            
            for img_path in image_paths:
                # Generate unique cell ID
                cell_id = f"MLL_{mll_class}_{img_path.stem}"
                
                # Extract pseudo-patient ID
                patient_id = extract_patient_id_from_filename(img_path.name, mll_class)
                
                # Output path
                output_path = class_output_dir / f"{cell_id}.png"
                
                try:
                    # Load and resize image if needed
                    img = Image.open(img_path).convert("RGB")
                    
                    # Resize to 224x224 if not already
                    if img.size != (resize_to, resize_to):
                        img = img.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
                    
                    # Save as PNG
                    img.save(output_path, "PNG")
                    
                    # Create metadata entry
                    metadata_entries.append({
                        "cell_id": cell_id,
                        "patient_id": patient_id,
                        "mapped_cell_category": mapped_class,
                        "original_cell_label": mll_class,
                        "extraction_status": "valid",
                        "crop_path": str(output_path.relative_to(output_crops_dir.parent.parent)),
                        "source_dataset": "MLL-Helmholtz-Fraunhofer",
                        "original_file": str(img_path.name),
                        "image_width": resize_to,
                        "image_height": resize_to,
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {e}")
                
                pbar.update(1)
    
    logger.info(f"Processed {len(metadata_entries)} valid images")
    return metadata_entries


def create_patient_splits(
    metadata_entries: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create patient-level train/val/test splits.
    """
    logger.info("Creating patient-level splits...")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Get unique patient IDs
    patient_ids = list(set(entry["patient_id"] for entry in metadata_entries))
    random.seed(seed)
    random.shuffle(patient_ids)
    
    # Calculate split indices
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]
    
    splits = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
    }
    
    # Count cells per split
    split_counts = {split: 0 for split in ["train", "val", "test"]}
    for entry in metadata_entries:
        pid = entry["patient_id"]
        if pid in train_patients:
            split_counts["train"] += 1
        elif pid in val_patients:
            split_counts["val"] += 1
        elif pid in test_patients:
            split_counts["test"] += 1
    
    logger.info(f"Split created:")
    logger.info(f"  Train: {len(train_patients)} patients ({split_counts['train']} cells)")
    logger.info(f"  Val:   {len(val_patients)} patients ({split_counts['val']} cells)")
    logger.info(f"  Test:  {len(test_patients)} patients ({split_counts['test']} cells)")
    
    return splits


def print_class_distribution(metadata_entries: List[Dict], splits: Dict[str, List[str]]):
    """Print per-class distribution across splits."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLASS DISTRIBUTION ACROSS SPLITS")
    logger.info("=" * 70)
    
    # Organize by split
    split_data = {"train": [], "val": [], "test": []}
    for entry in metadata_entries:
        pid = entry["patient_id"]
        if pid in splits["train"]:
            split_data["train"].append(entry)
        elif pid in splits["val"]:
            split_data["val"].append(entry)
        elif pid in splits["test"]:
            split_data["test"].append(entry)
    
    # Count per class
    all_classes = sorted(set(entry["mapped_cell_category"] for entry in metadata_entries))
    
    print(f"{'Class':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 70)
    
    for cell_class in all_classes:
        counts = {
            "train": sum(1 for e in split_data["train"] if e["mapped_cell_category"] == cell_class),
            "val": sum(1 for e in split_data["val"] if e["mapped_cell_category"] == cell_class),
            "test": sum(1 for e in split_data["test"] if e["mapped_cell_category"] == cell_class),
        }
        total = sum(counts.values())
        print(f"{cell_class:<25} {counts['train']:>10} {counts['val']:>10} {counts['test']:>10} {total:>10}")
    
    print("-" * 70)
    totals = {k: len(v) for k, v in split_data.items()}
    print(f"{'TOTAL':<25} {totals['train']:>10} {totals['val']:>10} {totals['test']:>10} {sum(totals.values()):>10}")
    print("=" * 70)
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MLL-Helmholtz dataset for multi-dataset training"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing MLL organized class folders"
    )
    parser.add_argument(
        "--output", type=str, default="data/external/mll_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--resize", type=int, default=224,
        help="Resize images to this size (default: 224x224)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("Did you run download_mll_dataset.py first?")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MLL-Helmholtz Dataset Preprocessing")
    logger.info("=" * 70)
    logger.info(f"Input:  {input_dir.absolute()}")
    logger.info(f"Output: {output_dir.absolute()}")
    logger.info(f"Resize: {args.resize}x{args.resize}")
    logger.info("")
    
    # Step 1: Collect all image paths
    images_by_class = collect_image_paths(input_dir)
    
    if not images_by_class:
        logger.error("No images found! Check input directory.")
        sys.exit(1)
    
    # Step 2: Create metadata and process images
    output_crops_dir = output_dir / "crops"
    metadata_entries = create_metadata_entries(
        images_by_class,
        output_crops_dir,
        resize_to=args.resize,
    )
    
    if not metadata_entries:
        logger.error("No valid metadata entries created!")
        sys.exit(1)
    
    # Step 3: Create patient-level splits
    splits = create_patient_splits(metadata_entries, seed=args.seed)
    
    # Step 4: Save metadata CSV
    metadata_path = output_dir / "metadata.csv"
    df = pd.DataFrame(metadata_entries)
    df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata saved: {metadata_path}")
    
    # Step 5: Save splits JSON
    splits_path = output_dir / "splits.json"
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Splits saved: {splits_path}")
    
    # Step 6: Print distribution
    print_class_distribution(metadata_entries, splits)
    
    # Step 7: Save summary
    summary = {
        "dataset_name": "MLL-Helmholtz-Fraunhofer",
        "source_url": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770",
        "processed_date": pd.Timestamp.now().isoformat(),
        "total_images": len(metadata_entries),
        "num_patients": len(set(e["patient_id"] for e in metadata_entries)),
        "num_classes": len(set(e["mapped_cell_category"] for e in metadata_entries)),
        "classes": sorted(set(e["mapped_cell_category"] for e in metadata_entries)),
        "image_size": f"{args.resize}x{args.resize}",
        "splits": {
            "train_patients": len(splits["train"]),
            "val_patients": len(splits["val"]),
            "test_patients": len(splits["test"]),
        },
        "label_mapping": LABEL_MAPPING,
    }
    
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed dataset: {output_dir.absolute()}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Splits:   {splits_path}")
    logger.info(f"  Crops:    {output_crops_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Validate:  python scripts/validate_mll_dataset.py")
    logger.info("  2. Train:     python vision/train_multi_dataset.py \\")
    logger.info(f"                  --additional-metadata {metadata_path} \\")
    logger.info(f"                  --additional-splits {splits_path} \\")
    logger.info(f"                  --additional-crops {output_crops_dir}")


if __name__ == "__main__":
    main()
