"""
Validate MLL-Helmholtz Processed Dataset

Checks that the processed MLL dataset is ready for multi-dataset training.

Usage:
    python scripts/validate_mll_dataset.py --dataset data/external/mll_processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

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
logger = logging.getLogger("validate_mll")

# Expected taxonomy (8 classes)
EXPECTED_CLASSES = [
    "blast",
    "mature_neutrophil",
    "immature_granulocyte",
    "lymphocyte",
    "erythroblast",
    "monocyte",
    "eosinophil",
    "basophil",
]


def check_file_structure(dataset_dir: Path) -> Dict[str, bool]:
    """Check that required files exist."""
    logger.info("Checking file structure...")
    
    required_files = {
        "metadata.csv": dataset_dir / "metadata.csv",
        "splits.json": dataset_dir / "splits.json",
        "crops": dataset_dir / "crops",
    }
    
    results = {}
    for name, path in required_files.items():
        exists = path.exists()
        results[name] = exists
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {name}: {path}")
    
    return results


def validate_metadata(metadata_path: Path) -> Dict[str, any]:
    """Validate metadata CSV."""
    logger.info("\nValidating metadata CSV...")
    
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        logger.error(f"Failed to read metadata: {e}")
        return {"valid": False, "error": str(e)}
    
    # Check required columns
    required_columns = [
        "cell_id",
        "patient_id",
        "mapped_cell_category",
        "extraction_status",
        "crop_path",
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        return {"valid": False, "missing_columns": missing_columns}
    
    logger.info(f"  ✓ All required columns present")
    logger.info(f"  ✓ Total rows: {len(df):,}")
    
    # Check cell classes
    found_classes = df["mapped_cell_category"].unique().tolist()
    missing_classes = set(EXPECTED_CLASSES) - set(found_classes)
    extra_classes = set(found_classes) - set(EXPECTED_CLASSES)
    
    if missing_classes:
        logger.warning(f"  ⚠ Missing classes: {missing_classes}")
    if extra_classes:
        logger.warning(f"  ⚠ Extra classes: {extra_classes}")
    
    logger.info(f"  ✓ Classes found: {len(found_classes)}")
    for cls in sorted(found_classes):
        count = len(df[df["mapped_cell_category"] == cls])
        logger.info(f"      {cls}: {count:,} cells")
    
    # Check extraction status
    valid_count = len(df[df["extraction_status"] == "valid"])
    logger.info(f"  ✓ Valid cells: {valid_count:,} / {len(df):,}")
    
    # Check for duplicates
    duplicate_ids = df["cell_id"].duplicated().sum()
    if duplicate_ids > 0:
        logger.warning(f"  ⚠ Duplicate cell IDs: {duplicate_ids}")
    else:
        logger.info(f"  ✓ No duplicate cell IDs")
    
    # Check unique patients
    unique_patients = df["patient_id"].nunique()
    logger.info(f"  ✓ Unique patients: {unique_patients:,}")
    
    return {
        "valid": True,
        "total_cells": len(df),
        "valid_cells": valid_count,
        "classes": found_classes,
        "unique_patients": unique_patients,
        "duplicate_ids": duplicate_ids,
    }


def validate_splits(splits_path: Path, metadata_path: Path) -> Dict[str, any]:
    """Validate splits JSON."""
    logger.info("\nValidating splits JSON...")
    
    try:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read splits: {e}")
        return {"valid": False, "error": str(e)}
    
    # Check required keys
    required_keys = ["train", "val", "test"]
    missing_keys = [key for key in required_keys if key not in splits]
    if missing_keys:
        logger.error(f"Missing split keys: {missing_keys}")
        return {"valid": False, "missing_keys": missing_keys}
    
    logger.info(f"  ✓ All split keys present")
    
    # Check split sizes
    for split_name, patient_ids in splits.items():
        logger.info(f"  ✓ {split_name}: {len(patient_ids)} patients")
    
    # Check for overlaps
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    overlaps = []
    if train_val_overlap:
        logger.error(f"  ✗ Train/Val overlap: {len(train_val_overlap)} patients")
        overlaps.append("train_val")
    if train_test_overlap:
        logger.error(f"  ✗ Train/Test overlap: {len(train_test_overlap)} patients")
        overlaps.append("train_test")
    if val_test_overlap:
        logger.error(f"  ✗ Val/Test overlap: {len(val_test_overlap)} patients")
        overlaps.append("val_test")
    
    if not overlaps:
        logger.info(f"  ✓ No patient overlap between splits")
    
    # Cross-check with metadata
    df = pd.read_csv(metadata_path)
    metadata_patients = set(df["patient_id"].unique())
    all_split_patients = train_set | val_set | test_set
    
    missing_in_splits = metadata_patients - all_split_patients
    missing_in_metadata = all_split_patients - metadata_patients
    
    if missing_in_splits:
        logger.warning(f"  ⚠ {len(missing_in_splits)} patients in metadata but not in splits")
    if missing_in_metadata:
        logger.warning(f"  ⚠ {len(missing_in_metadata)} patients in splits but not in metadata")
    
    if not missing_in_splits and not missing_in_metadata:
        logger.info(f"  ✓ All patients accounted for in both metadata and splits")
    
    return {
        "valid": len(overlaps) == 0,
        "overlaps": overlaps,
        "train_patients": len(splits["train"]),
        "val_patients": len(splits["val"]),
        "test_patients": len(splits["test"]),
    }


def validate_images(crops_dir: Path, metadata_path: Path, sample_size: int = 100) -> Dict[str, any]:
    """Validate image files (sample check for speed)."""
    logger.info(f"\nValidating image files (sampling {sample_size} images)...")
    
    df = pd.read_csv(metadata_path)
    
    # Sample images
    if len(df) > sample_size:
        sampled = df.sample(n=sample_size, random_state=42)
    else:
        sampled = df
    
    results = {
        "checked": 0,
        "missing": 0,
        "corrupted": 0,
        "wrong_size": 0,
        "valid": 0,
    }
    
    with tqdm(total=len(sampled), desc="Checking images") as pbar:
        for _, row in sampled.iterrows():
            crop_path = row["crop_path"]
            
            # Resolve path (could be relative or absolute)
            if Path(crop_path).is_absolute():
                full_path = Path(crop_path)
            else:
                # Try relative to crops_dir parent parent (dataset root)
                full_path = crops_dir.parent.parent / crop_path
            
            results["checked"] += 1
            
            if not full_path.exists():
                results["missing"] += 1
                pbar.update(1)
                continue
            
            try:
                img = Image.open(full_path)
                img.verify()  # Check if image is corrupted
                
                # Re-open for size check (verify() closes the file)
                img = Image.open(full_path)
                width, height = img.size
                
                if width != 224 or height != 224:
                    results["wrong_size"] += 1
                else:
                    results["valid"] += 1
                
            except Exception as e:
                results["corrupted"] += 1
            
            pbar.update(1)
    
    logger.info(f"  Checked: {results['checked']} images")
    logger.info(f"  ✓ Valid:     {results['valid']}")
    logger.info(f"  ✗ Missing:   {results['missing']}")
    logger.info(f"  ✗ Corrupted: {results['corrupted']}")
    logger.info(f"  ⚠ Wrong size: {results['wrong_size']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate processed MLL-Helmholtz dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to processed dataset directory"
    )
    parser.add_argument(
        "--sample-images", type=int, default=100,
        help="Number of images to sample for validation (default: 100)"
    )
    parser.add_argument(
        "--full-image-check", action="store_true",
        help="Check all images (slow for large datasets)"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset)
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("MLL-Helmholtz Dataset Validation")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset_dir.absolute()}")
    logger.info("")
    
    all_valid = True
    
    # Check 1: File structure
    structure = check_file_structure(dataset_dir)
    if not all(structure.values()):
        logger.error("Missing required files!")
        all_valid = False
    
    # Check 2: Metadata
    if structure.get("metadata.csv"):
        metadata_results = validate_metadata(dataset_dir / "metadata.csv")
        if not metadata_results.get("valid"):
            all_valid = False
    else:
        all_valid = False
    
    # Check 3: Splits
    if structure.get("splits.json") and structure.get("metadata.csv"):
        splits_results = validate_splits(
            dataset_dir / "splits.json",
            dataset_dir / "metadata.csv"
        )
        if not splits_results.get("valid"):
            all_valid = False
    else:
        all_valid = False
    
    # Check 4: Images
    if structure.get("crops") and structure.get("metadata.csv"):
        sample_size = None if args.full_image_check else args.sample_images
        image_results = validate_images(
            dataset_dir / "crops",
            dataset_dir / "metadata.csv",
            sample_size=sample_size or 100
        )
        if image_results["missing"] > 0 or image_results["corrupted"] > 0:
            logger.warning("Some images are missing or corrupted!")
            if image_results["missing"] / image_results["checked"] > 0.01:
                logger.error("More than 1% of checked images are missing!")
                all_valid = False
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    if all_valid:
        logger.info("✓ VALIDATION PASSED")
        logger.info("=" * 70)
        logger.info("Dataset is ready for training!")
        logger.info("")
        logger.info("Train with:")
        logger.info("  python vision/train_multi_dataset.py \\")
        logger.info(f"    --additional-metadata {dataset_dir}/metadata.csv \\")
        logger.info(f"    --additional-splits {dataset_dir}/splits.json \\")
        logger.info(f"    --additional-crops {dataset_dir}/crops")
    else:
        logger.error("✗ VALIDATION FAILED")
        logger.info("=" * 70)
        logger.error("Please fix the issues above before training.")
    logger.info("")


if __name__ == "__main__":
    main()
