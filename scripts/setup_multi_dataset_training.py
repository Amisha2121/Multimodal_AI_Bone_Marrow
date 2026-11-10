"""
Master Setup Script for Multi-Dataset Training

This script orchestrates the complete workflow:
1. Check if MLL dataset is downloaded
2. Preprocess the dataset
3. Validate the processed dataset
4. Provide training command

Usage:
    python scripts/setup_multi_dataset_training.py
    
    # Skip download instructions:
    python scripts/setup_multi_dataset_training.py --skip-download-check
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup")

REPO_ROOT = Path(__file__).resolve().parent.parent
MLL_DOWNLOAD_DIR = REPO_ROOT / "data" / "external" / "mll_helmholtz"
MLL_PROCESSED_DIR = REPO_ROOT / "data" / "external" / "mll_processed"


def check_dataset_downloaded() -> bool:
    """Check if MLL dataset has been downloaded."""
    download_dir = MLL_DOWNLOAD_DIR / "download"
    if not download_dir.exists():
        return False
    
    # Look for archive files
    archive_extensions = ['.zip', '.tar.gz', '.tgz', '.tar']
    for ext in archive_extensions:
        if list(download_dir.glob(f"*{ext}")):
            return True
    
    # Check if already extracted
    extracted_dir = MLL_DOWNLOAD_DIR / "extracted"
    organized_dir = MLL_DOWNLOAD_DIR / "organized"
    if extracted_dir.exists() or organized_dir.exists():
        return True
    
    return False


def run_download_script():
    """Run the download script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: Download MLL-Helmholtz Dataset")
    logger.info("=" * 70)
    logger.info("")
    
    script_path = REPO_ROOT / "scripts" / "download_mll_dataset.py"
    cmd = [sys.executable, str(script_path)]
    
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def run_preprocessing():
    """Run the preprocessing script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Preprocess Dataset")
    logger.info("=" * 70)
    logger.info("")
    
    # Find organized directory
    organized_dir = MLL_DOWNLOAD_DIR / "organized"
    if not organized_dir.exists():
        # Try extracted directory
        extracted_dir = MLL_DOWNLOAD_DIR / "extracted"
        if extracted_dir.exists():
            organized_dir = extracted_dir
        else:
            logger.error("Could not find extracted dataset directory!")
            logger.info("Expected locations:")
            logger.info(f"  - {MLL_DOWNLOAD_DIR / 'organized'}")
            logger.info(f"  - {MLL_DOWNLOAD_DIR / 'extracted'}")
            return False
    
    script_path = REPO_ROOT / "scripts" / "preprocess_mll_dataset.py"
    cmd = [
        sys.executable, str(script_path),
        "--input", str(organized_dir),
        "--output", str(MLL_PROCESSED_DIR),
    ]
    
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    logger.info("")
    
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def run_validation():
    """Run the validation script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Validate Dataset")
    logger.info("=" * 70)
    logger.info("")
    
    script_path = REPO_ROOT / "scripts" / "validate_mll_dataset.py"
    cmd = [
        sys.executable, str(script_path),
        "--dataset", str(MLL_PROCESSED_DIR),
        "--sample-images", "500",  # Check 500 images
    ]
    
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    logger.info("")
    
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def print_training_instructions():
    """Print final training instructions."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("READY TO TRAIN")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Your multi-dataset training is ready!")
    logger.info("")
    logger.info("Training command:")
    logger.info("-" * 70)
    
    metadata_path = MLL_PROCESSED_DIR / "metadata.csv"
    splits_path = MLL_PROCESSED_DIR / "splits.json"
    crops_path = MLL_PROCESSED_DIR / "crops"
    
    print(f"""
python vision/train_multi_dataset.py --config configs/default.yaml \\
    --additional-metadata {metadata_path} \\
    --additional-splits {splits_path} \\
    --additional-crops {crops_path}
""")
    
    logger.info("-" * 70)
    logger.info("")
    logger.info("Expected training time:")
    logger.info("  - GPU (RTX 3090 or similar): ~8-12 hours")
    logger.info("  - GPU (T4 or similar): ~15-20 hours")
    logger.info("  - CPU: Not recommended (days)")
    logger.info("")
    logger.info("Expected improvements:")
    logger.info("  - Macro F1: +8-12% (from adding 147K+ cells)")
    logger.info("  - Better generalization across patient populations")
    logger.info("  - More robust rare cell detection")
    logger.info("")
    logger.info("Output:")
    logger.info(f"  - Best model: models/checkpoints/best_vit_lora_morphology_multi_dataset.pt")
    logger.info(f"  - Taxonomy: models/checkpoints/taxonomy_multi_dataset.json")
    logger.info(f"  - Calibration: models/checkpoints/calibration_multi_dataset/")
    logger.info("")
    logger.info("Optional: Run training in background (Linux/Mac):")
    logger.info("  nohup python vision/train_multi_dataset.py ... > training.log 2>&1 &")
    logger.info("")
    logger.info("Optional: Run training in background (Windows PowerShell):")
    logger.info("  Start-Process python -ArgumentList 'vision/train_multi_dataset.py', ... -NoNewWindow -RedirectStandardOutput training.log")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Setup multi-dataset training with MLL-Helmholtz dataset"
    )
    parser.add_argument(
        "--skip-download-check", action="store_true",
        help="Skip checking if dataset is downloaded"
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Only run preprocessing step"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run validation step"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("MULTI-DATASET TRAINING SETUP")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This script will help you set up training with:")
    logger.info("  1. Your primary dataset (~4,000 cells)")
    logger.info("  2. MLL-Helmholtz dataset (~147,000 cells)")
    logger.info("")
    logger.info("Total: ~151,000 cells from 1,045+ patients")
    logger.info("")
    
    # Check if already processed
    if MLL_PROCESSED_DIR.exists() and (MLL_PROCESSED_DIR / "metadata.csv").exists():
        logger.info(f"✓ Processed dataset already exists: {MLL_PROCESSED_DIR}")
        logger.info("")
        
        if args.validate_only or not (args.preprocess_only):
            # Run validation
            validation_ok = run_validation()
            if validation_ok:
                print_training_instructions()
                return
        
        logger.info("To re-process, delete the processed directory:")
        logger.info(f"  rm -rf {MLL_PROCESSED_DIR}")
        logger.info("")
        return
    
    # Check if dataset downloaded
    if not args.skip_download_check:
        dataset_downloaded = check_dataset_downloaded()
        
        if not dataset_downloaded:
            logger.warning("MLL dataset not found!")
            logger.info("")
            logger.info("Please download it first:")
            logger.info(f"  python scripts/download_mll_dataset.py")
            logger.info("")
            logger.info("Or manually download from:")
            logger.info("  https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770")
            logger.info("")
            logger.info(f"Place downloaded archive in: {MLL_DOWNLOAD_DIR / 'download'}")
            logger.info("")
            return
        
        logger.info(f"✓ MLL dataset found: {MLL_DOWNLOAD_DIR}")
        logger.info("")
    
    # Run preprocessing
    if not args.validate_only:
        preprocessing_ok = run_preprocessing()
        if not preprocessing_ok:
            logger.error("Preprocessing failed!")
            return
    
    # Run validation
    if not args.preprocess_only:
        validation_ok = run_validation()
        if not validation_ok:
            logger.error("Validation failed!")
            return
    
    # Print training instructions
    if not args.preprocess_only and not args.validate_only:
        print_training_instructions()


if __name__ == "__main__":
    main()
