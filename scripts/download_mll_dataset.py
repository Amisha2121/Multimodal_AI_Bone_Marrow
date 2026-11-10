"""
Download MLL-Helmholtz-Fraunhofer Bone Marrow Dataset from TCIA

This script downloads the public bone marrow cytomorphology dataset
from The Cancer Imaging Archive (TCIA).

Dataset: Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer
Size: 6.8 GB (171,375 images)
License: CC BY 4.0
URL: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770

Requirements:
    pip install tcia-utils requests tqdm

Usage:
    python scripts/download_mll_dataset.py
    
    # With custom output directory:
    python scripts/download_mll_dataset.py --output data/external/mll_dataset
"""

import argparse
import hashlib
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing required packages. Install with:")
    print("  pip install requests tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_mll")

# TCIA dataset information
DATASET_NAME = "Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer"
DATASET_DOI = "10.7937/TCIA.AXH3-T579"
TCIA_URL = "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770"

# Direct download URLs (if available - TCIA may require manual download)
# Note: TCIA typically requires IBM Aspera Connect or their download manager
DOWNLOAD_INSTRUCTIONS = """
IMPORTANT: TCIA Download Instructions
======================================

The MLL-Helmholtz bone marrow dataset must be downloaded manually from TCIA.

Option 1: IBM Aspera Connect (RECOMMENDED - Fast)
--------------------------------------------------
1. Install IBM Aspera Connect:
   https://www.ibm.com/aspera/connect/

2. Visit the dataset page:
   {url}

3. Click "Tissue Slide Images (JPG, 6.8GB)"

4. The download will start automatically via Aspera Connect

5. Save the downloaded file to:
   {output_dir}/download/

6. Re-run this script to extract and process the dataset


Option 2: Direct HTTP Download (Slower)
----------------------------------------
1. Visit the dataset page:
   {url}

2. Click "Tissue Slide Images (JPG, 6.8GB)"

3. Choose "Download via Browser" if prompted

4. Save the downloaded ZIP/archive to:
   {output_dir}/download/

5. Re-run this script to extract and process the dataset


Option 3: NBIA Data Retriever (Alternative)
-------------------------------------------
1. Install NBIA Data Retriever from TCIA

2. Search for: {dataset_name}

3. Download all images

4. Move to: {output_dir}/download/

5. Re-run this script


After downloading, you should have one of these files:
  - *.zip
  - *.tar.gz
  - *.tcia (manifest file - use NBIA Data Retriever)

Re-run this script and it will automatically detect and process the download.
"""


def check_existing_download(download_dir: Path) -> Optional[Path]:
    """Check if dataset archive already exists."""
    archive_extensions = ['.zip', '.tar.gz', '.tgz', '.tar', '.7z']
    
    for archive_ext in archive_extensions:
        matches = list(download_dir.glob(f"*{archive_ext}"))
        if matches:
            logger.info(f"Found existing archive: {matches[0]}")
            return matches[0]
    
    return None


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract archive to output directory."""
    logger.info(f"Extracting {archive_path.name}...")
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Show progress
            members = zip_ref.namelist()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_dir)
                    pbar.update(1)
    
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        import tarfile
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar_ref.extract(member, extract_dir)
                    pbar.update(1)
    
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    logger.info(f"Extraction complete → {extract_dir}")
    return extract_dir


def find_image_directory(extract_dir: Path) -> Path:
    """Find the directory containing the cell images."""
    # Common patterns for TCIA datasets
    possible_names = [
        "Bone-Marrow-Cytomorphology",
        "BoneMarrowCytomorphology",
        "images",
        "cells",
        "data",
    ]
    
    for name in possible_names:
        matches = list(extract_dir.rglob(name))
        if matches:
            logger.info(f"Found image directory: {matches[0]}")
            return matches[0]
    
    # If not found, look for directory with most subdirectories
    subdirs = [d for d in extract_dir.rglob("*") if d.is_dir()]
    if subdirs:
        # Count PNG/JPG files in each directory
        best_dir = None
        max_images = 0
        for subdir in subdirs:
            num_images = len(list(subdir.glob("*.jpg"))) + len(list(subdir.glob("*.png")))
            if num_images > max_images:
                max_images = num_images
                best_dir = subdir
        
        if best_dir and max_images > 100:
            logger.info(f"Found image directory with {max_images} images: {best_dir}")
            return best_dir
    
    raise ValueError(
        "Could not locate image directory in extracted archive. "
        "Please manually specify the path."
    )


def verify_dataset_structure(image_dir: Path) -> dict:
    """Verify dataset structure and count images per class."""
    logger.info("Verifying dataset structure...")
    
    # Expected MLL classes
    expected_classes = [
        "ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "HAC", "KSC",
        "LYI", "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH",
        "PEB", "PLM", "PMO"
    ]
    
    class_counts = {}
    
    # Count images in each class directory
    for class_name in expected_classes:
        class_dir = image_dir / class_name
        if class_dir.exists():
            # Count images in subdirectories (MLL has numbered subdirs)
            num_images = len(list(class_dir.rglob("*.jpg"))) + len(list(class_dir.rglob("*.png")))
            class_counts[class_name] = num_images
            if num_images > 0:
                logger.info(f"  {class_name}: {num_images} images")
        else:
            logger.warning(f"  {class_name}: directory not found")
            class_counts[class_name] = 0
    
    total_images = sum(class_counts.values())
    logger.info(f"\nTotal images found: {total_images:,}")
    
    if total_images < 100000:
        logger.warning(
            f"Expected ~171,000 images but found {total_images:,}. "
            "Dataset may be incomplete."
        )
    
    return class_counts


def create_symlink_structure(image_dir: Path, output_dir: Path) -> Path:
    """Create organized symlink structure for easier access."""
    logger.info("Creating organized directory structure...")
    
    organized_dir = output_dir / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks to avoid duplicating 6.8GB
    for class_dir in image_dir.iterdir():
        if class_dir.is_dir():
            target = organized_dir / class_dir.name
            if not target.exists():
                # On Windows, may need to copy instead of symlink
                try:
                    target.symlink_to(class_dir, target_is_directory=True)
                    logger.info(f"  Linked {class_dir.name}")
                except OSError:
                    # Windows may require admin privileges for symlinks
                    logger.warning(f"  Could not create symlink for {class_dir.name}, skipping")
    
    return organized_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract MLL-Helmholtz bone marrow dataset from TCIA"
    )
    parser.add_argument(
        "--output", type=str, 
        default="data/external/mll_helmholtz",
        help="Output directory for downloaded dataset"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip extraction if already extracted"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "download"
    download_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MLL-Helmholtz-Fraunhofer Bone Marrow Dataset Downloader")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"DOI: {DATASET_DOI}")
    logger.info("")
    
    # Check if dataset already downloaded
    archive_path = check_existing_download(download_dir)
    
    if not archive_path:
        logger.warning("Dataset archive not found in download directory.")
        logger.info("")
        print(DOWNLOAD_INSTRUCTIONS.format(
            url=TCIA_URL,
            output_dir=output_dir.absolute(),
            dataset_name=DATASET_NAME
        ))
        logger.info("")
        logger.info(f"After downloading, place the archive in: {download_dir.absolute()}")
        logger.info("Then re-run this script.")
        return
    
    # Extract archive
    if not args.skip_extract:
        extract_dir = extract_archive(archive_path, output_dir)
    else:
        extract_dir = output_dir / "extracted"
        logger.info(f"Using existing extraction: {extract_dir}")
    
    # Find image directory
    try:
        image_dir = find_image_directory(extract_dir)
    except ValueError as e:
        logger.error(str(e))
        logger.info("Please inspect the extracted files manually:")
        logger.info(f"  {extract_dir.absolute()}")
        return
    
    # Verify dataset
    class_counts = verify_dataset_structure(image_dir)
    
    # Create organized structure
    organized_dir = create_symlink_structure(image_dir, output_dir)
    
    # Save dataset info
    info_file = output_dir / "dataset_info.json"
    import json
    with open(info_file, 'w') as f:
        json.dump({
            "dataset_name": DATASET_NAME,
            "doi": DATASET_DOI,
            "source_url": TCIA_URL,
            "download_date": str(Path(archive_path).stat().st_mtime),
            "total_images": sum(class_counts.values()),
            "class_counts": class_counts,
            "image_directory": str(image_dir.absolute()),
            "organized_directory": str(organized_dir.absolute()),
        }, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Dataset extracted to: {image_dir.absolute()}")
    logger.info(f"Organized view at:    {organized_dir.absolute()}")
    logger.info(f"Dataset info saved:   {info_file.absolute()}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run preprocessing: python scripts/preprocess_mll_dataset.py")
    logger.info("  2. Verify dataset:    python scripts/validate_mll_dataset.py")
    logger.info("  3. Train model:       python vision/train_multi_dataset.py ...")
    logger.info("")
    logger.info("Citation required:")
    logger.info("  Matek et al. (2021) Blood, 138(20):1917-1927")
    logger.info("  DOI: 10.1182/blood.2020010568")


if __name__ == "__main__":
    main()
