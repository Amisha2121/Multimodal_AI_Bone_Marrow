"""
Download MLL-Helmholtz-Fraunhofer Bone Marrow Dataset from Kaggle
==================================================================

This script downloads the expert-annotated bone marrow cell dataset
containing 171K+ cells from 945 patients using the Kaggle API.

Dataset Details:
- Name: Bone Marrow Cell Classification
- Kaggle: andrewmvd/bone-marrow-cell-classification
- Size: ~6.8 GB
- License: CC BY 4.0
- Original DOI: 10.7937/TCIA.AXH3-T579

Prerequisites:
1. Install Kaggle API:
   pip install kaggle

2. Set up Kaggle credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token" (downloads kaggle.json)
   - Place kaggle.json at: C:\\Users\\AMISHA\\.kaggle\\kaggle.json
   - Or set environment variables:
     KAGGLE_USERNAME=your_username
     KAGGLE_KEY=your_api_key

Usage:
    python scripts/download_mll_kaggle.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_kaggle_setup():
    """Check if Kaggle API is installed and configured"""
    print("=" * 70)
    print("KAGGLE SETUP CHECK")
    print("=" * 70)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✓ Kaggle API is installed")
    except ImportError:
        print("✗ Kaggle API not found")
        print("\nInstall with:")
        print("  pip install kaggle")
        return False
    
    # Check for credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    has_env_vars = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    has_json_file = kaggle_json.exists()
    
    if has_json_file:
        print(f"✓ Found kaggle.json at: {kaggle_json}")
    elif has_env_vars:
        print("✓ Found KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    else:
        print("✗ No Kaggle credentials found")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to:", kaggle_dir)
        print(f"4. Ensure directory exists: {kaggle_dir}")
        return False
    
    print()
    return True


def download_from_kaggle(output_dir: Path):
    """Download dataset from Kaggle using kaggle CLI"""
    print("=" * 70)
    print("DOWNLOADING DATASET FROM KAGGLE")
    print("=" * 70)
    
    dataset_slug = "andrewmvd/bone-marrow-cell-classification"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset: {dataset_slug}")
    print(f"Output directory: {output_dir}")
    print(f"Size: ~6.8 GB")
    print("\nThis will take 10-20 minutes depending on your internet speed...")
    print()
    
    # Download using kaggle CLI
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(output_dir),
        "--unzip"
    ]
    
    try:
        print("Running:", " ".join(cmd))
        print()
        subprocess.run(cmd, check=True)
        print("\n✓ Download completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ kaggle command not found")
        print("Try running: pip install kaggle")
        return False


def verify_download(output_dir: Path):
    """Verify the downloaded files"""
    print("\n" + "=" * 70)
    print("VERIFYING DOWNLOAD")
    print("=" * 70)
    
    # Expected directory structure from Kaggle
    expected_files = [
        "bone_marrow_cell_dataset"  # Main directory
    ]
    
    found_files = list(output_dir.glob("*"))
    print(f"\nFound {len(found_files)} items in {output_dir}")
    
    for item in found_files:
        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) if item.is_dir() else item.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"  {item.name}: {size_mb:.1f} MB")
    
    # Look for the main dataset directory
    dataset_dir = output_dir / "bone_marrow_cell_dataset"
    if not dataset_dir.exists():
        # Try alternative names
        alternatives = [d for d in output_dir.iterdir() if d.is_dir()]
        if alternatives:
            dataset_dir = alternatives[0]
            print(f"\n✓ Found dataset directory: {dataset_dir.name}")
        else:
            print("\n✗ Could not find dataset directory")
            return False
    else:
        print(f"\n✓ Found dataset directory: bone_marrow_cell_dataset")
    
    # Count images
    image_files = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.bmp"))
    print(f"✓ Found {len(image_files):,} image files")
    
    if len(image_files) < 100000:
        print(f"⚠ Warning: Expected ~171K images, found {len(image_files):,}")
    
    print("\n✓ Download verification passed!")
    return True


def main():
    """Main download workflow"""
    print("\n" + "=" * 70)
    print("MLL BONE MARROW DATASET DOWNLOADER (KAGGLE)")
    print("=" * 70)
    print("\nDataset: Bone Marrow Cell Classification")
    print("Source: Kaggle (andrewmvd/bone-marrow-cell-classification)")
    print("Size: ~6.8 GB (171K+ images)")
    print("License: CC BY 4.0")
    print()
    
    # Setup output directory
    output_dir = project_root / "data" / "external" / "mll_helmholtz" / "download"
    
    # Check if already downloaded
    dataset_dir = output_dir / "bone_marrow_cell_dataset"
    if dataset_dir.exists():
        image_count = len(list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.bmp")))
        if image_count > 100000:
            print(f"✓ Dataset already downloaded ({image_count:,} images)")
            print(f"  Location: {dataset_dir}")
            print("\nDelete the directory to re-download.")
            return 0
    
    # Step 1: Check Kaggle setup
    if not check_kaggle_setup():
        return 1
    
    # Step 2: Download from Kaggle
    if not download_from_kaggle(output_dir):
        return 1
    
    # Step 3: Verify download
    if not verify_download(output_dir):
        return 1
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\nDataset location: {output_dir}")
    print("\nNext steps:")
    print("1. Run preprocessing:")
    print("   python scripts/preprocess_mll_dataset.py")
    print("\n2. Or use the automated setup:")
    print("   python scripts/setup_multi_dataset_training.py")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
