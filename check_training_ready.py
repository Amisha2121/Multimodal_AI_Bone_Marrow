"""
check_training_ready.py
──────────────────────────────────────────────────────────────────────────────
Pre-flight check: Verify system is ready for detector training.

Run: python check_training_ready.py
"""

import sys
from pathlib import Path

def check_imports():
    """Check required packages are installed."""
    print("="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80 + "\n")
    
    checks = []
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA: Available (version {torch.version.cuda})")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA: Not available (will use CPU - training will be slow)")
        checks.append(True)
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
        print("  Install: pip install torch torchvision")
        checks.append(False)
    
    # Ultralytics
    try:
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Ultralytics: NOT INSTALLED")
        print("  Install: pip install ultralytics")
        checks.append(False)
    
    # Pandas
    try:
        import pandas
        print(f"✓ Pandas: {pandas.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Pandas: NOT INSTALLED")
        print("  Install: pip install pandas")
        checks.append(False)
    
    # NumPy
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ NumPy: NOT INSTALLED")
        print("  Install: pip install numpy")
        checks.append(False)
    
    # Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Matplotlib: NOT INSTALLED")
        print("  Install: pip install matplotlib")
        checks.append(False)
    
    # PIL
    try:
        import PIL
        print(f"✓ Pillow: {PIL.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Pillow: NOT INSTALLED")
        print("  Install: pip install pillow")
        checks.append(False)
    
    print()
    return all(checks)


def check_data():
    """Check ground-truth data is available."""
    print("="*80)
    print("CHECKING DATA")
    print("="*80 + "\n")
    
    checks = []
    
    # Cell metadata
    metadata_path = Path("data/processed/cell_metadata.csv")
    if metadata_path.exists():
        print(f"✓ Cell metadata: {metadata_path}")
        try:
            import pandas as pd
            df = pd.read_csv(metadata_path)
            total = len(df)
            valid = (df['extraction_status'] == 'valid').sum()
            print(f"  Total annotations: {total:,}")
            print(f"  Valid annotations: {valid:,}")
            if valid > 0:
                checks.append(True)
            else:
                print("  ✗ No valid annotations found")
                checks.append(False)
        except Exception as e:
            print(f"  ✗ Error reading metadata: {e}")
            checks.append(False)
    else:
        print(f"✗ Cell metadata NOT FOUND: {metadata_path}")
        checks.append(False)
    
    # Patient manifest
    manifest_path = Path("data/patient_manifest.csv")
    if manifest_path.exists():
        print(f"✓ Patient manifest: {manifest_path}")
        try:
            import pandas as pd
            df = pd.read_csv(manifest_path)
            splits = df['split'].value_counts()
            print(f"  Train patients: {splits.get('train', 0)}")
            print(f"  Val patients: {splits.get('val', 0)}")
            print(f"  Test patients: {splits.get('test', 0)}")
            checks.append(True)
        except Exception as e:
            print(f"  ✗ Error reading manifest: {e}")
            checks.append(False)
    else:
        print(f"✗ Patient manifest NOT FOUND: {manifest_path}")
        checks.append(False)
    
    print()
    return all(checks)


def check_detector_files():
    """Check detector training scripts exist."""
    print("="*80)
    print("CHECKING DETECTOR FILES")
    print("="*80 + "\n")
    
    detector_dir = Path("detector")
    required_files = [
        "prepare_dataset.py",
        "train_detector.py",
        "evaluate_detector.py",
        "visualize_predictions.py",
        "infer_detector.py",
        "run_full_pipeline.py",
    ]
    
    checks = []
    for file in required_files:
        file_path = detector_dir / file
        if file_path.exists():
            print(f"✓ {file}")
            checks.append(True)
        else:
            print(f"✗ {file} NOT FOUND")
            checks.append(False)
    
    print()
    return all(checks)


def main():
    """Run all checks."""
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECK: DETECTOR TRAINING READINESS")
    print("="*80 + "\n")
    
    results = []
    
    # Check dependencies
    results.append(("Dependencies", check_imports()))
    
    # Check data
    results.append(("Data Files", check_data()))
    
    # Check detector files
    results.append(("Detector Scripts", check_detector_files()))
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("="*80)
        print("✓ SYSTEM READY FOR TRAINING")
        print("="*80 + "\n")
        
        # Recommend training command
        try:
            import torch
            if torch.cuda.is_available():
                print("RECOMMENDED COMMAND (GPU):")
                print("  python detector/run_full_pipeline.py --model-size s --epochs 100")
                print()
                print("EXPECTED TIME: 1-3 hours")
            else:
                print("RECOMMENDED COMMAND (CPU):")
                print("  python detector/run_full_pipeline.py --model-size n --epochs 50")
                print()
                print("EXPECTED TIME: 4-8 hours")
                print()
                print("⚠ WARNING: Training on CPU is very slow.")
                print("  Consider using Google Colab or a machine with GPU.")
        except:
            print("RECOMMENDED COMMAND:")
            print("  python detector/run_full_pipeline.py --model-size s --epochs 100")
        
        print()
        print("For more options, see: detector/QUICKSTART.md")
        return 0
    else:
        print("="*80)
        print("✗ SYSTEM NOT READY")
        print("="*80 + "\n")
        print("Fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
