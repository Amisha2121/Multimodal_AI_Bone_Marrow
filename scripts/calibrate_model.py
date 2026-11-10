"""
scripts/calibrate_model.py
─────────────────────────────────────────────────────────────────────────────
Standalone script to perform temperature scaling calibration on trained model.

Usage:
    python scripts/calibrate_model.py --checkpoint models/checkpoints/best_vit_lora_morphology.pt

Output:
    - Temperature-scaled layer weights
    - ECE metrics before/after calibration
    - Reliability diagrams (PNG)
    - JSON calibration report
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from configs.config_loader import load_config
from vision.model import BoneMarrowViT
from vision.dataset import BoneMarrowCellDataset, load_cell_samples, get_default_transforms
from vision.calibration import calibrate_model, print_calibration_summary


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ViT model with temperature scaling"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_vit_lora_morphology.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/calibration",
        help="Directory to save calibration results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (optional)"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
        help="Number of confidence bins for ECE (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = REPO_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = BoneMarrowViT.load_checkpoint(checkpoint_path, device=str(device))
    model.eval()
    print("Model loaded successfully")
    
    # Load validation data
    cell_meta_path = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
    splits_path = REPO_ROOT / "data" / "processed" / "splits.json"
    
    if not cell_meta_path.exists():
        raise FileNotFoundError(f"cell_metadata.csv not found: {cell_meta_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")
    
    print("\nLoading validation samples...")
    val_samples = load_cell_samples(cell_meta_path, splits_path, split="val")
    print(f"Found {len(val_samples)} validation samples")
    
    if len(val_samples) == 0:
        raise RuntimeError("No validation samples found. Cannot calibrate.")
    
    # Create validation dataset
    img_size = cfg.dataset.image_size
    val_dataset = BoneMarrowCellDataset(
        samples=val_samples,
        transform=get_default_transforms(image_size=img_size, is_train=False),
        image_size=img_size,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    
    # Run calibration
    output_dir = REPO_ROOT / args.output_dir
    print(f"\nStarting calibration (output dir: {output_dir})")
    
    temp_layer = calibrate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        save_dir=output_dir,
        n_bins=args.n_bins,
    )
    
    # Print summary
    print_calibration_summary(output_dir / "calibration_report.json")
    
    print("\n✓ Calibration complete!")
    print(f"  Temperature layer: {output_dir / 'temperature_layer.pt'}")
    print(f"  Report:            {output_dir / 'calibration_report.json'}")
    print(f"  Diagrams:          {output_dir / 'reliability_*.png'}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review reliability diagrams to verify calibration improvement")
    print("2. Use CalibratedBoneMarrowViT for inference:")
    print()
    print("   from vision.calibration import CalibratedBoneMarrowViT")
    print("   model = CalibratedBoneMarrowViT.from_checkpoint(")
    print(f"       model_path='{args.checkpoint}',")
    print(f"       temp_path='{output_dir / 'temperature_layer.pt'}',")
    print("       device='cuda'")
    print("   )")
    print("   probs = model.predict_proba(images)  # Calibrated probabilities")
    print()
    print("3. Update pipeline to use calibrated model for cell classification")
    print("=" * 70)


if __name__ == "__main__":
    main()
