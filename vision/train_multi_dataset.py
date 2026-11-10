"""
Multi-Dataset Training Script for Bone Marrow Cell Classifier

This script extends the base training to support combining multiple datasets:
- Primary dataset: BoneMarrowWSI-PediatricLeukemia (current)
- Additional datasets: Can be added as external sources

Usage:
    # Single dataset (current behavior):
    python vision/train_multi_dataset.py --config configs/default.yaml

    # Multiple datasets:
    python vision/train_multi_dataset.py --config configs/default.yaml \
        --additional-metadata data/external/dataset2_metadata.csv \
        --additional-splits data/external/dataset2_splits.json \
        --additional-crops data/external/dataset2_crops

    # Multiple external datasets:
    python vision/train_multi_dataset.py --config configs/default.yaml \
        --additional-metadata dataset2.csv dataset3.csv \
        --additional-splits dataset2_splits.json dataset3_splits.json \
        --additional-crops dataset2_crops dataset3_crops

Requirements for additional datasets:
1. metadata CSV with columns: cell_id, patient_id, mapped_cell_category, extraction_status, crop_path
2. splits JSON with keys: train, val, test (arrays of patient IDs)
3. Cell crop directory with PNG files matching crop_path column
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config_loader import load_config, set_global_seed
from vision.taxonomy import MORPH_CLASSES, MORPH_TO_IDX, EXCLUDED_CATEGORIES, RARE_CLASSES
from vision.dataset import (
    BoneMarrowCellDataset,
    load_cell_samples,
    compute_weighted_sampler,
    compute_class_weights,
    get_default_transforms,
)
from vision.evaluate import evaluate_predictions, print_eval_report
from vision.model import BoneMarrowViT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_multi")


# ── Multi-dataset loading ─────────────────────────────────────────────────────

def load_multi_dataset_samples(
    primary_metadata: Path,
    primary_splits: Path,
    additional_metadata: List[Path],
    additional_splits: List[Path],
    additional_crops: List[Path],
    split: str,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Load and merge samples from multiple datasets.
    
    Args:
        primary_metadata: Path to primary dataset metadata CSV
        primary_splits: Path to primary dataset splits JSON
        additional_metadata: List of paths to additional metadata CSVs
        additional_splits: List of paths to additional splits JSONs
        additional_crops: List of paths to additional crop directories
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (merged_samples, dataset_counts)
    """
    # Load primary dataset
    logger.info(f"\n{'='*70}")
    logger.info(f"Loading PRIMARY dataset for split '{split}'")
    logger.info(f"{'='*70}")
    primary_samples = load_cell_samples(primary_metadata, primary_splits, split)
    logger.info(f"✓ Primary dataset: {len(primary_samples)} samples")
    
    dataset_counts = {"primary": len(primary_samples)}
    all_samples = primary_samples.copy()
    
    # Add dataset identifier to primary samples
    for sample in all_samples:
        sample['dataset_source'] = 'primary'
    
    # Load additional datasets
    if additional_metadata:
        logger.info(f"\n{'='*70}")
        logger.info(f"Loading {len(additional_metadata)} ADDITIONAL dataset(s)")
        logger.info(f"{'='*70}")
        
        for idx, (meta_path, splits_path, crops_path) in enumerate(
            zip(additional_metadata, additional_splits, additional_crops), start=1
        ):
            dataset_name = f"dataset_{idx}"
            logger.info(f"\n[{dataset_name}] Loading from:")
            logger.info(f"  Metadata: {meta_path}")
            logger.info(f"  Splits:   {splits_path}")
            logger.info(f"  Crops:    {crops_path}")
            
            try:
                samples = load_cell_samples(
                    meta_path, 
                    splits_path, 
                    split, 
                    crop_base_dir=crops_path
                )
                
                # Add dataset identifier
                for sample in samples:
                    sample['dataset_source'] = dataset_name
                
                all_samples.extend(samples)
                dataset_counts[dataset_name] = len(samples)
                logger.info(f"✓ [{dataset_name}] Added {len(samples)} samples")
                
            except Exception as e:
                logger.error(f"✗ [{dataset_name}] Failed to load: {e}")
                logger.warning(f"  Continuing without {dataset_name}...")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"MERGED DATASET SUMMARY ({split})")
    logger.info(f"{'='*70}")
    for dataset_name, count in dataset_counts.items():
        pct = (count / len(all_samples) * 100) if all_samples else 0
        logger.info(f"  {dataset_name:.<20} {count:>7} samples ({pct:>5.1f}%)")
    logger.info(f"  {'TOTAL':.<20} {len(all_samples):>7} samples")
    logger.info(f"{'='*70}\n")
    
    return all_samples, dataset_counts


def print_multi_dataset_taxonomy(
    samples_by_split: Dict[str, List[dict]],
    dataset_counts_by_split: Dict[str, Dict[str, int]],
    save_dir: Path
) -> None:
    """
    Prints taxonomy with per-dataset breakdown and saves to JSON.
    """
    from collections import Counter
    
    print("\n" + "=" * 90)
    print("MULTI-DATASET CELL MORPHOLOGY TAXONOMY")
    print("=" * 90)
    
    # Dataset sources summary
    all_datasets = set()
    for counts in dataset_counts_by_split.values():
        all_datasets.update(counts.keys())
    
    print("Dataset Sources:")
    for split_name, counts in dataset_counts_by_split.items():
        print(f"  {split_name.capitalize():>5}: ", end="")
        print(" + ".join(f"{ds}({cnt})" for ds, cnt in counts.items()))
    print()
    
    # Per-class breakdown
    print(f"{'Idx':<4} {'Category':<35} {'Train':>7} {'Val':>7} {'Test':>7}  Status")
    print("-" * 90)
    
    taxonomy_record = {}
    for idx, cat in enumerate(MORPH_CLASSES):
        counts = {}
        for split_name, samples in samples_by_split.items():
            counts[split_name] = sum(1 for s in samples if s["class_code"] == cat)
        
        is_rare = cat in RARE_CLASSES
        print(
            f"{idx:<4} {cat:<35} "
            f"{counts.get('train', 0):>7} "
            f"{counts.get('val', 0):>7} "
            f"{counts.get('test', 0):>7}  "
            f"{'rare' if is_rare else 'normal'}"
        )
        
        taxonomy_record[cat] = {
            "label_index": idx,
            "train_cells": counts.get("train", 0),
            "val_cells":   counts.get("val", 0),
            "test_cells":  counts.get("test", 0),
            "is_rare":     is_rare,
        }
    
    print()
    print("Excluded categories (NOT in classifier):")
    for cat in sorted(EXCLUDED_CATEGORIES):
        print(f"  - {cat}")
    print()
    
    total_train = sum(v["train_cells"] for v in taxonomy_record.values())
    total_val   = sum(v["val_cells"]   for v in taxonomy_record.values())
    total_test  = sum(v["test_cells"]  for v in taxonomy_record.values())
    print(f"Total cells — Train: {total_train}, Val: {total_val}, Test: {total_test}")
    print("=" * 90 + "\n")
    
    # Save taxonomy with dataset info
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "taxonomy_multi_dataset.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "taxonomy": taxonomy_record,
                "dataset_counts": dataset_counts_by_split,
                "excluded_categories": sorted(EXCLUDED_CATEGORIES),
                "num_classes": len(MORPH_CLASSES),
                "rare_classes": RARE_CLASSES,
                "num_datasets": len(all_datasets),
                "dataset_sources": list(all_datasets),
            },
            f, indent=2,
        )
    logger.info(f"Multi-dataset taxonomy saved → {save_dir / 'taxonomy_multi_dataset.json'}")


# ── Training functions (imported from train.py) ───────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    precision: str = "fp32",
) -> float:
    """Runs one training epoch; returns mean loss."""
    model.train()
    total_loss = 0.0
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype   = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp     = (device.type == "cuda") and precision in ("bf16", "fp16")

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                logits = model(images)
                loss   = criterion(logits, labels)
            if scaler is not None and precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates validation loss, accuracy, macro F1, and rare-class recall."""
    model.eval()
    total_loss = 0.0
    all_preds:   List[int] = []
    all_targets: List[int] = []

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    metrics = evaluate_predictions(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=MORPH_CLASSES,
    )

    return {
        "val_loss":        total_loss / max(1, len(loader)),
        "val_accuracy":    metrics["accuracy"],
        "val_macro_f1":    metrics["macro_f1"],
        "val_weighted_f1": metrics["weighted_f1"],
        "val_rare_recall": metrics["rare_classes_macro_recall"],
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_multi_dataset_training(
    config_override_path: Optional[str] = None,
    additional_metadata: Optional[List[str]] = None,
    additional_splits: Optional[List[str]] = None,
    additional_crops: Optional[List[str]] = None,
    skip_calibration: bool = False,
) -> None:
    """End-to-end multi-dataset training execution."""
    cfg = load_config(config_override_path)
    set_global_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Validate additional dataset arguments ─────────────────────────────────
    additional_metadata = additional_metadata or []
    additional_splits = additional_splits or []
    additional_crops = additional_crops or []
    
    if len(additional_metadata) != len(additional_splits):
        raise ValueError(
            f"Number of additional metadata files ({len(additional_metadata)}) must match "
            f"number of additional splits files ({len(additional_splits)})"
        )
    
    if len(additional_metadata) != len(additional_crops):
        raise ValueError(
            f"Number of additional metadata files ({len(additional_metadata)}) must match "
            f"number of additional crop directories ({len(additional_crops)})"
        )
    
    # Convert to Path objects
    additional_metadata = [Path(p) for p in additional_metadata]
    additional_splits = [Path(p) for p in additional_splits]
    additional_crops = [Path(p) for p in additional_crops]
    
    # Validate all files exist
    for meta_path in additional_metadata:
        if not meta_path.exists():
            raise FileNotFoundError(f"Additional metadata not found: {meta_path}")
    for splits_path in additional_splits:
        if not splits_path.exists():
            raise FileNotFoundError(f"Additional splits not found: {splits_path}")
    for crops_path in additional_crops:
        if not crops_path.exists():
            raise FileNotFoundError(f"Additional crops directory not found: {crops_path}")

    # ── Primary dataset paths ─────────────────────────────────────────────────
    primary_metadata = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
    primary_splits   = REPO_ROOT / "data" / "processed" / "splits.json"

    if not primary_metadata.exists():
        raise FileNotFoundError(
            f"Primary metadata not found at '{primary_metadata}'. "
            "Run the consensus DICOM parsing script to generate it."
        )
    if not primary_splits.exists():
        raise FileNotFoundError(
            f"Primary splits not found at '{primary_splits}'. "
            "Run scripts/generate_splits.py to generate patient-level splits."
        )

    # ── Load multi-dataset samples ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("LOADING MULTI-DATASET SAMPLES")
    logger.info("=" * 70)
    
    train_samples, train_counts = load_multi_dataset_samples(
        primary_metadata, primary_splits,
        additional_metadata, additional_splits, additional_crops,
        split="train"
    )
    
    val_samples, val_counts = load_multi_dataset_samples(
        primary_metadata, primary_splits,
        additional_metadata, additional_splits, additional_crops,
        split="val"
    )
    
    test_samples, test_counts = load_multi_dataset_samples(
        primary_metadata, primary_splits,
        additional_metadata, additional_splits, additional_crops,
        split="test"
    )

    if not train_samples:
        raise RuntimeError("No training samples found across all datasets!")
    if not val_samples:
        raise RuntimeError("No validation samples found across all datasets!")

    # ── Print and save multi-dataset taxonomy ─────────────────────────────────
    save_dir = REPO_ROOT / "models" / "checkpoints"
    print_multi_dataset_taxonomy(
        {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        },
        {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts
        },
        save_dir=save_dir,
    )

    # ── Build datasets and loaders ────────────────────────────────────────────
    img_size = cfg.dataset.image_size

    train_ds = BoneMarrowCellDataset(
        samples=train_samples,
        transform=get_default_transforms(image_size=img_size, is_train=True),
        image_size=img_size,
    )
    val_ds = BoneMarrowCellDataset(
        samples=val_samples,
        transform=get_default_transforms(image_size=img_size, is_train=False),
        image_size=img_size,
    )

    use_weighted_sampler = cfg.training.use_weighted_sampler
    if use_weighted_sampler:
        sampler = compute_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )
        class_weights_tensor = None
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )
        class_weights_tensor = compute_class_weights(train_ds).to(device)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BoneMarrowViT(
        backbone_name=cfg.vision.backbone,
        num_classes=len(MORPH_CLASSES),
        pretrained=cfg.vision.pretrained,
        use_lora=cfg.vision.lora.enabled,
        lora_r=cfg.vision.lora.r,
        lora_alpha=cfg.vision.lora.alpha,
        lora_dropout=cfg.vision.lora.dropout,
        target_modules=list(cfg.vision.lora.target_modules),
    ).to(device)

    param_stats = model.get_trainable_param_stats()
    logger.info(
        f"Model: {cfg.vision.backbone} | "
        f"{param_stats['trainable_params']:,} trainable / "
        f"{param_stats['total_params']:,} total "
        f"({param_stats['trainable_percentage']}%)"
    )

    # ── Optimizer and loss ────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.training.label_smoothing,
        weight=class_weights_tensor,
    )

    scaler = (
        torch.amp.GradScaler()
        if device.type == "cuda" and cfg.training.precision == "fp16"
        else None
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_macro_f1    = -1.0
    patience_counter = 0
    patience         = cfg.training.early_stopping_patience
    best_ckpt_path   = save_dir / "best_vit_lora_morphology_multi_dataset.pt"

    num_datasets = 1 + len(additional_metadata)
    logger.info(f"\nStarting multi-dataset training ({num_datasets} dataset(s))...")
    logger.info(f"  Train: {len(train_samples)} cells from {num_datasets} dataset(s)")
    logger.info(f"  Val:   {len(val_samples)} cells from {num_datasets} dataset(s)")
    logger.info(f"  Epochs: up to {cfg.training.epochs} | Patience: {patience}")
    logger.info(f"  Precision: {cfg.training.precision}")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            precision=cfg.training.precision,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{cfg.training.epochs} [{elapsed:.0f}s]  "
            f"TrainLoss={train_loss:.4f}  "
            f"ValLoss={val_metrics['val_loss']:.4f}  "
            f"MacroF1={val_metrics['val_macro_f1']:.4f}  "
            f"WtdF1={val_metrics['val_weighted_f1']:.4f}  "
            f"RareRec={val_metrics['val_rare_recall']:.4f}"
        )

        val_f1 = val_metrics["val_macro_f1"]
        if val_f1 > best_macro_f1:
            best_macro_f1    = val_f1
            patience_counter = 0
            model.save_checkpoint(
                best_ckpt_path,
                extra_metadata={
                    "epoch":                epoch,
                    "best_macro_f1":        best_macro_f1,
                    "val_weighted_f1":      val_metrics["val_weighted_f1"],
                    "val_rare_recall":      val_metrics["val_rare_recall"],
                    "num_datasets":         num_datasets,
                    "train_cells":          len(train_samples),
                    "val_cells":            len(val_samples),
                    "dataset_counts_train": train_counts,
                    "dataset_counts_val":   val_counts,
                    "imbalance_strategy":   (
                        "WeightedRandomSampler" if use_weighted_sampler else "class_weighted_loss"
                    ),
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    logger.info(f"\nMulti-dataset training complete!")
    logger.info(f"  Best val macro F1: {best_macro_f1:.4f}")
    logger.info(f"  Total datasets: {num_datasets}")
    logger.info(f"  Best checkpoint → {best_ckpt_path}")
    
    # ── Post-training calibration ─────────────────────────────────────────────
    if not skip_calibration and cfg.training.get("run_calibration", True):
        logger.info("\n" + "=" * 70)
        logger.info("POST-TRAINING CALIBRATION")
        logger.info("=" * 70)
        
        try:
            from vision.calibration import calibrate_model
            
            best_model = BoneMarrowViT.load_checkpoint(best_ckpt_path, device=str(device))
            best_model.eval()
            
            calibration_dir = save_dir / "calibration_multi_dataset"
            temp_layer = calibrate_model(
                model=best_model,
                val_loader=val_loader,
                device=device,
                save_dir=calibration_dir,
                n_bins=15,
            )
            
            logger.info(f"\n✓ Calibration complete → {calibration_dir}")
            logger.info(f"  Temperature: {temp_layer.get_temperature():.4f}")
            
        except ImportError as e:
            logger.warning(f"Calibration skipped: {e}")
        except Exception as e:
            logger.error(f"Calibration failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ViT morphology classifier on multiple bone marrow datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config override (e.g., configs/colab.yaml)"
    )
    parser.add_argument(
        "--additional-metadata", type=str, nargs='+', default=[],
        help="Path(s) to additional dataset metadata CSV files"
    )
    parser.add_argument(
        "--additional-splits", type=str, nargs='+', default=[],
        help="Path(s) to additional dataset splits JSON files"
    )
    parser.add_argument(
        "--additional-crops", type=str, nargs='+', default=[],
        help="Path(s) to additional dataset cell crop directories"
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip post-training calibration step"
    )
    
    args = parser.parse_args()
    
    run_multi_dataset_training(
        config_override_path=args.config,
        additional_metadata=args.additional_metadata,
        additional_splits=args.additional_splits,
        additional_crops=args.additional_crops,
        skip_calibration=args.skip_calibration,
    )
