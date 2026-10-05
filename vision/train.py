"""
vision/train.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 — Cell Morphology Baseline Training Pipeline.

Architecture: ViT-Small/16 (frozen) + LoRA adapters + 8-class linear head.
Taxonomy:     8 morphology categories from BoneMarrowWSI-PediatricLeukemia
              consensus annotations (see vision/taxonomy.py for details).

Key guarantees:
  - Patient-level split isolation enforced via data/processed/splits.json.
    Validation and test patients are NEVER touched during training.
  - Cell labels inherited from patient split — no independent cell-level splits.
  - Class imbalance addressed via WeightedRandomSampler (default) or
    class-weighted CrossEntropyLoss (if use_weighted_sampler=false in config).
  - Full taxonomy printed and saved before any gradient steps.
  - Mixed precision (bf16/fp16) and early stopping on validation macro F1.

Usage:
  python -m vision.train [--config configs/colab.yaml]

Prerequisites:
  Run vision/run_extraction.py to pre-extract all cell crops before training.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
logger = logging.getLogger("train")


# ── Taxonomy print ────────────────────────────────────────────────────────────

def print_taxonomy(samples_by_split: Dict[str, List[dict]], save_dir: Path) -> None:
    """
    Prints the morphology taxonomy with cell counts per split and saves to JSON.
    Called BEFORE any model construction or gradient steps.
    """
    from collections import Counter

    print("\n" + "=" * 70)
    print("CELL MORPHOLOGY TAXONOMY — BoneMarrowWSI-PediatricLeukemia baseline")
    print("=" * 70)
    print(f"{'Idx':<4} {'Category':<35} {'Train':>7} {'Val':>7} {'Test':>7}  Status")
    print("-" * 70)

    taxonomy_record = {}
    for idx, cat in enumerate(MORPH_CLASSES):
        counts = {}
        for split_name, samples in samples_by_split.items():
            counts[split_name] = sum(1 for s in samples if s["class_code"] == cat)
        is_rare = cat in RARE_CLASSES
        flag = " [RARE]" if is_rare else ""
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
    print(f"Total included classes: {len(MORPH_CLASSES)}")
    total_train = sum(v["train_cells"] for v in taxonomy_record.values())
    total_val   = sum(v["val_cells"]   for v in taxonomy_record.values())
    total_test  = sum(v["test_cells"]  for v in taxonomy_record.values())
    print(f"Total cells — Train: {total_train}, Val: {total_val}, Test: {total_test}")
    print("=" * 70 + "\n")

    # Save taxonomy to disk
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "taxonomy_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "taxonomy": taxonomy_record,
                "excluded_categories": sorted(EXCLUDED_CATEGORIES),
                "num_classes": len(MORPH_CLASSES),
                "rare_classes": RARE_CLASSES,
            },
            f, indent=2,
        )
    logger.info(f"Taxonomy saved → {save_dir / 'taxonomy_used.json'}")


# ── Train / validate ──────────────────────────────────────────────────────────

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

def run_training(config_override_path: Optional[str] = None) -> None:
    """End-to-end training execution."""
    cfg = load_config(config_override_path)
    set_global_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data paths ────────────────────────────────────────────────────────────
    cell_meta_path = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
    splits_path    = REPO_ROOT / "data" / "processed" / "splits.json"

    if not cell_meta_path.exists():
        raise FileNotFoundError(
            f"cell_metadata.csv not found at '{cell_meta_path}'. "
            "Run the consensus DICOM parsing script to generate it."
        )
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.json not found at '{splits_path}'. "
            "Run scripts/generate_splits.py to generate patient-level splits."
        )

    # ── Load samples (patient-level split enforcement) ────────────────────────
    logger.info("Loading cell samples from metadata...")
    train_samples = load_cell_samples(cell_meta_path, splits_path, split="train")
    val_samples   = load_cell_samples(cell_meta_path, splits_path, split="val")
    # Test samples loaded for reporting only — never touched during training
    test_samples  = load_cell_samples(cell_meta_path, splits_path, split="test")

    if not train_samples:
        raise RuntimeError(
            "No training samples found with existing crop files. "
            "Run vision/run_extraction.py to pre-extract all cell crops."
        )
    if not val_samples:
        raise RuntimeError(
            "No validation samples found with existing crop files. "
            "Run vision/run_extraction.py to pre-extract all cell crops."
        )

    # ── Print and save taxonomy BEFORE any gradient steps ────────────────────
    save_dir = REPO_ROOT / "models" / "checkpoints"
    print_taxonomy(
        {"train": train_samples, "val": val_samples, "test": test_samples},
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
        num_classes=len(MORPH_CLASSES),   # always 8 — from taxonomy
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
        weight=class_weights_tensor,  # None when using WeightedRandomSampler
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
    best_ckpt_path   = save_dir / "best_vit_lora_morphology.pt"

    logger.info(f"Starting training for up to {cfg.training.epochs} epochs ...")
    logger.info(f"  Train: {len(train_samples)} cells | Val: {len(val_samples)} cells")
    logger.info(f"  Imbalance strategy: {'WeightedRandomSampler' if use_weighted_sampler else 'Class-weighted loss'}")
    logger.info(f"  Precision: {cfg.training.precision} | Early stop patience: {patience}")

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
                    "epoch":             epoch,
                    "best_macro_f1":     best_macro_f1,
                    "val_weighted_f1":   val_metrics["val_weighted_f1"],
                    "val_rare_recall":   val_metrics["val_rare_recall"],
                    "train_cells":       len(train_samples),
                    "val_cells":         len(val_samples),
                    "imbalance_strategy": (
                        "WeightedRandomSampler" if use_weighted_sampler else "class_weighted_loss"
                    ),
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    logger.info(f"\nTraining complete. Best val macro F1 = {best_macro_f1:.4f}")
    logger.info(f"Best checkpoint → {best_ckpt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-Small/16 + LoRA morphology classifier")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config override (e.g., configs/colab.yaml)."
    )
    args = parser.parse_args()
    run_training(config_override_path=args.config)
