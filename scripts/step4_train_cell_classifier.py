"""
scripts/step4_train_cell_classifier.py
─────────────────────────────────────────────────────────────────────────────
STEP 4: Train the ViT-Small/16 + LoRA cell morphology classifier.

PREREQUISITES:
  - step3_verify_crops.py must pass (20,852 crops verified on disk).
  - timm, peft, omegaconf must be installed.
  - GPU strongly recommended (CUDA or MPS).

Architecture:
  - Backbone: vit_small_patch16_224 (ImageNet-21k pre-trained via timm)
  - PEFT:     LoRA on attention QKV projections (r=8, α=16)
  - Head:     Linear(384, 8) — 8 morphology classes

Guarantees:
  - Training data ONLY from training patients (splits.json).
  - Validation data ONLY from validation patients.
  - Test patients never seen during training.
  - No independent cell-level random splits.
  - WeightedRandomSampler handles class imbalance.
  - Early stopping on validation macro-F1.
  - Full taxonomy printed before any gradient steps.
  - Best checkpoint and hyperparameter JSON saved.

Usage:
  python scripts/step4_train_cell_classifier.py
  python scripts/step4_train_cell_classifier.py --epochs 30 --batch-size 64
  python scripts/step4_train_cell_classifier.py --config configs/colab.yaml

Outputs:
  models/checkpoints/best_vit_lora_morphology.pt
  models/checkpoints/taxonomy_used.json
  models/checkpoints/training_config.json
  models/checkpoints/train_log.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
logger = logging.getLogger("step4_train")


def print_taxonomy(samples_by_split: Dict[str, List[dict]], save_dir: Path) -> None:
    from collections import Counter
    print("\n" + "=" * 70)
    print("CELL MORPHOLOGY TAXONOMY — BoneMarrowWSI-PediatricLeukemia")
    print("=" * 70)
    print(f"{'Idx':<4} {'Category':<35} {'Train':>7} {'Val':>7} {'Test':>7}  Status")
    print("-" * 70)
    taxonomy_record = {}
    for idx, cat in enumerate(MORPH_CLASSES):
        counts = {sp: sum(1 for s in samps if s["class_code"] == cat)
                  for sp, samps in samples_by_split.items()}
        is_rare = cat in RARE_CLASSES
        print(
            f"{idx:<4} {cat:<35} "
            f"{counts.get('train', 0):>7} "
            f"{counts.get('val', 0):>7} "
            f"{counts.get('test', 0):>7}  "
            f"{'RARE' if is_rare else 'normal'}"
        )
        taxonomy_record[cat] = {
            "label_index": idx,
            "train_cells": counts.get("train", 0),
            "val_cells": counts.get("val", 0),
            "test_cells": counts.get("test", 0),
            "is_rare": is_rare,
        }
    print()
    print("Excluded categories (NOT in classifier):")
    for cat in sorted(EXCLUDED_CATEGORIES):
        print(f"  - {cat}")
    total_train = sum(v["train_cells"] for v in taxonomy_record.values())
    total_val   = sum(v["val_cells"]   for v in taxonomy_record.values())
    total_test  = sum(v["test_cells"]  for v in taxonomy_record.values())
    print(f"\nTotal cells — Train: {total_train:,}, Val: {total_val:,}, Test: {total_test:,}")
    print("=" * 70 + "\n")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "taxonomy_used.json", "w", encoding="utf-8") as f:
        json.dump({
            "taxonomy": taxonomy_record,
            "excluded_categories": sorted(EXCLUDED_CATEGORIES),
            "num_classes": len(MORPH_CLASSES),
            "rare_classes": RARE_CLASSES,
        }, f, indent=2)
    logger.info(f"Taxonomy saved → {save_dir / 'taxonomy_used.json'}")


def train_epoch(model, loader, criterion, optimizer, scaler, device, precision):
    model.train()
    total_loss = 0.0
    use_amp = (device.type == "cuda") and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for images, labels, _, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(images)
                loss   = criterion(logits, labels)
            if scaler and precision == "fp16":
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
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for images, labels, _, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
    metrics = evaluate_predictions(y_true=all_targets, y_pred=all_preds,
                                   class_names=MORPH_CLASSES)
    return {
        "val_loss": total_loss / max(1, len(loader)),
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_weighted_f1": metrics["weighted_f1"],
        "val_rare_recall": metrics["rare_classes_macro_recall"],
    }


def run_training(
    config_path: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
) -> Path:
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)

    if epochs_override:
        cfg.training.epochs = epochs_override
    if batch_override:
        cfg.training.batch_size = batch_override

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.warning("Device: CPU — training will be slow!")

    cell_meta_path = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
    splits_path    = REPO_ROOT / "data" / "processed" / "splits.json"

    for p in [cell_meta_path, splits_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    logger.info("Loading cell samples …")
    train_samples = load_cell_samples(cell_meta_path, splits_path, split="train")
    val_samples   = load_cell_samples(cell_meta_path, splits_path, split="val")
    test_samples  = load_cell_samples(cell_meta_path, splits_path, split="test")

    if not train_samples:
        raise RuntimeError("No training crops found on disk. Run step2 first.")
    if not val_samples:
        raise RuntimeError("No validation crops found on disk. Run step2 first.")

    save_dir = REPO_ROOT / "models" / "checkpoints"
    print_taxonomy(
        {"train": train_samples, "val": val_samples, "test": test_samples},
        save_dir=save_dir,
    )

    img_size = cfg.dataset.image_size
    train_ds = BoneMarrowCellDataset(
        samples=train_samples,
        transform=get_default_transforms(image_size=img_size, is_train=True),
    )
    val_ds = BoneMarrowCellDataset(
        samples=val_samples,
        transform=get_default_transforms(image_size=img_size, is_train=False),
    )

    num_workers = 4 if device.type != "cpu" else 0
    sampler = compute_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        sampler=sampler, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

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

    stats = model.get_trainable_param_stats()
    logger.info(
        f"Model: {cfg.vision.backbone} | "
        f"{stats['trainable_params']:,} trainable / {stats['total_params']:,} total "
        f"({stats['trainable_percentage']}%)"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    scaler = (torch.amp.GradScaler()
              if device.type == "cuda" and cfg.training.precision == "fp16"
              else None)

    # Save training config
    training_config = {
        "backbone": cfg.vision.backbone,
        "pretrained": cfg.vision.pretrained,
        "lora_r": cfg.vision.lora.r,
        "lora_alpha": cfg.vision.lora.alpha,
        "lora_dropout": cfg.vision.lora.dropout,
        "lora_target_modules": list(cfg.vision.lora.target_modules),
        "num_classes": len(MORPH_CLASSES),
        "class_names": MORPH_CLASSES,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "label_smoothing": cfg.training.label_smoothing,
        "precision": cfg.training.precision,
        "early_stopping_patience": cfg.training.early_stopping_patience,
        "imbalance_strategy": "WeightedRandomSampler",
        "seed": cfg.seed,
        "train_cells": len(train_samples),
        "val_cells": len(val_samples),
        "test_cells": len(test_samples),
        "device": str(device),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    best_macro_f1    = -1.0
    patience_counter = 0
    patience         = cfg.training.early_stopping_patience
    best_ckpt_path   = save_dir / "best_vit_lora_morphology.pt"
    train_log_path   = save_dir / "train_log.jsonl"

    logger.info(f"Starting training: {cfg.training.epochs} epochs max, patience={patience}")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg.training.precision,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()},
            "elapsed_s": round(elapsed, 1),
        }
        with open(train_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            f"Epoch {epoch:02d}/{cfg.training.epochs}  [{elapsed:.0f}s]  "
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
            model.save_checkpoint(best_ckpt_path, extra_metadata={
                "epoch": epoch,
                "best_val_macro_f1": best_macro_f1,
                "val_weighted_f1":   val_metrics["val_weighted_f1"],
                "val_rare_recall":   val_metrics["val_rare_recall"],
                "training_config":   training_config,
            })
            logger.info(f"  → New best checkpoint saved (macro F1={best_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    logger.info(f"\nTraining complete. Best val macro F1 = {best_macro_f1:.4f}")
    logger.info(f"Checkpoint → {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ViT-Small/16 + LoRA morphology classifier."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    run_training(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_override=args.batch_size,
    )
