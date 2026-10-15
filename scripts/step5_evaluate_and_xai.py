"""
scripts/step5_evaluate_and_xai.py
─────────────────────────────────────────────────────────────────────────────
STEP 5: Full evaluation on held-out test patients + XAI attention rollout.

Loads:
  models/checkpoints/best_vit_lora_morphology.pt

Evaluates on TEST SPLIT ONLY (17 held-out patients, 3,338 cells).

Reports:
  - Accuracy, Macro Precision/Recall/F1, Weighted F1
  - Per-class precision, recall, F1, support
  - Confusion matrix (CSV + ASCII)
  - Error analysis (most-confused class pairs)
  - Attention rollout XAI heatmaps for representative cells per class

Outputs:
  models/evaluation/cell_level_metrics.json
  models/evaluation/confusion_matrix.csv
  models/evaluation/xai/attention_rollout_{class}_{n}.png   (overlaid heatmaps)
  models/evaluation/error_analysis.json

Usage:
  python scripts/step5_evaluate_and_xai.py
  python scripts/step5_evaluate_and_xai.py --no-xai    # skip heatmap generation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from vision.taxonomy import MORPH_CLASSES, MORPH_TO_IDX, IDX_TO_MORPH, RARE_CLASSES
from vision.dataset import BoneMarrowCellDataset, load_cell_samples, get_default_transforms
from vision.evaluate import evaluate_predictions, print_eval_report
from vision.model import BoneMarrowViT
from vision.attention_rollout import ViTAttentionRollout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step5_eval")

CELL_META  = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
SPLITS     = REPO_ROOT / "data" / "processed" / "splits.json"
CKPT_PATH  = REPO_ROOT / "models" / "checkpoints" / "best_vit_lora_morphology.pt"
EVAL_DIR   = REPO_ROOT / "models" / "evaluation"
XAI_DIR    = EVAL_DIR / "xai"

XAI_SAMPLES_PER_CLASS = 3    # number of rollout heatmaps per morphology class


@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    """Returns (y_true, y_pred, y_prob, patient_ids, image_paths)."""
    model.eval()
    all_true, all_pred, all_prob, all_pids, all_paths = [], [], [], [], []

    for images, labels, class_codes, patient_ids in loader:
        images  = images.to(device)
        logits  = model(images)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        preds   = np.argmax(probs, axis=1)
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())
        all_pids.extend(list(patient_ids))

    return all_true, all_pred, all_prob, all_pids


def generate_xai(
    model: BoneMarrowViT,
    test_samples: List[dict],
    device: torch.device,
    n_per_class: int = XAI_SAMPLES_PER_CLASS,
) -> None:
    """Generate attention rollout heatmaps for representative test cells."""
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    rollout = ViTAttentionRollout(model, discard_ratio=0.1, head_fusion="mean")

    val_tf = get_default_transforms(image_size=224, is_train=False)

    # Group samples by class
    by_class: Dict[str, List[dict]] = {c: [] for c in MORPH_CLASSES}
    for s in test_samples:
        cc = s["class_code"]
        if cc in by_class:
            by_class[cc].append(s)

    for class_name, samples in by_class.items():
        if not samples:
            logger.warning(f"No test samples for class '{class_name}' — skipping XAI.")
            continue
        picked = samples[:n_per_class]
        for i, sample in enumerate(picked):
            img_path = Path(sample["image_path"])
            if not img_path.exists():
                continue
            try:
                raw_img = Image.open(img_path).convert("RGB")
                tensor  = val_tf(raw_img).unsqueeze(0)

                # Run rollout
                model.eval()
                heatmap = rollout(tensor)       # (224, 224) float32 [0,1]

                # Overlay heatmap on original image
                raw_np = np.array(raw_img.resize((224, 224)))
                heatmap_col = cm.jet(heatmap)[:, :, :3]  # (224,224,3) RGB
                overlay = 0.55 * raw_np / 255.0 + 0.45 * heatmap_col
                overlay = np.clip(overlay, 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
                axes[0].imshow(raw_np)
                axes[0].set_title("Original Cell", fontsize=9)
                axes[0].axis("off")
                axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title("Attention Rollout", fontsize=9)
                axes[1].axis("off")
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay", fontsize=9)
                axes[2].axis("off")

                safe_name = class_name.replace(" ", "_").replace("/", "_")
                fig.suptitle(
                    f"{class_name}  |  Patient: {sample['patient_id'][:8]}…  "
                    f"|  Label: {sample['original_label']}",
                    fontsize=8,
                )
                plt.tight_layout()
                out_path = XAI_DIR / f"xai_{safe_name}_{i+1:02d}.png"
                plt.savefig(out_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"XAI saved → {out_path.relative_to(REPO_ROOT)}")

            except Exception as e:
                logger.warning(f"XAI failed for {img_path.name}: {e}")

    rollout.remove_hooks()


def error_analysis(
    y_true: list, y_pred: list, test_samples: list
) -> dict:
    """Identify top confused class pairs and per-patient error rates."""
    confusion_pairs: Dict[str, int] = {}
    errors_by_patient: Dict[str, int] = {}

    for true_idx, pred_idx, sample in zip(y_true, y_pred, test_samples):
        if true_idx != pred_idx:
            pair_key = f"{IDX_TO_MORPH[true_idx]} → {IDX_TO_MORPH[pred_idx]}"
            confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1
            pid = sample["patient_id"]
            errors_by_patient[pid] = errors_by_patient.get(pid, 0) + 1

    top_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]
    return {
        "top_confused_pairs": [{"pair": k, "count": v} for k, v in top_pairs],
        "errors_by_patient": dict(sorted(errors_by_patient.items(), key=lambda x: -x[1])[:20]),
    }


def run_evaluation(skip_xai: bool = False) -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CKPT_PATH}. Run step4_train first."
        )

    # Device
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if hasattr(torch.backends, "mps") and
                   torch.backends.mps.is_available()
              else torch.device("cpu"))
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading checkpoint: {CKPT_PATH}")
    model = BoneMarrowViT.load_checkpoint(CKPT_PATH, device=str(device))
    model.eval()

    # Load test samples
    logger.info("Loading test samples …")
    test_samples = load_cell_samples(CELL_META, SPLITS, split="test")
    if not test_samples:
        raise RuntimeError("No test samples found. Run step2 to extract crops first.")
    logger.info(f"Test cells: {len(test_samples):,}")

    test_ds = BoneMarrowCellDataset(
        samples=test_samples,
        transform=get_default_transforms(image_size=224, is_train=False),
    )
    num_workers = 4 if device.type != "cpu" else 0
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=num_workers)

    # Inference
    logger.info("Running inference on test split …")
    y_true, y_pred, y_prob, patient_ids = run_inference(model, test_loader, device)

    # Metrics
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        class_names=MORPH_CLASSES,
        output_dir=str(EVAL_DIR),
    )
    print_eval_report(metrics, prefix="TEST")

    # Save full metrics
    metrics_path = EVAL_DIR / "cell_level_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics → {metrics_path.relative_to(REPO_ROOT)}")

    # Confusion matrix CSV
    cm = np.array(metrics["confusion_matrix"])
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=MORPH_CLASSES, columns=MORPH_CLASSES)
    cm_path = EVAL_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix → {cm_path.relative_to(REPO_ROOT)}")

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(MORPH_CLASSES)))
    ax.set_yticks(range(len(MORPH_CLASSES)))
    ax.set_xticklabels(MORPH_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(MORPH_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Cell-Level Morphology Confusion Matrix (Test Set)", fontsize=11)
    for i in range(len(MORPH_CLASSES)):
        for j in range(len(MORPH_CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix figure → {(EVAL_DIR / 'confusion_matrix.png').relative_to(REPO_ROOT)}")

    # Error analysis
    errors = error_analysis(y_true, y_pred, test_samples)
    with open(EVAL_DIR / "error_analysis.json", "w") as f:
        json.dump(errors, f, indent=2)
    logger.info("Top confused class pairs:")
    for item in errors["top_confused_pairs"][:10]:
        logger.info(f"  {item['pair']}: {item['count']}")

    # XAI heatmaps
    if not skip_xai:
        logger.info("Generating attention rollout heatmaps …")
        generate_xai(model, test_samples, device)
    else:
        logger.info("XAI skipped (--no-xai).")

    logger.info("\n[STEP 5 COMPLETE] Cell-level evaluation done. Run step6_patient_baseline.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-xai", action="store_true",
                        help="Skip attention rollout heatmap generation.")
    args = parser.parse_args()
    run_evaluation(skip_xai=args.no_xai)
