"""
scripts/step6_patient_baseline.py
─────────────────────────────────────────────────────────────────────────────
STEP 6: Patient-level 4-class baseline.

Uses the trained cell morphology model to:
  1. Infer cell-level probabilities for all test patients.
  2. Aggregate per-patient cell probability vectors (mean pooling).
  3. Train a logistic regression classifier on training patient vectors.
  4. Evaluate on the 17 held-out test patients.
  5. Flag patients with < 50 annotated cells as LOW-EVIDENCE.

Target classes:
  0 = ALL  (B-ALL, T-ALL, Burkitt Leukemia)
  1 = AML  (AML subtypes, APL)
  2 = CML  (CML chronic phase, CML blast phase)
  3 = Other (MPAL, missing diagnosis)

Important: blast fraction metric is ANNOTATED-CELL BLAST FRACTION, NOT
           a clinical marrow blast percentage.

Outputs:
  models/evaluation/patient_level_metrics.json
  models/evaluation/patient_level_predictions.csv
  models/evaluation/patient_confusion_matrix.csv
  docs/EXPERIMENT_002_RESULTS.md

Usage:
  python scripts/step6_patient_baseline.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support,
)

from vision.taxonomy import MORPH_CLASSES, IDX_TO_MORPH
from vision.dataset import BoneMarrowCellDataset, load_cell_samples, get_default_transforms
from vision.model import BoneMarrowViT
from vision.aggregate import aggregate_patient_crops

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step6_patient")

CELL_META    = REPO_ROOT / "data" / "processed" / "cell_metadata.csv"
SPLITS       = REPO_ROOT / "data" / "processed" / "splits.json"
MANIFEST     = REPO_ROOT / "data" / "patient_manifest.csv"
CKPT_PATH    = REPO_ROOT / "models" / "checkpoints" / "best_vit_lora_morphology.pt"
EVAL_DIR     = REPO_ROOT / "models" / "evaluation"

# Patient-level class mapping (must match REAL_DATA_SCHEMA.md)
PATIENT_CLASSES = ["ALL", "AML", "CML", "Other"]
LOW_EVIDENCE_THRESHOLD = 50    # flag patients with fewer annotated cells


def get_diagnosis_mapping(manifest_df: pd.DataFrame) -> Dict[str, str]:
    """Map patient_id → 4-class label from manifest.
    
    The manifest uses 'target_4class' (values: ALL, AML, CML, Other).
    Falls back to checking legacy column names for forward-compatibility.
    """
    mapping = {}
    for _, row in manifest_df.iterrows():
        pid = str(row.get("patient_id", row.get("PatientID", "")))
        # Try columns in priority order — target_4class is the real column
        diag = str(
            row.get("target_4class",
            row.get("primary_diagnosis_4class",
            row.get("diagnosis_group",
            row.get("primary_class", "Other"))))
        )
        if diag not in PATIENT_CLASSES or diag in ("nan", "None", ""):
            diag = "Other"
        mapping[pid] = diag
    return mapping



@torch.no_grad()
def infer_patient_vectors(
    model: torch.nn.Module,
    split_samples: List[dict],
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Run model on all cells for a split.
    Returns dict: patient_id → {'probs': (N,8) array, 'labels': list[int], 'class_codes': list[str]}
    """
    ds = BoneMarrowCellDataset(
        samples=split_samples,
        transform=get_default_transforms(image_size=224, is_train=False),
    )
    num_workers = 4 if device.type != "cpu" else 0
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)

    # Collect per-patient softmax probs
    patient_data: Dict[str, dict] = {}
    model.eval()

    for images, labels, class_codes, patient_ids in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        for i, pid in enumerate(patient_ids):
            if pid not in patient_data:
                patient_data[pid] = {"probs": [], "labels": [], "class_codes": []}
            patient_data[pid]["probs"].append(probs[i])
            patient_data[pid]["labels"].append(int(labels[i]))
            patient_data[pid]["class_codes"].append(str(class_codes[i]))

    # Stack arrays
    for pid in patient_data:
        patient_data[pid]["probs"] = np.stack(patient_data[pid]["probs"], axis=0)

    return patient_data


def build_feature_matrix(
    patient_data: Dict[str, dict],
    diag_map: Dict[str, str],
    split_name: str,
) -> tuple:
    """Returns (X, y, meta_rows) where X is (N_patients, 8) mean prob vectors."""
    X_rows, y_rows, meta = [], [], []
    for pid, data in patient_data.items():
        probs = data["probs"]           # (n_cells, 8)
        n_cells = len(probs)
        mean_vec = probs.mean(axis=0)   # (8,)

        predicted_classes = [IDX_TO_MORPH[int(np.argmax(probs[j]))]
                             for j in range(n_cells)]
        agg = aggregate_patient_crops(predicted_classes, min_crops_threshold=LOW_EVIDENCE_THRESHOLD)

        label = diag_map.get(pid, "Other")
        is_low_evidence = n_cells < LOW_EVIDENCE_THRESHOLD

        X_rows.append(mean_vec)
        y_rows.append(label)
        meta.append({
            "patient_id": pid,
            "split": split_name,
            "n_cells": n_cells,
            "is_low_evidence": is_low_evidence,
            "annotated_blast_fraction": agg["annotated_cell_blast_fraction"],
            "true_class": label,
            "blast_warning": agg.get("warning", ""),
        })

    return np.array(X_rows), np.array(y_rows), meta


def run_patient_baseline() -> None:
    logger.info("=" * 70)
    logger.info("STEP 6 — Patient-Level 4-Class Baseline")
    logger.info("=" * 70)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}. Run step4 first.")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if hasattr(torch.backends, "mps") and
                   torch.backends.mps.is_available()
              else torch.device("cpu"))
    logger.info(f"Device: {device}")

    model = BoneMarrowViT.load_checkpoint(CKPT_PATH, device=str(device))
    model.eval()

    # Load manifest for labels
    manifest_df = pd.read_csv(MANIFEST)
    diag_map = get_diagnosis_mapping(manifest_df)
    logger.info(f"Diagnosis map loaded: {len(diag_map)} patients")

    # Load cell samples per split
    logger.info("Loading cell samples …")
    train_samples = load_cell_samples(CELL_META, SPLITS, split="train")
    val_samples   = load_cell_samples(CELL_META, SPLITS, split="val")
    test_samples  = load_cell_samples(CELL_META, SPLITS, split="test")

    # Run inference per split
    logger.info("Inferring cell probabilities — train …")
    train_patient_data = infer_patient_vectors(model, train_samples, device)
    logger.info("Inferring cell probabilities — val …")
    val_patient_data   = infer_patient_vectors(model, val_samples, device)
    logger.info("Inferring cell probabilities — test …")
    test_patient_data  = infer_patient_vectors(model, test_samples, device)

    # Build feature matrices
    X_train, y_train, train_meta = build_feature_matrix(train_patient_data, diag_map, "train")
    X_val,   y_val,   val_meta   = build_feature_matrix(val_patient_data,   diag_map, "val")
    X_test,  y_test,  test_meta  = build_feature_matrix(test_patient_data,  diag_map, "test")

    logger.info(f"Train patients: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Log low-evidence patients
    low_evidence_test = [m for m in test_meta if m["is_low_evidence"]]
    if low_evidence_test:
        logger.warning(f"Low-evidence test patients (< {LOW_EVIDENCE_THRESHOLD} cells):")
        for m in low_evidence_test:
            logger.warning(f"  {m['patient_id']}: {m['n_cells']} cells [{m['true_class']}]")

    # Fit logistic regression on train (+val) patients
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_trainval, y_trainval)

    # Evaluate on test
    y_pred = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred, labels=clf.classes_, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )[2]
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    print("\n" + "=" * 60)
    print("PATIENT-LEVEL 4-CLASS EVALUATION (Test, N=17)")
    print("=" * 60)
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Macro Precision:     {macro_prec:.4f}")
    print(f"  Macro Recall:        {macro_rec:.4f}")
    print(f"  Macro F1:            {macro_f1:.4f}")
    print(f"  Weighted F1:         {weighted_f1:.4f}")
    print()
    print(f"  {'Class':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>4}")
    print("  " + "-" * 35)
    for i, cls in enumerate(clf.classes_):
        print(f"  {cls:<10} {prec[i]:>6.3f} {rec[i]:>6.3f} {f1[i]:>6.3f} {int(sup[i]):>4}")
    print()
    print("  Confusion Matrix:")
    print(f"  {'':>10}", end="")
    for cls in clf.classes_:
        print(f"  {cls:>6}", end="")
    print()
    for i, true_cls in enumerate(clf.classes_):
        print(f"  {true_cls:<10}", end="")
        for j in range(len(clf.classes_)):
            print(f"  {cm[i,j]:>6}", end="")
        print()
    print("=" * 60)

    if low_evidence_test:
        print(f"\n   Low-evidence patients (< {LOW_EVIDENCE_THRESHOLD} cells):")
        for m in low_evidence_test:
            print(f"    {m['patient_id'][:8]}…  n={m['n_cells']}  true={m['true_class']}")
        print("  These predictions should be interpreted with extra caution.")

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    patient_results = []
    for i, m in enumerate(test_meta):
        pred_class = y_pred[i]
        patient_results.append({
            **m,
            "predicted_class": pred_class,
            "correct": (pred_class == m["true_class"]),
            **{f"prob_{cls}": float(y_prob_test[i, j])
               for j, cls in enumerate(clf.classes_)},
        })
    results_df = pd.DataFrame(patient_results)
    results_df.to_csv(EVAL_DIR / "patient_level_predictions.csv", index=False)

    per_class = {}
    for i, cls in enumerate(clf.classes_):
        per_class[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }
    metrics_out = {
        "n_test_patients": len(X_test),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_order": list(clf.classes_),
        "low_evidence_patients": low_evidence_test,
        "low_evidence_threshold": LOW_EVIDENCE_THRESHOLD,
        "classifier": "LogisticRegression(C=1, class_weight=balanced)",
        "feature": "mean pooled cell softmax probabilities (8-dim)",
    }
    with open(EVAL_DIR / "patient_level_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(clf.classes_)))
    ax.set_yticks(range(len(clf.classes_)))
    ax.set_xticklabels(clf.classes_, fontsize=10)
    ax.set_yticklabels(clf.classes_, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Patient-Level Confusion Matrix (N={len(X_test)} test pts)", fontsize=10)
    for i in range(len(clf.classes_)):
        for j in range(len(clf.classes_)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "patient_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Results → {EVAL_DIR.relative_to(REPO_ROOT)}/")

    # Generate EXPERIMENT_002_RESULTS.md
    write_experiment_report(metrics_out, results_df)
    logger.info(f"\n[STEP 6 COMPLETE] Patient baseline done. See docs/EXPERIMENT_002_RESULTS.md")


def write_experiment_report(patient_metrics: dict, results_df: pd.DataFrame) -> None:
    """Write the final EXPERIMENT_002_RESULTS.md."""
    from datetime import datetime

    # Load cell-level metrics if available
    cell_metrics_path = EVAL_DIR / "cell_level_metrics.json"
    cell_metrics = {}
    if cell_metrics_path.exists():
        with open(cell_metrics_path) as f:
            cell_metrics = json.load(f)

    # Load training config if available
    config_path = REPO_ROOT / "models" / "checkpoints" / "training_config.json"
    train_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            train_cfg = json.load(f)

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    low_ev = patient_metrics.get("low_evidence_patients", [])

    lines = [
        "# EXPERIMENT_002 — Real-Data ViT-Small/16 + LoRA Baseline Results",
        "",
        f"> **Status:** Complete  ",
        f"> **Timestamp:** {ts}  ",
        f"> **Dataset:** BoneMarrowWSI-PediatricLeukemia (IDC, CC BY 4.0)  ",
        f"> **Checkpoint:** `models/checkpoints/best_vit_lora_morphology.pt`",
        "",
        "---",
        "",
        "## 1. Dataset & Splits",
        "",
        "| Split | Patients | Cells |",
        "|---|---|---|",
        f"| Train | 79 | {train_cfg.get('train_cells', 'N/A'):,} |",
        f"| Validation | 17 | {train_cfg.get('val_cells', 'N/A'):,} |",
        f"| Test | 17 | {train_cfg.get('test_cells', 'N/A'):,} |",
        f"| **Total** | **113** | **20,852** |",
        "",
        "> Patient overlap: Train∩Val = 0, Train∩Test = 0, Val∩Test = 0",
        "> Source: IDC `bonemarrowwsi_pediatricleukemia`, consensus annotations only.",
        "",
        "---",
        "",
        "## 2. Cell Morphology Taxonomy (8 Classes)",
        "",
        "| Idx | Class | Train | Val | Test | Type |",
        "|---|---|---|---|---|---|",
    ]

    # Taxonomy from file if available
    tax_path = REPO_ROOT / "models" / "checkpoints" / "taxonomy_used.json"
    if tax_path.exists():
        with open(tax_path) as f:
            tax = json.load(f)
        for idx, (cls, info) in enumerate(tax["taxonomy"].items()):
            flag = "Rare" if info.get("is_rare") else "Normal"
            lines.append(
                f"| {idx} | {cls} | {info['train_cells']:,} | "
                f"{info['val_cells']:,} | {info['test_cells']:,} | {flag} |"
            )
    else:
        lines.append("| *Run taxonomy_used.json not found* | | | | |")

    lines += [
        "",
        "**Excluded from classifier (artefacts / no-consensus):**",
        "- `Artefact (Excluded)` — 6,544 cells",
        "- `Ambiguous / No Consensus (Excluded)` — 683 cells",
        "- `Spicule (Tissue Fragment - Excluded)` — 316 cells",
        "",
        "---",
        "",
        "## 3. Training Configuration",
        "",
        "| Hyperparameter | Value |",
        "|---|---|",
        f"| Backbone | `{train_cfg.get('backbone', 'vit_small_patch16_224')}` |",
        f"| Pre-training | ImageNet-21k (via timm) |",
        f"| LoRA rank r | {train_cfg.get('lora_r', 8)} |",
        f"| LoRA α | {train_cfg.get('lora_alpha', 16)} |",
        f"| LoRA dropout | {train_cfg.get('lora_dropout', 0.05)} |",
        f"| LoRA target modules | `{train_cfg.get('lora_target_modules', ['qkv'])}` |",
        f"| Trainable params | ~147K / 21.8M (0.68%) |",
        f"| Epochs (max) | {train_cfg.get('epochs', 30)} |",
        f"| Batch size | {train_cfg.get('batch_size', 64)} |",
        f"| Learning rate | {train_cfg.get('lr', '3e-4')} |",
        f"| Weight decay | {train_cfg.get('weight_decay', 0.01)} |",
        f"| Label smoothing | {train_cfg.get('label_smoothing', 0.1)} |",
        f"| Precision | `{train_cfg.get('precision', 'bf16')}` |",
        f"| Imbalance strategy | WeightedRandomSampler |",
        f"| Early stopping | val macro-F1, patience=5 |",
        f"| Seed | {train_cfg.get('seed', 42)} |",
        "",
        "---",
        "",
        "## 4. Cell-Level Results (Test Split — 17 Patients, 3,338 Cells)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {cell_metrics.get('accuracy', 'TBD'):.4f} |" if cell_metrics else "| Accuracy | TBD |",
        f"| Macro Precision | {cell_metrics.get('macro_precision', 'TBD'):.4f} |" if cell_metrics else "| Macro Precision | TBD |",
        f"| Macro Recall | {cell_metrics.get('macro_recall', 'TBD'):.4f} |" if cell_metrics else "| Macro Recall | TBD |",
        f"| Macro F1 | {cell_metrics.get('macro_f1', 'TBD'):.4f} |" if cell_metrics else "| Macro F1 | TBD |",
        f"| Weighted F1 | {cell_metrics.get('weighted_f1', 'TBD'):.4f} |" if cell_metrics else "| Weighted F1 | TBD |",
        f"| Rare-class Macro Recall | {cell_metrics.get('rare_classes_macro_recall', 'TBD'):.4f} |" if cell_metrics else "| Rare-class Macro Recall | TBD |",
        "",
    ]

    if cell_metrics and "per_class" in cell_metrics:
        lines += [
            "### Per-Class Metrics",
            "",
            "| Class | Precision | Recall | F1 | Support | Rare? |",
            "|---|---|---|---|---|---|",
        ]
        from vision.taxonomy import RARE_CLASSES
        for cls, m in cell_metrics["per_class"].items():
            rare = "Yes" if m.get("is_rare_class") else "No"
            lines.append(
                f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1_score']:.4f} | {m['support']:,} | {rare} |"
            )
        lines.append("")

    lines += [
        "### Confusion Matrix",
        "",
        "See `models/evaluation/confusion_matrix.png`",
        "",
        "---",
        "",
        "## 5. Patient-Level Results (Test Split — 17 Patients)",
        "",
        "> **Classifier:** Logistic Regression on mean-pooled cell softmax probabilities  ",
        "> **Features:** 8-dim cell differential vector per patient  ",
        "> **Classes:** ALL / AML / CML / Other",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {patient_metrics.get('accuracy', 'TBD'):.4f} |",
        f"| Macro Precision | {patient_metrics.get('macro_precision', 'TBD'):.4f} |",
        f"| Macro Recall | {patient_metrics.get('macro_recall', 'TBD'):.4f} |",
        f"| Macro F1 | {patient_metrics.get('macro_f1', 'TBD'):.4f} |",
        f"| Weighted F1 | {patient_metrics.get('weighted_f1', 'TBD'):.4f} |",
        "",
        "### Per-Class Patient-Level Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]

    for cls, m in patient_metrics.get("per_class", {}).items():
        lines.append(
            f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['support']} |"
        )

    lines += [
        "",
        "### Confusion Matrix",
        "",
        "See `models/evaluation/patient_confusion_matrix.png`",
        "",
        "---",
        "",
        "## 6. Low-Evidence Cases",
        "",
        f"> Patients with fewer than {patient_metrics.get('low_evidence_threshold', 50)} "
        "annotated cells are flagged as LOW-EVIDENCE.  ",
        "> Their predictions should be interpreted with extra caution.",
        "> These patients are RETAINED in the analysis and NOT deleted.",
        "",
    ]

    if low_ev:
        lines += [
            "| Patient ID | Cells | True Class | Predicted | Blast Fraction |",
            "|---|---|---|---|---|",
        ]
        for m in low_ev:
            pid_pred = results_df[results_df["patient_id"] == m["patient_id"]]
            pred_cls = pid_pred["predicted_class"].values[0] if len(pid_pred) > 0 else "N/A"
            lines.append(
                f"| `{m['patient_id'][:12]}…` | {m['n_cells']} | {m['true_class']} | "
                f"{pred_cls} | {m['annotated_blast_fraction']:.3f} |"
            )
        lines.append("")
    else:
        lines.append("No low-evidence patients in the test split.")
        lines.append("")

    lines += [
        "---",
        "",
        "## 7. XAI — Attention Rollout Examples",
        "",
        "Representative attention rollout heatmaps (Abnar & Zuidema 2020) are saved in:",
        "```",
        "models/evaluation/xai/xai_{class}_{n}.png",
        "```",
        "",
        "---",
        "",
        "## 8. Scientific Disclaimers",
        "",
        "> [!WARNING]",
        "> **Annotated-Cell Blast Fraction ≠ Clinical Blast Percentage.**  ",
        "> The consensus annotations represent selected monolayer regions, NOT",
        "> an exhaustive whole-slide clinical aspirate count.  ",
        "> No clinical claims can be made from this metric alone.",
        "",
        "> [!NOTE]",
        "> This experiment represents the **vision morphology baseline** only.  ",
        "> It is NOT the final multimodal agentic system.  ",
        "> CBC, cytogenetics, and clinical notes are absent from this dataset.",
        "",
        "---",
        "",
        "## 9. Limitations",
        "",
        "1. **Pediatric-specific cohort** (ages 1.5–17 yrs). Cannot generalize to adults.",
        "2. **Extreme class imbalance**: CML=44%, ALL=33%, AML=15%, Other=8%.",
        "3. **No ancillary modalities**: CBC, cytogenetics, and molecular markers absent.",
        "4. **17 test patients**: Small test set; confidence intervals are wide.",
        "5. **Monolayer sampling bias**: Cells selected from pre-identified ROIs.",
        "",
        "---",
        "",
        "## 10. Next Steps",
        "",
        "- ☐ RAG layer: literature ingest, FAISS + BM25 hybrid indexing, BGE reranker",
        "- ☐ Agentic framework: Retriever, Synthesizer, Critic agents",
        "- ☐ Multimodal context assembly",
        "",
        "---",
        "",
        f"*Generated by `scripts/step6_patient_baseline.py` on {ts}.*",
        "",
    ]

    out_path = REPO_ROOT / "docs" / "EXPERIMENT_002_RESULTS.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run_patient_baseline()
