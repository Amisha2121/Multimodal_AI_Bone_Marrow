"""
vision/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Cell Classifier Evaluation Module — 8-class BoneMarrowWSI taxonomy.

Computes:
- Per-class precision, recall, F1-score, and support
- Macro and weighted average metrics
- Multiclass confusion matrix
- Rare class highlighting (Monocyte Series, Lymphoid, Other Morphological)
- JSON and CSV reporting
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import numpy as np

from vision.taxonomy import MORPH_CLASSES, RARE_CLASSES


def evaluate_predictions(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Union[Dict, float, int]]:
    """
    Evaluates cell-level morphology classification predictions.

    Args:
        y_true: Ground truth class indices (integers 0..N-1).
        y_pred: Predicted class indices (integers 0..N-1).
        class_names: Ordered list of class name strings.
            Defaults to MORPH_CLASSES (8-class real taxonomy).
        output_dir: Optional directory to save vision_eval_metrics.json.

    Returns:
        Structured dictionary of evaluation metrics.
    """
    if class_names is None:
        class_names = MORPH_CLASSES

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    total_samples = len(y_true)
    n_classes = len(class_names)

    # Per-class metrics
    per_class_metrics: Dict[str, Dict] = {}
    tp_total = 0

    for idx, c_name in enumerate(class_names):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall    = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[c_name] = {
            "precision":      round(float(precision), 4),
            "recall":         round(float(recall), 4),
            "f1_score":       round(float(f1), 4),
            "support":        support,
            "is_rare_class":  c_name in RARE_CLASSES,
        }
        tp_total += tp

    # Aggregate macro and accuracy
    macro_precision = float(np.mean([m["precision"] for m in per_class_metrics.values()]))
    macro_recall    = float(np.mean([m["recall"]    for m in per_class_metrics.values()]))
    macro_f1        = float(np.mean([m["f1_score"]  for m in per_class_metrics.values()]))
    accuracy        = (tp_total / total_samples) if total_samples > 0 else 0.0

    # Weighted macro (by support, excluding zero-support classes)
    supports = np.array([m["support"] for m in per_class_metrics.values()], dtype=float)
    weighted_sum = np.sum(supports)
    if weighted_sum > 0:
        weighted_f1 = float(
            np.sum([m["f1_score"] * m["support"] for m in per_class_metrics.values()]) / weighted_sum
        )
    else:
        weighted_f1 = 0.0

    # Rare-class macro recall (only over classes with support > 0)
    rare_recalls = [
        m["recall"]
        for c, m in per_class_metrics.items()
        if m["is_rare_class"] and m["support"] > 0
    ]
    rare_macro_recall = float(np.mean(rare_recalls)) if rare_recalls else 0.0

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    results = {
        "total_samples":        total_samples,
        "accuracy":             round(accuracy, 4),
        "macro_precision":      round(macro_precision, 4),
        "macro_recall":         round(macro_recall, 4),
        "macro_f1":             round(macro_f1, 4),
        "weighted_f1":          round(weighted_f1, 4),
        "rare_classes_macro_recall": round(rare_macro_recall, 4),
        "per_class":            per_class_metrics,
        "confusion_matrix":     cm.tolist(),
        "class_names":          class_names,
        "rare_classes":         RARE_CLASSES,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "vision_eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def print_eval_report(results: Dict, prefix: str = "") -> None:
    """Prints a formatted evaluation report to stdout."""
    tag = f"[{prefix}] " if prefix else ""
    n = results["total_samples"]
    print(f"\n{tag}Evaluation Report ({n} samples)")
    print("=" * 60)
    print(f"  Accuracy:              {results['accuracy']:.4f}")
    print(f"  Macro F1:              {results['macro_f1']:.4f}")
    print(f"  Weighted F1:           {results['weighted_f1']:.4f}")
    print(f"  Rare-class Macro Recall: {results['rare_classes_macro_recall']:.4f}")
    print()
    print(f"  {'Class':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print("  " + "-" * 60)
    for c, m in results["per_class"].items():
        flag = " *" if m["is_rare_class"] else ""
        print(
            f"  {c:<35} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1_score']:>6.3f} {m['support']:>6}{flag}"
        )
    print("  (* = rare class)")
    print("=" * 60)
