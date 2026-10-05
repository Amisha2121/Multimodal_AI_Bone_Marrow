"""
tests/test_vision.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the Vision Baseline Pipeline (8-class BoneMarrowWSI taxonomy):
- Taxonomy consistency: 8 classes, bidirectional index mappings
- Zero patient overlap verification in patient-level splits
- Wilson score confidence interval properties
- Patient-level differential aggregation and annotated-cell blast fraction
- Multiclass evaluation metrics and rare class tracking
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.taxonomy import (
    MORPH_CLASSES,
    MORPH_TO_IDX,
    IDX_TO_MORPH,
    BLAST_CATEGORIES,
    EXCLUDED_CATEGORIES,
    RARE_CLASSES,
    CELL_CLASSES,   # backward-compat alias
)
from vision.aggregate import (
    CELL_CLASSES as AGG_CELL_CLASSES,
    aggregate_patient_crops,
    wilson_score_interval,
)
from vision.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    create_patient_level_splits,
)
from vision.evaluate import evaluate_predictions


# ── Taxonomy consistency ──────────────────────────────────────────────────────

def test_8_classes_consistency():
    """Verify exactly 8 morphology classes and consistent bidirectional mappings."""
    assert len(MORPH_CLASSES) == 8, f"Expected 8 classes, got {len(MORPH_CLASSES)}"
    assert len(CLASS_TO_IDX) == 8
    assert len(IDX_TO_CLASS) == 8
    for i, c in enumerate(MORPH_CLASSES):
        assert CLASS_TO_IDX[c] == i, f"Forward mapping mismatch for '{c}'"
        assert IDX_TO_CLASS[i] == c, f"Reverse mapping mismatch for index {i}"


def test_cell_classes_backward_compat():
    """CELL_CLASSES alias must equal MORPH_CLASSES."""
    assert CELL_CLASSES == MORPH_CLASSES
    assert AGG_CELL_CLASSES == MORPH_CLASSES


def test_excluded_categories_disjoint_from_morph_classes():
    """Excluded categories must not appear in the classifier taxonomy."""
    for exc in EXCLUDED_CATEGORIES:
        assert exc not in MORPH_CLASSES, (
            f"Category '{exc}' appears in both MORPH_CLASSES and EXCLUDED_CATEGORIES"
        )


def test_blast_categories_in_morph_classes():
    """All blast categories must be valid MORPH_CLASSES members."""
    for b in BLAST_CATEGORIES:
        assert b in MORPH_CLASSES, f"Blast category '{b}' not in MORPH_CLASSES"


def test_rare_classes_in_morph_classes():
    """All rare classes must be valid MORPH_CLASSES members."""
    for r in RARE_CLASSES:
        assert r in MORPH_CLASSES, f"Rare class '{r}' not in MORPH_CLASSES"


# ── Patient-level splits ──────────────────────────────────────────────────────

def test_patient_level_splits_zero_leakage():
    """Verify patient-level splits have strictly ZERO patient overlap."""
    import warnings
    records = [{"patient_id": f"PATIENT_{i:03d}"} for i in range(100)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        splits = create_patient_level_splits(
            patient_records=records,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            output_path=None,
        )

    train_pts = set(splits["train"])
    val_pts   = set(splits["val"])
    test_pts  = set(splits["test"])

    assert len(train_pts & val_pts)  == 0, "Patient leakage between train and val!"
    assert len(train_pts & test_pts) == 0, "Patient leakage between train and test!"
    assert len(val_pts  & test_pts)  == 0, "Patient leakage between val and test!"

    total_assigned = len(train_pts) + len(val_pts) + len(test_pts)
    assert total_assigned == 100


# ── Wilson score ──────────────────────────────────────────────────────────────

def test_wilson_score_interval():
    """Verify Wilson score confidence interval properties."""
    low, high = wilson_score_interval(count=20, n=100, confidence=0.95)
    assert 0.0 <= low  < 0.20
    assert 0.20 < high <= 1.0

    # Boundary: zero successes
    low_0, high_0 = wilson_score_interval(0, 100)
    assert low_0 == 0.0
    assert high_0 > 0.0

    # Boundary: all successes
    low_n, high_n = wilson_score_interval(100, 100)
    assert low_n < 1.0
    assert high_n == 1.0


# ── Differential aggregation ──────────────────────────────────────────────────

def test_patient_crop_aggregation_basic():
    """Verify patient-level differential and annotated-cell blast fraction."""
    # 30 Blasts out of 100 cells → blast fraction = 0.30
    crops = ["Blast"] * 30 + ["Granulocyte (Mature/Band)"] * 70
    result = aggregate_patient_crops(
        predicted_classes=crops,
        min_crops_threshold=50,
    )

    assert result["total_crops"] == 100
    assert result["is_sufficient_sample"] is True
    assert result["annotated_cell_blast_fraction"] == 0.30
    assert result["differential"]["Blast"] == 0.30
    assert result["differential"]["Granulocyte (Mature/Band)"] == 0.70

    # Sum of differential across all 8 classes must equal 1.0
    total_diff = sum(result["differential"].values())
    assert abs(total_diff - 1.0) < 1e-4


def test_patient_crop_aggregation_all_classes():
    """Aggregate with one cell of each of the 8 classes → each gets 1/8 fraction."""
    crops = MORPH_CLASSES.copy()   # one example per class
    result = aggregate_patient_crops(predicted_classes=crops, min_crops_threshold=4)
    assert result["total_crops"] == 8
    for cat in MORPH_CLASSES:
        assert abs(result["differential"][cat] - 1/8) < 1e-4


def test_low_sample_count_warning():
    """Warning must be emitted when crop count is below threshold."""
    crops = ["Blast"] * 10
    result = aggregate_patient_crops(predicted_classes=crops, min_crops_threshold=50)
    assert result["is_sufficient_sample"] is False
    assert result["warning"] is not None
    assert "Low cell count warning" in result["warning"]


def test_blast_fraction_note_present():
    """Result must always carry the clinical-disclaimer note."""
    result = aggregate_patient_crops(predicted_classes=["Blast"] * 5)
    assert "blast_fraction_note" in result
    assert "RESEARCH NOTE" in result["blast_fraction_note"]


# ── Evaluation metrics ────────────────────────────────────────────────────────

def test_evaluate_predictions_metrics():
    """Verify per-class recall, precision, F1, and confusion matrix calculation."""
    # Use first 4 of the 8 classes for simplicity
    classes_4 = MORPH_CLASSES[:4]
    y_true = [0, 0, 1, 1, 2, 3]
    y_pred = [0, 1, 1, 1, 2, 0]  # one mistake in class 0, one in class 3

    metrics = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        class_names=classes_4,
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "rare_classes_macro_recall" in metrics
    assert "confusion_matrix" in metrics

    # Class 1 (index 1): true=[1,1], pred=[1,1] → recall 1.0
    assert metrics["per_class"][classes_4[1]]["recall"] == 1.0
    # Class 1 precision: 2 TP, 1 FP → precision = 2/3 ≈ 0.67
    assert round(metrics["per_class"][classes_4[1]]["precision"], 2) == 0.67


def test_evaluate_all_8_classes():
    """Evaluate predictions covering all 8 MORPH_CLASSES."""
    y_true = list(range(8))
    y_pred = list(range(8))   # perfect predictions
    metrics = evaluate_predictions(y_true=y_true, y_pred=y_pred, class_names=MORPH_CLASSES)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    for c in MORPH_CLASSES:
        assert metrics["per_class"][c]["f1_score"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
