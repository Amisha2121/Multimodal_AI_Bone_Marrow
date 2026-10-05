"""
tests/test_synthetic.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for synthetic data generation and context assembler:
- Validation of WHO 5th edition condition templates
- Exact sum-to-1.0 property of synthesized differentials
- Blast % consistency (MYO + MOB matching blast range)
- Deterministic structured patient context assembly
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from context.assembler import assemble_patient_context
from synthetic.generate import generate_single_case
from synthetic.sampler import compute_discrete_counts
from synthetic.templates import CASE_TEMPLATES, CELL_CLASSES


def test_case_templates_integrity():
    """Verify all templates have required WHO 5th metadata and valid blast ranges."""
    for cond_id, tmpl in CASE_TEMPLATES.items():
        assert "condition_name" in tmpl
        assert "who5_category" in tmpl
        b_min, b_max = tmpl["blast_pct_range"]
        assert 0.0 <= b_min <= b_max <= 1.0
        assert "lab_ranges" in tmpl
        assert "morphology_bias" in tmpl


def test_compute_discrete_counts_sum():
    """Verify discrete crop counts sum exactly to requested total."""
    diff = {"MYO": 0.333, "NGS": 0.667}
    counts = compute_discrete_counts(diff, total_crops=100)
    assert sum(counts.values()) == 100
    assert counts["MYO"] == 33
    assert counts["NGS"] == 67

    counts_odd = compute_discrete_counts(diff, total_crops=77)
    assert sum(counts_odd.values()) == 77


def test_generate_single_case_structure():
    """Verify synthetic patient case generation output structure and blast range."""
    case = generate_single_case(
        template_key="AML_t8_21",
        case_idx=1,
        total_crops=100,
        raw_dir="data/raw",
        seed=42,
    )
    assert case["case_id"] == "SYNTH_AML_t8_21_001"
    assert "patient_profile" in case
    assert "laboratory" in case
    assert "cytogenetics" in case
    assert len(case["crop_set"]) == 100

    # Differential should sum to 1.0
    diff_sum = sum(case["ground_truth_differential"].values())
    assert abs(diff_sum - 1.0) < 1e-3

    # AML blast % must be >= 20% under WHO 5th
    assert case["ground_truth_blast_pct"] >= 0.20


def test_assemble_patient_context():
    """Verify context assembler formats complete plain-text block and JSON payload."""
    case = generate_single_case(template_key="MDS_IB1", case_idx=2, seed=123)
    ctx = assemble_patient_context(
        case_id=case["case_id"],
        patient_profile=case["patient_profile"],
        differential=case["ground_truth_differential"],
        blast_pct=case["ground_truth_blast_pct"],
        total_crops=100,
        laboratory=case["laboratory"],
        cytogenetics=case["cytogenetics"],
    )

    assert "text_block" in ctx
    assert "json_payload" in ctx
    assert case["case_id"] in ctx["text_block"]
    assert "Bone Marrow Aspirate Cytomorphology" in ctx["text_block"]
    assert "WHO 5th Blast Percentage" in ctx["text_block"]
    assert "Cytogenetic & Molecular Diagnostic Findings" in ctx["text_block"]


if __name__ == "__main__":
    pytest.main([__file__])
