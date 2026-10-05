"""
synthetic/generate.py
─────────────────────────────────────────────────────────────────────────────
Generates synthetic patient cases from rule-based templates for AML and MDS
under the WHO 5th edition classification.

Outputs JSON records containing:
  - Case ID & demographic profile
  - Clinical narrative
  - Laboratory results (CBC, LDH, etc.)
  - Cytogenetic / molecular findings
  - Cell differential vector (ground truth)
  - Sampled crop set (real or simulated)
  - WHO 5th ground truth diagnosis & rationale

Usage:
  python synthetic/generate.py --num_cases 20 --output data/synthetic/cases.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from synthetic.sampler import sample_crops_for_differential
from synthetic.templates import CASE_TEMPLATES, CELL_CLASSES, PRIMARY_BLAST_CLASSES


def generate_single_case(
    template_key: str,
    case_idx: int,
    total_crops: int = 100,
    raw_dir: str = "data/raw",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates a single synthetic patient case from a condition template."""
    rng = random.Random(seed + case_idx)
    np_rng = np.random.default_rng(seed + case_idx)

    tmpl = CASE_TEMPLATES[template_key]

    # Demographic profile
    age = rng.randint(35, 78)
    sex = rng.choice(["Male", "Female"])

    # Sample blast %
    b_min, b_max = tmpl["blast_pct_range"]
    target_blast_pct = float(np_rng.uniform(b_min, b_max))

    # Construct differential
    bias = tmpl.get("morphology_bias", {})
    pblast = tmpl.get("primary_blast_class", "MYO")

    # Start with base distribution
    weights = []
    for c in CELL_CLASSES:
        if c == pblast:
            w = target_blast_pct
        else:
            w = bias.get(c, 0.01) + float(np_rng.uniform(0.005, 0.03))
        weights.append(w)

    # Normalize non-blast weights so that blast classes sum accurately
    weights = np.array(weights)
    blast_indices = [CELL_CLASSES.index(c) for c in PRIMARY_BLAST_CLASSES if c in CELL_CLASSES]
    non_blast_indices = [i for i in range(len(CELL_CLASSES)) if i not in blast_indices]

    non_blast_budget = max(0.01, 1.0 - target_blast_pct)
    non_blast_sum = np.sum(weights[non_blast_indices])
    if non_blast_sum > 0:
        weights[non_blast_indices] = (weights[non_blast_indices] / non_blast_sum) * non_blast_budget

    # If primary blast class was PMO (e.g. APL), ensure PMO has target fraction
    if pblast not in PRIMARY_BLAST_CLASSES and pblast in CELL_CLASSES:
        p_idx = CELL_CLASSES.index(pblast)
        weights[p_idx] = target_blast_pct
        other_indices = [i for i in range(len(CELL_CLASSES)) if i != p_idx]
        other_sum = np.sum(weights[other_indices])
        weights[other_indices] = (weights[other_indices] / other_sum) * (1.0 - target_blast_pct)

    differential = {c: round(float(weights[i]), 4) for i, c in enumerate(CELL_CLASSES)}
    # Re-normalize to exact 1.0 sum
    diff_sum = sum(differential.values())
    differential = {k: round(v / diff_sum, 4) for k, v in differential.items()}

    # Compute actual blast percentage (MYO + MOB)
    actual_blast_pct = round(sum(differential.get(b, 0.0) for b in PRIMARY_BLAST_CLASSES), 4)
    if pblast == "PMO" and "APL" in template_key:
        # In APL promyelocytes are the malignant counterpart
        promyelocyte_blast_equiv_pct = round(differential.get("PMO", 0.0) + differential.get("FGC", 0.0), 4)
    else:
        promyelocyte_blast_equiv_pct = None

    # Laboratory Panel
    labs = {}
    for lab_key, (l_min, l_max) in tmpl.get("lab_ranges", {}).items():
        val = float(np_rng.uniform(l_min, l_max))
        labs[lab_key] = round(val, 1)

    # Cytogenetics
    cytogenetics = tmpl.get("cytogenetics", ["Normal karyotype"])

    # Clinical presentation
    pres_template = rng.choice(tmpl["clinical_presentation_templates"])
    presentation = pres_template.format(age=age, sex=sex)

    # Sample crops
    crops = sample_crops_for_differential(
        differential=differential,
        total_crops=total_crops,
        raw_dir=raw_dir,
        rng=rng,
    )

    case_id = f"SYNTH_{template_key}_{case_idx:03d}"

    return {
        "case_id": case_id,
        "patient_profile": {
            "age": age,
            "sex": sex,
            "clinical_presentation": presentation,
        },
        "laboratory": labs,
        "cytogenetics": cytogenetics,
        "ground_truth_differential": differential,
        "ground_truth_blast_pct": actual_blast_pct,
        "apl_promyelocyte_equiv_pct": promyelocyte_blast_equiv_pct,
        "ground_truth_who5": {
            "condition_id": tmpl["condition_id"],
            "diagnosis": tmpl["condition_name"],
            "who5_category": tmpl["who5_category"],
        },
        "crop_set": crops,
    }


def generate_cases_suite(
    num_cases: int = 21,
    output_path: str = "data/synthetic/cases.json",
    raw_dir: str = "data/raw",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates a balanced suite of cases across all available templates."""
    templates = list(CASE_TEMPLATES.keys())
    cases = []

    for i in range(num_cases):
        template_key = templates[i % len(templates)]
        case = generate_single_case(
            template_key=template_key,
            case_idx=i + 1,
            total_crops=100,
            raw_dir=raw_dir,
            seed=seed,
        )
        cases.append(case)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"Generated {len(cases)} synthetic cases written to {out_file}")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic AML/MDS cases")
    parser.add_argument("--num_cases", type=int, default=21, help="Total cases to generate")
    parser.add_argument("--output", type=str, default="data/synthetic/cases.json", help="Output path")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw image dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_cases_suite(
        num_cases=args.num_cases,
        output_path=args.output,
        raw_dir=args.raw_dir,
        seed=args.seed,
    )
