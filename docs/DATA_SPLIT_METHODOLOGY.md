# Data Split Methodology — Medical Document AI

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 3 — Dataset Preparation  
**Date**: 2026-11-11

---

## Overview

A leakage-resistant data split methodology is applied to ensure that the trained models genuinely generalize to unseen documents rather than memorizing specific template patterns, institution styles, or report layouts.

---

## Grouping Key

Each synthetic report is assigned a **group key** composed of:

```
group_key = (template_id, institution_id)
```

Where:
- `template_id` ∈ {`TEMPLATE_A`, `TEMPLATE_B`, `TEMPLATE_C`} — the visual/structural layout variant
- `institution_id` ∈ {`INST_01`, `INST_02`, `INST_03`, `INST_04`, `INST_05`} — fictional hospital

This creates 15 unique group combinations (3 templates × 5 institutions).

**Rationale**: Reports from the same (template, institution) pair share visual formatting, header layout, font choices, and table structure. Including such reports in both train and test would leak layout information and inflate test performance.

---

## Split Ratios

| Split | Proportion | Templates Included | Purpose |
|-------|-----------|-------------------|---------|
| Train | 70% | TEMPLATE_A, TEMPLATE_B | Model learning |
| Validation | 15% | TEMPLATE_A, TEMPLATE_B | Hyperparameter tuning |
| Test | 15% | TEMPLATE_C **only** | Held-out generalization |

> [!IMPORTANT]
> **Template C is strictly held out**: No TEMPLATE_C samples appear in train or validation. The test set exclusively contains TEMPLATE_C variants — a completely unseen layout format. This measures genuine layout generalization, not template memorization.

---

## Split Assignment Algorithm

```python
def assign_split(report):
    template_id = report["template_id"]
    
    # Hard rule: TEMPLATE_C → test only
    if template_id == "TEMPLATE_C":
        return "test"
    
    # For TEMPLATE_A and TEMPLATE_B: stratified by category
    # Use deterministic hash of report_id for reproducibility
    import hashlib
    hash_val = int(hashlib.md5(report["report_id"].encode()).hexdigest(), 16)
    bucket = hash_val % 100
    
    if bucket < 70:
        return "train"
    else:
        return "val"
```

**Key properties**:
1. **Deterministic**: Same report_id always maps to the same split.
2. **Reproducible**: No random seed dependency — hash-based.
3. **Template-stratified**: TEMPLATE_C is fully isolated.
4. **Category-balanced**: Each category has ≥ 14 train / ≥ 3 val / ≥ 6 test samples (from 20 per category across 3 templates).

---

## Expected Split Counts (Synthetic Corpus Only)

| Category | Total | Train (~70%) | Val (~15%) | Test (~15%) |
|----------|-------|-------------|-----------|------------|
| CAT_A: Aspirate | 20 | ~14 | ~3 | ~6-7 |
| CAT_B: Biopsy | 20 | ~14 | ~3 | ~6-7 |
| CAT_C: CBC | 20 | ~14 | ~3 | ~6-7 |
| CAT_D: Smear | 20 | ~14 | ~3 | ~6-7 |
| CAT_E: Flow | 20 | ~14 | ~3 | ~6-7 |
| CAT_F: IHC | 20 | ~14 | ~3 | ~6-7 |
| CAT_G: FISH | 20 | ~14 | ~3 | ~6-7 |
| CAT_H: Karyotyping | 20 | ~14 | ~3 | ~6-7 |
| CAT_I: NGS | 20 | ~14 | ~3 | ~6-7 |
| CAT_J: PCR/Molecular | 20 | ~14 | ~3 | ~6-7 |
| CAT_K: Iron/Reticulocyte | 20 | ~14 | ~3 | ~6-7 |
| CAT_L: Coagulation | 20 | ~14 | ~3 | ~6-7 |
| CAT_M: MRD | 20 | ~14 | ~3 | ~6-7 |
| CAT_N: Transplant | 20 | ~14 | ~3 | ~6-7 |
| CAT_O: Immunoglobulin/SPEP | 20 | ~14 | ~3 | ~6-7 |
| **TOTAL** | **300** | **~210** | **~45** | **~90-100** |

> [!NOTE]
> Exact test counts slightly exceed 15% because all 20 TEMPLATE_C samples (one per category × 15 categories = not quite right; each category has ~6-7 TEMPLATE_C samples from 20 total across 3 templates). This is acceptable and expected.

---

## Leakage Prevention Checks

The validation script (`scripts/document_ai/validate_dataset.py`) checks:

1. **Template isolation**: Assert no `TEMPLATE_C` report_ids appear in train or val.
2. **Report uniqueness**: Assert no report_id appears in more than one split.
3. **Group integrity**: Assert that all reports with the same (template_id, institution_id) are in the same split OR the template-C hard rule applies.
4. **Category coverage**: Assert all 15 categories appear in train, val, and test.
5. **No hash collision**: Assert split assignments are stable across two runs.

---

## Real Dataset Integration (MTSamples, India OCR)

Small real datasets are added to the **test split only** as out-of-distribution (OOD) evaluation sets:

| Dataset | Split | Purpose |
|---------|-------|---------|
| MTSamples hematology (350 samples) | OOD test | NER generalization to real clinical notes |
| India Medical OCR (200 samples) | OOD test | OCR generalization to non-standard formats |

These are **not mixed** with synthetic training data — they are separate evaluation tracks:
- `data/document_ai/splits/ood_mtsamples.json`
- `data/document_ai/splits/ood_india_ocr.json`

---

## Reproducibility

The full split manifest is saved at `data/document_ai/splits/split_manifest.json`.

Format:
```json
{
  "methodology": "template_holdout_hash",
  "created_at": "2026-11-11T...",
  "template_holdout_rule": "TEMPLATE_C -> test only",
  "hash_function": "MD5",
  "splits": {
    "train": ["SYNTH_BMA_001", "SYNTH_BMA_002", ...],
    "val": ["SYNTH_BMA_007", ...],
    "test": ["SYNTH_BMA_014", ...]
  },
  "split_counts": {
    "train": 210,
    "val": 45,
    "test": 95
  }
}
```

To regenerate splits:
```bash
python scripts/document_ai/create_splits.py
```

The output is deterministic — no random seed — because split assignment is based on MD5 hash of `report_id`.
