"""
Dataset Validation & Quality Audit Suite
========================================
Stage 3 — Medical Document AI Training Pipeline

Performs comprehensive audits on the document dataset:
1. Annotation schema conformance across all 300 JSON files.
2. PDF and TXT file pairing consistency.
3. Class balance and category coverage across all 15 categories.
4. Split partition integrity and zero data leakage across templates (Template C test isolation).
5. Verification that all synthetic documents have is_synthetic = True.
"""

import json
from pathlib import Path

REQUIRED_KEYS = [
    "report_id", "is_synthetic", "category", "subtype", "template_id",
    "institution_id", "institution_name", "patient_name", "patient_dob",
    "patient_mrn", "patient_sex", "patient_age", "collection_date",
    "report_status", "is_abnormal", "requesting_clinician",
    "reporting_pathologist", "parameters", "conclusion"
]

def validate_dataset(base_dir: Path):
    annot_dir = base_dir / "annotations"
    corpus_dir = base_dir / "synthetic_corpus"
    splits_dir = base_dir / "splits"
    manifests_dir = base_dir / "manifests"

    errors = []
    warnings = []

    annot_files = list(annot_dir.glob("*_annotation.json"))
    if len(annot_files) < 300:
        errors.append(f"Expected at least 300 annotations, found {len(annot_files)}")

    categories = set()
    templates = set()
    category_counts = {}
    template_counts = {}

    for fpath in annot_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"Corrupt JSON in {fpath.name}: {e}")
            continue

        rid = data.get("report_id")
        if not rid:
            errors.append(f"Missing report_id in {fpath.name}")
            continue

        # Check required fields
        for k in REQUIRED_KEYS:
            if k not in data:
                errors.append(f"{rid} missing required key: {k}")

        if data.get("is_synthetic") is not True:
            errors.append(f"{rid} has is_synthetic != True")

        # Check file pairing
        pdf_path = corpus_dir / f"{rid}.pdf"
        txt_path = corpus_dir / f"{rid}.txt"
        if not pdf_path.exists():
            errors.append(f"Missing paired PDF for {rid}")
        if not txt_path.exists():
            errors.append(f"Missing paired TXT for {rid}")

        cat = data.get("category", "UNKNOWN")
        tpl = data.get("template_id", "UNKNOWN")

        categories.add(cat)
        templates.add(tpl)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        template_counts[tpl] = template_counts.get(tpl, 0) + 1

        # Check parameter schema
        params = data.get("parameters", [])
        if not isinstance(params, list) or len(params) == 0:
            errors.append(f"{rid} has no parameter items")
        else:
            for p in params:
                for pk in ["name", "reported_value", "unit", "reference_interval"]:
                    if pk not in p:
                        errors.append(f"{rid} parameter missing field '{pk}'")

    if len(categories) < 15:
        errors.append(f"Expected 15 categories, found {len(categories)}")

    # Check splits manifest and leakage
    split_file = splits_dir / "split_manifest.json"
    if not split_file.exists():
        errors.append("split_manifest.json does not exist")
    else:
        with open(split_file, "r", encoding="utf-8") as sf:
            split_data = json.load(sf)

        train_ids = set(split_data["splits"]["train"])
        val_ids = set(split_data["splits"]["val"])
        test_ids = set(split_data["splits"]["test"])

        # Check disjointness
        inter_tv = train_ids.intersection(val_ids)
        inter_tt = train_ids.intersection(test_ids)
        inter_vt = val_ids.intersection(test_ids)

        if inter_tv:
            errors.append(f"Data leakage between train and val: {len(inter_tv)} overlapping IDs")
        if inter_tt:
            errors.append(f"Data leakage between train and test: {len(inter_tt)} overlapping IDs")
        if inter_vt:
            errors.append(f"Data leakage between val and test: {len(inter_vt)} overlapping IDs")

        # Verify Template C isolation
        for tid in train_ids.union(val_ids):
            fpath = annot_dir / f"{tid}_annotation.json"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as tf:
                    tdata = json.load(tf)
                if tdata.get("template_id") == "TEMPLATE_C":
                    errors.append(f"LEAKAGE VIOLATION: TEMPLATE_C found in train/val split: {tid}")

    # Write Quality Report
    quality_report = {
        "validation_status": "PASSED" if len(errors) == 0 else "FAILED",
        "total_documents": len(annot_files),
        "total_categories": len(categories),
        "categories_found": sorted(list(categories)),
        "category_distribution": category_counts,
        "template_distribution": template_counts,
        "error_count": len(errors),
        "errors": errors[:50],  # cap at 50 to avoid massive logs
        "warning_count": len(warnings),
        "warnings": warnings
    }

    report_path = manifests_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    print(f"Validation status: {quality_report['validation_status']}")
    print(f"Total documents audited: {quality_report['total_documents']}")
    print(f"Categories: {quality_report['total_categories']}/15")
    print(f"Errors: {len(errors)}, Warnings: {len(warnings)}")
    print(f"Quality report logged to: {report_path}")

    return len(errors) == 0

if __name__ == "__main__":
    success = validate_dataset(Path("data/document_ai"))
    if not success:
        exit(1)
