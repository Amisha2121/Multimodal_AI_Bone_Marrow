"""
Pytest Test Suite for Medical Document AI Preparation (Stage 3)
===============================================================
"""

import json
from pathlib import Path
import pytest

BASE_DIR = Path("data/document_ai")
ANNOTATIONS_DIR = BASE_DIR / "annotations"
CORPUS_DIR = BASE_DIR / "synthetic_corpus"
SPLITS_DIR = BASE_DIR / "splits"
MANIFESTS_DIR = BASE_DIR / "manifests"

def test_annotation_files_count_and_format():
    files = list(ANNOTATIONS_DIR.glob("*_annotation.json"))
    assert len(files) >= 300, f"Expected at least 300 annotation files, got {len(files)}"

def test_paired_document_files():
    files = list(ANNOTATIONS_DIR.glob("*_annotation.json"))
    for f in files[:25]:  # verify sample of files
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        rid = data["report_id"]
        assert (CORPUS_DIR / f"{rid}.pdf").exists(), f"Missing PDF for {rid}"
        assert (CORPUS_DIR / f"{rid}.txt").exists(), f"Missing TXT for {rid}"

def test_required_annotation_schema_and_synthetic_flag():
    required_keys = [
        "report_id", "is_synthetic", "category", "subtype", "template_id",
        "institution_id", "patient_name", "parameters", "conclusion"
    ]
    for f in list(ANNOTATIONS_DIR.glob("*_annotation.json"))[:30]:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert data.get("is_synthetic") is True, f"{data['report_id']} is_synthetic != True"
        for k in required_keys:
            assert k in data, f"{data['report_id']} missing required key: {k}"
        assert len(data["parameters"]) > 0, f"{data['report_id']} has empty parameters"

def test_split_manifest_and_leakage_resistance():
    split_file = SPLITS_DIR / "split_manifest.json"
    assert split_file.exists(), "split_manifest.json missing"
    with open(split_file, "r", encoding="utf-8") as fp:
        split_data = json.load(fp)

    train_ids = set(split_data["splits"]["train"])
    val_ids = set(split_data["splits"]["val"])
    test_ids = set(split_data["splits"]["test"])

    # Disjointness
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap"

    # Template C holdout
    for rid in train_ids.union(val_ids):
        fpath = ANNOTATIONS_DIR / f"{rid}_annotation.json"
        with open(fpath, "r", encoding="utf-8") as af:
            ann = json.load(af)
        assert ann.get("template_id") != "TEMPLATE_C", f"Leakage: TEMPLATE_C found in train/val {rid}"

def test_quality_report_passed():
    q_file = MANIFESTS_DIR / "quality_report.json"
    assert q_file.exists(), "quality_report.json missing"
    with open(q_file, "r", encoding="utf-8") as fp:
        q_data = json.load(fp)
    assert q_data["validation_status"] == "PASSED"
    assert q_data["error_count"] == 0
    assert q_data["total_categories"] == 15
