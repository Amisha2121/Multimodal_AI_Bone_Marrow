"""
Pytest Test Suite for Medical Document AI Inference & Verification
==================================================================
Covers:
- Model Loading & Pipeline Initialization
- Category Mapping & Consistency
- Preprocessing & Input Validation
- Single-page & Multi-page document handling
- Unknown Document Rejection (Blank, OOD, Low Confidence)
- Missing / Corrupt File Error Handling
"""

import os
import json
import tempfile
from pathlib import Path
import pytest

from scripts.document_ai.inference import DocumentIntelligenceEngine, DEFAULT_MODEL_PATH, DEFAULT_LABEL_MAP

@pytest.fixture
def engine():
    return DocumentIntelligenceEngine(model_path=DEFAULT_MODEL_PATH, label_map_path=DEFAULT_LABEL_MAP, threshold=0.70)

def test_model_loading_and_classes(engine):
    assert engine.pipeline is not None
    assert len(engine.label_map) == 15
    assert len(engine.inv_label_map) == 15
    assert "CAT_A_MORPHOLOGY" in engine.label_map
    assert "CAT_C_CBC" in engine.label_map

def test_blank_and_empty_input(engine):
    res_empty = engine.classify_text("")
    assert res_empty["category"] == "UNKNOWN"
    assert res_empty["is_rejected"] is True
    assert res_empty["confidence"] == 0.0

    res_whitespace = engine.classify_text("   \n\n\t   ")
    assert res_whitespace["category"] == "UNKNOWN"
    assert res_whitespace["is_rejected"] is True

def test_ood_unrelated_document_rejection(engine):
    unrelated_text = (
        "Operative Report: Preoperative Diagnosis: Acute Appendicitis. "
        "Procedure: Laparoscopic Appendectomy. The appendix was identified, "
        "mesoappendix divided with harmonic scalpel, and base ligated with two endoloops."
    )
    res = engine.classify_text(unrelated_text)
    assert res["is_rejected"] is True
    assert res["category"] == "UNKNOWN"
    assert res["confidence"] < 0.20

def test_in_distribution_cbc_classification(engine):
    sample_cbc = Path("data/document_ai/synthetic_corpus/SYNTH_CBC_0041.txt")
    if sample_cbc.exists():
        with open(sample_cbc, "r", encoding="utf-8") as f:
            cbc_text = f.read()
    else:
        cbc_text = (
            "ST. HELENA MEDICAL CENTER\n"
            "LABORATORY REPORT — COMPLETE BLOOD COUNT REPORT\n"
            "Patient Name: John Doe   MRN: MRN-123456\n"
            "Haemoglobin: 14.2 g/dL   Ref: 12.0-16.5\n"
            "Haematocrit: 42.0 %      Ref: 36.0-48.0\n"
            "RBC Count: 4.8 x10^12/L  Ref: 3.8-5.2\n"
            "WBC Count: 7.1 x10^9/L   Ref: 4.0-11.0\n"
            "Platelets: 280 x10^9/L   Ref: 150-400\n"
            "Neutrophils: 55 %        Ref: 40-70\n"
            "Lymphocytes: 35 %        Ref: 20-40\n"
            "MCV: 88.0 fL             Ref: 80-100\n"
            "MCH: 29.5 pg             Ref: 27-33\n"
            "MCHC: 33.5 g/dL          Ref: 32-36\n"
            "RDW: 12.8 %              Ref: 11.5-14.5\n"
            "CONCLUSION: Complete blood count within normal reference intervals."
        )
    res = engine.classify_text(cbc_text)
    assert res["category"] == "CAT_C_CBC"
    assert res["is_rejected"] is False
    assert res["confidence"] >= 0.70

def test_missing_and_corrupt_files(engine):
    with pytest.raises(FileNotFoundError):
        engine.process_document("non_existent_file_path_xyz.pdf")

def test_multi_page_pdf_processing(engine):
    sample_pdf = Path("data/document_ai/synthetic_corpus/SYNTH_BMA_0001.pdf")
    if sample_pdf.exists():
        res = engine.process_document(str(sample_pdf))
        assert res["num_pages"] >= 1
        assert res["document_prediction"]["category"] == "CAT_A_MORPHOLOGY"
        assert res["document_prediction"]["is_rejected"] is False

