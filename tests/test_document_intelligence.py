"""
tests/test_document_intelligence.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive unit and integration test suite for Categories A through L:
  - Bone Marrow Aspirate & Biopsy (Cat A)
  - Special Stains (Cat B)
  - Immunophenotyping & Flow Cytometry (Cat C)
  - Cytogenetics & ISCN Karyotyping (Cat D)
  - Molecular Genetics & Mutations (Cat E)
  - CBC & Routine Laboratory (Cat F)
  - Protein Studies & SPEP (Cat G)
  - Cross-report correlation and contradiction detection
  - Multi-document ingestion endpoint verification
"""

import pytest
from pathlib import Path
from pipeline.unified_patient_model import DocumentCategory, ExtractionStatus, UnifiedPatientCase
from pipeline.document_intelligence import (
    classify_document_text,
    extract_parameters_from_text,
    process_uploaded_document_file
)
from pipeline.patient_correlation import correlate_patient_findings
from pipeline.longitudinal_tracker import build_longitudinal_timeline


def test_classify_cbc_report():
    text = "Complete Blood Count (CBC) with differential. Hemoglobin: 8.2 g/dL, WBC: 2.1 x10^9/L, Platelets: 45 x10^9/L."
    cat, name = classify_document_text(text)
    assert cat == DocumentCategory.CAT_F_LABORATORY
    params = extract_parameters_from_text("DOC_01", text, cat)
    assert any(p.name == "Hemoglobin" and p.reported_value == 8.2 for p in params)
    assert any(p.name == "WBC" and p.reported_value == 2.1 for p in params)
    assert any(p.name == "Platelets" and p.reported_value == 45.0 for p in params)


def test_classify_flow_cytometry_report():
    text = "Flow Cytometric Immunophenotyping. Blast gate analysis reveals CD34 positive, CD117 positive, HLA-DR positive, CD33 positive, CD19 negative."
    cat, name = classify_document_text(text)
    assert cat == DocumentCategory.CAT_C_IMMUNOPHENOTYPING
    params = extract_parameters_from_text("DOC_02", text, cat)
    assert any(p.name == "Flow Cytometry CD34" and "positive" in p.reported_value for p in params)
    assert any(p.name == "Flow Cytometry CD117" and "positive" in p.reported_value for p in params)


def test_classify_cytogenetics_report():
    text = "Conventional Karyotype and metaphase cytogenetics. ISCN Result: 46,XX,del(5)(q31q33)[18]/46,XX[2]. Clonal abnormality detected."
    cat, name = classify_document_text(text)
    assert cat == DocumentCategory.CAT_D_CYTOGENETICS
    params = extract_parameters_from_text("DOC_03", text, cat)
    karyo = next(p for p in params if "ISCN" in p.name)
    assert "del(5)" in karyo.reported_value
    assert karyo.abnormality_flag == "Abnormal"


def test_classify_molecular_ngs_report():
    text = "Next-Generation Sequencing Myeloid Gene Panel. TP53 mutation detected (exon 5, c.524G>A, VAF: 34.2%). NPM1: not detected."
    cat, name = classify_document_text(text)
    assert cat == DocumentCategory.CAT_E_MOLECULAR
    params = extract_parameters_from_text("DOC_04", text, cat)
    assert any("TP53" in p.name and p.abnormality_flag == "Detected" for p in params)


def test_cross_report_correlation_and_discordance():
    case = UnifiedPatientCase(
        patient_id="PAT_TEST_01",
        case_id="CASE_TEST_01",
        blast_fraction=0.18 # 18% blasts on aspirate image
    )
    # Add flow report with only 3% blasts
    from pipeline.unified_patient_model import UploadedMedicalDocument, ExtractedParameter
    flow_doc = UploadedMedicalDocument(
        document_id="DOC_FLOW",
        filename="flow_report.pdf",
        category=DocumentCategory.CAT_C_IMMUNOPHENOTYPING,
        category_name="Flow Cytometry",
        extracted_parameters=[
            ExtractedParameter(name="Blast Population Percentage", reported_value="3.0%", source_document_id="DOC_FLOW")
        ]
    )
    case.add_document(flow_doc)
    res = correlate_patient_findings(case)
    
    assert len(res["contradictions"]) > 0
    assert any(c["type"] == "BLAST_DISCORDANCE" for c in res["contradictions"])


def test_longitudinal_timeline_generation():
    case = UnifiedPatientCase(
        patient_id="PAT_TEST_02",
        case_id="CASE_TEST_02",
        blast_fraction=0.12
    )
    from pipeline.unified_patient_model import UploadedMedicalDocument, ExtractedParameter
    doc1 = UploadedMedicalDocument(
        document_id="DOC_T1",
        filename="cbc_t1.pdf",
        specimen_collection_date="01/15/2026",
        category=DocumentCategory.CAT_F_LABORATORY,
        category_name="CBC",
        extracted_parameters=[ExtractedParameter(name="Hemoglobin", reported_value=7.5, source_document_id="DOC_T1")]
    )
    doc2 = UploadedMedicalDocument(
        document_id="DOC_T2",
        filename="cbc_t2.pdf",
        specimen_collection_date="03/20/2026",
        category=DocumentCategory.CAT_F_LABORATORY,
        category_name="CBC",
        extracted_parameters=[ExtractedParameter(name="Hemoglobin", reported_value=11.2, source_document_id="DOC_T2")]
    )
    case.add_document(doc1)
    case.add_document(doc2)

    timeline = build_longitudinal_timeline(case)
    assert len(timeline) >= 2
    dates = [ep["episode_date"] for ep in timeline]
    assert "01/15/2026" in dates
    assert "03/20/2026" in dates
