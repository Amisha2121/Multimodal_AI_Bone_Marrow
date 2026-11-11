"""
tests/test_patient_report_engine.py
─────────────────────────────────────────────────────────────────────────────
Automated evaluation test suite for Modules 1-10:
  - Tests 2 distinct patient cases with differing cellularity and blast percentages.
  - Verifies claim provenance, source attribution, and evidence registry schema.
  - Verifies dynamic visualization manifest generation without fabricated data.
  - Verifies missing-data management and safety language guard.
  - Verifies pre-publication validation audit passes.
"""

import pytest
from pipeline.summary import build_morphology_summary
from pipeline.patient_context import extract_patient_context_from_case, DataStatus
from pipeline.evidence_registry import build_evidence_registry_for_case, EvidenceTier
from pipeline.dynamic_visualizer import build_patient_visualization_manifest
from pipeline.patient_report_engine import build_patient_specific_report
from pipeline.report_validator import validate_report


@pytest.fixture
def sample_case_low_blasts():
    """Case with low blasts (8.3%) and small sample size (12 cells)."""
    cell_images = [
        {"id": f"CELL_LOW_{i}", "class": "Blast" if i == 0 else "Myeloid (Maturing)", "confidence": 0.85, "image_path": f"data/uploads/cell_low_{i}.png"}
        for i in range(12)
    ]
    return {
        "case_id": "CASE-CECDBCC7",
        "id": "CASE-CECDBCC7",
        "total_cells": 12,
        "cell_images": cell_images,
        "metrics": {
            "blast_fraction": 0.0833,
            "cells_detected": 12,
            "cells_classified": 12,
            "blast_ci": [0.015, 0.354]
        }
    }


@pytest.fixture
def sample_case_moderate_blasts():
    """Case with moderate blasts (10.2%) and 49 cells."""
    cell_images = []
    for i in range(49):
        cls_name = "Blast" if i < 5 else ("Myeloid (Maturing)" if i < 35 else "Erythroid")
        cell_images.append({
            "id": f"CELL_MOD_{i}",
            "class": cls_name,
            "confidence": 0.90 if i < 5 else 0.75,
            "image_path": f"data/uploads/cell_mod_{i}.png"
        })
    return {
        "case_id": "CASE-F09AFAC4",
        "id": "CASE-F09AFAC4",
        "total_cells": 49,
        "cell_images": cell_images,
        "metrics": {
            "blast_fraction": 0.102,
            "cells_detected": 49,
            "cells_classified": 49,
            "blast_ci": [0.044, 0.218]
        }
    }


def test_missing_data_management(sample_case_low_blasts):
    """Module 1: Missing data must be explicitly tracked without fabrication."""
    ctx = extract_patient_context_from_case(sample_case_low_blasts)
    ctx_dict = ctx.to_dict()
    
    assert ctx_dict["is_image_only"] is True
    assert ctx_dict["has_cbc"] is False
    assert ctx_dict["parameters"]["hemoglobin"]["status"] == DataStatus.NOT_PROVIDED.value
    assert len(ctx_dict["missing_modalities"]) >= 4
    assert any("Flow" in m["modality"] for m in ctx_dict["missing_modalities"])


def test_evidence_hierarchy_ranking():
    """Module 2: Evidence registry establishes explicit evidence tiers."""
    chunks = [{"chunk_id": "WHO2022_AML_001", "score": 0.96}]
    multibank = {
        "pubmed_literature": [{"id": "3801234", "title": "MDS Differential Studies", "year": "2024", "snippet": "Study data"}],
        "clinvar_variants": [{"id": "12345", "title": "TP53 c.524G>A", "snippet": "Pathogenic"}]
    }
    reg = build_evidence_registry_for_case(chunks, multibank)
    
    assert "WHO2022_AML_001" in reg
    assert reg["WHO2022_AML_001"].tier == EvidenceTier.TIER_1_OFFICIAL_CLASSIFICATION
    
    assert "PMID_3801234" in reg
    assert reg["PMID_3801234"].tier == EvidenceTier.TIER_4_PEER_REVIEWED_RESEARCH

    assert "CLINVAR_12345" in reg
    assert reg["CLINVAR_12345"].tier == EvidenceTier.TIER_5_GENOMIC_DATABANK


def test_dynamic_patient_visualization_manifest(sample_case_moderate_blasts):
    """Module 5: Generates patient-specific visualization manifest."""
    summary = build_morphology_summary("CASE-F09AFAC4", case_dict=sample_case_moderate_blasts)
    patient_ctx = extract_patient_context_from_case(sample_case_moderate_blasts)
    
    manifest = build_patient_visualization_manifest(
        case_id="CASE-F09AFAC4",
        summary=summary,
        cells=sample_case_moderate_blasts["cell_images"],
        top4_sources=[{"chunk_id": "WHO2022_AML_001"}],
        patient_ctx=patient_ctx
    )
    
    assert len(manifest["visualizations"]) >= 3
    viz_ids = [v["visualization_id"] for v in manifest["visualizations"]]
    assert "D1_PROVENANCE" in viz_ids
    assert "D2_BLAST_RULER" in viz_ids
    assert "D5_INVESTIGATION_ROADMAP" in viz_ids


def test_report_synthesis_and_validation(sample_case_low_blasts, sample_case_moderate_blasts):
    """Module 3, 6, 7 & 8: Generates two distinct reports and audits validity."""
    cases = [sample_case_low_blasts, sample_case_moderate_blasts]
    reports = []

    for c in cases:
        summary = build_morphology_summary(c["case_id"], case_dict=c)
        rep = build_patient_specific_report(
            case_dict=c,
            summary=summary,
            retrieved_chunks=[{"chunk_id": "WHO2022_AML_001", "text": ">=20% blast threshold"}],
            multibank_evidence={},
            disease_differential=[]
        )
        reports.append(rep)
        
        # Validation checks
        assert rep["validation_audit"]["is_valid"] is True
        assert rep["validation_audit"]["grounding_ratio"] >= 0.90
        assert len(rep["sections"]) >= 10
        assert rep["has_claims"] is True
        assert rep["patient_clinical_context"]["case_id"] == c["case_id"]

    # Assert patient specificity: blast % and total cell numbers must differ between the reports
    r1, r2 = reports[0], reports[1]
    assert r1["case_id"] != r2["case_id"]
    assert "8.3%" in r1["sections"][0]["content"]
    assert "10.2%" in r2["sections"][0]["content"]
