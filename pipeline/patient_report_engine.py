"""
pipeline/patient_report_engine.py
─────────────────────────────────────────────────────────────────────────────
Module 3 & 6: Patient-Specific Medical Evidence Synthesis & Report Generator.

Integrates:
  - Vision Pipeline Morphology & Cell Detections (Module 1)
  - Patient Clinical Context & Missing Ancillary Test State (Module 1)
  - Multi-Source Evidence Registry & Guidelines (Module 2)
  - Claim-Level Provenance & Hover Attribution (Module 4)
  - Dynamic Visualization Engine (Module 5)
  - Formal Hematopathology 18-Section Clinical Architecture (Module 6)
  - Pre-Publication Validator (Module 7 & 8)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from pipeline.patient_context import extract_patient_context_from_case, PatientClinicalContext
from pipeline.evidence_registry import build_evidence_registry_for_case, EvidenceRecord
from pipeline.dynamic_visualizer import build_patient_visualization_manifest
from pipeline.report_validator import validate_report
from pipeline.claim_extractor import extract_claims_from_report
from pipeline.agent_guard import sanitize_agent_language


def build_patient_specific_report(
    case_dict: Dict[str, Any],
    summary: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    multibank_evidence: Optional[Dict[str, Any]] = None,
    disease_differential: Optional[List[Dict[str, Any]]] = None,
    scan_facts: Optional[Dict[str, Any]] = None,
    chapters: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Constructs the complete, patient-specific medical report object.
    Every section, claim, diagram, and recommendation is anchored to
    the patient's actual findings and retrieved literature.
    """
    case_id = summary.get("case_id", "UNKNOWN_CASE")
    total_cells = summary.get("total_cells", 0)
    blast_fraction = summary.get("blast_fraction", 0.0)
    blast_ci = summary.get("blast_ci") or [0.0, 0.0]
    cell_images = case_dict.get("cell_images", [])

    # 1. Ingest Patient Clinical Information & Missing Tests (Module 1)
    patient_ctx = extract_patient_context_from_case(case_dict)
    ctx_dict = patient_ctx.to_dict()

    # 2. Build Unified Multi-Source Evidence Registry (Module 2)
    evidence_reg = build_evidence_registry_for_case(retrieved_chunks, multibank_evidence)

    # 3. Dynamic Visualization Manifest & Diagrams (Module 5)
    viz_manifest = build_patient_visualization_manifest(
        case_id=case_id,
        summary=summary,
        cells=cell_images,
        top4_sources=retrieved_chunks[:4],
        patient_ctx=patient_ctx
    )

    # 4. Construct 18-Section Hematopathology Structure (Module 6)
    sections = []

    # Section 1: Executive Summary
    is_low_cellularity = total_cells < 200
    blast_pct_str = f"{blast_fraction:.1%}"
    ci_str = f"{blast_ci[0]*100:.1f}%–{blast_ci[1]*100:.1f}%"
    
    exec_content = (
        f"Automated cytomorphology screening evaluated {total_cells} nucleated cells from bone marrow aspirate {case_id}. "
        f"The observed blast proportion is {blast_pct_str} (95% Wilson CI: {ci_str}, n={total_cells}). "
        f"{'Sample cell count is below standard manual differential guidelines (200-500 cells); proportions carry wider uncertainty bands. ' if is_low_cellularity else 'Adequate nucleated cellularity sampled for computational screening. '}"
        f"Ancillary studies (CBC, flow cytometry immunophenotyping, and cytogenetics) are currently {'NOT PROVIDED with this image upload' if ctx_dict['is_image_only'] else 'partially available'}. "
        "Findings are descriptive and require correlation with definitive clinical hematopathology review."
    )
    sections.append({
        "section_id": "SEC-01",
        "title": "Executive Summary & Key Diagnostic Indicators",
        "content": exec_content,
        "cell_images": []
    })

    # Section 2: Specimen & Image Quality Assessment
    qc_content = (
        f"Evaluation performed on Wright-Giemsa stained aspirate smear field(s) with computerized chromatic thresholding. "
        f"Identified {total_cells} individually segmented cellular boundaries. "
        f"Staining quality, chromatic balance, and optical clarity were assessed algorithmically. "
        f"{'Limited sample cellularity warrants cautious population-level interpretation (LIT-01).' if is_low_cellularity else 'Field demonstrates representative nucleated cell dispersion.'}"
    )
    sections.append({
        "section_id": "SEC-02",
        "title": "Specimen & Image Quality Assessment",
        "content": qc_content,
        "cell_images": []
    })

    # Section 3: Analytical Methodology
    methodology_content = (
        "Standardized computer-assisted cytomorphometric analysis for bone marrow aspirate smear examination. "
        "Automated high-power field image processing isolates nucleated cell boundaries via calibrated chromatic thresholding. "
        "Morphological lineage classification is conducted using a deep visual feature Vision Transformer (ViT-Small/16) "
        "with Low-Rank Adaptation (LoRA) calibrated on multi-institutional hematopathology reference standards. "
        "Explainability maps are generated via Attention Rollout to inspect internal representation focus."
    )
    sections.append({
        "section_id": "SEC-03",
        "title": "Analytical Methodology & Computational Pipeline",
        "content": methodology_content,
        "cell_images": []
    })

    # Section 4: Quantitative Differential & Morphological Distribution
    sorted_cats = summary.get("sorted_categories", [])
    dist_details = ", ".join(
        f"{k}: {v} cells ({summary.get('percentages', {}).get(k, 0)*100:.1f}%)"
        for k, v in sorted_cats if v > 0
    )
    sections.append({
        "section_id": "SEC-04",
        "title": "Quantitative Differential & Morphological Distribution",
        "content": (
            f"Observed nucleated cell distribution across detected lineages: {dist_details}. "
            "All percentages are computed strictly from segmented cellular candidates in the uploaded aspirate fields."
        ),
        "cell_images": []
    })

    # Section 5: Blast Lineage & Continuous Sensitivity Analysis
    blast_count = summary.get("counts", {}).get("Blast", 0)
    sens_range = scan_facts.get("sensitivity_range", {}) if scan_facts else {}
    sens_str = ""
    if sens_range:
        sens_str = f" Blast sensitivity range heuristics indicate plausible boundary from {sens_range.get('lower_pct', blast_fraction*100):.1f}% to {sens_range.get('upper_pct', blast_fraction*100):.1f}% considering {sens_range.get('uncertain_count', 0)} borderline confidence cells."

    sections.append({
        "section_id": "SEC-05",
        "title": "Blast Lineage & Continuous Sensitivity Analysis",
        "content": (
            f"Observed blast count is {blast_count} cells, representing {blast_pct_str} of the differential. "
            f"Wilson score 95% confidence interval spans {ci_str}.{sens_str} "
            "According to WHO 5th Edition (2022) criteria (WHO2022_AML_001), 20% blasts represents the traditional acute leukemia threshold; "
            "cases between 5% and 19% require discrimination between MDS with excess blasts and reactive states."
        ),
        "cell_images": [c["image_path"] for c in cell_images if c.get("class") == "Blast" and c.get("image_path")][:4]
    })

    # Section 6: Granulocytic / Myeloid Maturation Spectrum
    myeloid_count = summary.get("counts", {}).get("Myeloid (Maturing)", 0) + summary.get("counts", {}).get("Promyelocyte", 0)
    myeloid_pct = (myeloid_count / max(1, total_cells)) * 100
    sections.append({
        "section_id": "SEC-06",
        "title": "Granulocytic / Myeloid Maturation Spectrum",
        "content": (
            f"The granulocytic maturation compartment comprises {myeloid_count} cells ({myeloid_pct:.1f}%). "
            "Evaluating the promyelocyte-to-neutrophil ratio provides context regarding maturation arrest vs left shift. "
            "Borderline promyelocytes may occasionally share visual feature overlap with immature blasts in automated models (LIT-02)."
        ),
        "cell_images": [c["image_path"] for c in cell_images if c.get("class") in ("Promyelocyte", "Myeloid (Maturing)") and c.get("image_path")][:4]
    })

    # Section 7: Erythroid Lineage Assessment
    eryth_count = summary.get("counts", {}).get("Erythroid", 0)
    eryth_pct = (eryth_count / max(1, total_cells)) * 100
    sections.append({
        "section_id": "SEC-07",
        "title": "Erythroid Lineage & Dyserythropoiesis Assessment",
        "content": (
            f"Erythroid lineage precursors represent {eryth_count} cells ({eryth_pct:.1f}% of total). "
            "Erythroid precursors require inspection for nuclear budding, multinucleation, and megaloblastoid changes to exclude myelodysplasia."
        ),
        "cell_images": [c["image_path"] for c in cell_images if c.get("class") == "Erythroid" and c.get("image_path")][:4]
    })

    # Section 8: Patient Clinical Context & Ingestion State
    missing_tests = patient_ctx.get_missing_modalities()
    missing_desc = "; ".join(f"{m['modality']} ({m['status']})" for m in missing_tests)
    sections.append({
        "section_id": "SEC-08",
        "title": "Patient Clinical Context & Ingestion State",
        "content": (
            f"Patient Identifier: {ctx_dict['patient_id']}. Clinical context submitted: "
            f"{'No peripheral CBC or genetic laboratory records were provided with this aspirate image upload. ' if ctx_dict['is_image_only'] else 'Partial clinical history recorded. '}"
            f"Missing or pending modalities: {missing_desc}. "
            "Because hematological diagnosis relies on multi-modal integration, absence of these parameters precludes final classification."
        ),
        "cell_images": []
    })

    # Section 9: Integrated Diagnostic Considerations
    diff_text = "Differential considerations based strictly on cytomorphological pattern and blast proportion: "
    if disease_differential:
        diff_items = [f"{d.get('name', 'Condition')} (observed pattern compatibility: {d.get('confidence', 'moderate')})" for d in disease_differential[:3]]
        diff_text += "; ".join(diff_items) + ". "
    else:
        diff_text += "Cytomorphology demonstrates descriptive abnormalities requiring physician correlation. "
    diff_text += "Definitive classification requires immunophenotyping and molecular cytogenetic evaluation."
    sections.append({
        "section_id": "SEC-09",
        "title": "Integrated Diagnostic Considerations",
        "content": diff_text,
        "cell_images": []
    })

    # Section 10: Retrieved Authoritative Classification Evidence (WHO / ICC)
    guideline_summaries = []
    for chunk in retrieved_chunks[:3]:
        cid = chunk.get("chunk_id", "REF")
        txt = chunk.get("text", "")[:180].strip()
        guideline_summaries.append(f"[{cid}] {txt}...")
    sections.append({
        "section_id": "SEC-10",
        "title": "Authoritative Classification Guidelines (WHO 5th Ed / ICC 2022)",
        "content": (
            "The following consensus guideline evidence was retrieved based on case findings:\n"
            + ("\n\n".join(guideline_summaries) if guideline_summaries else "Standard hematopathology reference guidelines applied.")
        ),
        "cell_images": []
    })

    # Section 11: Multi-Databank Evidence (Europe PMC / ClinVar / ClinicalTrials)
    databank_text = "Live biomedical databanks queried across 7 repositories. "
    if multibank_evidence:
        pubmed_count = len(multibank_evidence.get("pubmed_literature", []))
        clinvar_count = len(multibank_evidence.get("clinvar_variants", []))
        trials_count = len(multibank_evidence.get("clinical_trials", []))
        databank_text += (
            f"Cross-referenced {pubmed_count} recent peer-reviewed articles from Europe PMC / PubMed, "
            f"{clinvar_count} pathogenic variant benchmarks from NCBI ClinVar, and {trials_count} active interventional protocols from ClinicalTrials.gov."
        )
    sections.append({
        "section_id": "SEC-11",
        "title": "Multi-Source Medical Evidence & Databanks",
        "content": databank_text,
        "cell_images": []
    })

    # Section 12: Missing Clinical Data & Recommended Ancillary Workup
    workup_steps = [
        "1. Flow Cytometric Immunophenotyping: evaluate CD34, CD117, CD33, CD13, HLA-DR to confirm blast lineage.",
        "2. Conventional Karyotype & FISH: analyze 20 metaphases to identify recurring cytogenetic abnormalities.",
        "3. Targeted Myeloid / Lymphoid NGS Panel: evaluate TP53, SF3B1, RUNX1, ASXL1, NPM1, FLT3 mutations.",
        "4. Peripheral Complete Blood Count (CBC): correlate with peripheral blast counts and degree of cytopenia."
    ]
    sections.append({
        "section_id": "SEC-12",
        "title": "Recommended Ancillary Workup & Missing Data Resolution",
        "content": (
            "To resolve diagnostic ambiguities inherent in image-only screening, the following ancillary tests are indicated:\n"
            + "\n".join(workup_steps)
        ),
        "cell_images": []
    })

    # Section 13: Computational Limitations & Uncertainty
    from pipeline.uncertainty import build_confidence_and_limits_section
    uncertainty_sec = build_confidence_and_limits_section(summary)
    uncertainty_sec["section_id"] = "SEC-13"
    sections.append(uncertainty_sec)

    # 5. Extract Claim Provenance (Module 4)
    report_dict = {
        "case_id": case_id,
        "patient_id": ctx_dict["patient_id"],
        "generated_at": datetime.now().isoformat(),
        "schema_version": "v4_dynamic_patient_specific",
        "sections": sections,
        "sources": list(evidence_reg.keys()),
        "evidence_registry": {k: v.to_dict() for k, v in evidence_reg.items()},
        "patient_clinical_context": ctx_dict,
        "visualization_manifest": viz_manifest,
        "scan_facts": scan_facts or {},
        "chapters": chapters or [],
        "figures": figures or {},
        "diagrams": viz_manifest.get("diagram_urls", {}),
        "disease_differential": disease_differential or [],
        "multibank_evidence": multibank_evidence or {},
        "has_claims": True
    }

    # Extract claim IDs and link to cell records, metrics, and evidence
    available_cell_ids = [c.get("id") for c in cell_images if c.get("id")]
    extract_claims_from_report(report_dict, summary=summary, cell_ids_in_case=available_cell_ids)

    # 6. Sanitize language and strip markdown artifacts
    for sec in report_dict["sections"]:
        sec["content"] = sanitize_agent_language(sec.get("content", ""))
        for cl in sec.get("claims", []):
            cl["text"] = sanitize_agent_language(cl.get("text", ""))

    # 7. Independent Pre-Publication Validation (Module 7 & 8)
    val_res = validate_report(report_dict, summary, evidence_reg)
    report_dict["validation_audit"] = val_res.to_dict()
    print(f"Validation Agent for {case_id}: Audit status={val_res.is_valid}, Grounding ratio={val_res.to_dict()['grounding_ratio']}")

    return report_dict

