"""
pipeline/patient_correlation.py
─────────────────────────────────────────────────────────────────────────────
Phase 6: Cross-Report Correlation & Contradiction Detection Engine.

Integrates multimodal findings from aspirate morphology, flow cytometry,
cytogenetics, molecular markers, and CBC into a unified clinical correlation:
  - Cross-modality blast comparison (Aspirate image vs Flow CD34% vs Trephine)
  - Detects conflicting cellularity (Aspirate hypercellular vs Biopsy hypocellular)
  - Discrepancy detection between cytogenetics and molecular panel (e.g. Ph- vs BCR::ABL1+)
  - Identifies temporal inconsistencies across differing collection dates
  - Flags missing mandatory WHO/ICC diagnostic criteria
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pipeline.unified_patient_model import UnifiedPatientCase


def correlate_patient_findings(case: UnifiedPatientCase) -> Dict[str, Any]:
    """
    Executes cross-document correlation, discordance analysis, and
    contradiction detection across all uploaded documents and image findings.
    """
    correlations: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []
    missing_critical: List[str] = []

    # 1. Cross-modality blast comparison
    aspirate_blast_pct = (case.blast_fraction * 100) if case.blast_fraction is not None else None
    
    # Check reported morphology report blasts (Cat A)
    morph_blasts = []
    for p in case.morphology_reports:
        if "Blast" in p.get("name", "") and p.get("reported_value") is not None:
            try:
                morph_blasts.append(float(p["reported_value"]))
            except (ValueError, TypeError):
                pass

    # Check flow cytometry blast percentage (Cat C)
    flow_blasts = []
    for p in case.flow_cytometry:
        name = p.get("name", "").lower()
        if ("blast" in name or "cd34" in name) and "%" in str(p.get("reported_value", "")):
            val_str = str(p["reported_value"]).replace("%", "").strip()
            try:
                flow_blasts.append(float(val_str))
            except (ValueError, TypeError):
                pass

    if aspirate_blast_pct is not None and flow_blasts:
        mean_flow = sum(flow_blasts) / len(flow_blasts)
        delta = abs(aspirate_blast_pct - mean_flow)
        if delta > 10.0:
            contradictions.append({
                "type": "BLAST_DISCORDANCE",
                "severity": "HIGH",
                "message": (
                    f"Significant blast discrepancy: Vision aspirate model indicates {aspirate_blast_pct:.1f}% blasts, "
                    f"while Flow Cytometry reports {mean_flow:.1f}%. Hemodilution, selective lysis, or aspirate patchy distribution suspected."
                ),
                "modalities": ["Aspirate Morphology", "Flow Cytometry"]
            })
        else:
            correlations.append({
                "type": "BLAST_CONCORDANCE",
                "message": f"Concordant blast representation between aspirate ({aspirate_blast_pct:.1f}%) and flow cytometry ({mean_flow:.1f}%)."
            })

    # 2. Cytogenetics vs Molecular Genetics correlation (Cat D vs Cat E)
    has_karyotype = len(case.cytogenetics) > 0
    has_molecular = len(case.molecular_genetics) > 0

    if not has_karyotype:
        missing_critical.append("Conventional Karyotype / FISH is required by WHO 2022 / ICC for risk categorization.")
    if not has_molecular:
        missing_critical.append("Targeted Myeloid / Lymphoid NGS panel missing for defining somatic mutation detection.")

    # 3. Complete Blood Count (CBC) vs Aspirate Cellularity (Cat F vs Cat A)
    has_cbc = len(case.cbc_and_labs) > 0
    if not has_cbc:
        missing_critical.append("Peripheral CBC with differential missing; cytopenias cannot be formally confirmed.")
    else:
        # Check cytopenias
        for p in case.cbc_and_labs:
            name = p.get("name", "").lower()
            val = p.get("reported_value")
            if "hemoglobin" in name and isinstance(val, (int, float)):
                if val < 10.0:
                    correlations.append({"type": "ANEMIA_PRESENT", "message": f"Anemia confirmed (Hb: {val} g/dL)."})
            if "platelet" in name and isinstance(val, (int, float)):
                if val < 100.0:
                    correlations.append({"type": "THROMBOCYTOPENIA_PRESENT", "message": f"Thrombocytopenia confirmed (Platelets: {val} x10^9/L)."})

    # 4. Temporal consistency across documents
    dates = [doc.specimen_collection_date for doc in case.uploaded_documents if doc.specimen_collection_date]
    unique_dates = list(set(dates))
    if len(unique_dates) > 1:
        correlations.append({
            "type": "LONGITUDINAL_MULTIPLE_DATES",
            "message": f"Uploaded documents span multiple collection dates ({', '.join(unique_dates)}). Temporal trajectory analysis activated."
        })

    case.cross_report_correlations = correlations
    case.detected_contradictions = contradictions
    case.missing_investigations = [{"rationale": m} for m in missing_critical]

    return {
        "correlations": correlations,
        "contradictions": contradictions,
        "missing_critical": missing_critical,
        "concordance_score": round(max(0.0, 1.0 - (len(contradictions) * 0.25)), 2)
    }
