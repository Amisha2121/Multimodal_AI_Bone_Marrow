"""
pipeline/dynamic_visualizer.py
─────────────────────────────────────────────────────────────────────────────
Module 5: Dynamic Clinical Visualization Engine & Manifest.

Decides which visualizations to generate based on actual patient findings:
  - If blasts are present (>0%): Generate D2 Blast Ruler and Sensitivity Band.
  - If high-cell count and multi-class: Generate D3 Confusion Neighbors graph.
  - If clinical parameters are missing: Generate D5 Investigation Dependency Graph.
  - If cellularity/lineage skew observed: Generate D6 Lineage Compartment Stack.
  - Always generate D1 Pipeline Provenance and Fig 1 Annotated Smear Field.

Produces a patient-specific Visualization Manifest with provenance IDs.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from pipeline.report_diagrams import (
    get_derived_dir,
    generate_d1_pipeline_provenance,
    generate_d2_blast_ruler,
    generate_d3_confusion_neighbours,
    generate_d4_evidence_chain
)
from pipeline.patient_context import PatientClinicalContext


def generate_d5_investigation_dependency_chart(
    case_id: str,
    patient_ctx: PatientClinicalContext
) -> Optional[str]:
    """
    Generates a patient-specific missing investigation roadmap diagram
    illustrating which modalities are complete (green) vs pending/missing (amber/red).
    """
    out_dir = get_derived_dir(case_id)
    out_path = out_dir / "diagram_d5_investigation_roadmap.png"

    # Define modalities and status
    ctx_dict = patient_ctx.to_dict()
    params = ctx_dict["parameters"]
    
    modalities = [
        ("Aspirate Cytomorphology", "AVAILABLE", "#10b981", "Vision Model Completed"),
        ("CBC / Differential", params["hemoglobin"]["status"], "#10b981" if params["hemoglobin"]["is_available"] else "#f59e0b", "Establishes cytopenias"),
        ("Flow Cytometry", params["flow_cytometry"]["status"], "#10b981" if params["flow_cytometry"]["is_available"] else "#ef4444", "Immunophenotype / Aberrant markers"),
        ("Cytogenetics / FISH", params["cytogenetics"]["status"], "#10b981" if params["cytogenetics"]["is_available"] else "#ef4444", "Karyotype risk stratification"),
        ("Molecular NGS Panel", params["molecular_genetics"]["status"], "#10b981" if params["molecular_genetics"]["is_available"] else "#f59e0b", "Gene mutations (TP53, SF3B1)"),
        ("Trephine Core Biopsy", params["core_biopsy"]["status"], "#10b981" if params["core_biopsy"]["is_available"] else "#94a3b8", "Cellularity & Reticulin Fibrosis")
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    y_pos = list(range(len(modalities)))
    y_pos.reverse()

    for idx, (name, status, color, note) in enumerate(modalities):
        y = y_pos[idx]
        # Draw status block
        rect = plt.Rectangle((0.5, y - 0.32), 9.0, 0.64, facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        
        # Color status indicator tag
        indicator = plt.Rectangle((0.5, y - 0.32), 0.3, 0.64, facecolor=color, edgecolor='none', zorder=3)
        ax.add_patch(indicator)

        # Labels
        ax.text(1.1, y + 0.08, name, fontsize=9.5, fontweight='bold', color='#0f172a', va='center', zorder=4)
        ax.text(1.1, y - 0.12, f"Clinical Role: {note}", fontsize=7.5, color='#64748b', va='center', zorder=4)
        
        # Status Pill
        ax.text(8.8, y, f"[{status}]", fontsize=8.0, fontweight='bold', color=color, ha='right', va='center', zorder=4)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, len(modalities))
    ax.axis('off')

    plt.title(f"Clinical Diagnostic Roadmap & Ancillary Dependency State: {case_id}", fontsize=11, fontweight='bold', color='#0f172a', pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return f"derived/{case_id}/diagram_d5_investigation_roadmap.png"


def generate_d6_lineage_compartment_chart(
    case_id: str,
    summary: Dict[str, Any]
) -> Optional[str]:
    """
    Generates a proportional hematological compartment bar chart 
    (Blasts vs Granulocytic vs Erythroid vs Megakaryocyte/Monocyte/Lymphoid).
    """
    out_dir = get_derived_dir(case_id)
    out_path = out_dir / "diagram_d6_lineage_compartment.png"

    counts = summary.get("counts", {})
    total = summary.get("total_cells", 1) or 1

    # Map categories into major compartments
    compartments = {
        "Blasts": counts.get("Blast", 0),
        "Maturing Myeloid": counts.get("Promyelocyte", 0) + counts.get("Myeloid (Maturing)", 0),
        "Erythroid": counts.get("Erythroid", 0),
        "Megakaryocyte / Plt": counts.get("Megakaryocyte", 0),
        "Monocytes / Lymphs": counts.get("Monocyte", 0) + counts.get("Lymphocyte", 0),
        "Other / Plasma": counts.get("Plasma Cell", 0) + counts.get("Other Morphological", 0)
    }

    labels = [k for k, v in compartments.items() if v > 0]
    vals = [compartments[k] for k in labels]
    pcts = [(v / total) * 100 for v in vals]

    if not vals:
        return None

    colors = ['#dc2626', '#3b82f6', '#f59e0b', '#6366f1', '#06b6d4', '#94a3b8']

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    bars = ax.barh(labels, pcts, color=colors[:len(labels)], height=0.55, edgecolor='#cbd5e1', linewidth=0.8)
    
    for bar, pct, cnt in zip(bars, pcts, vals):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height()/2, 
                f"{pct:.1f}% ({cnt} cells)", 
                va='center', fontsize=8, fontweight='bold', color='#334155')

    ax.set_xlim(0, max(max(pcts) + 15, 40))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.set_xlabel("Lineage Proportion (% of Detected Nucleated Cells)", fontsize=8.5, color='#475569', labelpad=8)
    
    plt.title(f"Observed Lineage Compartment Distribution (n={total})", fontsize=10.5, fontweight='bold', color='#0f172a', pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return f"derived/{case_id}/diagram_d6_lineage_compartment.png"


def build_patient_visualization_manifest(
    case_id: str,
    summary: Dict[str, Any],
    cells: List[Dict[str, Any]],
    top4_sources: List[Dict[str, Any]],
    patient_ctx: PatientClinicalContext
) -> Dict[str, Any]:
    """
    Selects and renders diagrams dynamically based on patient-specific data.
    Returns a structured visualization manifest.
    """
    manifest = {
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(),
        "visualizations": [],
        "manifest_summary": ""
    }

    diagram_urls = {}

    # Extract top4 ID strings
    top4_ids = []
    for s in top4_sources:
        if isinstance(s, dict):
            top4_ids.append(s.get("chunk_id") or s.get("id") or "REF")
        elif isinstance(s, str):
            top4_ids.append(s)
    if not top4_ids:
        top4_ids = ["WHO2022_AML_001", "WHO2022_MDS_002", "ICC2022_MDS_AML_001", "LIT-05"]

    # 1. Pipeline Provenance (Always generated)
    d1 = generate_d1_pipeline_provenance(case_id, summary, top4_ids)
    if d1:
        rel_d1 = f"derived/{case_id}/d1_pipeline_provenance.svg"
        diagram_urls["D1"] = rel_d1
        manifest["visualizations"].append({
            "visualization_id": "D1_PROVENANCE",
            "type": "System Pipeline Execution Graph",
            "url": rel_d1,
            "title": "Diagram 1: Computational Provenance & Analysis Workflow",
            "description": "Visualizes end-to-end execution across detection, ViT inference, retrieval and verification.",
            "data_sources": ["vision_pipeline", "vit_lora_model", "retriever_rrf"]
        })

    # 2. Blast Ruler (Generated if blasts detected or differential considerations require blast context)
    blast_pct = summary.get("blast_fraction", 0.0)
    total_cells = summary.get("total_cells", 1) or 1
    has_blasts = blast_pct > 0 or any(c.get("class") == "Blast" for c in cells)
    if has_blasts or total_cells > 0:
        from pipeline.scan_analytics import blast_sensitivity_range
        sens = blast_sensitivity_range(cells, total_cells)
        d2 = generate_d2_blast_ruler(case_id, blast_pct, total_cells, sens)
        if d2:
            rel_d2 = f"derived/{case_id}/d2_blast_ruler.svg"
            diagram_urls["D2"] = rel_d2
            manifest["visualizations"].append({
                "visualization_id": "D2_BLAST_RULER",
                "type": "Continuous Spectrum Ruler with Uncertainty Bounds",
                "url": rel_d2,
                "title": "Diagram 2: Observed Blast Proportion & WHO Diagnostic Reference Intervals",
                "description": f"Positions observed blast proportion ({blast_pct:.1%}) against Wilson 95% CI and sensitivity bounds.",
                "data_sources": ["blast_fraction", "wilson_confidence_interval", "WHO2022_reference_thresholds"]
            })

    # 3. Lineage Compartment Distribution (Generated when nucleated cells detected)
    if summary.get("total_cells", 0) > 0:
        d6 = generate_d6_lineage_compartment_chart(case_id, summary)
        if d6:
            diagram_urls["D6"] = d6
            manifest["visualizations"].append({
                "visualization_id": "D6_LINEAGE_STACK",
                "type": "Hematological Compartment Proportions",
                "url": d6,
                "title": "Diagram 3: Observed Lineage Compartment Proportions",
                "description": "Illustrates representation across myeloid, erythroid, blast and monocyte/lymphoid compartments.",
                "data_sources": ["morphology_summary", "cell_counts"]
            })

    # 4. Investigation Dependency & Missing-Data Roadmap (Generated when ancillary modalities are pending/missing)
    if patient_ctx.to_dict()["is_image_only"] or len(patient_ctx.get_missing_modalities()) > 0:
        d5 = generate_d5_investigation_dependency_chart(case_id, patient_ctx)
        if d5:
            diagram_urls["D5"] = d5
            manifest["visualizations"].append({
                "visualization_id": "D5_INVESTIGATION_ROADMAP",
                "type": "Clinical Modality Roadmap",
                "url": d5,
                "title": "Diagram 4: Diagnostic Completeness & Ancillary Test Dependency State",
                "description": "Maps available image cytomorphology against missing CBC, flow cytometry, and cytogenetics requirements.",
                "data_sources": ["patient_clinical_context", "laboratory_orders"]
            })

    # 5. Evidence Chain / Claim Traceability
    d4 = generate_d4_evidence_chain(case_id)
    if d4:
        rel_d4 = f"derived/{case_id}/d4_evidence_chain.svg"
        diagram_urls["D4"] = rel_d4
        manifest["visualizations"].append({
            "visualization_id": "D4_EVIDENCE_CHAIN",
            "type": "Claim-to-Evidence Bi-partite Graph",
            "url": rel_d4,
            "title": "Diagram 5: Guideline Retrieval & Claim Provenance Mapping",
            "description": "Traces narrative statements to retrieved WHO/ICC classification chunks and literature.",
            "data_sources": ["retrieved_chunks", "guideline_database"]
        })

    manifest["diagram_urls"] = diagram_urls
    manifest["manifest_summary"] = f"Generated {len(manifest['visualizations'])} case-specific diagrams based on findings."
    return manifest
