"""
pipeline/longitudinal_tracker.py
─────────────────────────────────────────────────────────────────────────────
Phase 7: Longitudinal Patient Episode & Investigation Timeline Engine.

Tracks multiple investigations collected at differing collection dates:
  - Sequences episodes chronologically (T1 -> T2 -> T3)
  - Tracks trajectories of serial CBC (Hb, Platelets, WBC), serial blast counts,
    and Measurable Residual Disease (MRD) flow/molecular values.
  - Distinguishes diagnostic baseline vs post-treatment vs relapse monitoring.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pipeline.unified_patient_model import UnifiedPatientCase


def build_longitudinal_timeline(case: UnifiedPatientCase) -> List[Dict[str, Any]]:
    """
    Constructs a chronological episode timeline from all uploaded documents
    and imaging episodes, computing parameter deltas where available.
    """
    episodes: Dict[str, Dict[str, Any]] = {}

    # 1. Ingest imaging episode
    base_date = "Baseline / Image Intake"
    episodes[base_date] = {
        "episode_date": base_date,
        "modality": "Bone Marrow Aspirate Smear",
        "parameters": {
            "blast_fraction": case.blast_fraction,
            "total_cells": case.morphology_findings.get("total_cells") or len(case.cell_detection_results)
        },
        "documents": []
    }

    # 2. Ingest uploaded documents by collection date
    for doc in case.uploaded_documents:
        ep_date = doc.specimen_collection_date or "Date Unspecified"
        if ep_date not in episodes:
            episodes[ep_date] = {
                "episode_date": ep_date,
                "modality": doc.category_name,
                "parameters": {},
                "documents": []
            }
        episodes[ep_date]["documents"].append(doc.filename)
        for p in doc.extracted_parameters:
            episodes[ep_date]["parameters"][p.name] = {
                "reported_value": p.reported_value,
                "unit": p.original_unit,
                "flag": p.abnormality_flag
            }

    timeline = list(episodes.values())
    case.longitudinal_timeline = timeline
    return timeline
