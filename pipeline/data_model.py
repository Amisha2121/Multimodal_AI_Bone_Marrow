"""
pipeline/data_model.py
─────────────────────────────────────────────────────────────────────────────
Canonical Data Model for New-Case Cytomorphology Analysis.

Hierarchy:
  PatientAnalysis
    └── image_infos         : List[CaseImageInfo]   (one per uploaded image)
    └── cell_predictions    : List[CellInferenceResult]  (one per detected cell)
    └── xai_examples        : List[XAIExample]       (3 representative cells)

All outputs are keyed to the current case_id.
No hardcoded patient IDs. No cross-case data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


# ── Image-level info ──────────────────────────────────────────────────────────

@dataclass
class CaseImageInfo:
    """Metadata for one uploaded image."""
    image_id: str
    filename: str
    image_hash: str = ""
    width_px: int = 0
    height_px: int = 0
    cells_detected: int = 0
    status: str = "pending"       # ok | error | detection_failed
    error_message: str = ""
    original_image_path: str = ""
    annotated_image_path: str = ""
    debug_raw_path: str = ""
    debug_filtered_path: str = ""
    debug_contact_sheet_path: str = ""


# ── Detection-level info ──────────────────────────────────────────────────────

@dataclass
class CellDetectionResult:
    """Detection output for one image."""
    image_id: str
    cells: List[Any] = field(default_factory=list)   # List[DetectedCell]
    qc: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error_message: str = ""


# ── Cell-level inference result ───────────────────────────────────────────────

@dataclass
class CellInferenceResult:
    """ViT morphology prediction for a single detected cell."""
    cell_id: str                          # e.g. "IMAGE-01_CELL-007"
    image_id: str                         # source image
    bbox: List[int] = field(default_factory=list)  # [x, y, w, h]
    predicted_class: str = ""
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    is_uncertain: bool = False
    image_path: Optional[str] = None

    # Legacy compat
    @property
    def cell_index(self) -> int:
        try:
            return int(self.cell_id.split("-")[-1])
        except Exception:
            return 0

    @property
    def filename(self) -> str:
        return f"{self.cell_id}.png"


# ── XAI example ───────────────────────────────────────────────────────────────

@dataclass
class XAIExample:
    """Patient-specific attention rollout for a single detected cell."""
    cell_id: str
    image_id: str
    predicted_class: str
    confidence: float
    original_image_path: str
    overlay_path: str

    # Legacy compat
    @property
    def cell_index(self) -> int:
        try:
            return int(self.cell_id.split("-")[-1])
        except Exception:
            return 0

    @property
    def filename(self) -> str:
        return f"{self.cell_id}.png"


# ── Top-level case analysis ───────────────────────────────────────────────────

@dataclass
class PatientAnalysis:
    """
    Strongly typed case result passed between pipeline stages and rendered by the UI.
    All fields are populated from the current case only — no cross-case leakage.
    """
    case_id: str

    # Optional metadata (never fabricated)
    age_years: Optional[float] = None
    age_months: Optional[float] = None
    sex: Optional[str] = None
    sample_site: Optional[str] = None

    # Image-level
    image_infos: List[CaseImageInfo] = field(default_factory=list)
    per_image_cell_counts: Dict[str, int] = field(default_factory=dict)
    total_images_processed: int = 0
    total_images_failed: int = 0
    total_cells_detected: int = 0
    cells_classified: int = 0

    # Cell-level predictions
    cell_predictions: List[CellInferenceResult] = field(default_factory=list)

    # Sample-level morphology aggregation
    morphology_counts: Dict[str, int] = field(default_factory=dict)
    morphology_percentages: Dict[str, float] = field(default_factory=dict)
    blast_fraction: float = 0.0
    predominant_categories: List[str] = field(default_factory=list)
    confidence_summary: Dict[str, float] = field(default_factory=dict)

    # Evidence quality
    is_zero_cells: bool = False
    is_low_evidence: bool = False
    evidence_sufficiency: str = "UNKNOWN"  # HIGH | MODERATE | LOW | INSUFFICIENT

    # XAI
    xai_examples: List[XAIExample] = field(default_factory=list)

    # RAG
    retrieved_query: str = ""
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Agentic assessment
    agent_status: str = "pending"
    draft_differential: Optional[str] = None
    critic_grounding_score: float = 0.0
    critic_approved: bool = False
    requires_abstention: bool = False
    final_report: str = ""

    # Pipeline status
    status: str = "pending"        # ok | error
    status_message: str = ""

    # Missing modalities (never fabricated)
    missing_modalities: Dict[str, str] = field(default_factory=lambda: {
        "cbc": "Not provided",
        "cytogenetics": "Not provided",
        "molecular_genetics": "Not provided",
        "flow_cytometry": "Not provided",
    })

    # ── Legacy compatibility props ─────────────────────────────────────────
    @property
    def patient_id(self) -> str:
        return self.case_id

    @property
    def cells_analyzed(self) -> int:
        return self.cells_classified

    @property
    def input_files(self) -> List[str]:
        return [i.original_image_path for i in self.image_infos]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "demographics": {
                "age_years": self.age_years,
                "sex": self.sex,
                "sample_site": self.sample_site,
            },
            "images": {
                "total_uploaded": len(self.image_infos),
                "processed": self.total_images_processed,
                "failed": self.total_images_failed,
                "per_image_cell_counts": self.per_image_cell_counts,
            },
            "cells": {
                "detected": self.total_cells_detected,
                "classified": self.cells_classified,
                "is_zero_cells": self.is_zero_cells,
                "is_low_evidence": self.is_low_evidence,
            },
            "morphology": {
                "counts": self.morphology_counts,
                "percentages": self.morphology_percentages,
                "blast_fraction": self.blast_fraction,
                "predominant_categories": self.predominant_categories,
            },
            "evidence_quality": {
                "grounding_score": self.critic_grounding_score,
                "evidence_sufficiency": self.evidence_sufficiency,
                "approved": self.critic_approved,
                "requires_abstention": self.requires_abstention,
            },
            "missing_modalities": self.missing_modalities,
            "status": self.status,
        }
