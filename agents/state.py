"""
agents/state.py
─────────────────────────────────────────────────────────────────────────────
State schema for the Bone Marrow Agentic RAG Framework.

Evidence provenance categories (NEVER mixed):
  OBSERVED_FROM_IMAGE   - detected cells, morphology counts, blast fraction
  PROVIDED_BY_USER      - age, sex, sample site (if entered)
  RETRIEVED             - WHO/ICC guideline passages
  NOT_AVAILABLE         - CBC, cytogenetics, molecular, flow cytometry
"""

from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field


class PatientMorphologyInput(BaseModel):
    """Input payload for a new case's morphological profile."""
    patient_id: str
    age_months: Optional[float] = None       # None = not provided
    age_years: Optional[float] = None        # None = not provided
    sex: Optional[str] = None
    annotated_blast_fraction: float = Field(ge=0.0, le=1.0)
    predicted_class: str
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    cell_counts_by_category: Dict[str, int] = Field(default_factory=dict)
    total_cells: int = 0
    low_evidence_flag: bool = False


class AgentState(TypedDict, total=False):
    """
    Central state dict passed between LangGraph nodes.
    Separates evidence grounding (are claims supported by retrieved text?)
    from evidence sufficiency (is there enough image evidence to draw conclusions?).
    """
    # Case identity
    patient_id: str
    age_months: Optional[float]       # None = not provided
    age_years: Optional[float]        # None = not provided
    sex: Optional[str]

    # Image-derived morphology (OBSERVED_FROM_IMAGE)
    annotated_blast_fraction: float
    predicted_class: str
    class_probabilities: Dict[str, float]
    cell_counts_by_category: Dict[str, int]
    total_cells: int
    total_images: int
    low_evidence_flag: bool

    # Retrieval stage (RETRIEVED)
    search_query: str
    retrieved_evidence: List[Dict[str, Any]]
    formatted_guidelines_context: str

    # Synthesis stage
    draft_differential: str
    differential_diagnosis_hierarchy: List[str]
    morphological_justification: str

    # Critic & Verification stage — TWO SEPARATE CONCEPTS
    critic_grounding_score: float      # Are claims supported by retrieved literature? [0,1]
    evidence_sufficiency: str          # Is there enough image evidence? HIGH|MODERATE|LOW|INSUFFICIENT
    critic_hallucination_flags: List[str]
    critic_comments: str
    is_approved: bool
    revision_count: int

    # Finalization
    final_report: str
    requires_human_abstention: bool
