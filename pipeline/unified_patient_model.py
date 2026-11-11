"""
pipeline/unified_patient_model.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 & 5: Unified Patient Case Data Model & Multi-Category Document Schema.

Standardized schema supporting:
  - Categories A through L (Morphology, Histochemistry, Immunophenotyping,
    Cytogenetics, Molecular Genetics, CBC, Protein Studies, Microbiology,
    Integrated Assessments, Treatment Monitoring, Transplant, Specialized)
  - Full provenance tracking (source document ID, page, method, confidence)
  - Explicit representation of missing vs available vs unverified data.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentCategory(str, Enum):
    CAT_A_MORPHOLOGY = "CAT_A_MORPHOLOGY"
    CAT_B_STAINS = "CAT_B_STAINS"
    CAT_C_IMMUNOPHENOTYPING = "CAT_C_IMMUNOPHENOTYPING"
    CAT_D_CYTOGENETICS = "CAT_D_CYTOGENETICS"
    CAT_E_MOLECULAR = "CAT_E_MOLECULAR"
    CAT_F_LABORATORY = "CAT_F_LABORATORY"
    CAT_G_PROTEIN_STUDIES = "CAT_G_PROTEIN_STUDIES"
    CAT_H_MICROBIOLOGY = "CAT_H_MICROBIOLOGY"
    CAT_I_INTEGRATED_ASSESSMENT = "CAT_I_INTEGRATED_ASSESSMENT"
    CAT_J_TREATMENT_MONITORING = "CAT_J_TREATMENT_MONITORING"
    CAT_K_TRANSPLANT = "CAT_K_TRANSPLANT"
    CAT_L_SPECIALIZED = "CAT_L_SPECIALIZED"
    UNKNOWN = "UNKNOWN"


class ExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    VERIFIED = "VERIFIED"
    FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExtractedParameter(BaseModel):
    name: str
    reported_value: Any
    normalized_value: Optional[Any] = None
    original_unit: Optional[str] = None
    normalized_unit: Optional[str] = None
    reference_interval: Optional[str] = None
    abnormality_flag: Optional[str] = None     # Normal, High, Low, Abnormal, Positive, Negative
    source_document_id: str
    source_page: int = 1
    specimen_date: Optional[str] = None
    extraction_method: str = "rules_regex"
    confidence: float = 1.0
    verification_status: ExtractionStatus = ExtractionStatus.EXTRACTED
    clinical_notes: Optional[str] = None


class UploadedMedicalDocument(BaseModel):
    document_id: str
    filename: str
    category: DocumentCategory = DocumentCategory.UNKNOWN
    category_name: str = "Unclassified Medical Report"
    upload_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    specimen_collection_date: Optional[str] = None
    laboratory_or_institution: Optional[str] = None
    patient_identifier_in_doc: Optional[str] = None
    page_count: int = 1
    raw_text: Optional[str] = None
    extracted_parameters: List[ExtractedParameter] = Field(default_factory=list)
    extraction_status: ExtractionStatus = ExtractionStatus.EXTRACTED
    verification_comments: Optional[str] = None


class UnifiedPatientCase(BaseModel):
    patient_id: str
    case_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Document repository
    uploaded_documents: List[UploadedMedicalDocument] = Field(default_factory=list)

    # Image-derived vision analysis
    aspirate_images: List[str] = Field(default_factory=list)
    biopsy_images: List[str] = Field(default_factory=list)
    morphology_findings: Dict[str, Any] = Field(default_factory=dict)
    cell_detection_results: List[Dict[str, Any]] = Field(default_factory=list)
    blast_fraction: Optional[float] = None
    blast_ci: Optional[List[float]] = None

    # Categorized Clinical Investigations (A - L)
    morphology_reports: List[Dict[str, Any]] = Field(default_factory=list)      # Cat A
    special_stains: List[Dict[str, Any]] = Field(default_factory=list)          # Cat B
    flow_cytometry: List[Dict[str, Any]] = Field(default_factory=list)          # Cat C
    cytogenetics: List[Dict[str, Any]] = Field(default_factory=list)            # Cat D
    molecular_genetics: List[Dict[str, Any]] = Field(default_factory=list)      # Cat E
    cbc_and_labs: List[Dict[str, Any]] = Field(default_factory=list)            # Cat F
    protein_studies: List[Dict[str, Any]] = Field(default_factory=list)         # Cat G
    microbiology: List[Dict[str, Any]] = Field(default_factory=list)            # Cat H
    integrated_assessments: List[Dict[str, Any]] = Field(default_factory=list)  # Cat I
    treatment_monitoring: List[Dict[str, Any]] = Field(default_factory=list)    # Cat J
    transplant_reports: List[Dict[str, Any]] = Field(default_factory=list)      # Cat K
    specialized_studies: List[Dict[str, Any]] = Field(default_factory=list)     # Cat L

    # Cross-document correlation & longitudinal trends
    cross_report_correlations: List[Dict[str, Any]] = Field(default_factory=list)
    detected_contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    longitudinal_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    missing_investigations: List[Dict[str, Any]] = Field(default_factory=list)

    def add_document(self, doc: UploadedMedicalDocument):
        self.uploaded_documents.append(doc)
        # Dispatch extracted parameters to appropriate category buckets
        for p in doc.extracted_parameters:
            item = p.model_dump()
            if doc.category == DocumentCategory.CAT_A_MORPHOLOGY:
                self.morphology_reports.append(item)
            elif doc.category == DocumentCategory.CAT_B_STAINS:
                self.special_stains.append(item)
            elif doc.category == DocumentCategory.CAT_C_IMMUNOPHENOTYPING:
                self.flow_cytometry.append(item)
            elif doc.category == DocumentCategory.CAT_D_CYTOGENETICS:
                self.cytogenetics.append(item)
            elif doc.category == DocumentCategory.CAT_E_MOLECULAR:
                self.molecular_genetics.append(item)
            elif doc.category == DocumentCategory.CAT_F_LABORATORY:
                self.cbc_and_labs.append(item)
            elif doc.category == DocumentCategory.CAT_G_PROTEIN_STUDIES:
                self.protein_studies.append(item)
            elif doc.category == DocumentCategory.CAT_H_MICROBIOLOGY:
                self.microbiology.append(item)
            elif doc.category == DocumentCategory.CAT_I_INTEGRATED_ASSESSMENT:
                self.integrated_assessments.append(item)
            elif doc.category == DocumentCategory.CAT_J_TREATMENT_MONITORING:
                self.treatment_monitoring.append(item)
            elif doc.category == DocumentCategory.CAT_K_TRANSPLANT:
                self.transplant_reports.append(item)
            elif doc.category == DocumentCategory.CAT_L_SPECIALIZED:
                self.specialized_studies.append(item)
