"""
pipeline/patient_context.py
─────────────────────────────────────────────────────────────────────────────
Module 1: Complete Patient-Specific Data Ingestion & Missing-Data Management.

Explicitly tracks available vs unavailable patient information without ever
fabricating missing values. Distinguishes:
  - AVAILABLE: value provided and extracted
  - NOT_PROVIDED: modality not submitted with this scan
  - UNAVAILABLE: attempted but unrecorded in source data
  - PENDING: ordered/requested laboratory test awaiting completion
  - NOT_APPLICABLE: modality does not apply to this evaluation
  - EXTRACTION_FAILED: file/data submitted but parsing failed
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


class DataStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_PROVIDED = "NOT_PROVIDED"
    UNAVAILABLE = "UNAVAILABLE"
    PENDING = "PENDING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


@dataclass
class ClinicalParameter:
    name: str
    value: Any = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: DataStatus = DataStatus.NOT_PROVIDED
    source_document: Optional[str] = None
    extraction_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "status": self.status.value,
            "source_document": self.source_document,
            "extraction_notes": self.extraction_notes,
            "is_available": self.status == DataStatus.AVAILABLE
        }


@dataclass
class PatientClinicalContext:
    case_id: str
    patient_id: Optional[str] = None
    age_years: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Age", status=DataStatus.NOT_PROVIDED))
    sex: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Sex", status=DataStatus.NOT_PROVIDED))
    
    # Complete Blood Count (CBC)
    hemoglobin: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Hemoglobin", unit="g/dL", reference_range="13.0-17.5 (M) / 12.0-15.5 (F)"))
    white_blood_cells: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("WBC", unit="x10^9/L", reference_range="4.0-11.0"))
    platelets: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Platelets", unit="x10^9/L", reference_range="150-450"))
    absolute_neutrophil_count: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("ANC", unit="x10^9/L", reference_range="1.5-8.0"))
    
    # Ancillary Hematopathology Modalities
    flow_cytometry: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Flow Cytometry Immunophenotyping", status=DataStatus.NOT_PROVIDED))
    conventional_karyotype: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Cytogenetics / Karyotype", status=DataStatus.NOT_PROVIDED))
    molecular_genetics: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Molecular Panel (NGS / Mutations)", status=DataStatus.NOT_PROVIDED))
    bone_marrow_biopsy: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Trephine Core Biopsy Cellularity", status=DataStatus.NOT_PROVIDED))
    clinical_history: ClinicalParameter = field(default_factory=lambda: ClinicalParameter("Clinical Indication & History", status=DataStatus.NOT_PROVIDED))

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_missing_modalities(self) -> List[Dict[str, str]]:
        """Identify which ancillary tests are missing and required for definitive hematological evaluation."""
        missing = []
        checks = [
            (self.hemoglobin, "Peripheral Complete Blood Count (CBC)", "Required to establish cytopenias (anemia, thrombocytopenia, neutropenia)."),
            (self.flow_cytometry, "Flow Cytometric Immunophenotyping", "Required to detect blast aberrant marker expression (e.g., CD34, CD117, CD13, CD33, CD19, CD3)."),
            (self.conventional_karyotype, "Cytogenetic Karyotype / FISH", "Required for WHO 2022 / ICC risk stratification (e.g., del(5q), t(8;21), inv(16), complex karyotype)."),
            (self.molecular_genetics, "Myeloid / Lymphoid NGS Panel", "Required to identify defining genomic mutations (TP53, SF3B1, RUNX1, ASXL1, NPM1, FLT3, JAK2)."),
            (self.bone_marrow_biopsy, "Trephine Core Biopsy", "Required for architectural evaluation, fibrosis grading (reticulin/collagen), and topography.")
        ]
        for param, modality_name, clinical_rationale in checks:
            if param.status != DataStatus.AVAILABLE:
                missing.append({
                    "parameter": param.name,
                    "modality": modality_name,
                    "status": param.status.value,
                    "rationale": clinical_rationale,
                    "urgency": "High" if "Flow" in modality_name or "CBC" in modality_name else "Standard"
                })
        return missing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id or self.case_id,
            "parameters": {
                "age": self.age_years.to_dict(),
                "sex": self.sex.to_dict(),
                "hemoglobin": self.hemoglobin.to_dict(),
                "wbc": self.white_blood_cells.to_dict(),
                "platelets": self.platelets.to_dict(),
                "anc": self.absolute_neutrophil_count.to_dict(),
                "flow_cytometry": self.flow_cytometry.to_dict(),
                "cytogenetics": self.conventional_karyotype.to_dict(),
                "molecular_genetics": self.molecular_genetics.to_dict(),
                "core_biopsy": self.bone_marrow_biopsy.to_dict(),
                "clinical_history": self.clinical_history.to_dict()
            },
            "missing_modalities": self.get_missing_modalities(),
            "has_cbc": self.hemoglobin.status == DataStatus.AVAILABLE and self.platelets.status == DataStatus.AVAILABLE,
            "has_genetics": self.conventional_karyotype.status == DataStatus.AVAILABLE or self.molecular_genetics.status == DataStatus.AVAILABLE,
            "has_flow_cytometry": self.flow_cytometry.status == DataStatus.AVAILABLE,
            "is_image_only": all(
                p.status != DataStatus.AVAILABLE 
                for p in [self.hemoglobin, self.white_blood_cells, self.platelets, self.flow_cytometry, self.conventional_karyotype, self.molecular_genetics]
            )
        }


def extract_patient_context_from_case(case_dict: Dict[str, Any]) -> PatientClinicalContext:
    """
    Constructs a PatientClinicalContext from the stored case dictionary,
    extracting any clinical or patient demographic attributes without fabrication.
    """
    case_id = case_dict.get("case_id") or case_dict.get("id") or "UNKNOWN_CASE"
    ctx = PatientClinicalContext(case_id=case_id)

    # 1. Demographics
    if "patient_id" in case_dict:
        ctx.patient_id = str(case_dict["patient_id"])
    
    if "age" in case_dict and case_dict["age"] is not None:
        ctx.age_years = ClinicalParameter("Age", value=case_dict["age"], unit="years", status=DataStatus.AVAILABLE)
    elif "age_years" in case_dict and case_dict["age_years"] is not None:
        ctx.age_years = ClinicalParameter("Age", value=case_dict["age_years"], unit="years", status=DataStatus.AVAILABLE)

    if "sex" in case_dict and case_dict["sex"]:
        ctx.sex = ClinicalParameter("Sex", value=str(case_dict["sex"]).capitalize(), status=DataStatus.AVAILABLE)

    # 2. Check for uploaded lab parameters or clinical data dictionary
    clinical_data = case_dict.get("clinical_data") or case_dict.get("patient_metadata") or {}
    if isinstance(clinical_data, dict):
        if "hb" in clinical_data or "hemoglobin" in clinical_data:
            val = clinical_data.get("hb") or clinical_data.get("hemoglobin")
            ctx.hemoglobin = ClinicalParameter("Hemoglobin", value=val, unit="g/dL", status=DataStatus.AVAILABLE)
        
        if "wbc" in clinical_data:
            ctx.white_blood_cells = ClinicalParameter("WBC", value=clinical_data["wbc"], unit="x10^9/L", status=DataStatus.AVAILABLE)
            
        if "platelets" in clinical_data or "plt" in clinical_data:
            val = clinical_data.get("platelets") or clinical_data.get("plt")
            ctx.platelets = ClinicalParameter("Platelets", value=val, unit="x10^9/L", status=DataStatus.AVAILABLE)

        if "anc" in clinical_data:
            ctx.absolute_neutrophil_count = ClinicalParameter("ANC", value=clinical_data["anc"], unit="x10^9/L", status=DataStatus.AVAILABLE)

        if "flow_cytometry" in clinical_data:
            ctx.flow_cytometry = ClinicalParameter("Flow Cytometry", value=clinical_data["flow_cytometry"], status=DataStatus.AVAILABLE)

        if "karyotype" in clinical_data or "cytogenetics" in clinical_data:
            val = clinical_data.get("karyotype") or clinical_data.get("cytogenetics")
            ctx.conventional_karyotype = ClinicalParameter("Cytogenetics", value=val, status=DataStatus.AVAILABLE)

        if "molecular" in clinical_data or "ngs" in clinical_data:
            val = clinical_data.get("molecular") or clinical_data.get("ngs")
            ctx.molecular_genetics = ClinicalParameter("Molecular Panel", value=val, status=DataStatus.AVAILABLE)

        if "history" in clinical_data:
            ctx.clinical_history = ClinicalParameter("Clinical History", value=clinical_data["history"], status=DataStatus.AVAILABLE)

    return ctx

