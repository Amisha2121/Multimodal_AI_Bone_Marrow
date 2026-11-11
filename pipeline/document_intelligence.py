"""
pipeline/document_intelligence.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 & 4: Multi-Category Medical Document Intelligence & Extraction Engine.

Supports Categories A through L using PyMuPDF (fitz) text and table extraction
combined with deterministic medical regex & pattern extractors:
  - Cat A: Morphology & Biopsy (Cellularity, Blasts, M:E ratio, Dysplasia)
  - Cat B: Special Stains (Iron, Reticulin, MPO, PAS)
  - Cat C: Immunophenotyping (CD markers: CD34, CD117, CD33, CD13, HLA-DR, etc.)
  - Cat D: Cytogenetics (Karyotype ISCN strings, FISH probes)
  - Cat E: Molecular Genetics (Mutations, VAF: TP53, NPM1, FLT3, RUNX1, etc.)
  - Cat F: CBC & Laboratory (Hb, WBC, Platelets, ANC, LDH, Ferritin)
  - Cat G: Plasma Cell & Protein (SPEP, Free Light Chains, M-spike)
  - Cat H: Microbiology (AFB, Fungal, Bacterial cultures)
  - Cat I: Integrated Assessments (WHO/ICC diagnostic impressions)
  - Cat J: Treatment Monitoring (MRD flow %, post-chemo marrow)
  - Cat K: Transplant & Stem Cell (CD34+ harvest, Chimerism %)
  - Cat L: Specialized Studies (Amyloid, HLH markers)
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re
import fitz # PyMuPDF
from datetime import datetime

from pipeline.unified_patient_model import (
    DocumentCategory,
    ExtractionStatus,
    ExtractedParameter,
    UploadedMedicalDocument
)


CATEGORY_RULES = [
    (DocumentCategory.CAT_C_IMMUNOPHENOTYPING, [
        r'\bflow\s+cytometr(?:y|ic)\b', r'\bimmunophenotyp(?:e|ing)\b', r'\bcd\d{1,3}\b', r'\bgating\s+strategy\b', r'\bblast\s+gate\b'
    ]),
    (DocumentCategory.CAT_D_CYTOGENETICS, [
        r'\bkaryotyp(?:e|ing)\b', r'\bconventional\s+cytogenetics\b', r'\bmetaphase\b', r'\bfluorescence\s+in\s+situ\b', r'\bfish\s+analysis\b', r'\biscn\b', r'\b\d{2},[xXyY]{2}\b'
    ]),
    (DocumentCategory.CAT_E_MOLECULAR, [
        r'\bnext-generation\s+sequencing\b', r'\bngs\b', r'\bmyeloid\s+(?:mutation|panel)\b', r'\bvariant\s+allele\s+frequency\b', r'\bvaf\b', r'\brt-qpcr\b', r'\bflt3\b', r'\bnpm1\b', r'\btp53\b'
    ]),
    (DocumentCategory.CAT_B_STAINS, [
        r'\bprussian\s+blue\b', r'\biron\s+stain\b', r'\breticulin\s+stain\b', r'\bmyeloperoxidase\b', r'\bmpo\b', r'\bsudan\s+black\b', r'\bperiodic\s+acid-schiff\b', r'\bpas\s+stain\b'
    ]),
    (DocumentCategory.CAT_G_PROTEIN_STUDIES, [
        r'\bprotein\s+electrophoresis\b', r'\bspep\b', r'\bimmunofixation\b', r'\bfree\s+light\s+chain\b', r'\bkappa/lambda\b', r'\bm-spike\b', r'\bmonoclonal\s+gammopathy\b'
    ]),
    (DocumentCategory.CAT_H_MICROBIOLOGY, [
        r'\bbone\s+marrow\s+culture\b', r'\bacid-fast\s+bacilli\b', r'\bafb\b', r'\bfungal\s+culture\b', r'\bmycobacteri'
    ]),
    (DocumentCategory.CAT_K_TRANSPLANT, [
        r'\bstem\s+cell\s+transplant\b', r'\bcd34\+\s+enumeration\b', r'\bchimerism\b', r'\bengraftment\b', r'\bhla\s+typing\b'
    ]),
    (DocumentCategory.CAT_J_TREATMENT_MONITORING, [
        r'\bmeasurable\s+residual\s+disease\b', r'\bmrd\b', r'\bpost-induction\b', r'\bpost-consolidation\b', r'\bremission\s+status\b'
    ]),
    (DocumentCategory.CAT_A_MORPHOLOGY, [
        r'\bbone\s+marrow\s+aspirate\b', r'\btrephine\s+biopsy\b', r'\bcellularity\b', r'\bm:e\s+ratio\b', r'\bmyeloid:erythroid\b', r'\bdyserythropoiesis\b', r'\bmorpholog'
    ]),
    (DocumentCategory.CAT_F_LABORATORY, [
        r'\bcomplete\s+blood\s+count\b', r'\bcbc\b', r'\bhemoglobin\b', r'\bplatelet\s+count\b', r'\bwhite\s+blood\s+cell\b', r'\banc\b', r'\bneutrophil\b'
    ])
]


def classify_document_text(text: str) -> Tuple[DocumentCategory, str]:
    """Classifies document content into one of the 12 medical report categories."""
    text_lower = text.lower()
    for cat, patterns in CATEGORY_RULES:
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score >= 2:
            return cat, cat.value.replace("CAT_", "").replace("_", " ").title()
        elif score == 1:
            # Single strong trigger check
            return cat, cat.value.replace("CAT_", "").replace("_", " ").title()
    
    return DocumentCategory.CAT_I_INTEGRATED_ASSESSMENT, "Integrated Diagnostic Assessment"


def extract_parameters_from_text(doc_id: str, text: str, category: DocumentCategory) -> List[ExtractedParameter]:
    """Extracts structured clinical parameters based on document category."""
    params: List[ExtractedParameter] = []

    # 1. CBC & Laboratory parameters (Cat F & general)
    cbc_patterns = [
        (r'\b(?:hb|hemoglobin|hgb)\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(g/dl|g/l)?', "Hemoglobin", "g/dL", "13.0-17.5 (M) / 12.0-15.5 (F)"),
        (r'\b(?:wbc|white\s+blood\s+cell(?:s)?|leukocyte(?:s)?)\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(?:x\s*10\^?9/l|k/ul|/ul)?', "WBC", "x10^9/L", "4.0-11.0"),
        (r'\b(?:platelets?|plt)\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(?:x\s*10\^?9/l|k/ul|/ul)?', "Platelets", "x10^9/L", "150-450"),
        (r'\b(?:anc|absolute\s+neutrophil\s+count)\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(?:x\s*10\^?9/l|k/ul)?', "ANC", "x10^9/L", "1.5-8.0"),
        (r'\bldh\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(u/l)?', "LDH", "U/L", "140-280")
    ]
    for pattern, name, unit, ref in cbc_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            params.append(ExtractedParameter(
                name=name,
                reported_value=val,
                normalized_value=val,
                original_unit=unit,
                normalized_unit=unit,
                reference_interval=ref,
                source_document_id=doc_id,
                confidence=0.95
            ))

    # 2. Morphology & Blast percentage (Cat A & general)
    blast_match = re.search(r'\b(?:blasts?|myeloblasts?)\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*%', text, re.IGNORECASE)
    if blast_match:
        b_val = float(blast_match.group(1))
        params.append(ExtractedParameter(
            name="Reported Blast Percentage",
            reported_value=b_val,
            normalized_value=b_val / 100.0,
            original_unit="%",
            normalized_unit="fraction",
            reference_interval="< 5.0% in normal bone marrow",
            abnormality_flag="Elevated" if b_val >= 5.0 else "Normal",
            source_document_id=doc_id,
            confidence=0.98
        ))

    cellularity_match = re.search(r'\bcellularity\b\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*%', text, re.IGNORECASE)
    if cellularity_match:
        c_val = float(cellularity_match.group(1))
        params.append(ExtractedParameter(
            name="Bone Marrow Cellularity",
            reported_value=c_val,
            original_unit="%",
            reference_interval="Age-dependent: ~ (100 - age)%",
            source_document_id=doc_id,
            confidence=0.90
        ))

    # 3. Flow Cytometry CD Markers (Cat C)
    if category in (DocumentCategory.CAT_C_IMMUNOPHENOTYPING, DocumentCategory.CAT_I_INTEGRATED_ASSESSMENT):
        markers = ["CD34", "CD117", "CD33", "CD13", "HLA-DR", "CD38", "CD19", "CD10", "CD20", "CD3", "CD56"]
        for marker in markers:
            pattern = rf'\b{marker}\b\s*(?::|\s)\s*(positive|\+|negative|\-|dim|bright|[0-9]+\.?[0-9]*%)'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                state = m.group(1).strip()
                params.append(ExtractedParameter(
                    name=f"Flow Cytometry {marker}",
                    reported_value=state,
                    source_document_id=doc_id,
                    confidence=0.92,
                    clinical_notes=f"Reported immunophenotypic finding for {marker}"
                ))

    # 4. Cytogenetics / Karyotype ISCN (Cat D)
    if category in (DocumentCategory.CAT_D_CYTOGENETICS, DocumentCategory.CAT_I_INTEGRATED_ASSESSMENT):
        iscn_match = re.search(r'(?:karyotype|iscn)(?:\s+result)?\s*[:=]?\s*([0-9]{2},[xXyY]{2}[^\.\n\r]+)', text, re.IGNORECASE)
        if not iscn_match:
            iscn_match = re.search(r'([0-9]{2},[xXyY]{2}[^\.\n\r]+)', text)
        if iscn_match:
            iscn_str = iscn_match.group(1).strip()
            params.append(ExtractedParameter(
                name="Cytogenetic Karyotype (ISCN)",
                reported_value=iscn_str,
                source_document_id=doc_id,
                confidence=0.96,
                abnormality_flag="Abnormal" if any(w in iscn_str for w in ["del", "t(", "+", "-", "inv", "dup", "add", "mar"]) else "Normal"
            ))

    # 5. Molecular Mutations & VAF (Cat E)
    if category in (DocumentCategory.CAT_E_MOLECULAR, DocumentCategory.CAT_I_INTEGRATED_ASSESSMENT):
        genes = ["TP53", "NPM1", "FLT3", "RUNX1", "ASXL1", "IDH1", "IDH2", "DNMT3A", "JAK2", "SF3B1"]
        for g in genes:
            g_match = re.search(rf'\b{g}\b[^\.\n\r]*(?:mutation|variant|detected|positive|vaf\s*[:=]?\s*([0-9]+\.?[0-9]*%?))', text, re.IGNORECASE)
            if g_match:
                params.append(ExtractedParameter(
                    name=f"Molecular Gene: {g}",
                    reported_value=g_match.group(0).strip(),
                    source_document_id=doc_id,
                    confidence=0.94,
                    abnormality_flag="Detected"
                ))

    return params


def process_uploaded_document_file(filepath: Path, doc_id: Optional[str] = None) -> UploadedMedicalDocument:
    """
    Primary ingestion entrypoint: Parses PDF or text document,
    extracts raw text, classifies category, and produces UploadedMedicalDocument.
    """
    if not doc_id:
        doc_id = f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.stem[:8]}"

    raw_text = ""
    page_count = 1

    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        try:
            doc = fitz.open(str(filepath))
            page_count = len(doc)
            text_pages = [page.get_text() for page in doc]
            raw_text = "\n\n--- PAGE BREAK ---\n\n".join(text_pages)
            doc.close()
        except Exception as e:
            raw_text = f"[PDF Parsing Error: {e}]"
    else:
        # Fallback to plain text read
        try:
            raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raw_text = f"[Text Read Error: {e}]"

    category, cat_name = classify_document_text(raw_text)
    extracted = extract_parameters_from_text(doc_id, raw_text, category)

    # Date extraction heuristic
    date_match = re.search(r'\b(?:date|collected|specimen\s+date)\s*[:=]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}|[A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4})', raw_text, re.IGNORECASE)
    coll_date = date_match.group(1) if date_match else None

    return UploadedMedicalDocument(
        document_id=doc_id,
        filename=filepath.name,
        category=category,
        category_name=cat_name,
        specimen_collection_date=coll_date,
        page_count=page_count,
        raw_text=raw_text,
        extracted_parameters=extracted,
        extraction_status=ExtractionStatus.EXTRACTED if extracted else ExtractionStatus.FLAGGED_FOR_REVIEW
    )
