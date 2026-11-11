"""
pipeline/evidence_registry.py
─────────────────────────────────────────────────────────────────────────────
Module 2: Expanded Multi-Source Medical Knowledge Retrieval & Evidence Hierarchy.

Standardizes all retrieved evidence across:
  - Official Guidelines (WHO 5th Edition 2022, ICC 2022)
  - Peer-Reviewed Medical Literature (PubMed, Europe PMC)
  - Genomic & Variant Evidence (ClinVar, NCBI Entrez Gene, UniProtKB)
  - Clinical Trials & Regulatory Databanks (ClinicalTrials.gov, FDA)

Enforces an explicit Evidence Hierarchy:
  TIER_1: Official Diagnostic Classification Systems (WHO 5th Ed, ICC 2022)
  TIER_2: Professional Society Clinical Guidelines (ASH, NCCN, ELN)
  TIER_3: Peer-Reviewed Systematic Reviews & Meta-Analyses
  TIER_4: Original Biomedical Research Articles & Benchmark Cohorts
  TIER_5: Public Genomic & Molecular Variant Repositories (ClinVar, UniProt)
  TIER_6: Active Clinical Investigation Registries (ClinicalTrials.gov)
"""

from enum import IntEnum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


class EvidenceTier(IntEnum):
    TIER_1_OFFICIAL_CLASSIFICATION = 1
    TIER_2_CONSENSUS_GUIDELINE = 2
    TIER_3_SYSTEMATIC_REVIEW = 3
    TIER_4_PEER_REVIEWED_RESEARCH = 4
    TIER_5_GENOMIC_DATABANK = 5
    TIER_6_CLINICAL_TRIAL_REGISTRY = 6


@dataclass
class EvidenceRecord:
    evidence_id: str
    document_title: str
    issuing_organization: str
    source_type: str                  # e.g., "Official Classification", "Clinical Trial", "Genomic Variant"
    tier: EvidenceTier
    publication_year: Optional[int] = None
    document_version: Optional[str] = None
    doi_or_url: Optional[str] = None
    pmid: Optional[str] = None
    section_or_page: Optional[str] = None
    supporting_excerpt: str = ""
    relevance_score: float = 1.0
    access_status: str = "Verified Public Access"
    retrieval_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_title": self.document_title,
            "issuing_organization": self.issuing_organization,
            "source_type": self.source_type,
            "tier": int(self.tier),
            "tier_label": self.tier.name.replace("_", " ").title(),
            "publication_year": self.publication_year,
            "document_version": self.document_version,
            "doi_or_url": self.doi_or_url,
            "pmid": self.pmid,
            "section_or_page": self.section_or_page,
            "supporting_excerpt": self.supporting_excerpt,
            "relevance_score": round(self.relevance_score, 3),
            "access_status": self.access_status,
            "retrieval_timestamp": self.retrieval_timestamp
        }


# Standard static registry for core WHO / ICC guidelines chunks
CORE_GUIDELINE_REGISTRY: Dict[str, EvidenceRecord] = {
    "WHO2022_AML_001": EvidenceRecord(
        evidence_id="WHO2022_AML_001",
        document_title="WHO Classification of Tumours of Haematopoietic and Lymphoid Tissues (5th Edition)",
        issuing_organization="World Health Organization (WHO)",
        source_type="Official Classification Guideline",
        tier=EvidenceTier.TIER_1_OFFICIAL_CLASSIFICATION,
        publication_year=2022,
        document_version="5th Edition",
        doi_or_url="https://tumourclassification.iarc.who.int/",
        section_or_page="Acute Myeloid Leukaemia - Diagnostic Criteria",
        supporting_excerpt="In the absence of defining genetic abnormalities, AML diagnosis requires >=20% blasts in peripheral blood or bone marrow aspirate differential."
    ),
    "WHO2022_MDS_002": EvidenceRecord(
        evidence_id="WHO2022_MDS_002",
        document_title="WHO Classification of Tumours of Haematopoietic and Lymphoid Tissues (5th Edition)",
        issuing_organization="World Health Organization (WHO)",
        source_type="Official Classification Guideline",
        tier=EvidenceTier.TIER_1_OFFICIAL_CLASSIFICATION,
        publication_year=2022,
        document_version="5th Edition",
        doi_or_url="https://tumourclassification.iarc.who.int/",
        section_or_page="Myelodysplastic Neoplasms with excess blasts",
        supporting_excerpt="MDS-EB1: 5-9% bone marrow blasts or 2-4% peripheral blood blasts without Auer rods. MDS-EB2: 10-19% bone marrow blasts or 5-19% peripheral blood blasts, or presence of Auer rods."
    ),
    "ICC2022_MDS_AML_001": EvidenceRecord(
        evidence_id="ICC2022_MDS_AML_001",
        document_title="International Consensus Classification (ICC) of Myeloid Neoplasms and Acute Leukemias",
        issuing_organization="International Consensus Classification (ICC) Steering Committee",
        source_type="Official Classification Guideline",
        tier=EvidenceTier.TIER_1_OFFICIAL_CLASSIFICATION,
        publication_year=2022,
        document_version="ICC 2022",
        doi_or_url="https://doi.org/10.1182/blood.2022015850",
        section_or_page="MDS/AML Transition Category",
        supporting_excerpt="The ICC establishes MDS/AML for cases with 10-19% blasts in bone marrow or blood, recognizing rapid progression risk while retaining requirement for comprehensive cytogenetic and molecular characterization."
    ),
    "LIT-05": EvidenceRecord(
        evidence_id="LIT-05",
        document_title="Artificial Intelligence in Hematopathology: Bone Marrow Aspirate Cytomorphology Evaluation",
        issuing_organization="HemaSphere / European Hematology Association",
        source_type="Peer-Reviewed Review",
        tier=EvidenceTier.TIER_3_SYSTEMATIC_REVIEW,
        publication_year=2024,
        doi_or_url="https://doi.org/10.1002/hem3.70048",
        section_or_page="Page 2",
        supporting_excerpt="Acute leukemia is characterised by 20% or more blasts; chronic-phase CML typically shows blasts under 10% with left shift; MDS typically shows under 5% blasts, with MDS-EB up to 19%. Cytomorphology is only one component and must be integrated with other testing."
    ),
    "LIT-01": EvidenceRecord(
        evidence_id="LIT-01",
        document_title="Manual Differential Counting and Automated Cytomorphology in Hematology",
        issuing_organization="HemaSphere",
        source_type="Peer-Reviewed Review",
        tier=EvidenceTier.TIER_3_SYSTEMATIC_REVIEW,
        publication_year=2024,
        doi_or_url="https://doi.org/10.1002/hem3.70048",
        section_or_page="Introduction",
        supporting_excerpt="Manual differential counting classifies several hundred nucleated cells; continuous maturation causes subjectivity and high inter-observer variability."
    )
}


def build_evidence_registry_for_case(
    retrieved_chunks: List[Dict[str, Any]],
    multibank_evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, EvidenceRecord]:
    """
    Builds a unified, deduplicated dictionary of EvidenceRecord objects 
    mapping evidence_id -> EvidenceRecord for a specific patient report.
    """
    registry: Dict[str, EvidenceRecord] = {}

    # 1. Ingest core guidelines matched by retriever
    for chunk in retrieved_chunks:
        cid = chunk.get("chunk_id") or chunk.get("id")
        if not cid:
            continue
        if cid in CORE_GUIDELINE_REGISTRY:
            rec = CORE_GUIDELINE_REGISTRY[cid]
            # Copy with actual retrieval relevance score if present
            score = chunk.get("score") or chunk.get("relevance", 1.0)
            registry[cid] = EvidenceRecord(
                evidence_id=rec.evidence_id,
                document_title=rec.document_title,
                issuing_organization=rec.issuing_organization,
                source_type=rec.source_type,
                tier=rec.tier,
                publication_year=rec.publication_year,
                document_version=rec.document_version,
                doi_or_url=rec.doi_or_url,
                section_or_page=rec.section_or_page,
                supporting_excerpt=chunk.get("text", rec.supporting_excerpt)[:350],
                relevance_score=float(score)
            )
        else:
            # Generic guideline chunk from retriever index
            registry[cid] = EvidenceRecord(
                evidence_id=cid,
                document_title=chunk.get("metadata", {}).get("source", "WHO/ICC Classification Archive"),
                issuing_organization="WHO / ICC Hematopathology Guidelines",
                source_type="Authoritative Guideline Excerpt",
                tier=EvidenceTier.TIER_1_OFFICIAL_CLASSIFICATION,
                publication_year=2022,
                doi_or_url="https://tumourclassification.iarc.who.int/",
                section_or_page=chunk.get("metadata", {}).get("section", "Hematopathology Consensus"),
                supporting_excerpt=chunk.get("text", "")[:350],
                relevance_score=float(chunk.get("score", 0.95))
            )

    # 2. Ingest Multi-Databank Evidence (Europe PMC, PubMed, ClinVar, ClinicalTrials, etc.)
    if multibank_evidence:
        # PubMed / Europe PMC Literature
        for art in multibank_evidence.get("pubmed_literature", []):
            eid = f"PMID_{art.get('id', 'REF')}"
            registry[eid] = EvidenceRecord(
                evidence_id=eid,
                document_title=art.get("title", "Hematology Literature"),
                issuing_organization=art.get("source", "PubMed Central"),
                source_type="Peer-Reviewed Biomedical Article",
                tier=EvidenceTier.TIER_4_PEER_REVIEWED_RESEARCH,
                publication_year=int(art.get("year")) if str(art.get("year", "")).isdigit() else 2024,
                doi_or_url=art.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{art.get('id')}"),
                pmid=str(art.get("id")),
                supporting_excerpt=art.get("snippet", "")[:350],
                relevance_score=0.88
            )

        # ClinVar Pathogenic Control Records
        for cv in multibank_evidence.get("clinvar_variants", []):
            eid = f"CLINVAR_{cv.get('id', 'VAR')}"
            registry[eid] = EvidenceRecord(
                evidence_id=eid,
                document_title=f"ClinVar Variant: {cv.get('title', 'Pathogenic Alteration')}",
                issuing_organization="NCBI ClinVar Human Variant Database",
                source_type="Curated Genomic Pathogenicity Record",
                tier=EvidenceTier.TIER_5_GENOMIC_DATABANK,
                doi_or_url=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{cv.get('id')}",
                section_or_page=cv.get("classification", "Clinical Significance"),
                supporting_excerpt=cv.get("snippet", "")[:350],
                relevance_score=0.85
            )

        # Clinical Trials
        for ct in multibank_evidence.get("clinical_trials", []):
            eid = f"NCT_{ct.get('id', 'TRIAL')}"
            registry[eid] = EvidenceRecord(
                evidence_id=eid,
                document_title=ct.get("title", "Clinical Trial Protocol"),
                issuing_organization="U.S. National Library of Medicine (ClinicalTrials.gov)",
                source_type="Interventional / Observational Clinical Trial",
                tier=EvidenceTier.TIER_6_CLINICAL_TRIAL_REGISTRY,
                doi_or_url=ct.get("url", f"https://clinicaltrials.gov/study/{ct.get('id')}"),
                section_or_page=ct.get("phase", "Trial Phase"),
                supporting_excerpt=ct.get("snippet", "")[:350],
                relevance_score=0.82
            )

    return registry

