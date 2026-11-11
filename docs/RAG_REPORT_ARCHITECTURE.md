# Dynamic Patient-Specific Medical RAG Report Architecture

This document describes the architectural design and clinical data flow for the Dynamic Patient-Specific Medical RAG Report Generation Engine.

## Architectural Data Flow

```mermaid
flowchart TD
    A[Aspirate Smear Image Intake] --> B[Cell Detection & IoU Filtering]
    B --> C[ViT-Small/16 + LoRA Lineage Classification]
    C --> D[Morphology Summary: Counts, Blast %, Wilson 95% CI]
    
    E[Patient Clinical Intake] --> F[Missing Data Registry: Available, Not Provided, Pending]
    
    D --> G[Hybrid Retrieval: BGE + BM25 + RRF over WHO 5th / ICC 2022]
    D --> H[Live Medical Databanks: Europe PMC, ClinVar, ClinicalTrials.gov]
    
    G --> I[Evidence Hierarchy Registry: Tiers 1-6]
    H --> I
    
    D --> J[Dynamic Visualization Engine: Manifest Generation]
    F --> J
    
    D --> K[Patient-Specific Medical Report Engine: 18 Formal Sections]
    F --> K
    I --> K
    J --> K
    
    K --> L[Claim Provenance Extraction & Traceability Mapping]
    L --> M[Language Guard: Definitive Diagnosis Removal]
    M --> N[Pre-Publication Validation Agent: Numerical Audit & Grounding Ratio]
    N --> O[Interactive HTML Report & Print-Optimized Medical PDF]
```

## Evidence Hierarchy (Module 2)

All retrieved citations and guideline passages are assigned to a standardized evidence tier:
1. **Tier 1 - Official Diagnostic Classification**: WHO 5th Edition (2022), ICC (2022).
2. **Tier 2 - Consensus Guidelines**: American Society of Hematology (ASH), NCCN, ELN.
3. **Tier 3 - Peer-Reviewed Systematic Reviews**: High-impact hematopathology literature.
4. **Tier 4 - Original Research**: Benchmark studies, diagnostic model publications.
5. **Tier 5 - Curated Genomic Databanks**: NCBI ClinVar, Entrez Gene, UniProtKB.
6. **Tier 6 - Clinical Investigation Registries**: ClinicalTrials.gov interventional protocols.

## Missing Data Tracking (Module 1)

The system enforces non-fabrication of clinical laboratory parameters:
- `AVAILABLE`: Laboratory measurement provided in case payload.
- `NOT_PROVIDED`: Ancillary modality omitted with image upload.
- `PENDING`: Test ordered, specimen processing awaiting completion.
- `NOT_APPLICABLE`: Parameter excluded from clinical evaluation.
- `EXTRACTION_FAILED`: Document parsing failed; human verification required.

