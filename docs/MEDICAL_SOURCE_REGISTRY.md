# Medical Source Registry & Evidence Hierarchy

This document details the multi-source medical evidence framework integrated into the Patient-Specific RAG Reporting Engine.

## Evidence Tiers & Authority Hierarchy

| Tier | Category | Sources | Authority Level | Usage in Reporting |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Official Diagnostic Classification** | WHO Classification of Tumours (5th Ed, 2022); International Consensus Classification (ICC 2022) | Primary standard of care | Threshold definitions (e.g. $\ge 20\%$ blast threshold for AML; $5\text{--}19\%$ for MDS-EB). |
| **Tier 2** | **Consensus Clinical Guidelines** | American Society of Hematology (ASH), NCCN, European LeukemiaNet (ELN) | Clinical management consensus | Standard diagnostic pathways, risk stratification, confirmatory workup sequences. |
| **Tier 3** | **Peer-Reviewed Systematic Reviews** | High-impact hematopathology literature (e.g. HemaSphere, Blood, Leukemia) | Evaluated clinical evidence | Contextualizing diagnostic uncertainties and inter-observer cytomorphometric variance. |
| **Tier 4** | **Original Biomedical Research** | Peer-reviewed articles from Europe PMC & PubMed | Empirical research | Observational benchmarks and machine learning validation studies. |
| **Tier 5** | **Curated Genomic Databanks** | NCBI ClinVar, Entrez Gene, UniProtKB | Molecular reference | Pathogenicity classifications and molecular variant annotations (e.g. *TP53*, *NPM1*, *FLT3*). |
| **Tier 6** | **Clinical Investigation Registries** | U.S. NLM ClinicalTrials.gov | Experimental/Investigational | Active interventional protocol matching. |

## Integration Safeguards
- Research publications (Tier 4) are never presented as official diagnostic consensus guidelines (Tier 1).
- Disagreements between classification editions (e.g. WHO 5th Ed vs ICC 2022 on 10–19% blasts in MDS/AML) are preserved transparently.
- All retrieved records maintain persistent citation provenance (`evidence_id`, `document_title`, `issuing_organization`, `year`, `section`, `doi_or_url`, `excerpt`).

