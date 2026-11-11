# Synthetic Hematology Document Corpus Specification

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 3 — Dataset Preparation  
**Date**: 2026-11-11  
**Corpus Root**: `data/document_ai/synthetic_corpus/`  
**Ground-Truth Annotations**: `data/document_ai/annotations/`

---

## 1. Objectives & Compliance

1. **Patient Data Privacy**: All synthetic reports utilize procedural generation for patient names, dates of birth, MRNs, requesting physicians, and reporting hematopathologists. No actual protected health information (PHI) is present.
2. **Explicit Synthetic Labeling**: Every JSON annotation file contains `"is_synthetic": true`, and every generated PDF/text report contains clear disclaimers that the document is for research/evaluation purposes only.
3. **Medical Plausibility**: Clinical parameters, reference intervals, biological units, and pathological conclusions are rigorously structured using WHO 2022 (5th edition) hematolymphoid classification criteria, standard laboratory reference intervals, and real hematology report conventions.

---

## 2. Category Distribution (15 Categories, 300+ Documents)

The dataset covers 15 distinct hematology and bone marrow-related diagnostic report categories, with at least 20 balanced samples per category:

| Category ID | Clinical Report Category | Subtype Key | Target Samples | Parameter Schema Coverage |
|---|---|---|---|---|
| `CAT_A_MORPHOLOGY` | Bone Marrow Aspirate Report | `bone_marrow_aspirate` | 20 | Myeloblasts, Promyelocytes, Myelocytes, Metamyelocytes, Bands, Segs, Erythroblasts, Lymphocytes, Plasma cells, M:E ratio |
| `CAT_B_BIOPSY` | Bone Marrow Trephine Biopsy | `bone_marrow_biopsy` | 20 | Overall cellularity, Reticulin fibrosis grade (MF-0 to MF-3), CD34+ blast aggregates, Paratrabecular infiltration |
| `CAT_C_CBC` | Complete Blood Count (CBC) | `complete_blood_count` | 20 | Hb, Hct, MCV, MCH, MCHC, RBC, WBC, Platelets, Absolute Neutrophil/Lymphocyte counts, RDW |
| `CAT_D_SMEAR` | Peripheral Blood Smear Examination | `peripheral_blood_smear` | 20 | Manual WBC differential, blast presence, RBC poikilocytosis, toxic granulation, hypersegmentation |
| `CAT_E_FLOW` | Flow Cytometric Immunophenotyping | `flow_cytometry` | 20 | CD19+, CD3+, CD4+, CD8+, CD4:CD8 ratio, CD34+ blasts, Kappa/Lambda clonality, aberrant myeloid/lymphoid co-expression |
| `CAT_F_IHC` | Immunohistochemistry (IHC) Panel | `immunohistochemistry` | 20 | CD3, CD20, CD34, MPO, CD117, Ki-67 proliferation index |
| `CAT_G_FISH` | Cytogenetics FISH Report | `fish_cytogenetics` | 20 | BCR-ABL1 t(9;22), PML-RARA t(15;17), del(5q), del(7q), del(17p) TP53, MYC rearrangement |
| `CAT_H_KARYOTYPE` | Conventional Karyotyping / Cytogenetics | `karyotyping` | 20 | Mitoses analyzed, ISCN 2020 formula, clonal chromosomal aberrations |
| `CAT_I_NGS` | Myeloid Gene Mutation Panel (NGS) | `ngs_myeloid_panel` | 20 | DNMT3A, FLT3-ITD, NPM1, IDH1/2, TP53, ASXL1, RUNX1, VAF % |
| `CAT_J_PCR` | Molecular RT-PCR / Quantitative PCR | `molecular_pcr` | 20 | BCR-ABL1 %IS, PML-RARA transcript copies, JAK2 V617F allele burden |
| `CAT_K_IRON` | Iron Studies & Reticulocyte Assessment | `iron_reticulocyte` | 20 | Serum iron, TIBC, Transferrin saturation %, Ferritin, Absolute reticulocyte count, CHr |
| `CAT_L_COAG` | Coagulation / Hemostasis Profile | `coagulation` | 20 | PT, INR, aPTT, Fibrinogen, D-Dimer |
| `CAT_M_MRD` | Minimal Residual Disease (MRD) | `mrd_assessment` | 20 | Flow cytometric or molecular residual disease quantification, assay log-sensitivity |
| `CAT_N_TRANSPLANT` | Hematopoietic Stem Cell Engraftment | `transplant_engraftment` | 20 | Donor/host chimerism %, Day-to-engraftment metrics |
| `CAT_O_SPEP` | Serum Protein Electrophoresis & FLC | `spep_immunoglobulin` | 20 | Total protein, Albumin, M-spike band, IgG/A/M, Kappa, Lambda, FLC ratio |

---

## 3. Visual Layout Templates & Diversity

To prevent layout memorization, reports are generated across three distinct visual layout paradigms:

1. **`TEMPLATE_A` (Corporate Laboratory / Standard Clean Grid)**:
   - Primary blue/navy branding theme (`#1a3a5c`).
   - Clean tabular patient demographics header block.
   - Alternating row background shading (`#f0f4f8`).
   - Standard ISO reference range columns and explicit abnormality flags.

2. **`TEMPLATE_B` (Compact Clinical Hospital / Double-Column Condensed)**:
   - Deep burgundy/crimson accents (`#5c0a1a`).
   - Compact multi-column metadata grid.
   - Reduced padding, dense tabular lab outputs, and inline summary text.

3. **`TEMPLATE_C` (Academic Medical Center / Formal Boxed Layout)**:
   - Forest green institutional styling (`#0d4f1a`).
   - Strict boxed patient identification and specimen details headers.
   - Comprehensive interpretive commentary and formal ISCN/WHO diagnostic statements.
   - **Crucial Role**: Strictly held out as the unseen test template group for out-of-distribution evaluation.

---

## 4. Ground-Truth Annotation Schema

Every generated document in `synthetic_corpus/` has a 1-to-1 matching ground-truth JSON file in `annotations/{report_id}_annotation.json`:

```json
{
  "report_id": "SYNTH_BMA_0001",
  "is_synthetic": true,
  "category": "CAT_A_MORPHOLOGY",
  "subtype": "bone_marrow_aspirate",
  "template_id": "TEMPLATE_A",
  "institution_id": "INST_01",
  "institution_name": "St. Helena Medical Center",
  "patient_name": "David Miller",
  "patient_dob": "14 Mar 1968",
  "patient_mrn": "MRN-4918239",
  "patient_sex": "Male",
  "patient_age": 58,
  "collection_date": "18 Aug 2024",
  "report_status": "Final",
  "is_abnormal": true,
  "requesting_clinician": "Dr. A. Patel",
  "reporting_pathologist": "Dr. E. Schwartz, MD, PhD",
  "parameters": [
    {
      "name": "Myeloblasts",
      "reported_value": 14.8,
      "unit": "%",
      "reference_interval": "0.0–5.0",
      "flag": "H"
    }
  ],
  "conclusion": "Hypercellular marrow with increased blast percentage. Features consistent with myeloid neoplasm."
}
```
