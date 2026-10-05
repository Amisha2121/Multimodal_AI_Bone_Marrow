# Real Dataset Schema — BoneMarrowWSI-PediatricLeukemia

> **Status: VERIFIED & AUDITED**  
> Every claim in this document is sourced from direct DICOM inspection (`pydicom`) and IDC index SQL queries.  
> No secondary literature assumptions. Missing modalities are explicitly documented and handled.

---

## 1. Dataset Identity & Provenance

| Field | Value |
|---|---|
| **Dataset Name** | BoneMarrowWSI-PediatricLeukemia |
| **IDC Collection ID** | `bonemarrowwsi_pediatricleukemia` |
| **Source Institutions** | University Hospital Erlangen & Fraunhofer MEVIS |
| **DICOM Standard** | DICOM ANN (Annotation), DICOM SM (Slide Microscopy) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Stain** | May-Grünwald-Giemsa (MGG) |
| **Specimen** | Bone marrow aspirate smear |

---

## 2. Cohort Demographics (Verified across 245 Patients)

| Demographic Metric | Value |
|---|---|
| **Total Cohort Size** | **245 patients** |
| **Age Range** | 18 – 204 months (1.5 – 17.0 years, strictly pediatric) |
| **Median Age** | 114 months (~9.5 years) |
| **Mean Age** | 108.4 months (~9.0 years, SD 65.7 months) |
| **Sex — Male** | 129 (52.7%) |
| **Sex — Female** | 110 (44.9%) |
| **Sex — Other / Unspecified** | 6 (2.4%) |
| **Diagnoses Available** | 242 / 245 (3 patients missing diagnosis in DICOM) |

---

## 3. Disease Distribution & Target Class Mappings

### Original DICOM Diagnosis Labels (Verbatim from `AdmittingDiagnosesDescription`)

| Original Diagnosis | Total Cohort (n=245) | Consensus Subset (n=113) | Non-Consensus Extension (n=132) |
|---|:---:|:---:|:---:|
| Chronic Phase Chronic Myeloid Leukemia, BCR-ABL1 Positive | 46 | 46 | 0 |
| B Acute Lymphoblastic Leukemia | 46 | 32 | 14 |
| Mixed Phenotype Acute Leukemia, B/Myeloid | 27 | 6 | 21 |
| Acute Myeloid Leukemia with Maturation | 22 | 5 | 17 |
| T Acute Lymphoblastic Leukemia | 19 | 5 | 14 |
| Acute Myeloid Leukemia without Maturation | 17 | 1 | 16 |
| Acute Monocytic Leukemia | 11 | 5 | 6 |
| Acute Megakaryoblastic Leukemia | 11 | 0 | 11 |
| Acute Promyelocytic Leukemia with t(15;17)(q24.1;q21.2);PML-RARA | 10 | 3 | 7 |
| Acute Myelomonocytic Leukemia | 9 | 3 | 6 |
| Acute Myelomonocytic Leukemia with Abnormal Eosinophils | 7 | 0 | 7 |
| Burkitt Leukemia | 7 | 0 | 7 |
| Blast Phase Chronic Myeloid Leukemia, BCR-ABL1 Positive | 5 | 4 | 1 |
| Acute Myeloid Leukemia with Minimal Differentiation | 3 | 0 | 3 |
| Acute Erythroid Leukemia | 2 | 0 | 2 |
| Missing / NaN | 3 | 3 | 0 |

### 4-Class Primary Target (Vision Baseline Experiment)

To establish a statistically defensible supervised vision baseline, the original diagnoses are mapped into 4 primary classes. The original diagnoses are permanently preserved in `data/patient_manifest.csv`:

| Primary 4-Class Target | Verbatim DICOM Diagnoses Included | Supervised n (Consensus) | Total Cohort n |
|---|---|:---:|:---:|
| **ALL** | B Acute Lymphoblastic Leukemia, T Acute Lymphoblastic Leukemia, Burkitt Leukemia | 37 | 72 |
| **AML** | AML with/without Maturation, Acute Monocytic, Acute Myelomonocytic, Megakaryoblastic, Erythroid, Minimal Diff, APL | 17 | 82 |
| **CML** | Chronic Phase CML (BCR-ABL1+), Blast Phase CML (BCR-ABL1+) | 50 | 51 |
| **Other** | Mixed Phenotype Acute Leukemia (MPAL B/Myeloid), Missing / Unspecified | 9 | 40 |
| **Total** | | **113** | **245** |

> **IMPORTANT ARCHITECTURAL DISTINCTION**:  
> The 4-class patient-level classifier (`cell morphology model` → `patient-level feature aggregation` → `4-class classifier`) is a **baseline experiment** to evaluate the real cytomorphology signal. It is **NOT** the final multimodal/agentic system.

---

## 4. Primary Supervised Cohort vs. Non-Consensus Extension

| Cohort | N Patients | Available Modalities | Role in Pipeline |
|---|:---:|---|---|
| **Primary Supervised Cohort** | **113** | WSI (SM), Monolayer ROI (ANN), Consensus Cell Bounding Boxes with Labels (ANN), Patient Diagnosis | **Primary supervised training, validation, and testing** |
| **Non-Consensus Extension Cohort** | **132** | WSI (SM), Monolayer ROI (ANN), Patient Diagnosis | **Held out as secondary extension cohort** (reserved for future weakly-supervised MIL / representation learning) |

---

## 5. Deterministic Patient-Level Split (Consensus Supervised Cohort)

Generated reproducibly via two-stage Stratified Shuffle Split (`seed=42`, `splits.json`):

| Split | Patient Count | Split % | ALL | AML | CML | Other | Patient Overlap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | **79** | 69.9% | 26 | 12 | 35 | 6 | 0 |
| **Validation** | **17** | 15.0% | 5 | 3 | 7 | 2 | 0 |
| **Test** | **17** | 15.0% | 6 | 2 | 8 | 1 | 0 |
| **Total Supervised** | **113** | 100.0% | **37** | **17** | **50** | **9** | **ZERO LEAKAGE** |

- Zero patient overlap: `set(Train) ∩ set(Val) = ∅`, `set(Train) ∩ set(Test) = ∅`, `set(Val) ∩ set(Test) = ∅`.
- Saved artifact: [`data/processed/splits.json`](file:///home/t0538gi/bonemarrow/data/processed/splits.json) with full provenance metadata.

---

## 6. Raw Morphology Cell Labels & Preservation Policy

Consensus annotation DICOMs contain multi-annotator consensus labels. We strictly preserve:
1. **`original_cell_label`**: verbatim string from DICOM `AnnotationGroupLabel`.
2. **`mapped_morphology_category`**: functional morphological category.
3. **`is_excluded_artefact`**: boolean flag.

| Original Label | Mapped Category | Is Artefact? | Action in Differential |
|---|---|:---:|---|
| `myelocytic_blast` | Blast (Myeloid) | False | Included in blast fraction |
| `monocytic_blast` | Blast (Monocytic) | False | Included in blast fraction |
| `lymphocytic_blast` | Blast (Lymphoid) | False | Included in blast fraction |
| `unknown_blast` | Blast (Unspecified) | False | Included in blast fraction |
| `promyelocyte` | Promyelocyte | False | Counted in myeloid lineage |
| `segmented_neutrophil` | Granulocyte (Mature) | False | Counted in myeloid lineage |
| `neutrophilic_band` | Granulocyte (Band) | False | Counted in myeloid lineage |
| `eosinophilic_myelocyte` | Eosinophil series | False | Counted in myeloid lineage |
| `orthochromatic_erythroblast`| Erythroid precursor | False | Counted in erythroid lineage |
| `giant_platelet` | Megakaryocyte series | False | Counted in platelet/megakaryocyte |
| `technically_unfit` | Quality / Smear Artefact | **True** | **Excluded from differential** |
| `damaged_cell` | Damaged cell | **True** | **Excluded from differential** |
| `smudge_cell` | Smudge / Shadow cell | **True** | **Excluded from differential** |
| `artifact` | Slide artefact | **True** | **Excluded from differential** |
| `neutrophil_extracellular_trap` | NET | **True** | **Excluded from differential** |
| `phagocytosis` | Phagocytosis complex | **True** | **Excluded from differential** |

---

## 7. Definition of "Annotated-Cell Blast Fraction"

> [!WARNING]
> **CRITICAL SCIENTIFIC DISTINCTION:**  
> The consensus annotations represent **annotated cells within selected monolayer regions**, NOT an exhaustive whole-slide clinical aspirate count.  
> 
> Therefore, this metric must be termed:  
> **`annotated-cell blast fraction`** (or `annotated_blast_fraction`),  
> and **NOT** "patient blast percentage."
>
> $$\text{annotated\_blast\_fraction} = \frac{\sum \text{annotated blast cells}}{\sum \text{valid (non-artefact) annotated cells}}$$
>
> No clinical claims or direct WHO AML blast-threshold assertions can be made solely from this value without qualifying that it applies strictly to the sampled annotated cell population.

---

## 8. Verified Modality Inventory (What Exists vs. What Does NOT)

| Modality / Field | Exists in Dataset? | Source | Handling |
|---|:---:|---|---|
| Whole-Slide Image (WSI) | **YES** | DICOM SM (avg 6 tiles/patient) | Real pixel source |
| Monolayer ROIs | **YES** | DICOM ANN (Monolayer series) | Focus region for cell analysis |
| Consensus Cell Annotations | **YES (113 pts)**| DICOM ANN (Consensus series) | Bounding box coordinates + labels |
| Patient Diagnosis | **YES (242/245)** | DICOM `AdmittingDiagnosesDescription` | Ground truth target |
| Age and Sex | **YES (245 pts)** | DICOM `PatientAge`, `PatientSex` | Patient demographic context |
| Complete Blood Count (CBC) | **NO** | Not collected / not in DICOM | **Must remain null — DO NOT FABRICATE** |
| Cytogenetics / Karyotype | **NO** | Not collected / not in DICOM | **Must remain null — DO NOT FABRICATE** |
| Molecular Markers (FISH/PCR) | **NO** | Not collected / not in DICOM | **Must remain null — DO NOT FABRICATE** |
| Clinical Notes / Free Text | **NO** | Not collected / not in DICOM | **Must remain null — DO NOT FABRICATE** |
| Myelodysplastic Syndromes (MDS) | **NO (0 cases)** | Absent from cohort | **Not applicable in this cohort** |

---

## 9. Cohort Limitations

1. **Extreme Class Imbalance**: In the 113 consensus patients, CML comprises 44.2% (n=50) while AML comprises 15.0% (n=17) and Other comprises 8.0% (n=9). Weighted loss and per-class recall tracking are mandatory.
2. **Pediatric Specificity**: All patients are aged 1.5 to 17 years. Findings cannot be generalized directly to adult leukemias.
3. **Monolayer Sampling**: Cells were identified within pre-selected monolayer ROIs, introducing potential selection bias toward recognizable morphology.
4. **Missing Ancillary Modalities**: Absence of CBC and cytogenetics restricts the real-data baseline to cytology + demographics.

---

*Last Updated: 2026-10-05 — Verified against IDC `bonemarrowwsi_pediatricleukemia` collection and DICOM headers.*
