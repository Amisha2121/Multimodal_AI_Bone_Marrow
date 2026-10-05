# Data Provenance and Real Dataset Audit

This document provides a comprehensive audit of all biomedical datasets evaluated, downloaded, and utilized in this project. It enforces strict data provenance standards, prevents cross-modal fabrication, and establishes the verified real-data foundation for the research pipeline.

---

## 1. Verified Real Dataset: BoneMarrowWSI-PediatricLeukemia

### 1.1 Dataset Identification & Acquisition
- **Collection Name:** BoneMarrowWSI-PediatricLeukemia
- **IDC Collection ID:** `bonemarrowwsi_pediatricleukemia`
- **Originating Institution:** University Hospital Erlangen and Fraunhofer MEVIS
- **Data Repository:** NCI Imaging Data Commons (IDC) & TCIA / Zenodo
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Modality Standards:** DICOM Slide Microscopy (SM) and DICOM Comprehensive 2D Annotation (ANN)
- **Staining:** May-Grünwald-Giemsa (MGG) on bone marrow aspirate smears

### 1.2 Direct Inspection Findings (Audited via `pydicom` and IDC SQL)
Unlike initial secondary literature claims, exhaustive inspection of the 245 patient DICOM series confirmed the following verified facts:

| Modality / Feature | Status | Verification Source |
|---|:---:|---|
| Total Patients in Collection | **245** | IDC Index SQL (`PatientID` distinct count) |
| Whole Slide Microscopy (SM) | **245 / 245** | DICOM series with `Modality = SM` (avg 6 tile instances/pt) |
| Monolayer Regions of Interest | **245 / 245** | DICOM series with `Modality = ANN`, label `region_of_interest` |
| Consensus Cell Bounding Boxes | **113 / 245** | DICOM series `Consensus: cell bounding boxes with cell type labels` |
| Individual Annotator Sessions | 108–113 pts | Multi-session DICOM ANN series (Sessions 0–5) |
| Unlabeled Bounding Boxes | 97 pts | DICOM ANN series `Unlabeled cell bounding boxes` |
| Patient Diagnosis | **242 / 245** | DICOM tag `AdmittingDiagnosesDescription` (3 missing) |
| Patient Age | **245 / 245** | DICOM tag `PatientAge` (range 18–204 months, median 114m) |
| Patient Sex | **245 / 245** | DICOM tag `PatientSex` (129 M, 110 F, 6 O) |
| Complete Blood Count (CBC) | ❌ **UNAVAILABLE** | Verified absent from all DICOM headers |
| Cytogenetics / Karyotype | ❌ **UNAVAILABLE** | Verified absent from all DICOM headers |
| Molecular Genetics (FISH/PCR) | ❌ **UNAVAILABLE** | Verified absent from all DICOM headers |
| Adult MDS Patients | ❌ **ZERO CASES** | Entire cohort is pediatric leukemia (ALL, AML, CML, MPAL) |

---

## 2. Cohort Structure & Split Integrity

### 2.1 Primary Supervised Cohort (113 Patients with Consensus Labels)
Supervised cell-level training, feature aggregation, and patient-level baseline classification are performed exclusively on the 113 patients with multi-annotator consensus annotations:
- **ALL (n=37):** B-ALL (32), T-ALL (5)
- **AML (n=17):** AML with maturation (5), Acute monocytic (5), APL t(15;17) (3), Acute myelomonocytic (3), AML without maturation (1)
- **CML (n=50):** Chronic phase CML BCR-ABL1+ (46), Blast phase CML BCR-ABL1+ (4)
- **Other (n=9):** Mixed phenotype acute leukemia (6), Unspecified/NaN (3)

### 2.2 Secondary / Extension Cohort (132 Patients without Consensus Labels)
The remaining 132 patients possess verified WSI images, monolayer ROIs, and clinical diagnoses, but lack consensus cell bounding box labels.
- **Handling:** Retained in `data/patient_manifest.csv` under `split = "unlabeled_extension"`.
- **Policy:** Excluded from initial supervised training to prevent label noise. Reserved for future weakly supervised learning, Multiple Instance Learning (MIL), and representation learning extensions.

### 2.3 Patient-Level Split Manifest (`data/processed/splits.json`)
A deterministic Stratified Shuffle Split (`random_seed = 42`) yields:
- **Train Split:** 79 patients (35 CML, 26 ALL, 12 AML, 6 Other)
- **Validation Split:** 17 patients (7 CML, 5 ALL, 3 AML, 2 Other)
- **Test Split:** 17 patients (8 CML, 6 ALL, 2 AML, 1 Other)

**Zero Leakage Guarantee:**
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
Cell crops inherit their patient's partition membership, completely eliminating cross-split patch leakage.

---

## 3. Strict Rules for Data Integrity & Scientific Validity

1. **Real Data Only for Core Pipeline:** Model parameters are optimized and evaluated strictly on verified real biomedical data.
2. **Quarantine of Synthetic Fixtures:** The 14 mock cases in `synthetic/` exist solely for CI unit testing of code syntax. They are isolated in `data/synthetic/README.md` and explicitly forbidden from model training or reported results.
3. **No Cross-Modal Fabrication:** Because CBC and cytogenetics are absent from the DICOM files, the pipeline strictly sets `cbc_values = null` and `cytogenetics = null`. We never impute or fabricate laboratory data.
4. **Distinction of Annotated-Cell Blast Fraction:** The blast fraction extracted from consensus bounding boxes represents the proportion of blasts among *annotated* cells within sampled monolayer fields. It must be described as `annotated-cell blast fraction` and not conflated with exhaustive whole-slide clinical differential counts.
5. **Preservation of Raw Annotations:** Raw consensus cell labels (e.g. `myelocytic_blast`, `technically_unfit`) and original diagnoses are preserved verbatim in all manifests alongside high-level groupings.

---

*Verified by direct inspection on 2026-10-05.*
