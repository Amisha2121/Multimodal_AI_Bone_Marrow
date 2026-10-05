# Real Cell Extraction and Morphological Annotation Audit

> **Dataset:** BoneMarrowWSI-PediatricLeukemia (`bonemarrowwsi_pediatricleukemia`)  
> **Source Modality:** DICOM Microscopy Bulk Simple Annotations (ANN) + DICOM Slide Microscopy (SM)  
> **Cohort:** 113 Supervised Consensus-Annotated Patients  
> **Timestamp:** 2026-10-05  
> **Audit Status:** VERIFIED — 100% Parsed & Validated

---

## 1. Executive Summary

An exhaustive extraction audit was conducted across all **113 consensus DICOM annotation series** from the real pediatric leukemia cohort. Annotation bounding box coordinates were decoded directly from `PointCoordinatesData` (RECTANGLE graphic type).

| Metric | Count | Percentage |
|---|:---:|:---:|
| **Total Annotations Parsed** | **28,395** | 100.0% |
| **Excluded Artefacts** | **6,544** | 23.04% |
| **Excluded Marrow Fragments (Spicules)** | **316** | 1.11% |
| **Excluded Multi-Annotator Disagreements** | **683** | 2.41% |
| **Total Valid Single Cells for Training/Eval** | **20,852** | **73.44%** |
| **Patients Represented** | **113 / 113** | 100.0% |
| **Target Crop Standard** | **224 × 224 px** | RGB, 20% margin padding, bicubic resize |

---

## A. Total Valid Extracted Cells

- **Total Valid Single Cells:** **20,852**
- These cells represent legitimate cytomorphological entities suitable for single-cell computer vision classification.
- All valid cells strictly inherit their split partition (`train`, `val`, `test`) from their parent `patient_id` as recorded in [`data/processed/splits.json`](../data/processed/splits.json).

---

## B. Total Excluded Artefacts

A total of **6,544 annotations (23.04%)** were identified as non-cellular artefacts or unclassifiable debris and excluded from differential calculations:

| Artefact Category | Verbatim DICOM Label | Count | Reason for Exclusion |
|---|---|:---:|---|
| Smear Quality Debris | `damaged_cell` | 1,943 | Structurally disrupted cell membrane / ruptured cytoplasm |
| Smudge / Shadow Cell | `smudge_cell` | 1,820 | Disrupted lymphoid chromatin (Gumprecht shadows) |
| Inadequate Staining / Focus | `technically_unfit` | 1,340 | Optical blur, precipitation, or unclassifiable smear artifact |
| NETs | `neutrophil_extracellular_trap` | 719 | Extracellular chromatin fibril web (not a cell) |
| Slide Foreign Body | `artifact` | 663 | Slide dust, dye crystal precipitate, or glass scratch |
| Ingestion Complex | `phagocytosis` | 59 | Engulfed cell remnant complex |
| **Total Excluded Artefacts** | | **6,544** | |

*Additional Non-Cell Exclusions:*
- `spicule` (**316**): Macroscopic bone marrow fragments spanning 6,000 to 34,000 pixels (see Section K).
- `no_consensus_found` (**683**): Bounding boxes where expert annotators failed to achieve consensus agreement.

---

## C. Cell Count by Original Annotation Label

Complete census across all 49 original DICOM labels:

| Original Annotation Label | Total Count | Classification Category | Action |
|---|:---:|---|:---:|
| `lymphocytic_blast` | 6,241 | Blast (Lymphoid) | Valid Cell |
| `segmented_neutrophil` | 2,791 | Granulocyte (Mature) | Valid Cell |
| `damaged_cell` | 1,943 | Artefact | **Excluded** |
| `neutrophilic_band` | 1,825 | Granulocyte (Band) | Valid Cell |
| `smudge_cell` | 1,820 | Artefact | **Excluded** |
| `promyelocyte` | 1,573 | Promyelocyte | Valid Cell |
| `neutrophilic_myelocyte` | 1,512 | Myeloid (Maturing) | Valid Cell |
| `technically_unfit` | 1,340 | Artefact | **Excluded** |
| `thrombocyte` | 1,234 | Platelet series | Valid Cell |
| `neutrophilic_metamyelocyte` | 967 | Myeloid (Maturing) | Valid Cell |
| `megakaryocyte` | 738 | Megakaryocyte | Valid Cell |
| `neutrophil_extracellular_trap` | 719 | Artefact | **Excluded** |
| `no_consensus_found` | 683 | Ambiguous | **Excluded** |
| `artifact` | 663 | Artefact | **Excluded** |
| `myelocytic_blast` | 397 | Blast (Myeloid) | Valid Cell |
| `lymphocyte` | 382 | Lymphoid Series | Valid Cell |
| `monocytic_blast` | 376 | Blast (Monocytic) | Valid Cell |
| `spicule` | 316 | Tissue Fragment | **Excluded** |
| `orthochromatic_erythroblast` | 278 | Erythroid Precursor | Valid Cell |
| `segmented_eosinophil` | 263 | Eosinophil (Mature) | Valid Cell |
| `eosinophilic_myelocyte` | 245 | Eosinophil (Maturing) | Valid Cell |
| `erythrocyte` | 244 | Red Blood Cell | Valid Cell |
| `monocyte` | 209 | Monocyte Series | Valid Cell |
| `myeloid_precursor_cell` | 209 | Myeloid (Maturing) | Valid Cell |
| `polychromatic_erythroblast` | 162 | Erythroid Precursor | Valid Cell |
| `segmented_basophil` | 155 | Basophil (Mature) | Valid Cell |
| `degranulated_neutrophilic_myelocyte` | 86 | Myeloid (Maturing) | Valid Cell |
| `giant_platelet` | 84 | Platelet series | Valid Cell |
| `eosinophilic_metamyelocyte` | 83 | Eosinophil (Maturing) | Valid Cell |
| `eosinophilic_band` | 81 | Eosinophil (Band) | Valid Cell |
| `unknown_blast` | 79 | Blast (Unspecified) | Valid Cell |
| `proerythroblast` | 77 | Erythroid Precursor | Valid Cell |
| `mitosis` | 72 | Mitotic Figure | Valid Cell |
| `pseudo_gaucher_cell` | 71 | Histiocytic series | Valid Cell |
| `lymphoidocyte` | 66 | Lymphoid Series | Valid Cell |
| `phagocytosis` | 59 | Artefact | **Excluded** |
| `promonocyte` | 56 | Monocyte Series | Valid Cell |
| `basophilic_myelocyte` | 53 | Basophil (Maturing) | Valid Cell |
| `thrombocyte_aggregate` | 53 | Platelet series | Valid Cell |
| `prolymphocyte` | 51 | Lymphoid Series | Valid Cell |
| `degranulated_neutrophilic_metamyelocyte` | 51 | Myeloid (Maturing) | Valid Cell |
| `basophilic_erythroblast` | 39 | Erythroid Precursor | Valid Cell |
| `plasma_cell` | 16 | Plasma Cell | Valid Cell |
| `basophilic_metamyelocyte` | 12 | Basophil (Maturing) | Valid Cell |
| `lymphoid_precursor_cell` | 10 | Lymphoid Series | Valid Cell |
| `basophilic_band` | 7 | Basophil (Band) | Valid Cell |
| `macrophage` | 2 | Histiocytic series | Valid Cell |
| `micromegakaryocyte` | 1 | Megakaryocyte | Valid Cell |
| `promegakaryocyte` | 1 | Megakaryocyte | Valid Cell |

---

## D. Cell Count by Mapped Morphology Category

The 20,852 valid single cells map into 8 functional morphological categories:

| Mapped Morphology Category | Cell Count | Percentage of Valid Cells | Clinical Relevance |
|---|:---:|:---:|---|
| **Blast** | **7,649** | 36.68% | Hallmark of acute leukemia (ALL / AML) |
| **Granulocyte (Mature/Band)** | **5,122** | 24.56% | Segmented/band neutrophils, eosinophils, basophils |
| **Myeloid (Maturing)** | **3,228** | 15.48% | Myelocytes, metamyelocytes (prominent in CML) |
| **Megakaryocyte / Platelet** | **2,111** | 10.12% | Thrombocytes, giant platelets, megakaryocytes |
| **Promyelocyte** | **1,573** | 7.54% | Prominent in APL and early myeloid hyperplasia |
| **Lymphoid Series** | **515** | 2.47% | Mature lymphocytes, prolymphocytes, plasma cells |
| **Other Morphological** | **389** | 1.87% | Erythrocytes, mitotic figures, macrophages |
| **Monocyte Series** | **265** | 1.27% | Monocytes and promonocytes |
| **Total Valid Single Cells** | **20,852** | **100.00%** | |

---

## E. Cell Count by Patient Statistics

Distribution of valid cell annotations across the 113 consensus patients:

| Statistic | Value |
|---|---|
| **Mean Cells per Patient** | 184.5 cells |
| **Median Cells per Patient** | 176.0 cells |
| **Standard Deviation** | 111.2 cells |
| **Minimum** | 3 cells (Patient `2B24D495052A8CE66358EB576B8912C8`) |
| **25th Percentile (Q1)** | 97.0 cells |
| **75th Percentile (Q3)** | 261.0 cells |
| **Maximum** | 638 cells (Patient `A1D0C6E83F027327D8461063F4AC58A6`) |

---

## F. Cell Count by Train / Validation / Test Split

Partitioning strictly enforces the patient-level assignments from [`splits.json`](../data/processed/splits.json):

| Split | Patient Count | Patient % | Valid Cell Count | Cell % | Mean Cells/Pt |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train** | **79** | 69.9% | **14,563** | 69.84% | 184.3 |
| **Validation** | **17** | 15.0% | **2,951** | 14.15% | 173.6 |
| **Test** | **17** | 15.0% | **3,338** | 16.01% | 196.4 |
| **Total** | **113** | 100.0% | **20,852** | **100.00%** | **184.5** |

---

## G. Cell Count by Class Within Each Split

| Mapped Morphology Class | Train Split | Validation Split | Test Split | Total Valid Cells |
|---|:---:|:---:|:---:|:---:|
| **Blast** | 5,348 | 1,072 | 1,229 | **7,649** |
| **Granulocyte (Mature/Band)** | 3,558 | 759 | 805 | **5,122** |
| **Myeloid (Maturing)** | 2,284 | 436 | 508 | **3,228** |
| **Megakaryocyte / Platelet** | 1,446 | 316 | 349 | **2,111** |
| **Promyelocyte** | 1,124 | 194 | 255 | **1,573** |
| **Lymphoid Series** | 363 | 74 | 78 | **515** |
| **Other Morphological** | 254 | 63 | 72 | **389** |
| **Monocyte Series** | 186 | 37 | 42 | **265** |
| **Total** | **14,563** | **2,951** | **3,338** | **20,852** |

All 8 morphological classes are represented across all three splits in stable proportions.

---

## H. Number of Patients Represented in Each Morphology Class

| Mapped Morphology Class | Number of Patients ($N=113$) | Patient Representation % |
|---|:---:|:---:|
| **Blast** | 112 / 113 | 99.1% |
| **Lymphoid Series** | 100 / 113 | 88.5% |
| **Other Morphological** | 86 / 113 | 76.1% |
| **Granulocyte (Mature/Band)** | 84 / 113 | 74.3% |
| **Myeloid (Maturing)** | 83 / 113 | 73.5% |
| **Megakaryocyte / Platelet** | 81 / 113 | 71.7% |
| **Promyelocyte** | 64 / 113 | 56.6% |
| **Monocyte Series** | 60 / 113 | 53.1% |

---

## I. Failed / Missing / Corrupt Extractions

| Check | Result |
|---|---|
| DICOM Parse Failures | **0** (All 113 consensus DICOM files parsed successfully) |
| Missing PointCoordinatesData | **0** |
| Coordinate Decoding Errors | **0** |
| Truncated Byte Buffers | **0** |
| Zero-dimension Bounding Boxes | **0** |
| Extraction Pipeline Error Rate | **0.00%** |

---

## J. Example Crop Paths & Standards

Extracted crops are saved as standard PNG files:
- `data/processed/sample_crops/02522A2B2726FB0A03BB19F2D8D9524D_lymphocytic_blast_003.png` (224×224 RGB)
- `data/processed/sample_crops/02522A2B2726FB0A03BB19F2D8D9524D_eosinophilic_myelocyte_000.png` (224×224 RGB)
- `data/processed/sample_crops/02522A2B2726FB0A03BB19F2D8D9524D_giant_platelet_001.png` (224×224 RGB)

### Standardized Crop Procedure
1. Bounding box coordinates $(x_{min}, y_{min}, x_{max}, y_{max})$ decoded from floats.
2. Centroid computed: $c_x = (x_{min} + x_{max})/2$, $c_y = (y_{min} + y_{max})/2$.
3. Square crop dimension: $S = \max(width, height) \times 1.20$ (20% margin for cellular boundary context).
4. Top-left corner: $x_0 = \text{round}(c_x - S/2)$, $y_0 = \text{round}(c_y - S/2)$.
5. Remote tile streamed at Level 0 magnification via `wsidicom`.
6. Bicubic interpolation applied to standardize patch to $224 \times 224$ pixels.

---

## K. Coordinate & DICOM Interpretation Findings

1. **DICOM Microscopic Coordinate Space**: Coordinates in `PointCoordinatesData` are stored as IEEE 754 float32 values referencing the whole-slide Level 0 pixel coordinate grid.
2. **Macroscopic Spicules**:
   - 268 `spicule` annotations have dimensions exceeding 1,000 pixels (some up to $21,546 \times 34,587$ pixels).
   - In bone marrow aspirates, a spicule is an intact marrow tissue core/aggregate, not a single cell.
   - **Handling:** Spicules are classified as `excluded_spicule` in the master metadata and excluded from single-cell training.
3. **Absence of Ambiguity**: All coordinates strictly adhere to the 8-float RECTANGLE specification ($(x_1, y_1), (x_2, y_1), (x_2, y_2), (x_1, y_2)$).

---

## 10. Leakage & Duplication Anomaly Report

The dataset was analyzed for coordinate anomalies, overlaps, and patient outliers:

1. **Exact Duplicate Coordinates**:
   $$\text{Exact Duplicate Boxes Within Same Patient} = \mathbf{0}$$
   No identical duplicate bounding boxes exist in any patient slide.
2. **Overlapping Bounding Boxes (IoU > 0.5)**:
   Only **1 pair** of overlapping boxes with $\text{IoU} > 0.5$ exists in the entire collection of 28,395 annotations:
   - Patient `A1D0C6E83F027327D8461063F4AC58A6`: Two overlapping boxes labeled `artifact` ($\text{IoU} = 0.926$).
   - Both boxes are already marked as `excluded_artefact`.
3. **High-Annotation Patient Outlier**:
   - Patient `A1D0C6E83F027327D8461063F4AC58A6` has 782 total annotations (638 valid single cells).
   - *Recommendation:* Retained. This represents a highly cellular marrow field with dense annotator consensus; class-weighted loss prevents undue patient skew.
4. **Low-Annotation Patient Outliers**:
   - Patient `2B24D495052A8CE66358EB576B8912C8`: 6 annotations (3 valid cells).
   - Patient `7EF605FC8DBA5425D6965FBD4C8FBE1F`: 15 annotations (12 valid cells).
   - *Recommendation:* Retained in their respective splits (`train` / `val`), but patient-level aggregation flags them with a sample sufficiency warning (`N < 50`).

---

*Audit completed on 2026-10-05. Master metadata saved to `data/processed/cell_metadata.csv`.*
