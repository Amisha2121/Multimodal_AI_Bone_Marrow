# Dataset Preparation Report — Medical Document AI

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 3 — Dataset Preparation  
**Completion Date**: 2026-11-11  
**Execution Status**: Successfully Completed — 100% Validated

---

## 1. Executive Summary

In Stage 3 of the Medical Document AI roadmap, a comprehensive, clinically grounded, leakage-resistant document training and evaluation corpus was assembled, prepared, and validated.

All requirements have been met:
- **Zero Real PHI**: Fictional procedural patient identities with explicit `"is_synthetic": true` annotations.
- **15 Diagnostic Categories**: Exceeding the initial 12 categories to provide full coverage of bone marrow, morphology, flow cytometry, cytogenetics, molecular biology, and supportive hematology panels.
- **Strict Layout Disjointness**: Template C variants are 100% isolated to the test set, preventing visual memorization during future model training.
- **Zero Pipeline Interference**: Core Vision Transformer weights, FAISS retrieval index, BGE embeddings, and BM25 pickles were preserved with zero modifications.

---

## 2. Directory Structure & Inventory

The dataset is organized under `data/document_ai/`:

```
data/document_ai/
├── raw/
│   ├── mtsamples/
│   │   └── mtsamples_hematology.json (90 real-world hematology/oncology consult transcriptions)
│   └── india_ocr_test/
│       └── india_medical_ocr_metadata.json (verified OOD evaluation metadata)
├── processed/
│   └── (pre-tokenized tensors for Stage 4 training)
├── synthetic_corpus/
│   ├── 300 PDF reports (generated via ReportLab across Templates A, B, C)
│   └── 300 TXT reports (formatted paired plain-text representations)
├── annotations/
│   └── 300 JSON ground-truth annotation files ({report_id}_annotation.json)
├── splits/
│   └── split_manifest.json (Leakage-resistant train/val/test assignment)
└── manifests/
    ├── dataset_registry.json (Global provenance registry)
    ├── synthetic_annotations_master.json (Corpus master list)
    └── quality_report.json (Automated validation audit result)
```

---

## 3. Dataset Composition & Split Statistics

| Split | Document Count | Percentage | Template Variants | Role |
|---|---|---|---|---|
| **Train** | 180 | 60.0% | `TEMPLATE_A`, `TEMPLATE_B` | Primary model fine-tuning (DiT, LayoutLMv3, BioLinkBERT) |
| **Validation** | 30 | 10.0% | `TEMPLATE_A`, `TEMPLATE_B` | Hyperparameter selection, early stopping |
| **Test** | 90 | 30.0% | `TEMPLATE_C` **strictly** | Out-of-distribution visual layout generalization evaluation |
| **Total** | **300** | **100.0%** | **All 3 Templates** | **Zero leakage, zero overlap** |

### Per-Category Breakdown (20 Samples per Category)

1. `CAT_A_MORPHOLOGY` — Bone Marrow Aspirate Report (Differential count, M:E ratio, blast %)
2. `CAT_B_BIOPSY` — Bone Marrow Trephine Biopsy Report (Cellularity, MF fibrosis grading)
3. `CAT_C_CBC` — Complete Blood Count (CBC) with automated differential
4. `CAT_D_SMEAR` — Peripheral Blood Smear Examination (Manual differential & morphology)
5. `CAT_E_FLOW` — Flow Cytometric Immunophenotyping (Lymphoid & myeloid marker panels)
6. `CAT_F_IHC` — Immunohistochemistry (IHC) Panel (CD3, CD20, CD34, MPO, Ki-67)
7. `CAT_G_FISH` — Cytogenetics FISH (BCR-ABL1, PML-RARA, del(5q), del(17p))
8. `CAT_H_KARYOTYPE` — Conventional Cytogenetics / Karyotype (ISCN 2020 formula)
9. `CAT_I_NGS` — Next-Generation Sequencing (Myeloid mutational panel, VAF %)
10. `CAT_J_PCR` — Molecular RT-PCR / Quantitative PCR (%IS transcript levels)
11. `CAT_K_IRON` — Iron Studies and Reticulocyte Assessment
12. `CAT_L_COAG` — Coagulation Profile (PT, INR, aPTT, Fibrinogen, D-Dimer)
13. `CAT_M_MRD` — Minimal Residual Disease (MRD) Assessment
14. `CAT_N_TRANSPLANT` — Bone Marrow Transplant Chimerism & Engraftment Report
15. `CAT_O_SPEP` — Serum Protein Electrophoresis & Free Light Chain Assay

---

## 4. Real-World Benchmark Sets (OOD Evaluation)

- **MTSamples Hematology Subset**: 90 verified real-world clinical transcriptions loaded into `data/document_ai/raw/mtsamples/mtsamples_hematology.json` for biomedical NER vocabulary and boundary benchmarking.
- **India Medical OCR Dataset**: Evaluated for non-Western document formatting and degraded scan generalization (`data/document_ai/raw/india_ocr_test/`).

---

## 5. Automated Validation & Quality Assurance

The dataset was verified using `scripts/document_ai/validate_dataset.py`:
- **Errors**: 0
- **Warnings**: 0
- **Validation Status**: `PASSED`
- **File Pairing**: 300/300 paired PDFs and TXT files verified.
- **Annotation Completeness**: All 300 JSON annotations contain 100% of required clinical, demographic, and parameter fields.
- **Template Isolation**: Confirmed 0 samples of `TEMPLATE_C` in train or val splits.

---

## 6. Stage 3 Completion Statement

Stage 3 (Dataset Preparation) is completely finalized. In accordance with the system directives, **no model training will commence until explicit approval is granted by the user**.
