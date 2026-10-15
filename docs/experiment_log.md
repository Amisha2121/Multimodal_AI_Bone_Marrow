# docs/experiment_log.md — Experiment Log

Record all experiments here chronologically. One entry per training run, retrieval evaluation, or ablation.

---

## Template

```
### YYYY-MM-DD — [Phase] — [Short description]

**Config:** `configs/default.yaml` (+ any overrides)
**Seed:** 42

**What changed:** ...

**Results:**
| Metric | Value |
|---|---|
| Macro F1 | |
| Blast-class recall | |
| ... | |

**Observations:** ...

**Next steps:** ...
```

---

## Phase 0 — 2026-10-05 — Repository Scaffold

**Config:** N/A (scaffold only)  
**Seed:** N/A

**What was done:**
- Created directory structure per implementation plan (revised v2)
- Wrote `configs/default.yaml`, `configs/colab.yaml`, `configs/config_loader.py`
- Wrote `data/download_tcia.py` (TCIA download script)
- Wrote `data/dataset_cards/matek2021_bone_marrow.md`
- Wrote `requirements.txt`, `.env.example`, `README.md`
- Wrote `docs/clinical_disclaimer.md`, `docs/dataset_licenses.md`

**Dataset decision recorded:**
- Primary dataset: Matek 2021 (TCIA, CC BY 4.0, bone marrow aspirate, 21 classes, ~171k cells, 945 patients)
- Dropped: Matek 2019 (peripheral blood) and Acevedo 2020 (peripheral blood)

**Key design decisions locked:**
- Patient-level train/val/test splits (70/15/15)
- Vision output = cell-differential vector from N-crop aggregation
- No learned fusion module; structured context assembler instead
- NLI cross-encoder for Critic claim verification
- WHO 5th edition as primary classification standard
- 3 agents: Retriever, Synthesizer, Critic

**Next:** Phase 1b & Real-Data Cohort Audit, then ViT-Small LoRA baseline on real crops

---

## Phase 0c & 1 — 2026-10-05 — Vision Baseline Pipeline

**Config:** `configs/default.yaml`  
**Seed:** 42

**What was implemented & validated:**
1. **Vision Baseline Pipeline (`vision/`):**
   - Implemented `vision/dataset.py`: Patient-level stratified train/val/test splitting guaranteeing zero patient overlap across splits; class-imbalance `WeightedRandomSampler`; PyTorch `BoneMarrowCellDataset` with Wright-Giemsa augmentations.
   - Implemented `vision/model.py`: ViT-Small/16 (`vit_small_patch16_224`) wrapped with PEFT LoRA ($r=8, \alpha=16$) on attention projections. Validated only 0.68% trainable parameters (147,456 out of 21.8M params).
   - Implemented `vision/train.py`: Mixed-precision training loop, label smoothing ($\epsilon=0.1$), early stopping on macro F1.
   - Implemented `vision/evaluate.py`: Per-class recall, precision, F1, confusion matrix, and dedicated rare-class tracking (FGC, HAC, PEB, PLM, MOB).
   - Implemented `vision/aggregate.py`: Patient-level cell differential aggregation across 21 classes, WHO 5th blast percentage computation (`MYO` + `MOB`), and Wilson score 95% confidence intervals.
   - Implemented `vision/attention_rollout.py`: Abnar & Zuidema (2020) attention rollout producing 2D morphological spatial heatmaps.
2. **Structured Context Assembler (`context/assembler.py`):**
   - Deterministic multimodal context assembler uniting cell-differential vector and real morphology data; CBC/cytogenetics strictly `null` (never fabricated).
3. **Unit Test Suite (`tests/`):**
   - `tests/test_vision.py` (13 tests) passing at 100% on real data schema.

**Note:** A synthetic case generator (`synthetic/`) was created and subsequently deleted. No synthetic data remains in the repository.

**Next:** Phase 2 — RAG Layer: Literature ingest script, FAISS + BM25 hybrid indexing, and BGE reranker.

---

## Phase 1b — 2026-10-05 — Real-Data Cohort Audit & Deterministic Splitting

**Dataset:** BoneMarrowWSI-PediatricLeukemia (`bonemarrowwsi_pediatricleukemia`)  
**Seed:** 42  
**Manifest:** `data/patient_manifest.csv`  
**Splits:** `data/processed/splits.json`  

**Audit & Findings:**
1. **Full Cohort Verification (n=245):**
   - Downloaded and parsed all 245 patient DICOM series metadata.
   - Demographics: strictly pediatric (median age 9.5 years; range 1.5–17 years; 129 M, 110 F, 6 O).
   - Diagnoses: 242/245 annotated in `AdmittingDiagnosesDescription` (ALL, AML, CML, MPAL; 0 MDS cases).
   - Confirmed absence of CBC, cytogenetics, and molecular testing in DICOM headers.
2. **Supervised Consensus Cohort (n=113):**
   - 113 patients possess multi-annotator consensus cell bounding boxes with classification labels.
   - 132 patients without consensus labels held out as an unlabeled extension cohort (`split = "unlabeled_extension"`).
3. **Primary 4-Class Baseline Task Definition:**
   - Grouping: ALL (n=37), AML (n=17), CML (n=50), Other (n=9).
   - Established as a vision baseline experiment: `cell morphology model` → `feature aggregation` → `4-class classifier`.
   - Explicitly distinguished from the eventual multimodal agentic framework.
4. **Deterministic Stratified Patient Splits (seed=42):**
   - Train (n=79): 35 CML, 26 ALL, 12 AML, 6 Other
   - Val (n=17): 7 CML, 5 ALL, 3 AML, 2 Other
   - Test (n=17): 8 CML, 6 ALL, 2 AML, 1 Other
   - Zero patient leakage verified: `set(Train) ∩ set(Val) = ∅`, `set(Train) ∩ set(Test) = ∅`, `set(Val) ∩ set(Test) = ∅`.
5. **Scientific Clarifications:**
   - Defined metric as `annotated-cell blast fraction` to prevent misleading extrapolation to whole-marrow counts.
   - Preserved all raw consensus cell labels and original diagnoses verbatim in the manifest.

**Next:** Build real-data cell crop extractor from consensus DICOM bounding boxes and run initial cell morphology baseline training.
