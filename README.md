# An Agentic Multimodal Explainable RAG Framework for Personalized Pediatric Leukemia Assessment

> **⚠️ RESEARCH & DECISION-SUPPORT PROTOTYPE ONLY**  
> This software is an academic research prototype designed for computational pathology investigation. Outputs do **not** constitute medical advice or a clinical diagnosis. All outputs must be evaluated by a certified haematologist or pathologist.

---

## 1. Project Goal & Revised Scope

The goal of this research project is to build an explainable, evidence-grounded decision-support framework for bone marrow cytomorphology analysis. Following an exhaustive audit of publicly available biomedical cohorts, the core vision research pipeline is built upon the real, multi-annotated **BoneMarrowWSI-PediatricLeukemia** cohort.

### Revised Scope
- **Domain:** Pediatric acute and chronic leukaemias (age 1.5 – 17.0 years).
- **Primary Supervised Baseline:** 4-class patient-level classification (`ALL`, `AML`, `CML`, `Other`) derived from bone marrow aspirate cytomorphology.
- **Architecture Pipeline (Vision Baseline):**
  $$\text{Cell Morphology Model} \longrightarrow \text{Patient-Level Differential Aggregation} \longrightarrow \text{4-Class Baseline Classifier}$$
- **Modular Design:** While the current real-data cytology cohort lacks CBC and cytogenetics (which must remain `null` to preserve scientific integrity), the broader system architecture remains modular to accept laboratory, clinical, and genomic streams when available from legitimate sources.

---

## 2. Verified Real Dataset: BoneMarrowWSI-PediatricLeukemia

| Parameter | Verified Value | Documentation |
|---|---|---|
| **Collection Identifier** | `bonemarrowwsi_pediatricleukemia` (NCI IDC / Zenodo) | [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) |
| **Total Cohort Size** | **245 patients** | [`docs/REAL_DATA_SCHEMA.md`](docs/REAL_DATA_SCHEMA.md) |
| **Patient Age Range** | 18 – 204 months (1.5 – 17.0 years, strictly pediatric) | Verified in DICOM metadata |
| **WSI Microscopy** | 245 / 245 patients (DICOM SM) | May-Grünwald-Giemsa bone marrow aspirate |
| **Monolayer ROIs** | 245 / 245 patients (DICOM ANN) | Selected monolayer scanning fields |
| **Consensus Cell Annotations** | **113 patients** (multi-annotator consensus labels) | **Primary Supervised Cohort** |
| **Non-Consensus Extension** | **132 patients** (WSI + diagnosis available) | **Unlabeled Extension Cohort** |
| **Complete Blood Count (CBC)** | ❌ **UNAVAILABLE** in dataset | Explicitly set to `null` — no fabrication |
| **Cytogenetics / Karyotype** | ❌ **UNAVAILABLE** in dataset | Explicitly set to `null` — no fabrication |
| **Adult MDS** | ❌ **ZERO CASES** | Cohort is exclusively pediatric leukaemia |

---

## 3. Supervised Baseline Experiment: 4-Class Patient Task

To ensure statistical validity given small sample sizes in specific subtypes (e.g. APL $n=3$, CML-BP $n=4$), the supervised consensus cohort ($N=113$) is mapped to a 4-class target for the initial vision baseline. The original 7 granular diagnoses are permanently preserved in the master manifest:

- **ALL ($n=37$):** B-ALL ($n=32$), T-ALL ($n=5$)
- **AML ($n=17$):** AML with maturation, AML without maturation, Acute monocytic, Acute myelomonocytic, APL $t(15;17)$
- **CML ($n=50$):** Chronic phase CML ($n=46$), Blast phase CML ($n=4$)
- **Other ($n=9$):** Mixed phenotype acute leukaemia (MPAL, $n=6$), Unspecified/missing ($n=3$)

### Deterministic Stratified Patient-Level Split (`data/processed/splits.json`)

| Split | Patient Count | ALL | AML | CML | Other | Leakage Verification |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | **79** | 26 | 12 | 35 | 6 | $\text{Train} \cap \text{Val} = \emptyset$ |
| **Validation** | **17** | 5 | 3 | 7 | 2 | $\text{Train} \cap \text{Test} = \emptyset$ |
| **Test** | **17** | 6 | 2 | 8 | 1 | $\text{Val} \cap \text{Test} = \emptyset$ |
| **Total** | **113** | **37** | **17** | **50** | **9** | **Zero Patient Overlap** |

---

## 4. Key Scientific Definitions

- **Annotated-Cell Blast Fraction:** The proportion of annotated blasts among valid annotated cells within pre-selected monolayer fields. It is **not** an exhaustive whole-slide clinical differential count, and no clinical assertions are drawn directly from this number in isolation.
- **Raw Morphology Preservation:** Original consensus cell labels (`myelocytic_blast`, `promyelocyte`, etc.) and quality flags (`technically_unfit`, `artifact`, etc.) are retained verbatim to enable fine-grained morphological analysis and filtering.

---

## 5. Development Roadmap

1. **Phase 1: Real-Data Vision Baseline (CURRENT)**
   - Extract cell crops from DICOM consensus bounding boxes.
   - Train ViT-Small / LoRA on valid cell morphology classes.
   - Compute patient-level differential vectors and evaluate 4-class baseline classifier.
   - Conduct attention rollout and error analysis.
2. **Phase 2: Medical Knowledge Retrieval (RAG)**
   - Curate pediatric haematology literature index (WHO 5th edition).
   - Implement hybrid dense (FAISS) + sparse (BM25) retrieval with cross-encoder reranking.
3. **Phase 3: Agentic Orchestration & Verification**
   - Multi-agent coordination (Retriever, Synthesizer, Critic).
   - NLI claim-versus-evidence verification with calibrated abstention thresholds.
4. **Phase 4: Multimodal Decision Support & Interface**
   - Interactive Streamlit workstation displaying cell crops, attention maps, citations, and confidence scores.

---

## 6. Project Structure

```
bonemarrow/
├── configs/             # Hyperparameters, paths, random seeds
├── data/
│   ├── patient_manifest.csv    # Master manifest (all 245 patients, splits, targets)
│   ├── processed/splits.json   # Deterministic patient-level splits
│   ├── patient_metadata_dicom/ # Downloaded patient DICOM metadata
│   ├── sample_consensus_dicom/ # Sample consensus annotation series
│   └── synthetic/              # Quarantined unit-test fixtures only
├── vision/              # ViT/LoRA model, aggregation, attention rollout, dataset
├── context/             # Deterministic patient context assembler
├── retrieval/           # RAG retrieval layer (Phase 2)
├── agents/              # LangGraph agent orchestration (Phase 3)
├── explain/             # Attention rollout, citation linking, calibrated confidence
├── docs/                # REAL_DATA_SCHEMA.md, DATA_PROVENANCE.md, experiment_log.md
└── tests/               # Unit testing suite
```

---

## 7. Clinical Disclaimer

See [`docs/clinical_disclaimer.md`](docs/clinical_disclaimer.md) for full clinical and regulatory disclaimers.
