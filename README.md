# Agentic Multimodal Explainable RAG for Bone Marrow Assessment

> **Research & decision-support prototype only**  
> This software supports computational haematopathology research. It does **not** provide medical advice or a clinical diagnosis. All outputs require review by a qualified haematologist or pathologist.

---

## 1. What This Project Is

An **explainable, evidence-grounded framework** for bone marrow **cytomorphology** and related clinical context. The pipeline combines:

- **Vision:** cell detection, morphology classification, patient-level differential aggregation  
- **Retrieval:** WHO-aligned guideline and literature RAG (dense + sparse retrieval)  
- **Agents:** orchestrated synthesis, critique, and calibrated abstention  
- **Interface:** interactive review of crops, counts, citations, and confidence  

The design is **cohort-agnostic**: age group (pediatric, adult, geriatric), disease family (acute leukaemia, MDS/MPN, lymphoma involvement, reactive marrow, metastatic disease), and **optional modalities** (CBC, flow cytometry, cytogenetics, molecular panels) plug in when data exist—missing fields stay explicit `null` values, never imputed.

**Current reference benchmark:** the public [**BoneMarrowWSI-PediatricLeukemia**](docs/DATA_PROVENANCE.md) cohort (245 patients, MGG aspirate WSIs). It is the **default training and evaluation dataset** in this repo, not a limit on what the system can target.

---

## 2. Intended Use Cases (Beyond Pediatric Leukaemia)

| Use case | Role of this framework |
|----------|-------------------------|
| **Acute leukaemia workup** | Blast-aware morphology summaries + WHO citation-backed narrative |
| **MDS / MPN / CMML** | Dysplasia-aware cell typing when labels and slides are available |
| **Reactive / regenerating marrow** | Differential patterns without over-calling malignancy |
| **Lymphoma / metastatic marrow** | Extension via new label sets and retrieval corpora |
| **Multimodal correlation** | Fuse CBC, immunophenotype, and genetics when provided at case upload |
| **Teaching & QA** | Traceable evidence links and visual explanations for trainees |

Pediatric ALL/AML/CML remains a **strong supervised baseline** because of consensus cell labels; adult and mixed cohorts can be added as separate manifests and task heads without rewriting the core architecture.

---

## 3. System Pipeline

```
Upload / cohort case
    → Cell detection & morphology (YOLO / ViT paths)
    → Patient-level differential & blast-related metrics
    → Context assembly (demographics, labs, genetics — when present)
    → RAG over curated haematology knowledge
    → Agent synthesis + verification → structured report + citations
    → UI review (Streamlit / Flask)
```

Modular packages: `vision/`, `context/`, `retrieval/`, `agents/`, `explain/`, `pipeline/`, `app/`.

---

## 4. Reference Dataset (Pediatric Benchmark)

Verified statistics for the **default** public cohort. For schema and audit trails see [`docs/REAL_DATA_SCHEMA.md`](docs/REAL_DATA_SCHEMA.md) and [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

| Parameter | Verified value |
|-----------|----------------|
| **Collection ID** | `bonemarrowwsi_pediatricleukemia` (NCI IDC / Zenodo) |
| **Cohort size** | **245 patients** (pediatric only in this collection) |
| **Age range** | 18–204 months (1.5–17.0 years) |
| **Modality** | MGG bone marrow aspirate; DICOM SM + ANN |
| **Consensus cell labels** | **113 patients** (primary supervised subset) |
| **WSI + diagnosis, no consensus boxes** | **132 patients** (semi-supervised / weak-label extension) |
| **CBC / cytogenetics in IDC export** | **Not available** — stored as `null` in manifests |

### Supervised 4-class baseline (consensus *N* = 113)

Granular ICD-O-style diagnoses are preserved in `data/patient_manifest.csv`; training uses a collapsed 4-class target for stable metrics: **ALL**, **AML**, **CML**, **Other**.

| Split | Patients | ALL | AML | CML | Other |
|------:|:--------:|:---:|:---:|:---:|:-----:|
| Train | 79 | 26 | 12 | 35 | 6 |
| Val | 17 | 5 | 3 | 7 | 2 |
| Test | 17 | 6 | 2 | 8 | 1 |

Splits are **patient-level** with zero overlap (`data/processed/splits.json`).

### Definitions

- **Annotated-cell blast fraction:** blasts among valid annotated cells in selected monolayer fields—not a full clinical slide differential.  
- **Label fidelity:** consensus labels and quality flags are kept verbatim for filtering and fine-grained experiments.

---

## 5. Quick Start

### Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # add GROQ_API_KEY and/or GOOGLE_API_KEY
```

### Verify LLM + RAG

```bash
python setup_groq_rag.py
```

### Run the application

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Streamlit/Flask/Docker options. Typical local entry points: `run_app.bat` or the commands documented in the deployment guide.

### Vision baseline (reference cohort)

Documented in [`docs/ML_TRAINING_CODE_GUIDE.md`](docs/ML_TRAINING_CODE_GUIDE.md) and `scripts/` (crop extraction, training, patient-level evaluation).

---

## 6. LLM Modes

| Mode | When to use |
|------|-------------|
| **Groq (Llama 3.x)** | Fast cloud inference; set `GROQ_API_KEY` |
| **Google Gemini** | Alternative provider; set `GOOGLE_API_KEY` |
| **Deterministic templates** | Offline demos, CI, or when no API key is configured |

Setup details: [`docs/GROQ_LLAMA_RAG_SETUP.md`](docs/GROQ_LLAMA_RAG_SETUP.md).

RAG retrieves from a curated index under `data/knowledge_base/` and processed chunks in `data/processed/retrieval_index/`. Extend corpora for adult MDS, ICC/WHO updates, or institutional protocols—see [`docs/RAG_WHO_SOURCES.md`](docs/RAG_WHO_SOURCES.md).

---

## 7. Roadmap (General Bone Marrow Scope)

1. **Vision & metrics** — Robust cell detection on aspirate smears; calibration across stains and scanners; external validation cohorts (adult MDS, multi-centre).  
2. **Multimodal context** — First-class CBC, flow, cytogenetics, and molecular JSON schemas in `context/` with validation and abstention when incomplete.  
3. **Task registry** — Config-driven diagnosis tasks (pediatric 4-class, adult MDS risk, blast threshold flags) instead of hard-coded labels.  
4. **RAG & agents** — Age- and entity-aware retrieval; stronger claim–evidence NLI; human-in-the-loop edits logged to `docs/experiment_log.md`.  
5. **Governance** — Dataset cards per cohort, license tracking ([`docs/dataset_licenses.md`](docs/dataset_licenses.md)), and benchmark suites in `tests/` and `eval/`.

---

## 8. Project Layout

```
├── app/                 # Streamlit / Flask UI
├── agents/              # LangGraph-style orchestration
├── context/             # Patient context assembly (extensible modalities)
├── data/                # Manifests, knowledge base, processed indices
├── detector/            # YOLO training & inference utilities
├── explain/             # Attention, SHAP, citation linking
├── pipeline/            # End-to-end case pipeline & data models
├── retrieval/           # Indexing and hybrid search
├── vision/              # ViT/LoRA training and aggregation
├── scripts/             # Data prep, audits, batch jobs
├── docs/                # Architecture, deployment, schema, provenance
└── tests/
```

---

## 9. Documentation

| Document | Purpose |
|----------|---------|
| [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) | Components and data flow |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Local and production deployment |
| [`docs/ML_TRAINING_CODE_GUIDE.md`](docs/ML_TRAINING_CODE_GUIDE.md) | Training and evaluation |
| [`docs/GROQ_LLAMA_RAG_SETUP.md`](docs/GROQ_LLAMA_RAG_SETUP.md) | LLM configuration |
| [`docs/clinical_disclaimer.md`](docs/clinical_disclaimer.md) | Regulatory and clinical limits |
| [`detector/README.md`](detector/README.md) | Object-detection submodule |

---

## 10. Adding a New Cohort (Adult or Mixed)

1. **Provenance** — Document source, license, and IRB/status in `docs/DATA_PROVENANCE.md` (new section or sibling file).  
2. **Manifest** — Add rows to a cohort-specific CSV mirroring `patient_manifest.csv` fields (`age`, `sex`, diagnosis, split, nullable lab/genetics columns).  
3. **Labels** — Map ontology to a task config under `configs/` (do not overwrite pediatric splits).  
4. **Retrieval** — Append guideline chunks and re-run indexing (`retrieval/` scripts).  
5. **Evaluate** — Report patient-level metrics separately per cohort; do not pool pediatric and adult labels without explicit domain adaptation.

---

## 12. Where Data Lives & How to Back Up

All persistent clinical cases, analysis metrics, chat sessions, and uploaded images are saved under the `DATA_DIR` directory:

- **Environment Variable**: `DATA_DIR` (defaults to `./data` relative to the repository root).
- **SQLite Database**: `$DATA_DIR/cases.db` — Stores persistent case records, morphological counts, confidence metrics, RAG reports, and chat interactions.
- **Legacy Storage**: `$DATA_DIR/app_storage/` (`analyses.pkl`, `chats.json`) — Dual-written alongside SQLite for backward compatibility.
- **Uploads & Image Crops**:
  - `$DATA_DIR/uploads/` — Raw uploaded aspirate images.
  - `$DATA_DIR/new_patient_cache/` — Segmented cell crops, overlays, and contact sheets per case ID.

### Backup Procedure
To back up all patient analyses, chat histories, and database state safely without moving or modifying live files:
```bash
# Set your backup destination
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy SQLite database, storage files, and uploads
cp "$DATA_DIR/cases.db" "$BACKUP_DIR/" 2>/dev/null || true
cp -r "$DATA_DIR/app_storage" "$BACKUP_DIR/"
cp -r "$DATA_DIR/uploads" "$BACKUP_DIR/"
cp -r "$DATA_DIR/new_patient_cache" "$BACKUP_DIR/"
```

---

## 13. Citation & License

Cite the **BoneMarrowWSI-PediatricLeukemia** dataset and this codebase as appropriate for your publication. Dataset license: **CC BY 4.0** (see dataset card in `data/dataset_cards/`).

---

*Default experiments use the pediatric IDC cohort; the framework is intended for broader bone marrow cytology and multimodal haematopathology research.*
