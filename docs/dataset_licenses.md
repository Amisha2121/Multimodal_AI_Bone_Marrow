# docs/dataset_licenses.md — Data License & Provenance Registry

All data sources used in this project are documented here.
See individual dataset cards in `data/dataset_cards/` for full details.

---

## 1. Matek et al. 2021 — Bone Marrow Cytomorphology (TCIA)

| Field | Detail |
|---|---|
| **Modality** |  Bone marrow aspirate (single-cell) |
| **License** | **CC BY 4.0** |
| **DOI** | https://doi.org/10.7937/TCIA.AXH3-T579 |
| **Use in this project** | Vision model training (cell classification, 21 classes) |
| **Restrictions** | Attribution required. No patient re-identification. |

** Dropped datasets (wrong modality — peripheral blood, not bone marrow):**
- Matek et al. 2019 Munich AML Morphology (peripheral blood smear, 15 classes)
- Acevedo et al. 2020 (peripheral blood smear, 8 classes)

---

## 2. PubMed Central Open Access Subset

| Field | Detail |
|---|---|
| **License** | Varies by article; use CC BY / CC BY-NC subsets only |
| **Use in this project** | RAG knowledge base (AML + MDS articles) |
| **Restrictions** | Check per-article license; never include CC BY-NC-ND in commercial contexts |

---

## 3. OpenAlex API

| Field | Detail |
|---|---|
| **License** | CC0 (data) |
| **Use in this project** | Paper metadata and abstract augmentation for RAG |
| **Restrictions** | None; CC0 is fully open |

---

## 4. WHO Classification Summaries

| Field | Detail |
|---|---|
| **Source** | WHO Classification of Haematolymphoid Tumours, 5th edition (2022) |
| **License** | Copyrighted; fair-use excerpts only |
| **Use in this project** | Diagnostic criteria summaries in RAG knowledge base |
| **Restrictions** | Do not reproduce verbatim beyond brief fair-use excerpts. Always cite the source. Do not include full chapters. |

---

## 5. User-Provided Reference List

| Field | Detail |
|---|---|
| **Source** | Provided by user (Q12) |
| **License** | CC BY or CC BY-NC open-access articles only |
| **Use in this project** | RAG knowledge base (primary scholarly content) |
| **Restrictions** | Verify each paper's license before ingestion. Store in `data/knowledge_base/` (gitignored). |

---

*Provenance last verified: 2026-10-06*
