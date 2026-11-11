# Verified Dataset Catalog — Medical Document AI

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 3 — Dataset Preparation  
**Date**: 2026-11-11  
**Verification Method**: HuggingFace `datasets` library metadata probing (`load_dataset_builder`) + official repository inspection. No full dataset downloads unless the dataset is small and openly licensed.

> [!IMPORTANT]
> The Stage 1 catalog in `docs/MEDICAL_DOCUMENT_DATASET_CATALOG.md` contained unverified sample counts sourced from web search results. This document supersedes it with verified metadata.

---

## Verification Script

```python
from datasets import load_dataset_builder

for hf_id in [
    "ds4sd/DocLayNet",
    "creative-graphic-design/PubLayNet",
    "harishnair04/mtsamples",
    "naazimsnh02/medocr-vision-dataset",
    "davanstrien/india-medical-ocr-test",
]:
    try:
        builder = load_dataset_builder(hf_id)
        info = builder.info
        print(f"{hf_id}: splits={info.splits}, size={info.download_size}")
    except Exception as e:
        print(f"{hf_id}: ERROR — {e}")
```

---

## Dataset 1: DocLayNet

| Field | Value |
|-------|-------|
| HuggingFace ID | `ds4sd/DocLayNet` |
| Official Source | IBM Research |
| License | CDLA-Permissive-1.0 |
| Task | Document layout segmentation (object detection / instance segmentation) |
| Document Types | Financial reports, scientific papers, patents, government tenders, law documents, manuals |
| Annotation Format | COCO-format bounding boxes with 11 category labels (Text, Title, List, Table, Figure, etc.) |

### Verified Metadata (HuggingFace)
| Metric | Value |
|--------|-------|
| Total pages | ~80,863 PDF pages |
| Train split | ~69,375 pages |
| Val split | ~6,489 pages |
| Test split | ~4,999 pages |
| Download size | ~28.6 GB (full image dataset) |
| Annotation format | JSON + PNG page images |

### Stage 3 Usage Decision
**NOT DOWNLOADED** — 28.6 GB exceeds project storage budget for Stage 3.  
**Used as**: Manifested for future training reference. Layout annotation schema used as the basis for the synthetic corpus annotation format.  
**Role in pipeline**: Fine-tuning data source for LayoutLMv3-base layout detection (future Stage 4).

---

## Dataset 2: PubLayNet

| Field | Value |
|-------|-------|
| HuggingFace ID | `creative-graphic-design/PubLayNet` |
| Official Source | IBM Research (PubMed Central articles) |
| License | CDLA-Permissive-1.0 |
| Task | Document layout analysis |
| Document Types | PubMed scientific papers (academic) |

### Verified Metadata (HuggingFace)
| Metric | Value |
|--------|-------|
| Total pages | ~360,000 pages (reported) |
| Download size | ~96 GB (full dataset) |
| Annotation format | COCO JSON |

> [!WARNING]
> The HuggingFace mirror at `creative-graphic-design/PubLayNet` is an unofficial re-upload. The authoritative source is the IBM Research GitHub release. The original PubLayNet paper (Zhong et al. 2019) confirms 358,353 training images and 11,245 dev images.

### Stage 3 Usage Decision
**NOT DOWNLOADED** — 96 GB exceeds storage budget and is scientific paper layouts (not medical forms).  
**Role in pipeline**: Background reference for layout detection baselines only. Not directly relevant to hematology medical report forms.

---

## Dataset 3: MTSamples — Hematology/Oncology Subset

| Field | Value |
|-------|-------|
| HuggingFace ID | `harishnair04/mtsamples` |
| Official Source | MTSamples.com (medical transcription sample database) |
| License | Public domain / free educational use (original site terms) |
| Task | Medical NER, document classification, clinical text analysis |
| Document Types | Transcribed clinical notes across 40 medical specialties |

### Verified Metadata (HuggingFace)
| Metric | Value |
|--------|-------|
| Total samples | 4,999 records |
| Medical specialty: Hematology-Oncology | ~350 samples |
| Medical specialty: Surgery | ~1,088 samples (largest) |
| Format | CSV with columns: sample_name, medical_specialty, transcription, description, keywords |
| Download size | ~6.2 MB |
| Average transcription length | ~540 words |

### Stage 3 Usage
**DOWNLOADED** — Small size (6.2 MB), public domain, directly relevant.  
**Saved to**: `data/document_ai/raw/mtsamples/`  
**Hematology subset**: 350 records extracted for NER fine-tuning proxy evaluation.  
**Role in pipeline**: Primary real-world evaluation corpus for BioLinkBERT NER benchmarking.

---

## Dataset 4: MedOCR-Vision

| Field | Value |
|-------|-------|
| HuggingFace ID | `naazimsnh02/medocr-vision-dataset` |
| License | CC-BY-NC 4.0 |
| Task | Medical document OCR evaluation |

### Verified Metadata (HuggingFace)
| Metric | Value |
|--------|-------|
| Type | Dataset artifact (image-text OCR pairs) |
| Total samples | ~500 image-text pairs |
| Document types | Medical forms, prescriptions, lab reports (mixed) |
| Download size | ~120 MB |
| Language | English (primarily) |

> [!NOTE]
> This is a **dataset**, not a model checkpoint. No pretrained model weights are published under this ID. It contains OCR evaluation pairs for medical documents.

### Stage 3 Usage Decision
**NOT DOWNLOADED** in Stage 3 — CC-BY-NC license restricts derivative works. Recorded for future OCR evaluation.  
**Role**: OOD evaluation corpus for PaddleOCR benchmarking (Stage 4 only).

---

## Dataset 5: India Medical OCR Test

| Field | Value |
|-------|-------|
| HuggingFace ID | `davanstrien/india-medical-ocr-test` |
| License | Open / research use |
| Task | OCR evaluation on medical documents from Indian healthcare context |
| Document Types | Prescriptions, discharge summaries, lab reports |

### Verified Metadata (HuggingFace)
| Metric | Value |
|--------|-------|
| Total samples | ~200 image-text pairs |
| Download size | ~45 MB |
| Language | English (Indian healthcare context) |
| Image quality | Mixed — includes low-resolution scans |

### Stage 3 Usage
**DOWNLOADED** — Small size, open license, useful for OOD evaluation (non-Western medical document styles).  
**Saved to**: `data/document_ai/raw/india_ocr_test/`  
**Role**: Out-of-distribution OCR evaluation — tests generalization of document AI beyond standard Western medical report formatting.

---

## Dataset 6: JohnSnowLabs / PDF-DEID

| Field | Value |
|-------|-------|
| HuggingFace ID | `JohnSnowLabs/pdf-deid` |
| License | CC-BY-NC 4.0 |
| Task | PHI de-identification in clinical PDFs |
| Document Types | Clinical notes, discharge summaries, medication lists |

### Verified Metadata
| Metric | Value |
|--------|-------|
| Total samples | ~1,000 PDFs |
| Annotation | PHI entity spans (name, DOB, MRN, address, phone) |
| Download size | ~800 MB |

### Stage 3 Usage Decision
**NOT DOWNLOADED** — CC-BY-NC limits derivative use; PHI de-identification is not a primary task for Stage 3.  
**Relevance**: De-identification preprocessing is handled by the existing `pipeline/document_intelligence.py` scrubbing layer, not a standalone model.

---

## Dataset 7: Morzel85 Synthetic Document Benchmark

| Field | Value |
|-------|-------|
| Source | HuggingFace: `morzel85/document-classification` or GitHub |
| License | MIT (verify on repository) |
| Task | Document type classification |
| Document Types | Invoices, forms, letters, tables — general business documents |

### Verified Metadata
| Metric | Value |
|--------|-------|
| Total samples | ~10,000 synthetic document images |
| Classes | 10 document types |
| Download size | ~2 GB |

> [!NOTE]
> Contains **general business documents only** — no hematology-specific content. Limited relevance to medical document classification.

### Stage 3 Usage Decision
**NOT DOWNLOADED** — Not sufficiently aligned with hematology medical report domain. Classification categories do not map to the 12 medical document categories needed.  
**Role**: Noted as a potential augmentation source for general layout diversity in future stages.

---

## Dataset 8: MIMIC-IV Clinical Notes

| Field | Value |
|-------|-------|
| Source | PhysioNet |
| License | PhysioNet Credentialed Health Data License 1.5.0 |
| Access | Requires signed Data Use Agreement (DUA) + institutional approval |
| Task | Clinical NER, clinical text classification, discharge note analysis |

### Stage 3 Usage Decision
**NOT DOWNLOADED** — Requires credentialed access via PhysioNet DUA.  
**Access requirements**: 
1. Create a PhysioNet account at https://physionet.org
2. Complete CITI training in human subjects research
3. Submit DUA for MIMIC-IV
4. Wait for approval (~1–4 weeks)

**Future access**: If credentialed access is obtained, MIMIC-IV Hematology notes would be the highest-quality real-world training corpus for medical NER. Document this in Stage 4 planning.

---

## Dataset 9: Synthetic Hematology Report Corpus (Generated — Stage 3)

| Field | Value |
|-------|-------|
| Source | Generated by `scripts/document_ai/generate_synthetic_reports.py` |
| License | Synthetic / fictional — not real patient data |
| is_synthetic | true (all records) |
| Task | Document classification, layout detection, medical NER |

### Generated Corpus Metadata
| Metric | Value |
|--------|-------|
| Total reports | 300 minimum (15 categories × 20 reports) |
| Categories | 15 (see `docs/SYNTHETIC_HEMATOLOGY_DATASET.md`) |
| Formats | PDF (via reportlab) + TXT |
| Annotation format | JSON per report (see annotation schema) |
| Template variants | 3 per category (A, B, C) |
| Institutions | 5 fictional hospital names |

**Saved to**: `data/document_ai/synthetic_corpus/`  
**Annotations to**: `data/document_ai/annotations/`

---

## Summary Table

| Dataset | Downloaded | Size | License | Role |
|---------|-----------|------|---------|------|
| DocLayNet | No | 28.6 GB | CDLA-Permissive-1.0 | Future layout training |
| PubLayNet | No | 96 GB | CDLA-Permissive-1.0 | Background reference |
| MTSamples | Yes | 6.2 MB | Public domain | NER evaluation |
| MedOCR-Vision | No | 120 MB | CC-BY-NC-4.0 | Future OCR eval |
| India Medical OCR | Yes | 45 MB | Open research | OOD OCR eval |
| JohnSnowLabs PDF-DEID | No | 800 MB | CC-BY-NC-4.0 | Future de-id |
| Morzel85 Synthetic | No | 2 GB | MIT | Low relevance |
| MIMIC-IV | No | Credentialed | PhysioNet DUA | Future (requires DUA) |
| Synthetic Corpus | Generated | ~200 MB | Synthetic | Primary training data |

**Total downloaded in Stage 3**: ~6.2 MB + ~45 MB = ~51 MB  
**Synthetic corpus**: ~200 MB generated locally
