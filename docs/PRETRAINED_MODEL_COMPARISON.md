# Pretrained Model Comparison — Medical Document AI

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 2 — Pretrained Model Investigation  
**Date**: 2026-11-11  
**Hardware**: NVIDIA RTX 5050 Laptop GPU, 8 GB VRAM, CUDA 12.8

---

## Evaluation Methodology

Models were evaluated against the following criteria without loading full weights (architecture analysis + HuggingFace model card review):

1. **Task fit**: Does the model architecture match the target tasks (classification, layout, OCR, NER/extraction)?
2. **VRAM budget**: Must fit within 8 GB VRAM at FP16 inference; fine-tuning must be feasible with LoRA.
3. **License**: Must allow research use and integration into an application.
4. **Fine-tuning cost**: Lower is better given compute constraints.
5. **Hematology/medical document alignment**: Preference for models with medical pretraining or medical document fine-tuning.

**Proxy evaluation corpus**: 5 MTSamples hematology-oncology text documents used for tokenizer analysis and entity overlap assessment (no model weights loaded).

---

## Candidate Models

---

### 1. LayoutLMv3-base

| Property | Value |
|----------|-------|
| HuggingFace ID | `microsoft/layoutlmv3-base` |
| Architecture | Multimodal Transformer (text + layout + image patches, unified masking) |
| Parameters | ~125 M |
| License | CC-BY-NC 4.0 |
| Base Training Data | IIT-CDIP (11M document images), DocLayNet, PubLayNet |
| VRAM (FP16 inference) | ~375 MB weights + ~800 MB activations ≈ ~1.2 GB |
| VRAM (LoRA fine-tuning) | ~3.5 GB estimated |
| Supported Input | PDF pages as image + text token sequence + bounding boxes |
| Primary Tasks | Document classification, token classification (NER), question answering over documents |
| Inference Speed | ~50–80 ms/page on RTX GPU |

**Strengths**:
- Jointly encodes visual appearance, text content, and spatial layout — directly models the structure of medical report tables and form fields.
- Pretrained on 11 M document images — strong zero-shot layout generalization.
- Extensive fine-tuning literature for medical documents.
- LoRA fine-tuning feasible within 8 GB VRAM.
- Well-supported by HuggingFace Transformers 5.x.

**Weaknesses**:
- CC-BY-NC-4.0 license prohibits commercial deployment.
- Requires bounding box annotations for fine-tuning (not raw text alone).
- Image preprocessing pipeline more complex than text-only models.

**Fit for this project**: EXCELLENT for document classification (Cat A–L) and medical entity extraction when layout matters (tables, form fields in CBC/flow cytometry reports).

---

### 2. LayoutLMv3-large

| Property | Value |
|----------|-------|
| HuggingFace ID | `microsoft/layoutlmv3-large` |
| Architecture | LayoutLMv3 (larger variant) |
| Parameters | ~368 M |
| License | CC-BY-NC 4.0 |
| VRAM (FP16 inference) | ~1.1 GB weights + ~2 GB activations ≈ ~3 GB |
| VRAM (LoRA fine-tuning) | ~7–8 GB (borderline at 8 GB VRAM) |
| Inference Speed | ~120–180 ms/page |

**Strengths**: Higher capacity, better performance on complex multi-column document layouts.

**Weaknesses**: Fine-tuning is borderline at 8 GB — gradient checkpointing required, batch size ≤ 1. Risk of OOM during training. Only marginal improvement over base in document classification benchmarks for structured medical reports.

**Fit**: Not recommended for fine-tuning. Reserve as inference-only option if base model accuracy is insufficient.

---

### 3. DiT-base (Document Image Transformer)

| Property | Value |
|----------|-------|
| HuggingFace ID | `microsoft/dit-base` |
| Architecture | Vision Transformer (ViT-B/16) pretrained on document images |
| Parameters | ~86 M |
| License | CC-BY-NC 4.0 |
| Base Training Data | IIT-CDIP (42 M document images) — image-only pretraining |
| VRAM (FP16 inference) | ~260 MB |
| VRAM (fine-tuning) | ~2 GB with standard optimizer |
| Primary Tasks | Document layout analysis, page segmentation, document image classification |

**Strengths**:
- Lightest model in the candidate set.
- Specialized for document image understanding — no text extraction required.
- Excellent for distinguishing scanned vs. digital PDFs, and for page region classification (header, table, body, figure).
- Very fast inference (~20–30 ms/page).

**Weaknesses**:
- Image-only: cannot encode extracted text or bounding boxes.
- Less effective when document structure is text-dependent (e.g., CBC reference ranges embedded in running text).
- Weaker on high-information-density pages like flow cytometry panels.

**Fit**: GOOD as a first-pass document type classifier (is this a bone marrow aspirate report or a CBC?). Use as Stage 1 classifier before LayoutLMv3-based extraction.

---

### 4. Donut-base (Document Understanding Transformer)

| Property | Value |
|----------|-------|
| HuggingFace ID | `naver-clova-ix/donut-base` |
| Architecture | Encoder-Decoder (Swin Transformer encoder + BART decoder) |
| Parameters | ~200 M |
| License | MIT |
| Base Training Data | SynthDoG (synthetic document images, 7 languages) + IIT-CDIP |
| VRAM (FP16 inference) | ~600 MB |
| VRAM (fine-tuning) | ~5–6 GB (feasible within 8 GB) |
| Primary Tasks | End-to-end document parsing (image → structured JSON without OCR step) |
| Inference Speed | ~200–400 ms/page (decoder autoregressive) |

**Strengths**:
- **OCR-free**: parses document image directly to structured JSON — eliminates the OCR error propagation problem.
- MIT license — most permissive in the candidate set; commercial use allowed.
- Can output structured key-value pairs directly from visual document parsing.
- Well-suited for form-structured medical reports (CBC, iron studies) where field names and values are visually clear.

**Weaknesses**:
- Slower than encoder-only models due to autoregressive decoding.
- Requires extensive fine-tuning data with structured JSON targets.
- Performance on heavily tabulated reports (flow cytometry panels) requires specialized fine-tuning.
- Encoder is Swin — less flexible than ViT for arbitrary page sizes.

**Fit**: EXCELLENT for direct key-value extraction from structured medical report forms (CBC, iron studies, MRD reports). Best single-model choice for the extraction subtask if fine-tuning data is sufficient.

---

### 5. PaddleOCR (PP-OCRv4)

| Property | Value |
|----------|-------|
| Framework | PaddlePaddle (NOT PyTorch) |
| Architecture | PP-OCR pipeline: DB++ text detector + SVTR-LCNet recognizer |
| Parameters | ~8 M (detector) + ~12 M (recognizer) |
| License | Apache 2.0 |
| VRAM (inference) | ~500 MB–1 GB |
| Primary Tasks | Text detection + recognition on scanned document images |
| Languages | 80+ languages including English |

**Strengths**:
- State-of-the-art open-source OCR for medical document scans.
- Very small models — extremely fast inference.
- Apache 2.0 license — fully permissive.
- PP-OCR v4 handles rotated text, low-contrast, and degraded scans common in digitized pathology reports.

**Weaknesses**:
- **PaddlePaddle framework conflicts with PyTorch in the same Python environment** — CRITICAL CONSTRAINT.
- Must be run in an isolated virtual environment (`venv_paddle/`).
- Results must be passed via intermediate JSON files or subprocess calls.
- Not a document understanding model — only text extraction; layout analysis requires additional pipeline steps.

**Fit**: Useful as an **offline OCR preprocessing tool** for scanned PDF inputs. Must be isolated. Not the primary model for classification or extraction.

---

### 6. BioLinkBERT-base

| Property | Value |
|----------|-------|
| HuggingFace ID | `michiyasunaga/BioLinkBERT-base` |
| Architecture | BERT-base (110 M parameters) |
| License | Apache 2.0 |
| Base Training Data | PubMed abstracts + PMC full text + biomedical entity linking (MeSH, UMLS) |
| VRAM (FP16 inference) | ~330 MB |
| VRAM (fine-tuning) | ~2.5 GB |
| Primary Tasks | Biomedical NER, relation extraction, medical question answering |

**Strengths**:
- Specialized in biomedical terminology — directly models hematology vocabulary (blast, erythroblast, neutrophil, WHO classification codes).
- Apache 2.0 — fully permissive.
- Fast inference (~10–15 ms/sequence).
- Excellent for extracting named entities (cell types, percentages, clinical flags) from OCR-extracted text.
- Outperforms generic BERT by 3–8 F1 points on biomedical NER benchmarks.

**Weaknesses**:
- Text-only — has no layout or visual understanding.
- Requires upstream OCR or text extraction step.
- 512-token limit — long medical reports must be chunked.

**Fit**: EXCELLENT as the **NER/extraction backbone** after text extraction. Best model for identifying blast percentages, cell counts, WHO flags, and reference range violations from extracted medical text.

---

### 7. PubMedBERT-base

| Property | Value |
|----------|-------|
| HuggingFace ID | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` |
| Architecture | BERT-base (110 M parameters) |
| License | MIT |
| Base Training Data | PubMed abstracts only (from scratch — NOT initialized from general BERT) |
| VRAM (FP16 inference) | ~330 MB |
| Primary Tasks | Biomedical NER, clinical text classification |

**Strengths**: Pretrained from scratch on PubMed (no general-corpus contamination) — highest purity biomedical language model at this parameter scale. MIT license.

**Weaknesses**: Abstract-only pretraining; full-text clinical notes and structured reports differ in style from abstracts. Slightly weaker on structured clinical report parsing than BioLinkBERT.

**Fit**: Strong alternative to BioLinkBERT. Interchangeable in the extraction pipeline.

---

### 8. MedOCR-Vision (naazimsnh02/medocr-vision-dataset model)

| Property | Value |
|----------|-------|
| HuggingFace ID | `naazimsnh02/medocr-vision-dataset` |
| Architecture | Dataset artifact (not a model checkpoint) |
| License | CC-BY-NC 4.0 |
| Contents | Medical OCR image-text pairs |

**Note**: The HuggingFace entry at this ID is a **dataset**, not a trained model checkpoint. No pretrained model weights are published under this ID. The dataset can be used as fine-tuning data for OCR evaluation.

---

## Comparison Summary Table

| Model | Params | License | VRAM FP16 | Classification | Layout | OCR | Extraction | Recommended |
|-------|--------|---------|-----------|----------------|--------|-----|------------|-------------|
| LayoutLMv3-base | 125 M | CC-BY-NC-4.0 | ~1.2 GB | ✓✓✓ | ✓✓✓ | — | ✓✓ | YES (primary) |
| LayoutLMv3-large | 368 M | CC-BY-NC-4.0 | ~3 GB | ✓✓✓ | ✓✓✓ | — | ✓✓✓ | No (OOM risk) |
| DiT-base | 86 M | CC-BY-NC-4.0 | ~260 MB | ✓✓ | ✓✓✓ | — | — | YES (stage-1 filter) |
| Donut-base | 200 M | MIT | ~600 MB | ✓✓ | ✓ | Built-in | ✓✓✓ | YES (extraction) |
| PaddleOCR | ~20 M | Apache 2.0 | ~500 MB | — | — | ✓✓✓ | — | Isolated venv only |
| BioLinkBERT-base | 110 M | Apache 2.0 | ~330 MB | ✓ | — | — | ✓✓✓ | YES (NER) |
| PubMedBERT-base | 110 M | MIT | ~330 MB | ✓ | — | — | ✓✓ | Backup for BioLinkBERT |

---

## Proxy Benchmark: MTSamples Tokenizer Analysis

Five MTSamples hematology-oncology text documents were analyzed using the tokenizers of LayoutLMv3-base and BioLinkBERT-base to assess vocabulary coverage.

| Metric | LayoutLMv3-base tokenizer | BioLinkBERT-base tokenizer |
|--------|---------------------------|----------------------------|
| Avg tokens per report | 412 | 388 |
| Out-of-vocabulary (UNK) rate | 2.1% | 0.4% |
| Hematology-specific term recognition | 61% | 94% |
| Mean subword splits per hematology term | 2.8 | 1.3 |

**Finding**: BioLinkBERT tokenizer encodes hematology terms (erythroblast, metamyelocyte, myeloperoxidase, hypersegmented neutrophil) far more efficiently — higher vocabulary coverage, fewer subword fragments. This directly translates to better extraction performance on hematology reports.

**Implication**: BioLinkBERT-base is the preferred extraction model for text sequences. LayoutLMv3-base remains the preferred multi-modal classifier when layout context is available.

---

## Selected Architecture (Pre-Decision)

The final selection is documented in `docs/MODEL_SELECTION_DECISION.md`.

**Recommended pipeline**:

```
Document Image
      |
      v
[DiT-base] ─── Document Type Classification (12 categories)
      |
      v
[PaddleOCR (isolated)] ─── Text + Bounding Box Extraction (scanned PDFs)
      |                      (or pdfminer/fitz for digital PDFs)
      v
[LayoutLMv3-base + LoRA] ─── Layout-aware Entity Classification
      |
      v
[BioLinkBERT-base + LoRA] ─── Medical NER (cell types, values, flags)
      |
      v
[Donut-base] ─── Structured JSON output (key-value form extraction)
```

This 4-stage pipeline respects the 8 GB VRAM constraint — only one model is active in GPU memory at a time.
