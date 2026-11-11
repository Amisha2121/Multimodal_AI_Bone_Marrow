# Model Selection Decision — Medical Document AI

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 2 — Final Model Selection  
**Date**: 2026-11-11  
**Decision Maker**: Architecture analysis + VRAM constraint + proxy benchmark (see `PRETRAINED_MODEL_COMPARISON.md`)

---

## Decision Summary

Three primary models are selected for the Medical Document AI pipeline, operating sequentially. No single model is selected for all tasks.

| Role | Selected Model | HF ID | License | Rationale |
|------|---------------|--------|---------|-----------|
| Document Type Classifier | **DiT-base** | `microsoft/dit-base` | CC-BY-NC-4.0 | Lightest model, visual-only, fastest inference for 12-class doc type routing |
| Layout-Aware Extraction | **LayoutLMv3-base + LoRA** | `microsoft/layoutlmv3-base` | CC-BY-NC-4.0 | Multimodal (text + layout + image) — best for structured medical report forms |
| Medical NER | **BioLinkBERT-base + LoRA** | `michiyasunaga/BioLinkBERT-base` | Apache 2.0 | 94% hematology vocabulary coverage vs. 61% for LayoutLMv3 tokenizer; superior entity extraction |
| Structured Parsing (secondary) | **Donut-base** | `naver-clova-ix/donut-base` | MIT | OCR-free end-to-end parsing for clean digital PDFs; MIT license is most permissive |
| OCR (scanned inputs) | **PaddleOCR PP-OCRv4** | PaddlePaddle hub | Apache 2.0 | Best open-source OCR for degraded scans; runs in isolated venv |

---

## Decision Rationale — Task by Task

### Task 1: Document Type Classification

**Goal**: Given a medical report image/PDF, classify it into one of the 12 document categories (CAT_A through CAT_L: aspirate, biopsy, CBC, smear, flow, IHC, FISH, karyotyping, NGS, PCR/molecular, iron/reticulocyte, MRD/transplant).

**Selected**: `microsoft/dit-base` (86 M parameters)

**Why DiT-base**:
- IIT-CDIP pretraining (42 M document images) gives it strong visual priors for document layout patterns.
- At ~86 M params and ~260 MB VRAM, it can be kept loaded during the full inference pipeline alongside other models.
- Document classification is a visual task — the structural appearance of a bone marrow aspirate report (header layout, table structure) is visually distinct from a flow cytometry printout.
- Fast inference (~20–30 ms/page) enables real-time classification on upload.

**Why not LayoutLMv3 for this task**:
- LayoutLMv3 requires bounding boxes and text tokens — adds complexity for a pure classification step.
- DiT-base achieves comparable classification accuracy at lower cost.

---

### Task 2: Layout-Aware Information Extraction

**Goal**: Extract named fields from medical reports with layout context — e.g., detect that "12.5" appearing in the "Blast %" column of a table is a blast percentage, not an unrelated number.

**Selected**: `microsoft/layoutlmv3-base` with LoRA fine-tuning (125 M parameters)

**Why LayoutLMv3-base**:
- Trimodal input (text + 2D position + visual patch) captures table structure — critical for medical reports where the same number has different meanings depending on its position (column header context).
- DocLayNet + IIT-CDIP pretraining provides strong layout understanding baseline.
- LoRA fine-tuning at rank-8 reduces trainable parameters to ~1–2 M while keeping full model context.
- Fits comfortably within 8 GB VRAM during fine-tuning (estimated ~3.5 GB peak with LoRA).

**Why not LayoutLMv3-large**:
- 368 M params → ~7–8 GB peak during LoRA fine-tuning → OOM risk on 8 GB GPU.
- Base model provides sufficient capacity for the medical report domain when fine-tuned on the synthetic corpus.

---

### Task 3: Medical Named Entity Recognition

**Goal**: Extract specific medical entities from text: cell type names, percentage values, WHO classification codes, clinical flags (H/L/Critical), reference range violations, diagnosis keywords.

**Selected**: `michiyasunaga/BioLinkBERT-base` with LoRA fine-tuning (110 M parameters)

**Why BioLinkBERT-base**:
- Pretrained on PubMed + PMC full text with UMLS entity linking — directly encodes hematology vocabulary.
- Tokenizer analysis on MTSamples shows 94% hematology term recognition vs. 61% for LayoutLMv3 tokenizer — fewer subword splits = better entity boundary detection.
- Apache 2.0 license — most permissive among the medically-pretrained candidates; allows full pipeline commercialization.
- Fine-tunes efficiently at ~2.5 GB VRAM.

**Why not PubMedBERT**:
- PubMedBERT is abstract-only pretraining; BioLinkBERT is trained on full PMC text + entity linking, which better matches structured clinical report language.
- BioLinkBERT consistently outperforms PubMedBERT on biomedical NER benchmarks (NER F1 +1.5–3.0 points).

---

### Task 4: Structured Form Parsing (Secondary Path)

**Goal**: For well-structured digital PDFs (CBC reports, iron studies forms), parse the full document to a structured JSON in one pass.

**Selected**: `naver-clova-ix/donut-base` (200 M parameters, MIT license)

**Why Donut-base**:
- OCR-free — eliminates the OCR error propagation problem for digital PDFs.
- MIT license — the most permissive in the entire candidate set.
- End-to-end parsing to JSON is exactly what the patient ingestion pipeline needs.
- Fits within 8 GB VRAM (~600 MB at FP16 inference; ~5–6 GB during fine-tuning).

**Usage scope**: Donut is activated only for clean digital PDFs where the form structure is regular. Scanned/photographed PDFs route through the PaddleOCR → LayoutLMv3 path instead.

---

### OCR: PaddleOCR PP-OCRv4 (Auxiliary, Isolated)

**Usage**: Activated only when input is a scanned PDF or image with poor text layer. Runs in `venv_paddle/` subprocess — results returned as JSON bounding box + text pairs.

**Not selected** as a primary model due to framework conflict risk.

---

## License Compatibility Matrix

| Model | License | Research Use | Application Integration | Commercial Deployment |
|-------|---------|-------------|------------------------|----------------------|
| DiT-base | CC-BY-NC-4.0 | Yes | Yes (non-commercial) | No |
| LayoutLMv3-base | CC-BY-NC-4.0 | Yes | Yes (non-commercial) | No |
| BioLinkBERT-base | Apache 2.0 | Yes | Yes | Yes |
| Donut-base | MIT | Yes | Yes | Yes |
| PaddleOCR | Apache 2.0 | Yes | Yes (isolated) | Yes |

**Project context**: This is a research application for hematological diagnosis support. CC-BY-NC-4.0 permits research use. No commercial deployment is planned at this stage.

---

## Rejected Candidates

| Model | Reason for Rejection |
|-------|---------------------|
| LayoutLMv3-large | OOM risk during LoRA fine-tuning at 8 GB VRAM |
| naazimsnh02/medocr-vision | Dataset artifact only — no model checkpoint published |
| Generic BERT / RoBERTa | Insufficient biomedical vocabulary coverage |
| Llama / Mistral 7B+ | Exceeds VRAM budget for fine-tuning; overkill for structured extraction |

---

## Next Steps (Stage 4 — Training, Not Started)

1. Fine-tune DiT-base on synthetic corpus classification labels.
2. Fine-tune LayoutLMv3-base + LoRA on synthetic corpus with bounding box annotations.
3. Fine-tune BioLinkBERT-base + LoRA on MTSamples NER + synthetic corpus entity annotations.
4. Optional: Fine-tune Donut-base on clean synthetic PDFs with JSON targets.
5. Evaluate each model on held-out TEMPLATE_*C test set.
6. Integrate inference pipeline into `pipeline/document_intelligence.py`.

**Stage 4 is NOT started. User approval required before training begins.**
