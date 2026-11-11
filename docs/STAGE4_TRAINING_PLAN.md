# Stage 4: Medical Document AI Fine-Tuning Specification & Training Plan

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 4 — Model Training & Evaluation  
**Hardware Environment**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, CUDA 12.8)  
**Frameworks**: PyTorch 2.11, Transformers 5.5.4, PEFT 0.21.2 (LoRA)

---

## 1. Multi-Model Pipeline Architecture

In alignment with [MODEL_SELECTION_DECISION.md](file:///c:/Users/AMISHA/Desktop/Codes/Multimodal_AI_Bone_Marrow/docs/MODEL_SELECTION_DECISION.md), the document intelligence workflow uses three specialized models:

```
Uploaded Document (PDF / Scan / Image)
                 │
                 ▼
      [STAGE 4.1: DiT-base] ──► Document Classification (15 Diagnostic Categories)
                 │
                 ▼
 [STAGE 4.2: LayoutLMv3-base + LoRA] ──► Spatial Layout & Tabular Structure Recognition
                 │
                 ▼
 [STAGE 4.3: BioLinkBERT-base + LoRA] ──► Medical Entity Extraction (Biomarkers, Values, Units, Flags)
                 │
                 ▼
   Integrated Clinical Case Record (pipeline/document_intelligence.py)
```

---

## 2. Model 1: Document Classification (DiT-base)

* **Base Model**: `microsoft/dit-base` (Document Image Transformer)
* **Pretrained Weights**: Pretrained on 42M document images (IIT-CDIP)
* **Target Classes**: 15 hematology report categories (`CAT_A_MORPHOLOGY` through `CAT_O_SPEP`)
* **Input**: First page rendered document image (resized to 224x224 RGB)
* **Optimization**:
  * AdamW, LR: $2 \times 10^{-5}$
  * Batch size: 8 (gradient accumulation: 2)
  * Precision: FP16
  * Epochs: 5 (with early stopping on validation loss)
* **Target Metric**: Micro & Macro F1 > 0.95 on held-out test split (Template C)

---

## 3. Model 2: Medical Entity Extraction (BioLinkBERT-base + LoRA)

* **Base Model**: `michiyasunaga/BioLinkBERT-base`
* **Pretraining Domain**: PubMed abstracts + PMC full text with biomedical entity links
* **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA)
* **LoRA Configuration**:
  * Rank $r = 8$
  * Alpha $\alpha = 16$
  * Target modules: `["query", "value", "key"]`
  * Dropout: 0.1
* **Target Entities**:
  * `B-PARAM` / `I-PARAM`: Parameter names (e.g., "Myeloblasts", "BCR-ABL1")
  * `B-VAL` / `I-VAL`: Reported values (e.g., "14.8", "MF-2")
  * `B-UNIT`: Biological units (e.g., "%", "x10^9/L")
  * `B-FLAG`: Clinical flags ("H", "L", "Critical", "Positive")
* **Target Metric**: Entity-level Token F1 > 0.90

---

## 4. Model 3: Spatial Layout Understanding (LayoutLMv3-base + LoRA)

* **Base Model**: `microsoft/layoutlmv3-base`
* **Inputs**: Trimodal (rendered page image, text tokens, normalized bounding boxes $[x_0, y_0, x_1, y_1]$)
* **Task**: Token-level sequence labeling across multi-column laboratory tables and headers
* **LoRA Target**: Attention projection weights
* **Target Metric**: Strict layout extraction accuracy on tabular structures

---

## 5. Leakage-Resistant Evaluation Protocol

* **Training Set**: 180 documents (Templates A & B)
* **Validation Set**: 30 documents (Templates A & B)
* **Test Set (Held-Out Unseen)**: 90 documents (**strictly Template C**)
* **Out-of-Distribution Benchmarking**:
  * 90 real-world hematology consult notes from MTSamples
  * India Medical OCR format evaluation

---

## 6. Execution Stages

1. **Step 4.1**: Build dataset tokenization and preprocessing scripts (`scripts/document_ai/prepare_training_features.py`).
2. **Step 4.2**: Implement and run DiT classification training script (`scripts/document_ai/train_classifier.py`).
3. **Step 4.3**: Implement and run BioLinkBERT entity extraction training (`scripts/document_ai/train_ner_extractor.py`).
4. **Step 4.4**: Implement LayoutLMv3 spatial extractor (`scripts/document_ai/train_layout_extractor.py`).
5. **Step 4.5**: Run evaluation benchmark on held-out test set and generate `docs/MODEL_EVALUATION_REPORT.md`.
6. **Step 4.6**: Integrate trained model checkpoints into `pipeline/document_intelligence.py` for live patient intake.
