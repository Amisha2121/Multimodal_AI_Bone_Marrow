# Medical Document AI Model Evaluation & Benchmark Report

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Stage**: 4 — Model Training & Evaluation  
**Date**: 2026-11-11  
**Hardware Executed**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)  
**Evaluation Constraint**: Zero Data Leakage (Held-Out `TEMPLATE_C` Unseen Layouts Only)

---

## 1. Executive Summary

Three document intelligence models were implemented, trained, and benchmarked against the held-out test split of 90 synthetic hematology reports representing 15 clinical categories and unseen visual layout formats (`TEMPLATE_C`).

| Task | Architecture Evaluated | Accuracy | Macro F1 | Status |
|---|---|---|---|---|
| **Visual Document Classification** | EfficientNet-B0 / DiT Backbone | 8.89% | 0.0665 | Failed across layout shifts (memorized visual borders) |
| **Multimodal Vision-Text Fusion** | EfficientNet-B0 + Text Keyword MLP | 16.67% | 0.0837 | Partially robust, visual dominance created layout bias |
| **Semantic Clinical Classifier** | N-Gram TF-IDF + High-Dimensional Logistic Head | **100.00%** | **1.0000** | **Passed** — Immune to visual layout changes |
| **Structured Entity Extraction** | Rule-Guided Medical Biomarker Parser | **93.85% (Prec)** | **0.8299 (F1)** | **Passed** — 366/492 exact parameter-value matches |

---

## 2. Key Insights & Methodological Learnings

1. **The Vision Layout Trap**: Pure visual classification models (CNNs and Vision Transformers) trained on small template datasets heavily overfit to template framing (e.g., border thickness, hospital logos, table grid lines). When evaluated on the unseen `TEMPLATE_C` (which utilizes forest green boxes and distinct margins), visual accuracy collapsed to 8.89%.
2. **Clinical Text Invariance**: Medical clinical vocabulary (e.g., "myeloblasts", "reticulin MF-2", "BCR-ABL1 %IS", "FLT3-ITD", "immunofixation") is 100% invariant to layout styling. The semantic language classifier achieved 100% accuracy across all 90 held-out test cases.
3. **Hybrid Architecture Recommendation**:
   - Primary document category routing should be driven by **semantic language parsing**.
   - Spatial layout models (LayoutLMv3) should be restricted to **bounding box detection** and cell coordinate association rather than whole-document category classification.

---

## 3. Production Model Artifacts

* **Trained Classifier**: `data/document_ai/models/semantic_classifier/semantic_document_classifier.joblib`
* **Category Label Map**: `data/document_ai/processed/category_label_map.json`
* **Evaluation Manifests**:
  * [classification_evaluation_results.json](file:///c:/Users/AMISHA/Desktop/Codes/Multimodal_AI_Bone_Marrow/data/document_ai/manifests/classification_evaluation_results.json)
  * [multimodal_classification_evaluation.json](file:///c:/Users/AMISHA/Desktop/Codes/Multimodal_AI_Bone_Marrow/data/document_ai/manifests/multimodal_classification_evaluation.json)
  * [semantic_classification_evaluation.json](file:///c:/Users/AMISHA/Desktop/Codes/Multimodal_AI_Bone_Marrow/data/document_ai/manifests/semantic_classification_evaluation.json)
  * [extraction_evaluation_results.json](file:///c:/Users/AMISHA/Desktop/Codes/Multimodal_AI_Bone_Marrow/data/document_ai/manifests/extraction_evaluation_results.json)

---

## 4. Preservation & Compliance

* **Vision Transformer Checkpoint**: Untouched in `data/processed/checkpoints/`.
* **FAISS Vector Store & BGE Embeddings**: Intact in `data/processed/retrieval_index/`.
* **BM25 Search Index**: Preserved.
* **Test Suite**: Ready for continuous integration.
