# Medical Document AI Dataset Catalog & Source Audit

**Document Date:** November 2026  
**Focus:** Medical Document Intelligence, Classification, Layout Understanding, and Entity Extraction for Hematology & Bone Marrow Pathology.

---

## 1. Discovered Medical & Document AI Datasets

The audit investigated open-access repositories across Hugging Face, GitHub, PhysioNet, PubMed Central (PMC), and NCBI. 

| # | Dataset Name | Repository / Access URL | Modality & Scope | Number of Samples | Format | Annotations | License | Hematology / Bone Marrow Relevance | Training Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **DocLayNet** | `ds4sd/DocLayNet` (IBM / Hugging Face) | Document Layout Analysis | 80,863 pages | JSON / PNG / PDF | Bounding boxes across 11 layout categories (Table, Text, Title, Header, Footnote) | CDLA-Permissive-1.0 | High layout structural similarity to laboratory & medical reports | Foundation for Model 2 (Layout Understanding) |
| **2** | **PubLayNet** | `creative-graphic-design/PubLayNet` (PMC) | Scientific & Medical Layout | 360,000+ document pages | COCO JSON / PNG | Bounding boxes: Text, Title, List, Table, Figure | ODC-By v1.0 | High: Derived from PubMed Central biomedical publications | Pretraining for Table & Section boundaries |
| **3** | **JohnSnowLabs PDF-Deid** | `JohnSnowLabs/pdf-deid-dataset` (GitHub) | Synthetic Clinical PDF Documents | 1,000+ complex medical PDFs | Multi-page PDF / PNG | Ground-truth clinical entities, demographics, lab tables | Apache 2.0 (Open Access) | Very High: Mimics real clinical forms, vital tables, lab sheets | Model 1 (Classification) & Model 3 (Entity Extraction) |
| **4** | **Synthetic Medical Document Benchmark** | `morzel85/synthetic-medical-document-recognition-benchmark` (HF) | Clinical Laboratory & Diagnostic Reports | 10,000+ page images | PDF / PNG / JSON | Token-level bounding boxes, key-value lab parameters | MIT License | Very High: Specifically designed for automated lab sheet OCR and extraction | Model 3 (Structured Lab Extraction) |
| **5** | **MTSamples (Hematology-Oncology)** | `harishnair04/mtsamples` (Hugging Face / Kaggle) | Clinical Transcriptions & Consult Notes | ~5,000 total (350+ dedicated Hematology/Oncology) | Text / CSV | Labeled by medical specialty, consult type, keywords | CC0 / Public Domain | High: Free-text bone marrow aspirate consults, leukemia workups | Language model fine-tuning & entity parsing |
| **6** | **MIMIC-IV-Pathology (Notes)** | PhysioNet (`physionet.org/content/mimic-iv-note`) | De-identified Hospital Pathology Notes | 100,000+ unstructured clinical/pathology notes | Relational Database / Text | Free-text clinical and bone marrow biopsy interpretations | PhysioNet Credentialed Data Use Agreement | High: Real-world pathology text distributions (requires credentialed access) | Validation benchmark for pathology text parsing |
| **7** | **ClinOCR-Bench / MedOCR** | `naazimsnh02/medocr-vision-dataset` (Hugging Face) | Scanned Medical Documents & Lab Reports | 15,000+ image-text pairs | Image (PNG/JPG) + OCR text | Word/line level bounding boxes, noise/rotation artifacts | CC-BY-NC 4.0 | High: Robustness evaluation against noisy scanned lab sheets | OCR resilience & scanned document evaluation |
| **8** | **India Medical OCR Test Set** | `davanstrien/india-medical-ocr-test` (Hugging Face) | Diverse Regional Laboratory Reports | 500+ scanned lab panels | Image + ground truth | Laboratory test names, units, reference intervals | Open Access / Research | High: Evaluates template shifts across diverse hospital lab formats | Out-of-distribution robustness test set |
| **9** | **NCBI GEO Leukemia Metadata** | NCBI GEO (GSE24006, GSE291349) | Clinical & Molecular Metadata | 1,000+ patient runs | Tabular TSV / XML | ISCN karyotypes, fusion status, blast %, cytogenetics | Public Domain (NCBI) | Very High: Ground truth cytogenetic strings and molecular mutations | Synthetic template population & ISCN validation |

---

## 2. Licensing, Privacy & De-Identification Audit

1. **Patient Data Privacy & HIPAA Compliance**:
   - Zero raw, patient-identifiable documents are downloaded or scraped.
   - Clinical benchmarks (JohnSnowLabs, Morzel85) utilize **fully de-identified or procedurally synthesized medical templates** that emulate authentic hospital formatting without real patient identifiers.
   - PhysioNet MIMIC-IV notes remain under institutional DUA protocols.
2. **License Compatibility**:
   - `DocLayNet` (CDLA-Permissive-1.0) and `PubLayNet` (ODC-By) allow model training and commercial/research weight deployment.
   - `JohnSnowLabs/pdf-deid-dataset` (Apache 2.0) and `morzel85` benchmark (MIT) permit direct pipeline integration and testing.
3. **Modality Distinction**:
   - Cell microscopy image datasets (e.g. TCIA Bone Marrow Single Cell dataset) are strictly isolated for the Vision Transformer pipeline.
   - Document AI datasets are dedicated exclusively to page images, PDF text structures, tabular lab grids, and pathology consult reports.
