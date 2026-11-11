# Environment and Hardware Audit

**Project**: Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment  
**Audit Date**: 2026-11-11  
**Purpose**: Document the full hardware and software environment to constrain model selection for Stage 2.

---

## 1. Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM | 7.96 GB (nominally 8 GB) |
| CUDA Version | 12.8 |
| CPU | (host system CPU) |
| System RAM | (host system RAM) |
| Available Disk | ~121 GB free of 952 GB total |

### VRAM Capacity Planning

At **FP16 precision**, model VRAM footprint ≈ 2 × parameter count in bytes.

| Parameter Count | FP16 VRAM (weights only) | With Activations (~1.5× overhead) | Fits in 8 GB? |
|---|---|---|---|
| 86 M (DiT-base) | ~172 MB | ~260 MB | Yes |
| 110 M (BioLinkBERT/PubMedBERT) | ~220 MB | ~330 MB | Yes |
| 125 M (LayoutLMv3-base) | ~250 MB | ~375 MB | Yes |
| 200 M (Donut-base) | ~400 MB | ~600 MB | Yes |
| 368 M (LayoutLMv3-large) | ~736 MB | ~1.1 GB | Yes (inference only) |
| 1 B+ | ~2 GB | ~3+ GB | Marginal |

> **Note**: Fine-tuning adds optimizer states (Adam: 3–4× weights). LayoutLMv3-large fine-tuning (~3–4 GB) is borderline. LoRA/PEFT reduces this to ~1.2× weights.

**Hard VRAM ceiling for safe fine-tuning (with LoRA)**: models up to ~368 M parameters.  
**Hard VRAM ceiling for inference only**: models up to ~1 B parameters.

---

## 2. Software Environment

| Package | Version |
|---------|---------|
| Python | 3.14.4 |
| PyTorch | 2.11.0+cu128 |
| CUDA (via torch) | 12.8 |
| transformers | 5.5.4 |
| accelerate | 1.13.0 |
| timm | 1.0.30 |
| scikit-learn | 1.8.0 |
| Pillow | 12.2.0 |
| PyMuPDF (fitz) | 1.27.2.3 |
| OpenCV (cv2) | Available |
| datasets (HuggingFace) | Installed Stage 3 |
| reportlab | Installed Stage 3 |
| peft | To be verified |
| bitsandbytes | To be verified |

### Python Executable
```
C:\Users\AMISHA\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

---

## 3. Framework Compatibility Notes

### PyTorch 2.11 + CUDA 12.8
- Full BF16/FP16 support on RTX 5050 (Blackwell/Ada architecture).
- Flash Attention 2 available if `flash-attn` is installed.
- `torch.compile()` available for inference acceleration.

### PaddleOCR Risk
PaddlePaddle uses its own GPU runtime that **conflicts** with PyTorch CUDA context sharing. If PaddleOCR is evaluated:
- Must run in an **isolated virtual environment** (`venv_paddle/`).
- Must NOT be installed system-wide alongside PyTorch.
- Results captured via subprocess calls only.

### HuggingFace Transformers 5.x
- Full LayoutLMv3, DiT, Donut, BioLinkBERT compatibility confirmed.
- `AutoProcessor`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` all available.
- PEFT/LoRA: requires `peft>=0.6.0` — install separately.

---

## 4. Storage Plan

| Path | Purpose | Estimated Size |
|------|---------|----------------|
| `data/document_ai/raw/` | Raw downloaded datasets | ~500 MB – 2 GB |
| `data/document_ai/processed/` | Preprocessed features | ~1–3 GB |
| `data/document_ai/annotations/` | JSON ground-truth annotations | ~50 MB |
| `data/document_ai/synthetic_corpus/` | Synthetic PDFs + text | ~200 MB |
| `data/document_ai/splits/` | Split manifests | ~5 MB |
| Model checkpoints (future) | LoRA adapters per model | ~100–500 MB each |

**Total estimated storage for Stage 3**: ~500 MB – 1 GB (synthetic + small public datasets only).  
Large datasets (DocLayNet, PubLayNet) are **not downloaded**; only metadata is recorded.

---

## 5. Network and Licensing Constraints

| Constraint | Detail |
|-----------|--------|
| MIMIC-IV | NOT downloaded — requires PhysioNet DUA credentialed access |
| PaddleOCR weights | Downloadable (Apache 2.0) but isolated venv required |
| CC-BY-NC-4.0 models | Permitted for research use; cannot be redistributed commercially |
| HuggingFace Hub access | Available — `datasets` library used for metadata probing |
| Internet access | Available for model card inspection |

---

## 6. Existing Pipeline — Preservation Requirements

The following components must remain **completely unmodified** throughout Stages 2–3:

| Component | Location |
|-----------|----------|
| ViT checkpoint | `data/processed/checkpoints/` |
| FAISS index | `data/processed/retrieval_index/` |
| BGE-small embeddings | `data/processed/retrieval_index/` |
| BM25 pickle | `data/processed/retrieval_index/` |
| Flask app routes | `app/flask_app.py` (existing routes) |
| RAG pipeline | `pipeline/rag_*.py` |
| Patient context | `pipeline/patient_context.py` |
| Evidence registry | `pipeline/evidence_registry.py` |

The Document AI training pipeline is **additive only** — it creates new files in `data/document_ai/`, `scripts/document_ai/`, and `docs/` without touching existing infrastructure.
