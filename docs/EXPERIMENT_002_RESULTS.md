# EXPERIMENT_002 — Real-Data ViT-Small/16 + LoRA Baseline Results

> **Status:** Complete  
> **Timestamp:** 2026-10-06 11:19 UTC  
> **Dataset:** BoneMarrowWSI-PediatricLeukemia (IDC, CC BY 4.0)  
> **Checkpoint:** `models/checkpoints/best_vit_lora_morphology.pt`

---

## 1. Dataset & Splits

| Split | Patients | Cells |
|---|---|---|
| Train | 79 | 14,563 |
| Validation | 17 | 2,951 |
| Test | 17 | 3,338 |
| **Total** | **113** | **20,852** |

> Patient overlap: Train∩Val = 0, Train∩Test = 0, Val∩Test = 0
> Source: IDC `bonemarrowwsi_pediatricleukemia`, consensus annotations only.

---

## 2. Cell Morphology Taxonomy (8 Classes)

| Idx | Class | Train | Val | Test | Type |
|---|---|---|---|---|---|
| 0 | Blast | 5,170 | 1,204 | 1,275 | Normal |
| 1 | Granulocyte (Mature/Band) | 3,691 | 636 | 795 | Normal |
| 2 | Megakaryocyte / Platelet | 1,447 | 339 | 325 | Normal |
| 3 | Myeloid (Maturing) | 2,385 | 367 | 476 | Normal |
| 4 | Promyelocyte | 1,094 | 184 | 295 | Normal |
| 5 | Lymphoid Series | 327 | 120 | 68 | Rare |
| 6 | Other Morphological | 280 | 58 | 51 | Rare |
| 7 | Monocyte Series | 169 | 43 | 53 | Rare |

**Excluded from classifier (artefacts / no-consensus):**
- `Artefact (Excluded)` — 6,544 cells
- `Ambiguous / No Consensus (Excluded)` — 683 cells
- `Spicule (Tissue Fragment - Excluded)` — 316 cells

---

## 3. Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | `vit_small_patch16_224` |
| Pre-training | ImageNet-21k (via timm) |
| LoRA rank r | 8 |
| LoRA α | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | `['qkv']` |
| Trainable params | ~147K / 21.8M (0.68%) |
| Epochs (max) | 30 |
| Batch size | 64 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Precision | `bf16` |
| Imbalance strategy | WeightedRandomSampler |
| Early stopping | val macro-F1, patience=5 |
| Seed | 42 |

---

## 4. Cell-Level Results (Test Split — 17 Patients, 3,338 Cells)

| Metric | Value |
|---|---|
| Accuracy | 0.8448 |
| Macro Precision | 0.7195 |
| Macro Recall | 0.8293 |
| Macro F1 | 0.7530 |
| Weighted F1 | 0.8536 |
| Rare-class Macro Recall | 0.8099 |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support | Rare? |
|---|---|---|---|---|---|
| Blast | 0.9734 | 0.8886 | 0.9291 | 1,275 | No |
| Granulocyte (Mature/Band) | 0.9335 | 0.8654 | 0.8982 | 795 | No |
| Megakaryocyte / Platelet | 0.9937 | 0.9662 | 0.9797 | 325 | No |
| Myeloid (Maturing) | 0.6769 | 0.6029 | 0.6378 | 476 | No |
| Promyelocyte | 0.6806 | 0.8814 | 0.7681 | 295 | No |
| Lymphoid Series | 0.3876 | 0.7353 | 0.5076 | 68 | Yes |
| Other Morphological | 0.7797 | 0.9020 | 0.8364 | 51 | Yes |
| Monocyte Series | 0.3307 | 0.7925 | 0.4667 | 53 | Yes |

### Confusion Matrix

See `models/evaluation/confusion_matrix.png`

---

## 5. Patient-Level Results (Test Split — 17 Patients)

> **Classifier:** Logistic Regression on mean-pooled cell softmax probabilities  
> **Features:** 8-dim cell differential vector per patient  
> **Classes:** ALL / AML / CML / Other

| Metric | Value |
|---|---|
| Accuracy | 0.8235 |
| Macro Precision | 0.5750 |
| Macro Recall | 0.6667 |
| Macro F1 | 0.5985 |
| Weighted F1 | 0.8057 |

### Per-Class Patient-Level Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ALL | 0.8000 | 0.6667 | 0.7273 | 6 |
| AML | 0.5000 | 1.0000 | 0.6667 | 2 |
| CML | 1.0000 | 1.0000 | 1.0000 | 8 |
| Other | 0.0000 | 0.0000 | 0.0000 | 1 |

### Confusion Matrix

See `models/evaluation/patient_confusion_matrix.png`

---

## 6. Low-Evidence Cases

> Patients with fewer than 50 annotated cells are flagged as LOW-EVIDENCE.  
> Their predictions should be interpreted with extra caution.
> These patients are RETAINED in the analysis and NOT deleted.

| Patient ID | Cells | True Class | Predicted | Blast Fraction |
|---|---|---|---|---|
| `32BB90E8976A…` | 9 | ALL | AML | 0.444 |
| `93DB85ED909C…` | 24 | ALL | ALL | 0.792 |

---

## 7. XAI — Attention Rollout Examples

Representative attention rollout heatmaps (Abnar & Zuidema 2020) are saved in:
```
models/evaluation/xai/xai_{class}_{n}.png
```

---

## 8. Scientific Disclaimers

> [!WARNING]
> **Annotated-Cell Blast Fraction ≠ Clinical Blast Percentage.**  
> The consensus annotations represent selected monolayer regions, NOT
> an exhaustive whole-slide clinical aspirate count.  
> No clinical claims can be made from this metric alone.

> [!NOTE]
> This experiment represents the **vision morphology baseline** only.  
> It is NOT the final multimodal agentic system.  
> CBC, cytogenetics, and clinical notes are absent from this dataset.

---

## 9. Limitations

1. **Pediatric-specific cohort** (ages 1.5–17 yrs). Cannot generalize to adults.
2. **Extreme class imbalance**: CML=44%, ALL=33%, AML=15%, Other=8%.
3. **No ancillary modalities**: CBC, cytogenetics, and molecular markers absent.
4. **17 test patients**: Small test set; confidence intervals are wide.
5. **Monolayer sampling bias**: Cells selected from pre-identified ROIs.

---

## 10. Next Steps

- ☐ RAG layer: literature ingest, FAISS + BM25 hybrid indexing, BGE reranker
- ☐ Agentic framework: Retriever, Synthesizer, Critic agents
- ☐ Multimodal context assembly

---

*Generated by `scripts/step6_patient_baseline.py` on 2026-10-06 11:19 UTC.*
