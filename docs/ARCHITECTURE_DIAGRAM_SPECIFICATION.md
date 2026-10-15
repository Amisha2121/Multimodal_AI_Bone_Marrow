# 🎨 Architecture Diagram Specification

**Purpose**: Complete, accurate specification for creating the system architecture diagram  
**Based On**: Actual implementation verified against codebase  
**Last Updated**: October 7, 2026  

---

## 📋 Table of Contents

1. [Overall Layout](#overall-layout)
2. [Component 1: Input](#component-1-input)
3. [Component 2: Preprocessing](#component-2-preprocessing)
4. [Component 3: Vision Pipeline](#component-3-vision-pipeline)
5. [Component 4: Patient-Level Aggregation](#component-4-patient-level-aggregation)
6. [Component 5: Agentic Multi-Agent System](#component-5-agentic-multi-agent-system)
7. [Component 6: Knowledge Base](#component-6-knowledge-base)
8. [Component 7: RAG System](#component-7-rag-system)
9. [Component 8: Explainability Module](#component-8-explainability-module)
10. [Component 9: Output & Interaction](#component-9-output--interaction)
11. [Data Flow Arrows](#data-flow-arrows)
12. [Color Scheme](#color-scheme)
13. [Visual Examples](#visual-examples)

---

## 🎯 Overall Layout

### Diagram Structure (Left to Right):
```
┌────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────────┐   ┌────────┐
│   1.   │→→→│      2.      │→→→│    3.    │→→→│     4.      │→→→│   5.   │
│ Input  │   │Preprocessing │   │  Vision  │   │ Aggregation │   │ Agents │
└────────┘   └──────────────┘   └──────────┘   └─────────────┘   └────┬───┘
                                                                        │
┌─────────────────────────────────────────────────────────────────────┘
│
├→ 6. Knowledge Base ─┐
├→ 7. RAG System ─────┤→→→ 8. Explainability ─→→→ 9. Output
├→ (Feedback Loop) ───┘
```

### Dimensions:
- **Width**: ~1200-1400px
- **Height**: ~700-900px
- **9 main components** arranged in logical flow

---

## Component 1: Input

### Box Properties:
- **Color**: Light Purple (#E8E4F3)
- **Border**: Purple (#7C3AED)
- **Size**: 200×300px
- **Position**: Far left

### Content:

```
┌─────────────────────────────────────┐
│ 1. INPUT                             │
├─────────────────────────────────────┤
│                                      │
│ ✅ REQUIRED:                         │
│ ┌─────┐                              │
│ │🖼️   │ Bone Marrow Microscopy      │
│ │     │ Image                        │
│ └─────┘ • PNG / JPG / TIFF          │
│         • 224×224 to 4000×4000 px    │
│         • RGB color                  │
│         • Single or multiple images  │
│                                      │
│ ⚠️ OPTIONAL (Enhances Quality):     │
│ ┌─────┐                              │
│ │👤   │ Patient Demographics        │
│ └─────┘ • Age (years or months)     │
│         • Sex (M/F)                  │
│         • Patient ID                 │
│                                      │
│ Used for:                            │
│   • Age-aware RAG queries           │
│   • Pediatric vs adult context      │
│   • Report display only             │
│                                      │
│ ❌ NOT USED (Dataset limitation):   │
│   • CBC / Cytogenetics              │
│   • Clinical notes                  │
│   • Prior test results              │
│                                      │
└─────────────────────────────────────┘
```

### Key Notes:
- **Only image is required**
- Age/sex are optional, improve quality
- No clinical data needed
- No research query needed

---

## Component 2: Preprocessing

### Box Properties:
- **Color**: Light Blue (#E0F2FE)
- **Border**: Blue (#0284C7)
- **Size**: 250×350px
- **Position**: After Input

### Content:

```
┌─────────────────────────────────────┐
│ 2. PREPROCESSING                     │
├─────────────────────────────────────┤
│                                      │
│ Step 1: Image Validation             │
│ ┌─────┐                              │
│ │✓    │ • Check format (PNG/JPG/TIFF)│
│ └─────┘ • Validate dimensions        │
│         • Check file size (<50MB)    │
│         • Convert to PIL Image       │
│                                      │
│         ↓                            │
│                                      │
│ Step 2: Cell Detection               │
│ ┌─────┐ Heuristic Detector:          │
│ │🔍   │ 1. RGB → Optical Density     │
│ └─────┘ 2. Nuclear stain separation  │
│ [img]   3. Distance transform        │
│         4. Peak detection            │
│         5. Watershed segmentation    │
│         6. Morphological filtering   │
│         7. NMS (IoU threshold 0.40)  │
│                                      │
│ Parameters:                          │
│   • Min cell area: 250 px²          │
│   • Max cell area: 30,000 px²       │
│   • Min dimension: 15 px            │
│   • Max dimension: 180 px           │
│                                      │
│ Output: ~154 detected cells          │
│                                      │
│         ↓                            │
│                                      │
│ Step 3: Crop Extraction              │
│ ┌─────┐                              │
│ │✂️   │ Extract 224×224 crops        │
│ └─────┘ Per detected cell            │
│ [grid]  Center on cell centroid      │
│         RGB format                   │
│                                      │
│         ↓                            │
│                                      │
│ Step 4: Normalization                │
│ ┌─────┐                              │
│ │📊   │ Model preprocessing:         │
│ └─────┘ • Resize to 224×224          │
│ [img]   • Normalize (ImageNet stats) │
│         • ToTensor transformation    │
│                                      │
│ NO DATA AUGMENTATION (inference)     │
│                                      │
└─────────────────────────────────────┘
```

### Key Changes from Original:
- ❌ Removed "DICOM Processing"
- ✅ Added "Cell Detection" as first step
- ✅ Shows extraction happens on-the-fly
- ✅ No augmentation during inference

---

## Component 3: Vision Pipeline

### Box Properties:
- **Color**: Light Yellow (#FEF9C3)
- **Border**: Yellow/Gold (#CA8A04)
- **Size**: 280×400px
- **Position**: Center

### Content:

```
┌─────────────────────────────────────┐
│ 3. VISION PIPELINE                   │
│ (Cell Classification)                │
├─────────────────────────────────────┤
│                                      │
│ Model Architecture:                  │
│ ┌─────────────────────────────┐    │
│ │ ViT-Small/16 + LoRA         │    │
│ │                             │    │
│ │ Backbone:                   │    │
│ │   vit_small_patch16_224    │    │
│ │   • 22M parameters (frozen) │    │
│ │   • 12 transformer layers   │    │
│ │   • Patch size: 16×16       │    │
│ │   • Hidden dim: 384         │    │
│ │                             │    │
│ │ LoRA Adaptation:            │    │
│ │   • Rank: 8                 │    │
│ │   • Alpha: 16               │    │
│ │   • Target: QKV projections │    │
│ │   • 0.3M trainable params   │    │
│ │                             │    │
│ │ Classification Head:        │    │
│ │   Linear(384 → 8 classes)  │    │
│ │   • 0.2M trainable params   │    │
│ └─────────────────────────────┘    │
│                                      │
│         ↓                            │
│                                      │
│ Processing (Batch):                  │
│ ┌─────┐                              │
│ │🔄   │ • Batch size: 32 cells       │
│ └─────┘ • Forward pass through ViT   │
│         • Softmax for probabilities  │
│         • Confidence threshold: 0.60 │
│                                      │
│         ↓                            │
│                                      │
│ Cell Classification Output:          │
│ ┌─────────────────────────────┐    │
│ │ 8 Morphology Classes:       │    │
│ │                             │    │
│ │ [img] Blast                 │    │
│ │ [img] Lymphoid Series       │    │
│ │ [img] Granulocyte (Mature)  │    │
│ │ [img] Granulocyte (Immature)│    │
│ │ [img] Erythroid Precursor   │    │
│ │ [img] Monocyte              │    │
│ │ [img] Megakaryocyte/Platelet│    │
│ │ [img] Other/Artifact        │    │
│ └─────────────────────────────┘    │
│                                      │
│ Per-Cell Output:                     │
│ ┌─────────────────────────────┐    │
│ │ Cell #1:                    │    │
│ │   Class: Blast              │    │
│ │   Confidence: 0.92          │    │
│ │   Probabilities:            │    │
│ │     Blast: 0.92             │    │
│ │     Lymphoid: 0.05          │    │
│ │     Others: 0.03            │    │
│ │                             │    │
│ │ Cell #2:                    │    │
│ │   Class: Lymphoid Series    │    │
│ │   Confidence: 0.88          │    │
│ │   ...                       │    │
│ └─────────────────────────────┘    │
│                                      │
│ Total: ~154 classified cells         │
│                                      │
└─────────────────────────────────────┘
```

### Key Features:
- Specific model name: ViT-Small/16 + LoRA
- Exact parameter counts
- 8-class taxonomy
- Per-cell predictions with confidence

---

## Component 4: Patient-Level Aggregation

### Box Properties:
- **Color**: Light Green (#D1FAE5)
- **Border**: Green (#059669)
- **Size**: 250×400px
- **Position**: After Vision

### Content:

```
┌─────────────────────────────────────┐
│ 4. PATIENT-LEVEL AGGREGATION         │
├─────────────────────────────────────┤
│                                      │
│ Input: 154 cell predictions          │
│                                      │
│ Step 1: Cell Counting                │
│ ┌─────────────────────────────┐    │
│ │ Count by Category:          │    │
│ │                             │    │
│ │ • Blast: 8 cells (5.2%)    │    │
│ │ • Megakaryocyte: 87 (56.5%)│    │
│ │ • Lymphoid: 34 (22.1%)     │    │
│ │ • Granulocyte: 25 (16.2%)  │    │
│ │ • Others: minimal           │    │
│ │                             │    │
│ │ Total: 154 cells            │    │
│ └─────────────────────────────┘    │
│                                      │
│         ↓                            │
│                                      │
│ Step 2: Blast Fraction               │
│ ┌─────────────────────────────┐    │
│ │ Annotated Blast Fraction:   │    │
│ │                             │    │
│ │   8 blasts / 154 total      │    │
│ │ = 5.2%                      │    │
│ │                             │    │
│ │ WHO Criteria:               │    │
│ │   • <20%: Not acute leukemia│    │
│ │   • ≥20%: Acute leukemia    │    │
│ │                             │    │
│ │ Result: 5.2% (<20%)         │    │
│ └─────────────────────────────┘    │
│                                      │
│         ↓                            │
│                                      │
│ Step 3: Disease Prediction           │
│ ┌─────────────────────────────┐    │
│ │ Aggregate Probabilities:    │    │
│ │                             │    │
│ │ • Megakaryocyte: 56.5%      │    │
│ │   → Predicted Class         │    │
│ │                             │    │
│ │ Differential List:          │    │
│ │   1. Reactive/Other         │    │
│ │   2. CML (chronic phase)    │    │
│ │   3. ET (need cytogenetics) │    │
│ │                             │    │
│ │ Confidence: MODERATE        │    │
│ │ (154 cells detected)        │    │
│ └─────────────────────────────┘    │
│                                      │
│         ↓                            │
│                                      │
│ Step 4: Quality Flags                │
│ ┌─────────────────────────────┐    │
│ │ Evidence Sufficiency:       │    │
│ │                             │    │
│ │ Total Cells: 154            │    │
│ │ Status: MODERATE            │    │
│ │                             │    │
│ │ Thresholds:                 │    │
│ │   <5: INSUFFICIENT          │    │
│ │   5-49: LOW                 │    │
│ │   50-149: MODERATE          │    │
│ │   ≥150: HIGH                │    │
│ │                             │    │
│ │ Low Evidence Flag: FALSE    │    │
│ └─────────────────────────────┘    │
│                                      │
│ Output: PatientAnalysis object       │
│                                      │
└─────────────────────────────────────┘
```

### Visual Addition:
```
┌────────────────────────┐
│ Cell Distribution      │
│ ███████████ 56.5%      │ Megakaryocyte
│ █████ 22.1%            │ Lymphoid
│ ████ 16.2%             │ Granulocyte
│ █ 5.2%                 │ Blast
└────────────────────────┘
```

---

## Component 5: Agentic Multi-Agent System

### Box Properties:
- **Color**: Light Red/Pink (#FEE2E2)
- **Border**: Red (#DC2626)
- **Size**: 900×200px (wide, spans multiple columns)
- **Position**: Below aggregation, above knowledge base

### Content:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. AGENTIC MULTI-AGENT SYSTEM (LangGraph)                                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐│
│  │  START   │ →  │  RETRIEVER   │ →  │ SYNTHESIZER  │ →  │   CRITIC     │ →  │  END   ││
│  │          │    │    AGENT     │    │    AGENT     │    │    AGENT     │    │        ││
│  └──────────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └────────┘│
│                         │                    │                    │                      │
│                         │                    │                    │                      │
│                  ┌──────▼────────────┐┌──────▼──────────┐ ┌──────▼────────────┐        │
│                  │ Search WHO        ││ Generate        │ │ Validate          │        │
│                  │ Guidelines        ││ Assessment      │ │ Evidence          │        │
│                  │                   ││                 │ │                   │        │
│                  │ Input:            ││ Input:          │ │ Input:            │        │
│                  │  • Blast: 5.2%    ││  • Retrieved    │ │  • Draft report   │        │
│                  │  • Cells: Mega    ││    guidelines   │ │  • Citations      │        │
│                  │  • Age: 6 yrs     ││  • Patient data │ │  • Total cells    │        │
│                  │                   ││                 │ │                   │        │
│                  │ Query Build:      ││ LLM:            │ │ Checks:           │        │
│                  │  "Bone marrow     ││  Groq API       │ │  • Citation valid?│        │
│                  │   for Megakaryo   ││  Model: openai/ │ │  • Data fabrica...

ted?│        │
│                  │   blast 5.2%      ││  gpt-oss-120b   │ │  • Evidence suff..?│        │
│                  │   age 6 years"    ││  120B params    │ │                   │        │
│                  │                   ││  Temp: 0.1      │ │ Grounding Score:  │        │
│                  │ Hybrid Retrieval: ││                 │ │  0.0 - 1.0        │        │
│                  │  • FAISS (dense)  ││ Mode:           │ │                   │        │
│                  │  • BM25 (sparse)  ││  LLM-Powered ✓  │ │ Sufficiency:      │        │
│                  │  • RRF fusion     ││  Deterministic  │ │  • <5: INSUFF     │        │
│                  │                   ││  (fallback)     │ │  • 5-49: LOW      │        │
│                  │ Output:           ││                 │ │  • 50-149: MOD    │        │
│                  │  Top-4 passages   ││ Output:         │ │  • ≥150: HIGH     │        │
│                  │  WHO 5th Ed       ││  Clinical       │ │                   │        │
│                  │                   ││  assessment     │ │ Decision:         │        │
│                  └───────────────────┘│  ~800 words     │ │  APPROVED ✓       │        │
│                                       │  WHO citations  │ │  or REJECTED      │        │
│                                       └─────────────────┘ │  (→ retry)        │        │
│                                                            └───────────────────┘        │
│                                                                      │                  │
│                                                                      │ if rejected      │
│                                                                      │ & revision < 2   │
│                                                                      │                  │
│                                                                      └──────────┐       │
│                  ┌─────────────────────────────────────────────────────────────┘       │
│                  │ Iterative Feedback Loop (max 2 revisions)                            │
│                  └───────────────────────→ SYNTHESIZER ───────────────────────┐         │
│                                                                                │         │
└────────────────────────────────────────────────────────────────────────────────┴─────────┘
```

### CRITICAL: This is the CORRECTED version with 3 agents:
1. **Retriever Agent** - Searches guidelines
2. **Synthesizer Agent** - Generates assessment with Groq LLM
3. **Critic Agent** - Validates evidence (IMPORTANT!)

### Removed from Original Diagram:
- ❌ Orchestrator Agent (it's LangGraph workflow, not an agent)
- ❌ Vision Agent (vision runs before agents)
- ❌ Clinical Agent (metadata passed through state)
- ❌ Explanation Agent (part of synthesizer)
- ❌ Reasoning Agent (this IS the synthesizer)

---

## Component 6: Knowledge Base for RAG

### Box Properties:
- **Color**: Light Cyan (#CFFAFE)
- **Border**: Cyan (#0891B2)
- **Size**: 220×300px
- **Position**: Below agents, left side

### Content:

```
┌─────────────────────────────────────┐
│ 6. KNOWLEDGE BASE FOR RAG            │
├─────────────────────────────────────┤
│                                      │
│ Medical Literature:                  │
│ ┌─────┐                              │
│ │📚   │ WHO 5th Edition (2022)       │
│ │     │ • Leukemia classification    │
│ │     │ • Diagnostic criteria        │
│ │     │ • Blast thresholds           │
│ │     │                              │
│ │     │ ICC 2022                     │
│ │     │ • Morphology descriptions    │
│ │     │ • Clinical guidelines        │
│ └─────┘                              │
│                                      │
│ Content Structure:                   │
│ ┌─────────────────────────────┐    │
│ │ 12 Clinical Chunks          │    │
│ │                             │    │
│ │ Each chunk contains:        │    │
│ │  • chunk_id (unique)        │    │
│ │  • title                    │    │
│ │  • text (512 tokens)        │    │
│ │  • source (WHO/ICC)         │    │
│ │  • disease_category         │    │
│ │  • specific_entity          │    │
│ │  • blast_threshold_rule     │    │
│ │                             │    │
│ │ Example chunk IDs:          │    │
│ │  • WHO2022_BALL_001         │    │
│ │  • WHO2022_AML_SUB_005      │    │
│ │  • WHO2022_CML_CP_007       │    │
│ │  • WHO2022_HAEMATOGONES_009 │    │
│ │  • WHO2022_REACTIVE_010     │    │
│ │  • DISCLAIMER_MONOLAYER_012 │    │
│ └─────────────────────────────┘    │
│                                      │
│ Fallback System:                     │
│ ┌─────────────────────────────┐    │
│ │ Built-in WHO Guidelines     │    │
│ │ (agents/fallback_guidelines)│    │
│ │                             │    │
│ │ Used when:                  │    │
│ │  • Retrieval fails          │    │
│ │  • No results found         │    │
│ │  • FAISS index unavailable  │    │
│ │                             │    │
│ │ Ensures: Never empty results│    │
│ └─────────────────────────────┘    │
│                                      │
│ ❌ NOT USED:                         │
│  • Internal case database           │
│  • Synthetic cases                  │
│  • Previous patient data            │
│                                      │
└─────────────────────────────────────┘
```

---

## Component 7: RAG System

### Box Properties:
- **Color**: Light Purple (#F3E8FF)
- **Border**: Purple (#9333EA)
- **Size**: 300×350px
- **Position**: Below agents, center

### Content:

```
┌─────────────────────────────────────┐
│ 7. RETRIEVAL-AUGMENTED GENERATION    │
│         (RAG)                        │
├─────────────────────────────────────┤
│                                      │
│ Step 1: Query Embedding              │
│ ┌─────────────────────────────┐    │
│ │ Embedder:                   │    │
│ │   BAAI/bge-small-en-v1.5    │    │
│ │   • 384 dimensions          │    │
│ │   • Cosine similarity       │    │
│ │                             │    │
│ │ Input Query:                │    │
│ │   "Bone marrow cytomorpho..│    │
│ │    for Megakaryocyte |     │    │
│ │    blast 5.2% | age 6 yrs" │    │
│ │                             │    │
│ │ Output: 384-dim vector      │    │
│ └─────────────────────────────┘    │
│         ↓                            │
│         ↓                            │
│ Step 2: Hybrid Search                │
│ ┌─────────────────────────────┐    │
│ │ A. Dense Search (FAISS):    │    │
│ │    • Vector similarity      │    │
│ │    • FlatL2 index           │    │
│ │    • Top-10 results         │    │
│ │                             │    │
│ │ B. Sparse Search (BM25):    │    │
│ │    • Lexical matching       │    │
│ │    • Token-based            │    │
│ │    • Top-10 results         │    │
│ └─────────────────────────────┘    │
│         ↓                            │
│         ↓                            │
│ Step 3: Reciprocal Rank Fusion      │
│ ┌─────────────────────────────┐    │
│ │ RRF Formula:                │    │
│ │                             │    │
│ │ score = dense_weight        │    │
│ │         ───────────── +     │    │
│ │         k + dense_rank      │    │
│ │                             │    │
│ │         sparse_weight       │    │
│ │         ────────────────    │    │
│ │         k + sparse_rank     │    │
│ │                             │    │
│ │ Parameters:                 │    │
│ │   k = 60                    │    │
│ │   dense_weight = 0.6        │    │
│ │   sparse_weight = 0.4       │    │
│ │                             │    │
│ │ Rerank candidates           │    │
│ │ Select top-4                │    │
│ └─────────────────────────────┘    │
│         ↓                            │
│         ↓                            │
│ Step 4: Pass to LLM                  │
│ ┌─────────────────────────────┐    │
│ │ Top-4 Retrieved Documents:  │    │
│ │                             │    │
│ │ 1. [WHO2022_CML_CP_007]     │    │
│ │    "Chronic phase CML..."   │    │
│ │    (RRF: 0.245)             │    │
│ │                             │    │
│ │ 2. [WHO2022_REACTIVE_010]   │    │
│ │    "Reactive changes..."    │    │
│ │    (RRF: 0.189)             │    │
│ │                             │    │
│ │ 3. [WHO2022_HAEMATOGONES..] │    │
│ │    "Pediatric B-cell..."    │    │
│ │    (RRF: 0.156)             │    │
│ │                             │    │
│ │ 4. [DISCLAIMER_MONOLAYER..] │    │
│ │    "Monolayer field..."     │    │
│ │    (RRF: 0.134)             │    │
│ └─────────────────────────────┘    │
│         ↓                            │
│         ↓                            │
│ Step 5: LLM Generation               │
│ ┌─────────────────────────────┐    │
│ │ Groq API                    │    │
│ │                             │    │
│ │ Endpoint:                   │    │
│ │   api.groq.com/openai/v1/   │    │
│ │   chat/completions          │    │
│ │                             │    │
│ │ Model:                      │    │
│ │   openai/gpt-oss-120b       │    │
│ │   (120 billion parameters)  │    │
│ │                             │    │
│ │ Parameters:                 │    │
│ │   max_tokens: 2048          │    │
│ │   temperature: 0.1          │    │
│ │                             │    │
│ │ Input: System prompt +      │    │
│ │        Retrieved docs +     │    │
│ │        Patient data         │    │
│ │                             │    │
│ │ Output: Clinical assessment │    │
│ │         with citations      │    │
│ │         (~800 words)        │    │
│ └─────────────────────────────┘    │
│                                      │
│ Latency: 1-3 seconds                 │
│ Cost: $0 (free tier)                 │
│                                      │
└─────────────────────────────────────┘
```

### Key Corrections:
- ✅ BGE-small (not Gemini)
- ✅ FAISS (not ChromaDB)
- ✅ Added BM25 + RRF
- ✅ Specific model: openai/gpt-oss-120b (120B)

---

## Component 8: Explainability Module

### Box Properties:
- **Color**: Light Lavender (#EDE9FE)
- **Border**: Purple (#7C3AED)
- **Size**: 280×350px
- **Position**: Below agents, right side

### Content:

```
┌─────────────────────────────────────┐
│ 8. EXPLAINABILITY & EVIDENCE MODULE  │
├─────────────────────────────────────┤
│                                      │
│ A. Visual Explanations               │
│ ┌─────────────────────────────┐    │
│ │ Attention Rollout:          │    │
│ │                             │    │
│ │ ┌────────┐   ┌────────┐    │    │
│ │ │Original│ → │Heatmap │    │    │
│ │ │  Cell  │   │(colored│    │    │
│ │ └────────┘   └────────┘    │    │
│ │  [cell]      [attention]   │    │
│ │                             │    │
│ │ Shows:                      │    │
│ │  • High attention (red/hot) │    │
│ │  • Low attention (blue/cold)│    │
│ │                             │    │
│ │ Highlights:                 │    │
│ │  ✓ Nuclear chromatin        │    │
│ │  ✓ Nucleoli                 │    │
│ │  ✓ Cytoplasmic features     │    │
│ │  ✓ Cell boundaries          │    │
│ │                             │    │
│ │ Method:                     │    │
│ │  • ViT attention matrices   │    │
│ │  • Rollout through layers   │    │
│ │  • CLS token attribution    │    │
│ │  • 224×224 heatmap          │    │
│ └─────────────────────────────┘    │
│                                      │
│ B. Evidence Citations                │
│ ┌─────────────────────────────┐    │
│ │ Retrieved WHO Guidelines:   │    │
│ │                             │    │
│ │ Shown in report:            │    │
│ │  [WHO2022_CML_CP_007]       │    │
│ │  "Chronic myeloid leukemia  │    │
│ │   chronic phase..."         │    │
│ │                             │    │
│ │  [WHO2022_HAEMATOGONES_009] │    │
│ │  "Benign B-cell precursors  │    │
│ │   in pediatric marrow..."   │    │
│ │                             │    │
│ │ User can:                   │    │
│ │  • Click chunk ID           │    │
│ │  • See full passage         │    │
│ │  • Verify citation accuracy │    │
│ └─────────────────────────────┘    │
│                                      │
│ C. Cell-Level Explanations           │
│ ┌─────────────────────────────┐    │
│ │ Per-Cell Reasoning:         │    │
│ │                             │    │
│ │ Cell #1:                    │    │
│ │  [thumbnail]                │    │
│ │  Classified as: Blast       │    │
│ │  Confidence: 0.92           │    │
│ │                             │    │
│ │  Key features:              │    │
│ │   • High N/C ratio          │    │
│ │   • Fine chromatin          │    │
│ │   • Prominent nucleoli      │    │
│ │   • Scant cytoplasm         │    │
│ │                             │    │
│ │  Attention focused on:      │    │
│ │   ██████ Nucleus (80%)      │    │
│ │   ██ Cytoplasm (15%)        │    │
│ │   █ Boundary (5%)           │    │
│ └─────────────────────────────┘    │
│                                      │
│ D. Patient-Level Reasoning           │
│ ┌─────────────────────────────┐    │
│ │ Diagnostic Logic:           │    │
│ │                             │    │
│ │ 1. Blast fraction: 5.2%     │    │
│ │    → Below WHO threshold    │    │
│ │    → Not acute leukemia     │    │
│ │                             │    │
│ │ 2. Megakaryocytic excess    │    │
│ │    → Consider reactive      │    │
│ │    → Rule out CML           │    │
│ │    → Need BCR::ABL1         │    │
│ │                             │    │
│ │ 3. Age: 6 years             │    │
│ │    → Pediatric context      │    │
│ │    → Consider haematogones  │    │
│ │                             │    │
│ │ 4. Cell count: 154          │    │
│ │    → MODERATE evidence      │    │
│ │    → Sufficient for initial │    │
│ │       assessment            │    │
│ └─────────────────────────────┘    │
│                                      │
│ ❌ NOT IMPLEMENTED:                  │
│  • Grad-CAM (use Attention instead) │
│                                      │
└─────────────────────────────────────┘
```

---

## Component 9: Output & Interaction

### Box Properties:
- **Color**: Light Blue (#DBEAFE)
- **Border**: Blue (#2563EB)
- **Size**: 250×400px
- **Position**: Far right

### Content:

```
┌─────────────────────────────────────┐
│ 9. OUTPUT & INTERACTION              │
├─────────────────────────────────────┤
│                                      │
│ ✅ IMPLEMENTED:                      │
│                                      │
│ A. Interactive Report                │
│ ┌─────────────────────────────┐    │
│ │ Streamlit Web UI            │    │
│ │                             │    │
│ │ Sections:                   │    │
│ │ 1. Detection Summary        │    │
│ │    • Cells detected: 154    │    │
│ │    • Processing time: 8s    │    │
│ │    • Status: Success        │    │
│ │                             │    │
│ │ 2. Cell Distribution Chart  │    │
│ │    [Bar chart]              │    │
│ │    • By morphology class    │    │
│ │    • Percentages shown      │    │
│ │    • Color-coded            │    │
│ │                             │    │
│ │ 3. AI Clinical Assessment   │    │
│ │    • Evidence provenance    │    │
│ │    • Morphological findings │    │
│ │    • Clinical context       │    │
│ │    • Limitations clearly    │    │
│ │      stated                 │    │
│ │    • Recommended workup     │    │
│ │                             │    │
│ │ 4. Retrieved Evidence       │    │
│ │    • WHO guideline passages │    │
│ │    • Chunk IDs clickable    │    │
│ │    • Source citations       │    │
│ │                             │    │
│ │ 5. Detected Cells Gallery   │    │
│ │    [Thumbnail grid]         │    │
│ │    • All detected cells     │    │
│ │    • Click for detail       │    │
│ │    • Color-coded by class   │    │
│ │                             │    │
│ │ 6. Medical Disclaimers      │    │
│ │    • Not for diagnosis      │    │
│ │    • Research prototype     │    │
│ │    • Expert review required │    │
│ └─────────────────────────────┘    │
│                                      │
│ B. Technical Details (Expandable)    │
│ ┌─────────────────────────────┐    │
│ │ Hidden by default:          │    │
│ │ • Model confidence scores   │    │
│ │ • Detection parameters      │    │
│ │ • Processing timestamps     │    │
│ │ • System information        │    │
│ └─────────────────────────────┘    │
│                                      │
│ C. Highlighted Image Regions         │
│ ┌─────────────────────────────┐    │
│ │ Original image with:        │    │
│ │ • Bounding boxes (detected) │    │
│ │ • Color by classification   │    │
│ │ • Cell numbers              │    │
│ │ • Zoom capability           │    │
│ └─────────────────────────────┘    │
│                                      │
│ ❌ NOT IMPLEMENTED:                  │
│                                      │
│ D. Chat Interface                    │
│ ┌─────────────────────────────┐    │
│ │ ❌ Follow-up questions       │    │
│ │ ❌ Request clarifications    │    │
│ │ ❌ Ask about specific cells  │    │
│ │                             │    │
│ │ Future feature              │    │
│ └─────────────────────────────┘    │
│                                      │
│ E. Export Report                     │
│ ┌─────────────────────────────┐    │
│ │ ❌ PDF export                │    │
│ │ ❌ JSON export               │    │
│ │                             │    │
│ │ Future feature              │    │
│ └─────────────────────────────┘    │
│                                      │
│ Current Access:                      │
│  • Web browser (localhost:8501)     │
│  • Mobile responsive                │
│  • No authentication                │
│                                      │
└─────────────────────────────────────┘
```

---

## Data Flow Arrows

### Arrow Specifications:

1. **Input → Preprocessing**: Solid blue arrow, thick (3px)
2. **Preprocessing → Vision**: Solid blue arrow, thick (3px)
3. **Vision → Aggregation**: Solid blue arrow, thick (3px)
4. **Aggregation → Agents**: Solid blue arrow, thick (3px)
5. **Agents → Knowledge Base**: Dashed arrow (2px), blue (query)
6. **Knowledge Base → Agents**: Dashed arrow (2px), green (retrieved docs)
7. **Agents → RAG**: Solid arrow (3px), purple
8. **RAG → Agents**: Solid arrow (3px), purple
9. **Agents → Explainability**: Solid arrow (2px), gray
10. **Agents → Output**: Solid arrow (3px), blue
11. **Explainability → Output**: Solid arrow (2px), gray
12. **Output → Agents**: Dashed arrow (1px), red (feedback loop, if rejected)

### Feedback Loop:
- **Critic → Synthesizer**: Red dashed arrow with label "If rejected & revision < 2"
- Shows iterative refinement

---

## Color Scheme

### Component Colors:

| Component | Background | Border | Use |
|-----------|-----------|--------|-----|
| **Input** | `#E8E4F3` | `#7C3AED` | Light purple |
| **Preprocessing** | `#E0F2FE` | `#0284C7` | Light blue |
| **Vision** | `#FEF9C3` | `#CA8A04` | Light yellow |
| **Aggregation** | `#D1FAE5` | `#059669` | Light green |
| **Agents** | `#FEE2E2` | `#DC2626` | Light red/pink |
| **Knowledge Base** | `#CFFAFE` | `#0891B2` | Cyan |
| **RAG** | `#F3E8FF` | `#9333EA` | Purple |
| **Explainability** | `#EDE9FE` | `#7C3AED` | Lavender |
| **Output** | `#DBEAFE` | `#2563EB` | Blue |

### Icon Colors:
- Success: Green `#10B981`
- Warning: Yellow `#F59E0B`
- Error: Red `#EF4444`
- Info: Blue `#3B82F6`
- Optional: Gray `#6B7280`
- Not implemented: Red with strikethrough

---

## Visual Examples

### Cell Images:
```
Include actual cell microscopy images:
- Blast cell
- Lymphoid cell
- Granulocyte (mature)
- Erythroid precursor
- Megakaryocyte

Size: 64×64px thumbnails
```

### Attention Maps:
```
Show side-by-side:
[Original Cell]  [Attention Heatmap]
  Gray scale       Color (blue→red)
  64×64px          64×64px
```

### Charts:
```
Bar Chart for Cell Distribution:
- Horizontal bars
- Different colors per class
- Percentages on right
- Count labels inside bars
```

---

## Annotations & Labels

### Add These Labels:

1. **On Preprocessing → Vision arrow**:
   "~154 cells, 224×224 crops each"

2. **On Vision → Aggregation arrow**:
   "154 classified cells with confidence scores"

3. **On Aggregation → Agents arrow**:
   "PatientAnalysis: blast 5.2%, 154 cells, MODERATE"

4. **On Knowledge Base → RAG arrow**:
   "12 WHO guideline chunks available"

5. **On RAG → Agents arrow**:
   "Top-4 retrieved passages + LLM synthesis"

6. **On Agents → Output arrow**:
   "Final clinical assessment report (validated)"

7. **On Feedback Loop arrow**:
   "If evidence insufficient / citations invalid (max 2 revisions)"

---

## Legend/Key (Bottom Right)

```
┌──────────────────────────────────┐
│ LEGEND                            │
├──────────────────────────────────┤
│                                   │
│ ✅ Implemented & Working          │
│ ⚠️  Optional / Partial            │
│ ❌ Not Implemented                │
│                                   │
│ ──→  Data flow                   │
│ ──→  Control flow                │
│ ─ ─→ Query/retrieval             │
│ ─ ─→ Feedback/iteration          │
│                                   │
│ [img] Includes visual examples    │
│                                   │
└──────────────────────────────────┘
```

---

## Summary Checklist

Use this to verify your diagram:

### ✅ Inputs:
- [ ] Image only (required)
- [ ] Age/Sex (optional, marked as such)
- [ ] No clinical data (marked as not available)
- [ ] No research query (marked as not implemented)

### ✅ Preprocessing:
- [ ] No DICOM processing
- [ ] Cell detection shown first
- [ ] 224×224 crop extraction
- [ ] No augmentation during inference

### ✅ Vision:
- [ ] Model name: ViT-Small/16 + LoRA
- [ ] 8-class taxonomy shown
- [ ] Per-cell predictions

### ✅ Aggregation:
- [ ] Cell counts by category
- [ ] Blast fraction calculation
- [ ] Disease prediction
- [ ] Evidence sufficiency levels

### ✅ Agents:
- [ ] Only 3 agents: Retriever, Synthesizer, Critic
- [ ] No Vision/Clinical/Explanation agents
- [ ] Feedback loop shown (Critic → Synthesizer)
- [ ] LangGraph orchestration mentioned

### ✅ Knowledge Base:
- [ ] WHO 5th Edition + ICC 2022
- [ ] 12 clinical chunks
- [ ] Fallback system mentioned
- [ ] No internal case database

### ✅ RAG:
- [ ] BGE-small embeddings (not Gemini)
- [ ] FAISS (not ChromaDB)
- [ ] BM25 + RRF shown
- [ ] Groq openai/gpt-oss-120b (120B) specified

### ✅ Explainability:
- [ ] Attention Rollout (not Grad-CAM)
- [ ] WHO citations shown
- [ ] Cell-level explanations
- [ ] Patient-level reasoning

### ✅ Output:
- [ ] Implemented features marked
- [ ] Chat interface marked as not implemented
- [ ] PDF/JSON export marked as not implemented
- [ ] Streamlit UI shown

---

## Final Notes

### This specification is:
- ✅ Based on actual implementation
- ✅ Verified against codebase
- ✅ Corrected from original diagram
- ✅ Complete and detailed
- ✅ Ready for visual diagram creation

### Tools You Can Use:
- **Draw.io / Diagrams.net** (recommended)
- **Lucidchart**
- **Figma**
- **PowerPoint/Google Slides**
- **Mermaid** (for text-based diagrams)

### Recommended Diagram Size:
- **Width**: 1400px (landscape)
- **Height**: 900px
- **Format**: PNG or SVG
- **Resolution**: 300 DPI for print

---

**Use this specification to create a 100% accurate architecture diagram!** 🎨
