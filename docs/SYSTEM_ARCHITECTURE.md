# 🏗️ System Architecture Documentation

**Project**: Multimodal AI Bone Marrow Cytology Analysis  
**Version**: 1.0  
**Last Updated**: October 7, 2026  

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Module Dependencies](#module-dependencies)
7. [API Interfaces](#api-interfaces)
8. [Configuration Management](#configuration-management)

---

## 🎯 System Overview

### Purpose
AI-powered clinical decision support system for bone marrow cytomorphology analysis using vision models, retrieval-augmented generation (RAG), and multi-agent orchestration.

### Key Capabilities
- Automated cell detection and morphology classification
- Evidence-based clinical assessment generation
- WHO guideline retrieval and citation
- Multi-agent quality assurance
- Medical-grade user interface

### Architecture Style
- **Pattern**: Modular pipeline with agent orchestration
- **Processing**: Sequential with parallel retrieval
- **Deployment**: Single-machine with cloud LLM inference
- **Scalability**: Horizontal (batch processing capable)

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
│                     (Streamlit Web Application)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                            │
│              (Pipeline Coordinator & Agent Graph)                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         New Patient Engine (Entry Point)                  │  │
│  │  • Image validation                                       │  │
│  │  • Workflow coordination                                  │  │
│  │  • Result aggregation                                     │  │
│  └──────────────────────┬───────────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   VISION     │  │     RAG      │  │    AGENT     │
│  PIPELINE    │  │   SYSTEM     │  │  FRAMEWORK   │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 🔍 Component Details

### 1. VISION PIPELINE LAYER

```
┌─────────────────────────────────────────────────────────┐
│               VISION PROCESSING PIPELINE                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  1. CELL DETECTION                             │    │
│  │     (pipeline/cell_detector.py)                │    │
│  │                                                 │    │
│  │  Input: RGB microscopy image (any size)        │    │
│  │  Algorithm: Heuristic morphological detector   │    │
│  │                                                 │    │
│  │  Steps:                                         │    │
│  │    a) Optical density conversion (RGB → OD)    │    │
│  │    b) Nuclear stain separation (H&E/Giemsa)   │    │
│  │    c) Distance transform peak detection        │    │
│  │    d) Watershed segmentation                   │    │
│  │    e) Morphological filtering                  │    │
│  │    f) Non-maximum suppression (NMS)            │    │
│  │                                                 │    │
│  │  Output: List of cell bounding boxes + crops   │    │
│  │  Performance: ~154 cells/image                 │    │
│  └─────────────────┬───────────────────────────────┘    │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────────────┐    │
│  │  2. CELL CLASSIFICATION                        │    │
│  │     (vision/model.py + trained weights)        │    │
│  │                                                 │    │
│  │  Model: ViT-Small/16 + LoRA adapters           │    │
│  │  Input: 224×224 RGB cell crop                  │    │
│  │  Classes: 8 morphology categories              │    │
│  │    • Blast                                      │    │
│  │    • Lymphoid Series                           │    │
│  │    • Granulocyte (Mature/Band)                 │    │
│  │    • Granulocyte (Immature)                    │    │
│  │    • Erythroid Precursor                       │    │
│  │    • Monocyte                                   │    │
│  │    • Megakaryocyte / Platelet                  │    │
│  │    • Other/Artifact                            │    │
│  │                                                 │    │
│  │  Output: Class probabilities per cell          │    │
│  │  Confidence threshold: 0.60                    │    │
│  └─────────────────┬───────────────────────────────┘    │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────────────┐    │
│  │  3. PATIENT-LEVEL AGGREGATION                  │    │
│  │     (pipeline/new_patient_engine.py)           │    │
│  │                                                 │    │
│  │  Aggregation:                                   │    │
│  │    • Cell counts by category                   │    │
│  │    • Blast fraction calculation                │    │
│  │    • Class probability averaging               │    │
│  │    • Low-evidence flagging (<50 cells)         │    │
│  │                                                 │    │
│  │  Output: PatientAnalysis object                │    │
│  │    - total_cells: int                          │    │
│  │    - annotated_blast_fraction: float           │    │
│  │    - predicted_class: str                      │    │
│  │    - class_probabilities: dict                 │    │
│  │    - cell_counts_by_category: dict             │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Key Files:**
- `pipeline/cell_detector.py` - Cell detection logic
- `vision/model.py` - ViT-Small model definition
- `vision/train.py` - Training pipeline
- `vision/dataset.py` - Data loading
- `pipeline/data_model.py` - Data structures

**Configuration:**
- Cell detection parameters: Lines 30-40 in `cell_detector.py`
- Model: `configs/default.yaml` → `vision` section
- Training: `configs/default.yaml` → `training` section

---

### 2. RAG RETRIEVAL LAYER

```
┌─────────────────────────────────────────────────────────┐
│                  RAG RETRIEVAL SYSTEM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  KNOWLEDGE BASE                                 │    │
│  │  (data/knowledge_base/)                         │    │
│  │                                                 │    │
│  │  Content:                                       │    │
│  │    • WHO 5th Edition guidelines                │    │
│  │    • ICC 2022 classification                   │    │
│  │    • 12 clinical chunks                        │    │
│  │                                                 │    │
│  │  Metadata per chunk:                            │    │
│  │    - chunk_id: str                             │    │
│  │    - title: str                                │    │
│  │    - text: str (512 tokens)                    │    │
│  │    - source: str                               │    │
│  │    - disease_category: str                     │    │
│  │    - specific_entity: str                      │    │
│  │    - blast_threshold_rule: dict (optional)     │    │
│  └─────────────────────────────────────────────────┘    │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────────────┐    │
│  │  INDEXING LAYER                                 │    │
│  │  (retrieval/index.py)                           │    │
│  │                                                 │    │
│  │  Dense Index (FAISS):                           │    │
│  │    • Embeddings: BAAI/bge-small-en-v1.5        │    │
│  │    • Dimension: 384                             │    │
│  │    • Index type: FlatL2 (cosine similarity)    │    │
│  │    • File: dense_index.faiss                   │    │
│  │                                                 │    │
│  │  Sparse Index (BM25):                           │    │
│  │    • Algorithm: BM25Okapi                       │    │
│  │    • Tokenization: word-level                  │    │
│  │    • File: sparse_bm25.pkl                     │    │
│  └─────────────────────────────────────────────────┘    │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────────────┐    │
│  │  HYBRID RETRIEVAL ENGINE                        │    │
│  │  (retrieval/retrieve.py)                        │    │
│  │                                                 │    │
│  │  Query Processing:                              │    │
│  │    1. Build clinical query from morphology     │    │
│  │       "Bone marrow cytomorphology for {dx}     │    │
│  │        | blast {%} | cells {types} | age"      │    │
│  │                                                 │    │
│  │    2. Dense search (FAISS)                      │    │
│  │       - Embed query with BGE-small             │    │
│  │       - Cosine similarity search               │    │
│  │       - Top-K=10 results                       │    │
│  │                                                 │    │
│  │    3. Sparse search (BM25)                      │    │
│  │       - Tokenize query                         │    │
│  │       - BM25 scoring                           │    │
│  │       - Top-K=10 results                       │    │
│  │                                                 │    │
│  │    4. Reciprocal Rank Fusion (RRF)             │    │
│  │       score = dense_w/(k+rank) +               │    │
│  │               sparse_w/(k+rank)                 │    │
│  │       k=60, dense_w=0.6, sparse_w=0.4          │    │
│  │                                                 │    │
│  │    5. Rerank by combined score                  │    │
│  │       - Return top-4 passages                  │    │
│  │                                                 │    │
│  │  Fallback:                                      │    │
│  │    • If retrieval fails → fallback_guidelines  │    │
│  │    • Ensures never empty results               │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Key Files:**
- `retrieval/retrieve.py` - Hybrid retrieval engine
- `retrieval/index.py` - FAISS/BM25 indexing
- `agents/fallback_guidelines.py` - Fallback content
- `data/knowledge_base/` - WHO guideline chunks

**Configuration:**
- `configs/default.yaml` → `retrieval` section
- Embedder model, reranker, top-K settings

---

### 3. AGENT ORCHESTRATION LAYER

```
┌───────────────────────────────────────────────────────────────┐
│                    AGENT FRAMEWORK                             │
│                  (LangGraph Orchestration)                     │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                    AGENT GRAPH                       │     │
│  │              (agents/graph.py)                       │     │
│  │                                                       │     │
│  │    START                                              │     │
│  │      │                                                │     │
│  │      ▼                                                │     │
│  │  ┌─────────────┐                                     │     │
│  │  │  RETRIEVER  │  Searches WHO guidelines            │     │
│  │  │    AGENT    │  Based on morphology profile        │     │
│  │  └──────┬──────┘                                     │     │
│  │         │                                             │     │
│  │         ▼                                             │     │
│  │  ┌─────────────┐                                     │     │
│  │  │ SYNTHESIZER │  Generates clinical assessment     │     │
│  │  │    AGENT    │  (LLM or deterministic)             │     │
│  │  └──────┬──────┘                                     │     │
│  │         │                                             │     │
│  │         ▼                                             │     │
│  │  ┌─────────────┐                                     │     │
│  │  │   CRITIC    │  Validates evidence grounding       │     │
│  │  │    AGENT    │  Checks citations & hallucinations  │     │
│  │  └──────┬──────┘                                     │     │
│  │         │                                             │     │
│  │    ┌────┴────┐                                       │     │
│  │    │         │                                       │     │
│  │    ▼         ▼                                       │     │
│  │ APPROVED  REJECTED                                   │     │
│  │    │         │                                       │     │
│  │    │    (revision_count < 2)                         │     │
│  │    │         │                                       │     │
│  │    │         └──────▶ SYNTHESIZER (retry)            │     │
│  │    │                                                  │     │
│  │    ▼                                                  │     │
│  │   END                                                 │     │
│  │                                                       │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  AGENT DETAILS                                       │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │                                                       │     │
│  │  1. RETRIEVER AGENT (agents/retriever.py)           │     │
│  │     Input:                                           │     │
│  │       - patient_id                                   │     │
│  │       - predicted_class                              │     │
│  │       - blast_fraction                               │     │
│  │       - cell_counts                                  │     │
│  │       - age (optional)                               │     │
│  │     Process:                                          │     │
│  │       - Build clinical query                         │     │
│  │       - Call HybridRetriever                         │     │
│  │       - Format retrieved evidence                    │     │
│  │     Output:                                           │     │
│  │       - search_query: str                            │     │
│  │       - retrieved_evidence: list[dict]               │     │
│  │       - formatted_guidelines_context: str            │     │
│  │                                                       │     │
│  │  2. SYNTHESIZER AGENT (agents/synthesizer.py)       │     │
│  │     Modes:                                            │     │
│  │       a) LLM Mode (USE_LLM=true)                     │     │
│  │          - Uses Groq openai/gpt-oss-120b            │     │
│  │          - Prompt engineering with system prompt     │     │
│  │          - Evidence-based generation                 │     │
│  │          - Automatic fallback on error               │     │
│  │                                                       │     │
│  │       b) Deterministic Mode (USE_LLM=false)          │     │
│  │          - Rule-based templates                      │     │
│  │          - If/else logic by diagnosis                │     │
│  │          - Faster but less flexible                  │     │
│  │                                                       │     │
│  │     Input:                                            │     │
│  │       - All patient data                             │     │
│  │       - Retrieved evidence                           │     │
│  │       - Formatted guidelines                         │     │
│  │     Process:                                          │     │
│  │       - Generate clinical assessment                 │     │
│  │       - Cite specific guideline chunks               │     │
│  │       - Mark limitations clearly                     │     │
│  │     Output:                                           │     │
│  │       - draft_differential: str (markdown)           │     │
│  │       - morphological_justification: str             │     │
│  │                                                       │     │
│  │  3. CRITIC AGENT (agents/critic.py)                 │     │
│  │     Validation Axes:                                 │     │
│  │       A. Evidence Grounding (0.0-1.0)                │     │
│  │          - Are claims supported by retrieved docs?   │     │
│  │          - Citation correctness check                │     │
│  │          - Hallucination detection                   │     │
│  │                                                       │     │
│  │       B. Evidence Sufficiency (enum)                 │     │
│  │          - INSUFFICIENT: <5 cells                    │     │
│  │          - LOW: 5-49 cells                           │     │
│  │          - MODERATE: 50-149 cells                    │     │
│  │          - HIGH: ≥150 cells                          │     │
│  │                                                       │     │
│  │     Decision Logic:                                   │     │
│  │       - APPROVED: grounding≥0.75 AND sufficient      │     │
│  │       - INSUFFICIENT: <5 cells (override)            │     │
│  │       - REVIEW_REQUIRED: grounding<0.75              │     │
│  │       - ABSTAINED: max revisions exceeded            │     │
│  │                                                       │     │
│  │     Output:                                           │     │
│  │       - is_approved: bool                            │     │
│  │       - critic_grounding_score: float                │     │
│  │       - evidence_sufficiency: str                    │     │
│  │       - final_report: str                            │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `agents/graph.py` - LangGraph workflow
- `agents/retriever.py` - Retriever agent
- `agents/synthesizer.py` - Synthesizer agent (dual mode)
- `agents/critic.py` - Critic agent
- `agents/state.py` - Shared state schema
- `agents/tools.py` - Utility functions

**State Schema:**
```python
AgentState = TypedDict({
    "patient_id": str,
    "age_months": float | None,
    "age_years": float | None,
    "sex": str | None,
    "annotated_blast_fraction": float,
    "predicted_class": str,
    "class_probabilities": dict,
    "cell_counts_by_category": dict,
    "total_cells": int,
    "low_evidence_flag": bool,
    "search_query": str,
    "retrieved_evidence": list,
    "formatted_guidelines_context": str,
    "draft_differential": str,
    "revision_count": int,
    "is_approved": bool,
    "critic_grounding_score": float,
    "final_report": str,
    # ... additional fields
})
```

---

### 4. LLM INTEGRATION LAYER

```
┌───────────────────────────────────────────────────────────┐
│                  LLM PROVIDER ABSTRACTION                  │
│              (agents/llm_provider.py)                      │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │           LLMProvider Class                       │    │
│  │                                                    │    │
│  │  Supported Providers:                             │    │
│  │    • Groq (CURRENT)                               │    │
│  │    • OpenAI                                       │    │
│  │    • Local HuggingFace                            │    │
│  │                                                    │    │
│  │  Configuration:                                    │    │
│  │    - Provider: configs/default.yaml → llm.provider│    │
│  │    - Model: llm.model                             │    │
│  │    - API Key: .env → GROQ_API_KEY                │    │
│  │                                                    │    │
│  │  Methods:                                          │    │
│  │    __init__(config)                               │    │
│  │      • Load config                                │    │
│  │      • Initialize client                          │    │
│  │                                                    │    │
│  │    generate(messages, **kwargs) → str             │    │
│  │      • Takes OpenAI-format messages               │    │
│  │      • Returns generated text                     │    │
│  │      • Handles errors gracefully                  │    │
│  │                                                    │    │
│  │  Singleton Pattern:                                │    │
│  │    get_llm_provider() → cached instance           │    │
│  └──────────────────────────────────────────────────┘    │
│                      │                                    │
│                      ▼                                    │
│  ┌──────────────────────────────────────────────────┐    │
│  │         GROQ CLOUD INFERENCE                      │    │
│  │                                                    │    │
│  │  Model: openai/gpt-oss-120b                       │    │
│  │  Parameters: 120 billion                           │    │
│  │  Context: 8,192 tokens                             │    │
│  │  Latency: 1-3 seconds                              │    │
│  │                                                    │    │
│  │  API Endpoint:                                     │    │
│  │    https://api.groq.com/openai/v1/chat/completions│    │
│  │                                                    │    │
│  │  Rate Limits (Free Tier):                          │    │
│  │    • 14,400 requests/day                          │    │
│  │    • 30 requests/minute                           │    │
│  │                                                    │    │
│  │  Request Format:                                   │    │
│  │    {                                               │    │
│  │      "model": "openai/gpt-oss-120b",              │    │
│  │      "messages": [                                 │    │
│  │        {"role": "system", "content": "..."},      │    │
│  │        {"role": "user", "content": "..."}         │    │
│  │      ],                                            │    │
│  │      "max_tokens": 2048,                           │    │
│  │      "temperature": 0.1                            │    │
│  │    }                                               │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `agents/llm_provider.py` - LLM abstraction
- `configs/default.yaml` - LLM configuration
- `.env` - API keys

**Configuration:**
```yaml
llm:
  provider: groq
  model: openai/gpt-oss-120b
  max_tokens: 2048
  temperature: 0.1
```

---

### 5. USER INTERFACE LAYER

```
┌───────────────────────────────────────────────────────────┐
│               STREAMLIT WEB APPLICATION                    │
│                 (app/streamlit_app.py)                     │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  Layout Structure:                                         │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │  HEADER                                           │    │
│  │  • Title: "AI-Powered Diagnostic Support"        │    │
│  │  • Subtitle: For bone marrow cytomorphology      │    │
│  │  • Medical disclaimer (collapsible)              │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │  SIDEBAR                                          │    │
│  │  • Patient demographics input                     │    │
│  │    - Age (optional)                               │    │
│  │    - Sex (optional)                               │    │
│  │  • System information (collapsible)              │    │
│  │    - Python version                               │    │
│  │    - PyTorch version                              │    │
│  │    - Device info                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │  MAIN CONTENT AREA                                │    │
│  │                                                    │    │
│  │  1. Image Upload                                  │    │
│  │     • Drag & drop or file picker                  │    │
│  │     • Supported: PNG, JPG, JPEG, TIFF             │    │
│  │     • Preview uploaded image                      │    │
│  │                                                    │    │
│  │  2. Processing Button                             │    │
│  │     • "Analyze Image" (gradient blue button)      │    │
│  │     • Triggers full pipeline                      │    │
│  │                                                    │    │
│  │  3. Results Display (after processing)            │    │
│  │                                                    │    │
│  │     a) Detection Summary Card                     │    │
│  │        • Total cells detected                     │    │
│  │        • Processing time                          │    │
│  │        • Status indicator                         │    │
│  │                                                    │    │
│  │     b) Morphology Distribution                    │    │
│  │        • Bar chart of cell categories             │    │
│  │        • Percentage breakdown                     │    │
│  │        • Blast fraction highlight                 │    │
│  │                                                    │    │
│  │     c) AI Clinical Assessment                     │    │
│  │        • Evidence provenance table                │    │
│  │        • Morphological findings                   │    │
│  │        • Clinical context (NOT diagnosis)         │    │
│  │        • Limitations clearly stated               │    │
│  │        • Recommended workup                       │    │
│  │        • WHO guideline citations                  │    │
│  │                                                    │    │
│  │     d) Retrieved Medical Evidence                 │    │
│  │        • WHO guideline passages                   │    │
│  │        • Chunk IDs for traceability               │    │
│  │        • Source information                       │    │
│  │                                                    │    │
│  │     e) Technical Details (expandable)             │    │
│  │        • Model confidence scores                  │    │
│  │        • Detection parameters                     │    │
│  │        • Processing timestamps                    │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │  FOOTER                                           │    │
│  │  • Disclaimers                                    │    │
│  │  • Research prototype notice                      │    │
│  │  • Not for clinical decision-making               │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Styling:**
- Color scheme: Deep blue (#1A237E) primary
- Typography: Inter font family, 32px headers
- Cards: Shadow effects, rounded corners
- Buttons: Gradient blue with hover effects
- Hidden: Streamlit branding, menu, footer

**Key Files:**
- `app/streamlit_app.py` - Main UI application
- Custom CSS embedded for medical-grade styling

---

## 🔄 Data Flow Diagram

### Complete Processing Flow:

```
┌─────────────────┐
│  USER UPLOADS   │
│     IMAGE       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  IMAGE VALIDATION                                │
│  • Check format (PNG/JPG/TIFF)                   │
│  • Check size (max 50MB)                         │
│  • Convert to PIL Image                          │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  CELL DETECTION                                  │
│  Input: PIL Image (any size)                    │
│  Process:                                         │
│    1. Convert RGB → Optical Density              │
│    2. Nuclear stain separation                   │
│    3. Distance transform                         │
│    4. Peak detection                             │
│    5. Watershed segmentation                     │
│    6. Morphological filtering                    │
│    7. NMS (IoU threshold 0.40)                   │
│  Output: List[DetectedCell]                      │
│    - bbox: (x, y, w, h)                          │
│    - crop: 224×224 RGB                           │
│    - confidence: float                           │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  CELL CLASSIFICATION (Batch)                    │
│  Input: List of 224×224 crops                   │
│  Model: ViT-Small/16 + LoRA                      │
│  Batch size: 32                                  │
│  Process:                                         │
│    1. Normalize images                           │
│    2. Forward pass through ViT                   │
│    3. Get class probabilities                    │
│    4. Apply confidence threshold (0.60)          │
│  Output: List[CellInferenceResult]               │
│    - predicted_class: str                        │
│    - class_probabilities: dict[str, float]       │
│    - confidence: float                           │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  PATIENT AGGREGATION                             │
│  Input: List[CellInferenceResult]                │
│  Process:                                         │
│    1. Count cells by category                    │
│    2. Calculate blast fraction                   │
│    3. Average class probabilities                │
│    4. Determine primary diagnosis                │
│    5. Check low-evidence flag (<50 cells)        │
│  Output: PatientAnalysis                         │
│    - total_cells: 154                            │
│    - annotated_blast_fraction: 0.052             │
│    - predicted_class: "Megakaryocyte/Platelet"   │
│    - cell_counts_by_category: {...}              │
│    - low_evidence_flag: false                    │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  AGENT ORCHESTRATION (START)                    │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  RETRIEVER AGENT                                 │
│  Input: PatientAnalysis                          │
│  Process:                                         │
│    1. Build query:                               │
│       "Bone marrow cytomorphology for            │
│        Megakaryocyte | blast 5.2% | cells        │
│        Megakaryocyte, Lymphoid, Granulocyte"     │
│    2. Hybrid retrieval:                          │
│       - Dense search (FAISS + BGE)               │
│       - Sparse search (BM25)                     │
│       - RRF fusion                               │
│       - Top-4 results                            │
│    3. Fallback if empty                          │
│  Output:                                          │
│    - retrieved_evidence: 4 WHO passages          │
│    - search_query: str                           │
│    - formatted_guidelines_context: str           │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  SYNTHESIZER AGENT (LLM MODE)                   │
│  Input:                                           │
│    - PatientAnalysis data                        │
│    - Retrieved evidence (4 passages)             │
│    - Formatted guidelines context                │
│  Process:                                         │
│    1. Build system prompt (clinical rules)       │
│    2. Build user prompt with:                    │
│       - Patient demographics                     │
│       - Cell detection results                   │
│       - Blast evaluation                         │
│       - Retrieved guidelines                     │
│    3. Call Groq API:                             │
│       POST https://api.groq.com/...              │
│       Model: openai/gpt-oss-120b                 │
│       Max tokens: 2048                           │
│       Temperature: 0.1                           │
│    4. Parse response                             │
│    5. Fallback to deterministic if error         │
│  Output:                                          │
│    - draft_differential: str (markdown, ~800w)   │
│    - morphological_justification: str            │
│  Latency: 1-3 seconds                            │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  CRITIC AGENT                                    │
│  Input:                                           │
│    - draft_differential: str                     │
│    - retrieved_evidence: list                    │
│    - total_cells: 154                            │
│    - low_evidence_flag: false                    │
│  Process:                                         │
│    1. Evidence Sufficiency:                      │
│       154 cells → MODERATE                       │
│    2. Grounding Check:                           │
│       - Extract cited chunks                     │
│       - Verify against retrieved evidence        │
│       - Check for fabricated data                │
│       - Calculate penalty score                  │
│       grounding_score = 1.0 - penalties          │
│    3. Decision:                                   │
│       if grounding ≥ 0.75 AND sufficient:        │
│         APPROVED                                 │
│       elif insufficient:                          │
│         ABSTAIN                                  │
│       else:                                       │
│         REVIEW_REQUIRED → retry synthesizer      │
│  Output:                                          │
│    - is_approved: true                           │
│    - critic_grounding_score: 1.0                 │
│    - evidence_sufficiency: "MODERATE"            │
│    - final_report: str (with disclaimers)        │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  AGENT ORCHESTRATION (END)                      │
│  Final state returned to engine                  │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  RESULT PACKAGING                                │
│  Process:                                         │
│    1. Extract final_report from agent state      │
│    2. Add cell distribution chart                │
│    3. Add retrieved evidence section             │
│    4. Format for UI display                      │
│  Output: PatientAnalysis (complete)              │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  UI RENDERING                                    │
│  Display:                                         │
│    • Detection summary card                      │
│    • Morphology distribution chart               │
│    • AI clinical assessment (markdown)           │
│    • Retrieved medical evidence                  │
│    • Technical details (expandable)              │
└──────────────────────────────────────────────────┘
```

**Total Processing Time**: 5-10 seconds
- Cell detection: 1-2s
- Classification: 1-2s
- RAG retrieval: <1s
- LLM synthesis: 1-3s
- Critic validation: <1s
- UI rendering: <1s

---

## 💻 Technology Stack

### Core Technologies:

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Backend** | Python | 3.10+ | Core language |
| **Deep Learning** | PyTorch | 2.2.0+ | Neural networks |
| | timm | 0.9.16+ | Vision models |
| | PEFT | 0.10.0+ | LoRA adapters |
| **RAG** | FAISS | 1.8.0+ | Vector search |
| | BM25 | 0.2.2+ | Sparse retrieval |
| | Sentence Transformers | 2.6.0+ | Embeddings |
| **LLM** | Groq | 0.11.0+ | Cloud inference |
| | LangChain | 0.1.0+ | LLM framework |
| | LangGraph | 0.0.28+ | Agent orchestration |
| **UI** | Streamlit | 1.35.0+ | Web interface |
| | Plotly | 5.22.0+ | Visualizations |
| **Data** | NumPy | 1.26.0+ | Numerical computing |
| | Pandas | 2.2.0+ | Data manipulation |
| | Pillow | 10.3.0+ | Image processing |
| **Config** | python-dotenv | 1.0.0+ | Environment vars |
| | OmegaConf | 2.3.0+ | YAML configs |

### System Requirements:

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 2 GB
- GPU: Not required (optional for training)
- Internet: Required (for Groq API)

**Recommended**:
- CPU: 8 cores
- RAM: 16 GB
- Storage: 5 GB
- GPU: NVIDIA GPU with 8GB VRAM (for training only)
- Internet: Required

---

## 🔗 Module Dependencies

### Dependency Graph:

```
app/streamlit_app.py
    ↓
pipeline/new_patient_engine.py
    ├── pipeline/cell_detector.py
    │       └── PIL, numpy, cv2
    ├── vision/model.py
    │       ├── timm
    │       ├── peft
    │       └── torch
    ├── vision/dataset.py
    │       └── torch.utils.data
    └── agents/graph.py
            ├── agents/retriever.py
            │       ├── retrieval/retrieve.py
            │       │       ├── faiss
            │       │       ├── sentence_transformers
            │       │       └── rank_bm25
            │       └── agents/fallback_guidelines.py
            ├── agents/synthesizer.py
            │       ├── agents/llm_provider.py
            │       │       ├── groq
            │       │       └── configs/config_loader.py
            │       └── agents/tools.py
            └── agents/critic.py
                    └── agents/tools.py
```

### Import Structure:

**Top-Level Modules**:
- `app/` - UI layer (no imports from other modules)
- `pipeline/` - Orchestration (imports vision, agents)
- `vision/` - Vision models (standalone)
- `retrieval/` - RAG system (standalone)
- `agents/` - Agent framework (imports retrieval, llm)
- `configs/` - Configuration (standalone)

**Circular Dependency Prevention**:
- No module imports from `app/`
- `vision/` and `retrieval/` are independent
- `agents/` depends on both but they don't depend on `agents/`
- `pipeline/` is top-level coordinator

---

## 🌐 API Interfaces

### 1. Main Entry Point

```python
# Function: analyze_new_case
# File: pipeline/new_patient_engine.py

def analyze_new_case(
    image_path: str | Path,
    patient_id: Optional[str] = None,
    age_years: Optional[float] = None,
    age_months: Optional[float] = None,
    sex: Optional[str] = None,
    use_agents: bool = True,
) -> PatientAnalysis:
    """
    End-to-end analysis pipeline.
    
    Args:
        image_path: Path to microscopy image
        patient_id: Optional patient identifier
        age_years: Patient age in years
        age_months: Patient age in months
        sex: Patient sex (M/F)
        use_agents: Use agent framework (default: True)
    
    Returns:
        PatientAnalysis object with all results
        
    Raises:
        DetectionFailure: If cell detection fails
        ModelLoadError: If model cannot be loaded
    """
```

### 2. Agent Framework API

```python
# Function: run_agentic_assessment
# File: agents/graph.py

def run_agentic_assessment(
    patient_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute multi-agent assessment workflow.
    
    Args:
        patient_data: Dictionary with patient morphology data
            Required keys:
                - patient_id: str
                - annotated_blast_fraction: float
                - predicted_class: str
                - total_cells: int
            Optional keys:
                - age_years: float
                - age_months: float
                - sex: str
                - class_probabilities: dict
                - cell_counts_by_category: dict
    
    Returns:
        Final agent state dictionary with final_report
    """
```

### 3. LLM Provider API

```python
# Function: generate
# File: agents/llm_provider.py

def generate(
    messages: list[dict[str, str]],
    **kwargs
) -> str:
    """
    Generate text using configured LLM provider.
    
    Args:
        messages: OpenAI-format message list
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]
        **kwargs: Provider-specific parameters
            - max_tokens: int
            - temperature: float
    
    Returns:
        Generated text string
        
    Raises:
        ValueError: If API key not configured
        Exception: On API errors
    """
```

---

## ⚙️ Configuration Management

### Configuration Files:

```
configs/
├── default.yaml       # Main configuration
├── colab.yaml         # Google Colab overrides
└── config_loader.py   # Config loading utility

.env                   # Environment variables (API keys)
.env.example           # Template for .env
```

### Configuration Schema:

```yaml
# configs/default.yaml

seed: 42

data:
  raw_dir: data/raw
  processed_dir: data/processed
  kb_dir: data/knowledge_base

dataset:
  name: bonemarrowwsi_pediatricleukemia_consensus
  num_classes: 8
  image_size: 224
  augmentation: medium

vision:
  backbone: vit_small_patch16_224
  pretrained: true
  lora:
    enabled: true
    r: 8
    alpha: 16

training:
  epochs: 30
  batch_size: 64
  lr: 3.0e-4
  precision: bf16

retrieval:
  embedder: BAAI/bge-small-en-v1.5
  chunk_size: 512
  top_k_reranked: 5

llm:
  provider: groq
  model: openai/gpt-oss-120b
  max_tokens: 2048
  temperature: 0.1

critic:
  max_revision_iterations: 3
  abstention_threshold: 0.5

ui:
  port: 8501
  max_upload_size_mb: 50
```

### Environment Variables:

```bash
# .env

# Groq API (REQUIRED for LLM mode)
GROQ_API_KEY=gsk_...

# Hugging Face (optional)
HF_TOKEN=hf_...

# TCIA (for data download)
TCIA_USERNAME=...
TCIA_PASSWORD=...

# Weights & Biases (optional)
WANDB_API_KEY=...
WANDB_PROJECT=bonemarrow-vision

# Mode toggles
SYNTHESIZER_USE_LLM=true
```

---

## 📁 Directory Structure

```
project_root/
│
├── app/                          # User interface
│   ├── streamlit_app.py          # Main UI application
│   └── __init__.py
│
├── agents/                       # Agent framework
│   ├── graph.py                  # LangGraph workflow
│   ├── retriever.py              # Retriever agent
│   ├── synthesizer.py            # Synthesizer agent
│   ├── critic.py                 # Critic agent
│   ├── state.py                  # State definitions
│   ├── tools.py                  # Utility functions
│   ├── llm_provider.py           # LLM abstraction
│   ├── fallback_guidelines.py    # Fallback content
│   └── __init__.py
│
├── pipeline/                     # Orchestration layer
│   ├── new_patient_engine.py     # Main engine
│   ├── cell_detector.py          # Cell detection
│   ├── data_model.py             # Data structures
│   └── __init__.py
│
├── vision/                       # Vision models
│   ├── model.py                  # ViT-Small + LoRA
│   ├── train.py                  # Training pipeline
│   ├── dataset.py                # Data loading
│   └── __init__.py
│
├── retrieval/                    # RAG system
│   ├── retrieve.py               # Hybrid retrieval
│   ├── index.py                  # FAISS/BM25 indexing
│   └── __init__.py
│
├── configs/                      # Configuration
│   ├── default.yaml              # Main config
│   ├── colab.yaml                # Colab overrides
│   ├── config_loader.py          # Config loader
│   └── __init__.py
│
├── data/                         # Data directory
│   ├── knowledge_base/           # WHO guidelines
│   ├── processed/                # Processed data
│   │   ├── cell_metadata.csv
│   │   ├── splits.json
│   │   └── retrieval_index/
│   └── raw/                      # Raw DICOM (gitignored)
│
├── detector/                     # YOLO detector (unused)
│   ├── prepare_dataset.py
│   ├── train_detector.py
│   └── ...
│
├── tests/                        # Unit tests
│   ├── test_vision.py
│   ├── test_retrieval.py
│   └── ...
│
├── .env                          # API keys (gitignored)
├── .env.example                  # Template
├── requirements.txt              # Dependencies
├── setup_groq_rag.py            # Setup verification
├── README.md                     # Main documentation
├── PROJECT_STATUS_COMPLETE.md   # Status report
├── SYSTEM_ARCHITECTURE.md       # This file
└── ...
```

---

## 🔐 Security Considerations

### API Key Management:
- Store in `.env` file (gitignored)
- Never commit to version control
- Use environment variables only
- Rotate keys periodically

### Data Privacy:
- No patient data uploaded to cloud (except image processing)
- LLM inference is stateless (no data retention)
- Local processing for detection/classification
- HIPAA compliance considerations if deployed clinically

### Input Validation:
- Image format verification
- File size limits (50MB)
- SQL injection prevention (no SQL used)
- Command injection prevention (no shell commands with user input)

---

## 📈 Performance Optimization

### Current Optimizations:
1. **Singleton pattern** for model loading (avoid reloading)
2. **Batch inference** for cell classification (batch_size=32)
3. **FAISS indexing** for fast vector search
4. **Caching** of embeddings and retrieval results
5. **Lazy loading** of heavy models
6. **GPU acceleration** (when available)

### Future Optimizations:
1. Model quantization (INT8)
2. ONNX runtime conversion
3. Async processing
4. Redis caching for frequent queries
5. CDN for static assets

---

## 🎯 Summary

This system architecture implements a complete multimodal AI pipeline for bone marrow cytology analysis with the following key characteristics:

### Design Principles:
- ✅ Modularity (independent components)
- ✅ Extensibility (easy to add features)
- ✅ Reliability (fallback mechanisms)
- ✅ Transparency (evidence citations)
- ✅ Medical safety (disclaimers, limitations)

### Technical Highlights:
- Vision: ViT-Small + LoRA for cell classification
- RAG: Hybrid FAISS + BM25 retrieval
- LLM: Groq 120B cloud inference
- Agents: LangGraph orchestration with self-reflection
- UI: Medical-grade Streamlit interface

### Performance:
- Processing: 5-10 seconds per patient
- Throughput: ~600 patients/day (free tier)
- Accuracy: Depends on model training
- Reliability: 100% (with fallbacks)

---

**For creating detailed architecture diagrams, use this document as a comprehensive reference. All components, data flows, and interactions are documented here.**

---

**Document Version**: 1.0  
**Last Updated**: October 7, 2026  
**Maintainer**: Development Team  
