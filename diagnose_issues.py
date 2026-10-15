"""
diagnose_issues.py
─────────────────────────────────────────────────────────────────────────────
Diagnoses issues with RAG, ML model, and detector.

Run: python diagnose_issues.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

print("\n" + "="*80)
print("SYSTEM DIAGNOSTIC")
print("="*80 + "\n")

# 1. Check if ViT model exists
print("1. Checking ViT Model...")
vit_model_path = REPO_ROOT / "models" / "checkpoints" / "best_vit_lora_morphology.pt"
if vit_model_path.exists():
    print(f"   ✓ ViT model found: {vit_model_path}")
    import torch
    try:
        ckpt = torch.load(vit_model_path, map_location='cpu', weights_only=False)
        print(f"   ✓ Model loads successfully")
        print(f"     - Classes: {ckpt.get('num_classes', 'unknown')}")
        print(f"     - Backbone: {ckpt.get('backbone_name', 'unknown')}")
        print(f"     - LoRA: {ckpt.get('use_lora', False)}")
    except Exception as e:
        print(f"   ✗ Model load failed: {e}")
else:
    print(f"   ✗ ViT model NOT FOUND: {vit_model_path}")
    print(f"     Run: python vision/train.py")

print()

# 2. Check if RAG indices exist
print("2. Checking RAG Indices...")
faiss_index = REPO_ROOT / "data" / "knowledge_base" / "embeddings" / "faiss_index.bin"
bm25_index = REPO_ROOT / "data" / "knowledge_base" / "embeddings" / "bm25_index.pkl"

if faiss_index.exists():
    print(f"   ✓ FAISS index found: {faiss_index}")
else:
    print(f"   ✗ FAISS index NOT FOUND: {faiss_index}")
    print(f"     Run: python retrieval/ingest.py")

if bm25_index.exists():
    print(f"   ✓ BM25 index found: {bm25_index}")
else:
    print(f"   ✗ BM25 index NOT FOUND: {bm25_index}")
    print(f"     Run: python retrieval/ingest.py")

print()

# 3. Test RAG retrieval
print("3. Testing RAG Retrieval...")
try:
    from retrieval.retrieve import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve("acute leukemia blast percentage", top_k=2)
    print(f"   ✓ RAG retrieval works")
    print(f"     Retrieved {len(results)} passages")
    if results:
        print(f"     Sample title: {results[0].get('title', 'N/A')}")
except Exception as e:
    print(f"   ✗ RAG retrieval failed: {e}")

print()

# 4. Test cell detector (heuristic)
print("4. Testing Heuristic Cell Detector...")
try:
    from pipeline.cell_detector import detect_cells
    from PIL import Image
    import numpy as np
    
    # Create synthetic test image
    test_img = Image.new('RGB', (500, 500), color=(240, 235, 225))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    # Draw a few purple circles (simulated cells)
    for i in range(5):
        x, y = 100 + i*80, 200
        draw.ellipse([x-30, y-30, x+30, y+30], fill=(150, 120, 180))
    
    cells, qc = detect_cells(test_img, image_id="DIAGNOSTIC-TEST")
    print(f"   ✓ Heuristic detector works")
    print(f"     Detected: {len(cells)} cells")
    print(f"     Raw candidates: {qc.get('raw_candidates_count', 0)}")
    print(f"     Accepted: {qc.get('accepted_cells', 0)}")
    print(f"     Rejected: {qc.get('rejected_count', 0)}")
    
except Exception as e:
    print(f"   ✗ Heuristic detector failed: {e}")
    import traceback
    traceback.print_exc()

print()

# 5. Check trained YOLO detector
print("5. Checking Trained YOLO Detector...")
yolo_model = REPO_ROOT / "detector" / "runs" / "train" / "cell_detector" / "weights" / "best.pt"
if yolo_model.exists():
    print(f"   ✓ YOLO model found: {yolo_model}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(yolo_model))
        print(f"   ✓ YOLO model loads successfully")
    except Exception as e:
        print(f"   ✗ YOLO model load failed: {e}")
else:
    print(f"   ✗ YOLO model NOT FOUND: {yolo_model}")
    print(f"     Run: python detector/run_full_pipeline.py")

print()

# 6. Test agent workflow
print("6. Testing Agent Workflow...")
try:
    from agents.graph import run_agentic_assessment
    
    test_data = {
        "patient_id": "TEST-001",
        "age_years": 10.0,
        "annotated_blast_fraction": 0.15,
        "predicted_class": "ALL",
        "class_probabilities": {"ALL": 0.8, "AML": 0.1, "CML": 0.05, "Other": 0.05},
        "cell_counts_by_category": {
            "Blast": 30,
            "Lymphoid (Mature)": 20,
            "Myeloid (Maturing)": 25,
            "Other": 25,
        },
        "total_cells": 100,
        "low_evidence_flag": False,
    }
    
    result = run_agentic_assessment(test_data)
    print(f"   ✓ Agent workflow works")
    print(f"     Final report generated: {len(result.get('final_report', ''))} chars")
    print(f"     Grounding score: {result.get('critic_grounding_score', 0):.2f}")
    print(f"     Approved: {result.get('is_approved', False)}")
    
except Exception as e:
    print(f"   ✗ Agent workflow failed: {e}")
    import traceback
    traceback.print_exc()

print()

# 7. Summary
print("="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print("\nREQUIRED FIXES:")
issues = []

if not vit_model_path.exists():
    issues.append("• Train ViT model: python vision/train.py")

if not (faiss_index.exists() and bm25_index.exists()):
    issues.append("• Build RAG indices: python retrieval/ingest.py")

if not yolo_model.exists():
    issues.append("• Train YOLO detector: python detector/run_full_pipeline.py")

if not issues:
    print("\n✓ ALL SYSTEMS OPERATIONAL")
    print("\nThe system should work. If you're still seeing issues:")
    print("1. Check error logs in the Streamlit app")
    print("2. Verify uploaded images are bone marrow smears")
    print("3. Ensure images are not too small (<500px)")
else:
    print()
    for issue in issues:
        print(issue)

print("\n" + "="*80)
