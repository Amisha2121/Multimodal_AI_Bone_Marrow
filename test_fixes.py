"""
test_fixes.py
─────────────────────────────────────────────────────────────────────────────
Quick test to verify all fixes are working.

Run: python test_fixes.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

print("\n" + "="*80)
print("TESTING FIXES")
print("="*80 + "\n")

passed = []
failed = []

# Test 1: Fallback guidelines exist and load
print("Test 1: RAG Fallback Guidelines")
try:
    from agents.fallback_guidelines import get_fallback_evidence
    evidence = get_fallback_evidence(
        blast_fraction=0.25,
        age_years=10.0,
        predominant_morphology="Blast",
        total_cells=100
    )
    assert len(evidence) > 0, "No fallback evidence returned"
    assert any("blast" in str(e).lower() for e in evidence), "No blast-related evidence"
    print("  ✓ Fallback guidelines work")
    print(f"    Generated {len(evidence)} evidence chunks")
    passed.append("Fallback Guidelines")
except Exception as e:
    print(f"  ✗ Fallback guidelines failed: {e}")
    failed.append("Fallback Guidelines")

print()

# Test 2: Retriever uses fallback
print("Test 2: Retriever Fallback Integration")
try:
    from agents.retriever import retriever_node
    from agents.state import AgentState
    
    test_state: AgentState = {
        "patient_id": "TEST",
        "predicted_class": "ALL",
        "annotated_blast_fraction": 0.30,
        "age_months": 120.0,
        "cell_counts_by_category": {"Blast": 30, "Other": 20},
        "total_cells": 50,
    }
    
    result = retriever_node(test_state)
    assert "retrieved_evidence" in result, "No evidence retrieved"
    assert len(result["retrieved_evidence"]) > 0, "Empty evidence"
    
    print("  ✓ Retriever integration works")
    print(f"    Retrieved {len(result['retrieved_evidence'])} evidence chunks")
    if result.get("used_fallback_guidelines"):
        print("    ⚠️  Used fallback (RAG may be empty)")
    passed.append("Retriever Integration")
except Exception as e:
    print(f"  ✗ Retriever failed: {e}")
    failed.append("Retriever Integration")

print()

# Test 3: Detector parameters changed
print("Test 3: Detector Parameter Changes")
try:
    with open(REPO_ROOT / "pipeline" / "cell_detector.py", 'r') as f:
        content = f.read()
    
    checks = [
        ("MIN_CELL_AREA_PX = 250", "MIN_CELL_AREA_PX"),
        ("MAX_DIM_PX = 180", "MAX_DIM_PX"),
        ("MIN_NUCLEUS_PIXELS = 80", "MIN_NUCLEUS_PIXELS"),
    ]
    
    all_found = True
    for expected, name in checks:
        if expected in content:
            print(f"  ✓ {name} updated")
        else:
            print(f"  ✗ {name} not found or not updated")
            all_found = False
    
    if all_found:
        passed.append("Detector Parameters")
    else:
        failed.append("Detector Parameters")
        
except Exception as e:
    print(f"  ✗ Could not check detector: {e}")
    failed.append("Detector Parameters")

print()

# Test 4: Agent workflow
print("Test 4: Full Agent Workflow")
try:
    from agents.graph import run_agentic_assessment
    
    test_data = {
        "patient_id": "TEST-001",
        "age_years": 8.0,
        "annotated_blast_fraction": 0.25,
        "predicted_class": "ALL",
        "class_probabilities": {"ALL": 0.75, "AML": 0.15, "CML": 0.05, "Other": 0.05},
        "cell_counts_by_category": {
            "Blast": 25,
            "Lymphoid (Mature)": 20,
            "Myeloid (Maturing)": 30,
            "Other": 25,
        },
        "total_cells": 100,
        "low_evidence_flag": False,
    }
    
    result = run_agentic_assessment(test_data)
    
    assert "final_report" in result, "No final report"
    assert len(result["final_report"]) > 100, "Report too short"
    assert "critic_grounding_score" in result, "No grounding score"
    
    report_length = len(result["final_report"])
    grounding = result["critic_grounding_score"]
    
    print("  ✓ Agent workflow completes")
    print(f"    Report length: {report_length} characters")
    print(f"    Grounding score: {grounding:.2f}")
    print(f"    Approved: {result.get('is_approved', False)}")
    
    # Check if report has specific content
    report_lower = result["final_report"].lower()
    if "blast" in report_lower and ("threshold" in report_lower or "percentage" in report_lower):
        print("    ✓ Report mentions blast threshold")
    if "pediatric" in report_lower or "age" in report_lower:
        print("    ✓ Report includes age-specific content")
    
    passed.append("Agent Workflow")
    
except Exception as e:
    print(f"  ✗ Agent workflow failed: {e}")
    import traceback
    traceback.print_exc()
    failed.append("Agent Workflow")

print()

# Test 5: Detector runs
print("Test 5: Cell Detector Execution")
try:
    from pipeline.cell_detector import detect_cells
    from PIL import Image, ImageDraw
    
    # Create synthetic test image with purple circles
    test_img = Image.new('RGB', (800, 800), color=(240, 235, 225))
    draw = ImageDraw.Draw(test_img)
    
    # Draw 10 simulated cells
    for i in range(10):
        x = 100 + (i % 3) * 200
        y = 100 + (i // 3) * 200
        # Purple nucleus
        draw.ellipse([x-40, y-40, x+40, y+40], fill=(130, 100, 160))
        # Lighter cytoplasm
        draw.ellipse([x-50, y-50, x+50, y+50], fill=(180, 160, 200), outline=(130, 100, 160))
    
    cells, qc = detect_cells(test_img, image_id="FIXTURE-TEST")
    
    print(f"  ✓ Detector runs")
    print(f"    Detected: {len(cells)} cells")
    print(f"    Raw candidates: {qc.get('raw_candidates_count', 0)}")
    print(f"    Accepted: {qc.get('accepted_cells', 0)}")
    print(f"    Median dims: {qc.get('median_bbox_dims', (0,0))}")
    
    passed.append("Detector Execution")
    
except Exception as e:
    print(f"  ✗ Detector failed: {e}")
    failed.append("Detector Execution")

print()

# Summary
print("="*80)
print("TEST SUMMARY")
print("="*80)

print(f"\n✓ Passed: {len(passed)}/{len(passed)+len(failed)}")
for test in passed:
    print(f"  • {test}")

if failed:
    print(f"\n✗ Failed: {len(failed)}/{len(passed)+len(failed)}")
    for test in failed:
        print(f"  • {test}")
    print("\n⚠️  Some tests failed. Check errors above.")
else:
    print("\n🎉 ALL TESTS PASSED!")
    print("\nYour system is ready. Run: streamlit run app/streamlit_app.py")

print("\n" + "="*80 + "\n")
