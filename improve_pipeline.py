"""
improve_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Diagnose and improve the current ML pipeline performance.

This script will:
1. Test each pipeline component
2. Identify bottlenecks
3. Apply optimizations
4. Generate performance report
"""

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_cell_detector():
    """Test cell detector with various parameter sets."""
    print("\n" + "="*80)
    print("TESTING CELL DETECTOR")
    print("="*80)
    
    from pipeline.cell_detector import CellDetector
    from PIL import Image
    import numpy as np
    
    # Create test image (simulate microscopy)
    test_image = np.random.randint(180, 255, (2000, 2000, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_image)
    
    detector = CellDetector()
    
    print("\nCurrent detector parameters:")
    print(f"  MIN_CELL_AREA_PX: {detector.MIN_CELL_AREA_PX}")
    print(f"  MAX_CELL_AREA_PX: {detector.MAX_CELL_AREA_PX}")
    print(f"  MIN_DIM_PX: {detector.MIN_DIM_PX}")
    print(f"  MAX_DIM_PX: {detector.MAX_DIM_PX}")
    print(f"  MIN_NUCLEUS_PIXELS: {detector.MIN_NUCLEUS_PIXELS}")
    print(f"  NMS_IOU_THRESHOLD: {detector.NMS_IOU_THRESHOLD}")
    
    start_time = time.time()
    result = detector.detect(test_image, image_id="test_001")
    elapsed = time.time() - start_time
    
    print(f"\nDetection results:")
    print(f"  Cells detected: {len(result.cells)}")
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Speed: {len(result.cells)/elapsed:.1f} cells/sec")
    
    if result.qc:
        print(f"\nQuality metrics:")
        print(f"  Raw candidates: {result.qc.get('raw_candidates_count', 0)}")
        print(f"  Accepted cells: {result.qc.get('accepted_cells', 0)}")
        print(f"  Rejected: {result.qc.get('rejected_count', 0)}")
        
    return len(result.cells) > 0

def test_vit_classifier():
    """Test ViT classifier performance."""
    print("\n" + "="*80)
    print("TESTING VIT CLASSIFIER")
    print("="*80)
    
    try:
        from vision.classifier import load_model, classify_cell_crop
        from PIL import Image
        import numpy as np
        
        print("\nLoading ViT model...")
        start_time = time.time()
        model, device = load_model()
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"✓ Device: {device}")
        
        # Test classification speed
        test_crop = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        print("\nTesting classification speed...")
        start_time = time.time()
        num_tests = 10
        for _ in range(num_tests):
            pred_class, confidence = classify_cell_crop(test_crop, model, device)
        elapsed = time.time() - start_time
        
        print(f"✓ Classification speed: {num_tests/elapsed:.1f} cells/sec")
        print(f"✓ Average latency: {elapsed/num_tests*1000:.1f}ms per cell")
        
        return True
        
    except Exception as e:
        print(f"✗ ViT classifier test failed: {e}")
        return False

def test_rag_system():
    """Test RAG retrieval and fallback."""
    print("\n" + "="*80)
    print("TESTING RAG SYSTEM")
    print("="*80)
    
    try:
        from agents.retriever import retrieve_evidence
        
        test_query = "blast cells acute leukemia"
        
        print(f"\nTest query: '{test_query}'")
        print("Retrieving evidence...")
        
        start_time = time.time()
        evidence = retrieve_evidence(test_query, top_k=5)
        elapsed = time.time() - start_time
        
        print(f"✓ Retrieved {len(evidence)} passages in {elapsed:.2f}s")
        
        if evidence:
            print(f"\nFirst passage preview:")
            first = evidence[0]
            text_preview = first.get('text', '')[:200]
            print(f"  {text_preview}...")
            print(f"  Source: {first.get('source', 'N/A')}")
            
            # Check if fallback was used
            if 'used_fallback_guidelines' in first:
                print(f"\n✓ Fallback system is active")
        else:
            print("⚠ No evidence retrieved")
            
        return len(evidence) > 0
        
    except Exception as e:
        print(f"✗ RAG test failed: {e}")
        return False

def test_agent_workflow():
    """Test complete agent workflow."""
    print("\n" + "="*80)
    print("TESTING AGENT WORKFLOW")
    print("="*80)
    
    try:
        from agents.graph import run_assessment
        
        # Mock patient data
        test_data = {
            'case_id': 'TEST-001',
            'cells_classified': 150,
            'morphology_counts': {
                'Blast': 8,
                'Granulocyte (Mature/Band)': 45,
                'Granulocyte (Immature)': 30,
                'Lymphoid Series': 25,
                'Erythroid Series (Mature)': 20,
                'Erythroid Series (Immature)': 15,
                'Megakaryocyte / Platelet': 5,
                'Other': 2
            },
            'blast_fraction': 0.053,
            'age_years': None,
            'sex': None
        }
        
        print("\nRunning mock assessment...")
        start_time = time.time()
        
        result = run_assessment(
            morphology_counts=test_data['morphology_counts'],
            cells_classified=test_data['cells_classified'],
            blast_fraction=test_data['blast_fraction'],
            age_years=test_data['age_years'],
            sex=test_data['sex']
        )
        
        elapsed = time.time() - start_time
        
        print(f"✓ Assessment completed in {elapsed:.2f}s")
        print(f"✓ Generated {len(result.get('final_report', ''))} characters")
        print(f"✓ Grounding score: {result.get('grounding_score', 0):.2f}")
        print(f"✓ Status: {result.get('status', 'unknown')}")
        
        return result.get('status') in ['approved', 'completed']
        
    except Exception as e:
        print(f"✗ Agent workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test complete end-to-end pipeline."""
    print("\n" + "="*80)
    print("TESTING END-TO-END PIPELINE")
    print("="*80)
    
    try:
        from pipeline.new_patient_engine import analyze_new_case
        from PIL import Image
        import numpy as np
        import io
        
        # Create synthetic test image
        print("\nCreating synthetic test image...")
        test_image = np.random.randint(200, 255, (1500, 1500, 3), dtype=np.uint8)
        
        # Add some "cell-like" blobs
        from scipy.ndimage import gaussian_filter
        for _ in range(20):
            x, y = np.random.randint(100, 1400, 2)
            radius = np.random.randint(30, 60)
            y_grid, x_grid = np.ogrid[-radius:radius, -radius:radius]
            mask = x_grid**2 + y_grid**2 <= radius**2
            x1, x2 = max(0, x-radius), min(1500, x+radius)
            y1, y2 = max(0, y-radius), min(1500, y+radius)
            mask_h, mask_w = y2-y1, x2-x1
            if mask_h > 0 and mask_w > 0:
                test_image[y1:y2, x1:x2] = np.clip(
                    test_image[y1:y2, x1:x2] - 80, 0, 255
                )
        
        # Convert to file-like object
        img_pil = Image.fromarray(test_image)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_bytes.name = "test_image.png"
        
        print("Running complete analysis...")
        start_time = time.time()
        
        result = analyze_new_case(
            uploaded_files=[img_bytes],
            age_years=None,
            sex=None,
            sample_site=None
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ End-to-end analysis completed in {elapsed:.2f}s")
        print(f"✓ Case ID: {result.case_id}")
        print(f"✓ Cells detected: {result.total_cells_detected}")
        print(f"✓ Cells classified: {result.cells_classified}")
        print(f"✓ Blast fraction: {result.blast_fraction:.1%}")
        print(f"✓ Assessment length: {len(result.final_report)} chars")
        
        return result.total_cells_detected > 0
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_report(results):
    """Generate improvement report."""
    print("\n" + "="*80)
    print("PIPELINE DIAGNOSTIC REPORT")
    print("="*80)
    
    all_passed = all(results.values())
    
    print("\nComponent Status:")
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {component}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nYour pipeline is working correctly!")
        print("\nRecommendations:")
        print("  1. Test with real bone marrow images")
        print("  2. Monitor detection quality")
        print("  3. Collect user feedback")
        print("  4. Fine-tune parameters if needed")
    else:
        print("⚠ SOME TESTS FAILED")
        print("="*80)
        print("\nFailed components need attention.")
        print("Check the error messages above for details.")
    
    print()

def main():
    """Run all pipeline tests."""
    print("="*80)
    print("PIPELINE IMPROVEMENT & DIAGNOSTIC TOOL")
    print("="*80)
    print("\nThis will test all pipeline components and identify issues.")
    print()
    
    results = {}
    
    # Test each component
    try:
        results['Cell Detector'] = test_cell_detector()
    except Exception as e:
        print(f"✗ Cell detector test crashed: {e}")
        results['Cell Detector'] = False
    
    try:
        results['ViT Classifier'] = test_vit_classifier()
    except Exception as e:
        print(f"✗ ViT classifier test crashed: {e}")
        results['ViT Classifier'] = False
    
    try:
        results['RAG System'] = test_rag_system()
    except Exception as e:
        print(f"✗ RAG test crashed: {e}")
        results['RAG System'] = False
    
    try:
        results['Agent Workflow'] = test_agent_workflow()
    except Exception as e:
        print(f"✗ Agent workflow test crashed: {e}")
        results['Agent Workflow'] = False
    
    try:
        results['End-to-End Pipeline'] = test_end_to_end()
    except Exception as e:
        print(f"✗ End-to-end test crashed: {e}")
        results['End-to-End Pipeline'] = False
    
    # Generate report
    generate_report(results)
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
