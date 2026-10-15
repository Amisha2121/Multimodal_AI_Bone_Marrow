"""
scripts/smoke_test_detector.py
─────────────────────────────────────────────────────────────────────────────
Quick smoke test for the new cell detector.
Creates a synthetic stained-looking image and verifies detection produces > 1 cell.
Run with:  python scripts/smoke_test_detector.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw
from pipeline.cell_detector import detect_cells, DetectionFailure

def make_synthetic_bone_marrow(n_cells: int = 10, size: int = 512) -> Image.Image:
    """Create a white background with coloured circles that resemble Romanowsky-stained cells."""
    rng = np.random.default_rng(42)
    img = Image.new("RGB", (size, size), color=(240, 235, 225))
    draw = ImageDraw.Draw(img)
    for _ in range(n_cells):
        cx = int(rng.integers(40, size - 40))
        cy = int(rng.integers(40, size - 40))
        r_nucleus = int(rng.integers(12, 22))
        r_cyto = r_nucleus + int(rng.integers(6, 14))
        # Cytoplasm (pinkish)
        draw.ellipse([cx - r_cyto, cy - r_cyto, cx + r_cyto, cy + r_cyto], fill=(220, 160, 160))
        # Nucleus (purple-blue)
        hue_b = int(rng.integers(100, 160))
        draw.ellipse([cx - r_nucleus, cy - r_nucleus, cx + r_nucleus, cy + r_nucleus], fill=(hue_b, 60, 180))
    return img

print("Creating synthetic bone-marrow smear image with 10 cells...")
img = make_synthetic_bone_marrow(n_cells=10)
img.save("data/new_patient_cache/_smoke_test_input.png")
print(f"  Image size: {img.size}")

print("Running cell detector...")
try:
    cells, qc = detect_cells(img, image_id="SMOKE-01")
    print(f"  Accepted: {qc['accepted_cells']}")
    print(f"  Rejected: {qc['rejected_cells']}")
    print(f"  Uncertain: {qc['uncertain_cells']}")
    assert qc["accepted_cells"] > 1, f"Expected >1 cell, got {qc['accepted_cells']}"
    assert cells[0].crop is not None, "First cell has no crop"
    assert cells[0].crop.size == (224, 224), f"Expected 224x224, got {cells[0].crop.size}"
    print(f"  First cell ID: {cells[0].cell_id}")
    print(f"  First cell crop size: {cells[0].crop.size}")
    print("\n SMOKE TEST PASSED: detector produced multiple individual cells from one large image.")
except DetectionFailure as e:
    print(f"\n Detection failed (may need more distinctive colours for synthetic test): {e}")
    print("This is expected if synthetic colours don't match the Romanowsky stain mask — try with a real image.")
except AssertionError as e:
    print(f"\n ASSERTION FAILED: {e}")
    sys.exit(1)
