"""
scripts/check_status.py
─────────────────────────────────────────────────────────────────────────────
Quick progress inspector for the Pediatric Leukemia Vision Baseline pipeline.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

CROPS_DIR = REPO_ROOT / "data" / "processed" / "cell_crops"
MODEL_CKPT = REPO_ROOT / "models" / "checkpoints" / "best_vit_lora_morphology.pt"
AUDIT_DOC = REPO_ROOT / "docs" / "CROP_RESTORATION_AUDIT.md"
RESULTS_DOC = REPO_ROOT / "docs" / "EXPERIMENT_002_RESULTS.md"
TOTAL_EXPECTED_CROPS = 20_852


def main():
    print("=" * 65)
    print("  PIPELINE PROGRESS & TRAINING STATUS")
    print("=" * 65)

    # 1. Crop Extraction Status
    if CROPS_DIR.exists():
        crops = list(CROPS_DIR.rglob("*.png"))
        n_crops = len(crops)
    else:
        n_crops = 0

    pct_crops = (n_crops / TOTAL_EXPECTED_CROPS) * 100
    print(f"\n[1] Real Cell Crops on Disk:")
    print(f"    - Extracted: {n_crops:,} / {TOTAL_EXPECTED_CROPS:,} ({pct_crops:.1f}%)")
    if n_crops == TOTAL_EXPECTED_CROPS:
        print("    - Status:    [COMPLETE] All 20,852 real crops extracted.")
    elif n_crops == 0:
        print("    - Status:    [NOT STARTED]")
    else:
        print("    - Status:    [IN PROGRESS / PAUSED]")

    # 2. Verification Status
    print(f"\n[2] Step 3 Verification Gate:")
    if AUDIT_DOC.exists():
        print(f"    - Status:    [PASSED] Audit report exists at {AUDIT_DOC.name}")
    else:
        print("    - Status:    [PENDING] Runs automatically once all crops are extracted.")

    # 3. Training Status
    print(f"\n[3] Step 4 ViT-Small/16 + LoRA Training:")
    if MODEL_CKPT.exists():
        size_mb = MODEL_CKPT.stat().st_size / (1024 * 1024)
        print(f"    - Status:    [DONE] Model weights saved ({size_mb:.1f} MB)")
        print(f"    - File:      {MODEL_CKPT.relative_to(REPO_ROOT)}")
    else:
        print("    - Status:    [NOT YET DONE] (Waiting for extraction to finish)")

    # 4. Evaluation & Results Status
    print(f"\n[4] Step 6 Patient Baseline & Final Results:")
    if RESULTS_DOC.exists():
        print(f"    - Status:    [COMPLETE] Report available at {RESULTS_DOC.name}")
    else:
        print("    - Status:    [PENDING]")

    print("\n" + "=" * 65)
    if MODEL_CKPT.exists() and RESULTS_DOC.exists():
        print(">> ALL TRAINING & EVALUATION ARE COMPLETE! <<")
    elif n_crops < TOTAL_EXPECTED_CROPS:
        print(f">> EXTRACTION INCOMPLETE ({n_crops:,}/{TOTAL_EXPECTED_CROPS:,}).")
        print("   To resume from where it left off, run:")
        print("   setup_and_run_windows.bat   OR   python scripts/run_all.py --start-step 2")
    else:
        print(">> Crops extracted! Ready to train:")
        print("   python scripts/step4_train_cell_classifier.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
