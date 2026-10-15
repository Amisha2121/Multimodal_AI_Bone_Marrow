"""
scripts/run_all.py
─────────────────────────────────────────────────────────────────────────────
Master orchestrator for the Pediatric Leukemia Bone Marrow Vision Baseline:

Step 1: scripts/step1_delete_incomplete_crops.py  (Safe reset of crop cache)
Step 2: scripts/step2_extract_all_crops.py        (Re-extract 20,852 real crops from IDC S3)
Step 3: scripts/step3_verify_crops.py             (Verify real dataset integrity)
Step 4: scripts/step4_train_cell_classifier.py    (Train ViT-Small/16 + LoRA morphology classifier)
Step 5: scripts/step5_evaluate_and_xai.py         (Evaluate on test patients & generate XAI heatmaps)
Step 6: scripts/step6_patient_baseline.py         (Patient-level 4-class baseline & EXPERIMENT_002_RESULTS.md)

Usage:
  python scripts/run_all.py                      # Run entire pipeline from step 1 to 6
  python scripts/run_all.py --start-step 2       # Skip step 1 (if cache already reset)
  python scripts/run_all.py --dry-run            # Preview steps without network/training
  python scripts/run_all.py --patients P1 P2     # Extract/test specific patients only
  python scripts/run_all.py --epochs 10          # Override training epochs
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    (1, "Delete Incomplete Crops", REPO_ROOT / "scripts" / "step1_delete_incomplete_crops.py"),
    (2, "Extract All Real Crops from IDC S3", REPO_ROOT / "scripts" / "step2_extract_all_crops.py"),
    (3, "Verify Crop Integrity & Overlap", REPO_ROOT / "scripts" / "step3_verify_crops.py"),
    (4, "Train ViT-Small/16 + LoRA Classifier", REPO_ROOT / "scripts" / "step4_train_cell_classifier.py"),
    (5, "Cell Evaluation & Attention Rollout XAI", REPO_ROOT / "scripts" / "step5_evaluate_and_xai.py"),
    (6, "Patient 4-Class Baseline & Generate Docs", REPO_ROOT / "scripts" / "step6_patient_baseline.py"),
]


def run_step(step_num: int, title: str, script_path: Path, args_list: list) -> bool:
    print("\n" + "=" * 75)
    print(f"  RUNNING STEP {step_num}: {title.upper()}")
    print("=" * 75)
    cmd = [sys.executable, str(script_path)] + args_list
    print(f"Command: {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed with return code {ret.returncode}.")
        return False
    print(f"\n[SUCCESS] Step {step_num} completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Master pipeline runner for Bone Marrow Vision Baseline.")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 7),
                        help="Step to start execution from (default: 1).")
    parser.add_argument("--stop-step", type=int, default=6, choices=range(1, 7),
                        help="Step to end execution at (default: 6).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run steps in dry-run/preview mode where supported.")
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Pass specific patient IDs to step 2 extraction.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs in step 4.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size in step 4.")
    parser.add_argument("--fast-verify", action="store_true",
                        help="Skip pixel-level verification in step 3.")
    args = parser.parse_args()

    print("*" * 75)
    print(" MULTIMODAL AI BONE MARROW - PEDIATRIC LEUKEMIA VISION PIPELINE")
    print(f" Executing steps {args.start_step} through {args.stop_step}")
    print("*" * 75)

    for num, title, script in STEPS:
        if num < args.start_step or num > args.stop_step:
            continue

        step_args = []
        if num == 1:
            if args.dry_run:
                step_args.append("--dry-run")
            else:
                step_args.append("--yes")
        elif num == 2:
            if args.dry_run:
                step_args.append("--dry-run")
            if args.patients:
                step_args.extend(["--patients"] + args.patients)
        elif num == 3:
            if args.fast_verify:
                step_args.append("--fast")
        elif num == 4:
            if args.epochs:
                step_args.extend(["--epochs", str(args.epochs)])
            if args.batch_size:
                step_args.extend(["--batch-size", str(args.batch_size)])

        success = run_step(num, title, script, step_args)
        if not success:
            print(f"\n[PIPELINE HALTED] Pipeline halted at Step {num}.")
            sys.exit(1)

    print("\n" + "*" * 75)
    print(" PIPELINE FINISHED SUCCESSFULLY!")
    print("*" * 75)


if __name__ == "__main__":
    main()
