"""
detector/run_full_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Master script to run complete detector training and validation pipeline.

PIPELINE:
1. Prepare YOLO dataset from ground-truth annotations
2. Train YOLO detector
3. Evaluate on test set
4. Generate visual validation (50-crop contact sheet)
5. Report final metrics and readiness

Run: python detector/run_full_pipeline.py
"""

import os
import sys
from pathlib import Path
import argparse
import json
import subprocess
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class DetectorPipelineRunner:
    """Runs complete detector training and validation pipeline."""
    
    def __init__(self,
                 model_size: str = "n",
                 epochs: int = 100,
                 batch_size: int = 16,
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.5,
                 skip_training: bool = False,
                 model_path: str = None):
        """
        Args:
            model_size: YOLO model size (n, s, m, l, x)
            epochs: Training epochs
            batch_size: Batch size
            img_size: Input image size
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for evaluation
            skip_training: Skip training (use existing model)
            model_path: Path to existing model (if skip_training=True)
        """
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.skip_training = skip_training
        self.model_path = model_path
        
        self.detector_dir = Path("detector")
        self.dataset_config = self.detector_dir / "yolo_dataset" / "dataset.yaml"
        
        print("\n" + "="*80)
        print("CELL DETECTOR TRAINING PIPELINE")
        print("="*80 + "\n")
        print(f"Configuration:")
        print(f"  Model size: {self.model_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.img_size}")
        print(f"  Confidence threshold: {self.conf_threshold}")
        print(f"  IoU threshold: {self.iou_threshold}")
        print(f"  Skip training: {self.skip_training}")
        if self.skip_training:
            print(f"  Model path: {self.model_path}")
    
    def step_1_prepare_dataset(self):
        """Step 1: Prepare YOLO dataset."""
        print("\n" + "="*80)
        print("STEP 1: PREPARE YOLO DATASET")
        print("="*80 + "\n")
        
        if self.dataset_config.exists():
            print(f"Dataset config already exists: {self.dataset_config}")
            response = input("Recreate dataset? (y/n): ")
            if response.lower() != 'y':
                print("Skipping dataset preparation.")
                return
        
        print("Running prepare_dataset.py...")
        
        from prepare_dataset import main as prepare_main
        prepare_main()
        
        print("\n✓ Dataset preparation complete")
    
    def step_2_train_detector(self):
        """Step 2: Train YOLO detector."""
        print("\n" + "="*80)
        print("STEP 2: TRAIN YOLO DETECTOR")
        print("="*80 + "\n")
        
        if self.skip_training:
            print("Training skipped (using existing model)")
            if self.model_path and Path(self.model_path).exists():
                print(f"Model: {self.model_path}")
                self.trained_model_path = Path(self.model_path)
            else:
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}\n"
                    f"Either provide valid --model path or remove --skip-training"
                )
            return
        
        print(f"Training YOLOv8{self.model_size} for {self.epochs} epochs...")
        
        # Import and run training
        sys.argv = [
            'train_detector.py',
            '--dataset', str(self.dataset_config),
            '--model-size', self.model_size,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--img-size', str(self.img_size),
        ]
        
        from train_detector import main as train_main
        train_main()
        
        # Get trained model path
        self.trained_model_path = (
            self.detector_dir / "runs" / "train" / "cell_detector" / "weights" / "best.pt"
        )
        
        if not self.trained_model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {self.trained_model_path}")
        
        print(f"\n✓ Training complete")
        print(f"Trained model: {self.trained_model_path}")
    
    def step_3_evaluate_detector(self):
        """Step 3: Evaluate on test set."""
        print("\n" + "="*80)
        print("STEP 3: EVALUATE ON TEST SET")
        print("="*80 + "\n")
        
        print("Running evaluation...")
        
        sys.argv = [
            'evaluate_detector.py',
            '--model', str(self.trained_model_path),
            '--conf', str(self.conf_threshold),
            '--iou', str(self.iou_threshold),
        ]
        
        from evaluate_detector import main as eval_main
        eval_main()
        
        # Load results
        results_file = self.detector_dir / "evaluation" / "evaluation_results.json"
        with open(results_file) as f:
            self.eval_results = json.load(f)
        
        print("\n✓ Evaluation complete")
    
    def step_4_visualize_predictions(self):
        """Step 4: Generate visual validation."""
        print("\n" + "="*80)
        print("STEP 4: VISUAL VALIDATION (MANDATORY)")
        print("="*80 + "\n")
        
        print("Generating visualizations...")
        
        sys.argv = [
            'visualize_predictions.py',
            '--model', str(self.trained_model_path),
            '--conf', str(self.conf_threshold),
        ]
        
        from visualize_predictions import main as viz_main
        viz_main()
        
        self.contact_sheet_path = (
            self.detector_dir / "visualizations" / "contact_sheet_50_crops.png"
        )
        
        print(f"\n✓ Visualization complete")
        print(f"\n⚠️  MANDATORY VISUAL INSPECTION REQUIRED:")
        print(f"   Open: {self.contact_sheet_path}")
        print(f"   Verify each crop contains ONE actual individual cell")
    
    def step_5_final_report(self):
        """Step 5: Generate final readiness report."""
        print("\n" + "="*80)
        print("STEP 5: FINAL REPORT")
        print("="*80 + "\n")
        
        metrics = self.eval_results['detailed_metrics']
        
        # Check success criteria
        criteria_met = []
        criteria_failed = []
        
        # 1. Recall > 0.80
        recall = metrics['recall']
        if recall >= 0.80:
            criteria_met.append(f"✓ Recall: {recall:.3f} >= 0.80")
        else:
            criteria_failed.append(f"✗ Recall: {recall:.3f} < 0.80 (target: >= 0.80)")
        
        # 2. Precision > 0.70
        precision = metrics['precision']
        if precision >= 0.70:
            criteria_met.append(f"✓ Precision: {precision:.3f} >= 0.70")
        else:
            criteria_failed.append(f"✗ Precision: {precision:.3f} < 0.70 (target: >= 0.70)")
        
        # 3. mAP@50 > 0.75
        map50 = metrics['map50']
        if map50 >= 0.75:
            criteria_met.append(f"✓ mAP@50: {map50:.3f} >= 0.75")
        else:
            criteria_failed.append(f"✗ mAP@50: {map50:.3f} < 0.75 (target: >= 0.75)")
        
        # Print report
        print("SUCCESS CRITERIA EVALUATION:\n")
        
        if criteria_met:
            print("PASSED:")
            for item in criteria_met:
                print(f"  {item}")
            print()
        
        if criteria_failed:
            print("FAILED:")
            for item in criteria_failed:
                print(f"  {item}")
            print()
        
        # Additional metrics
        print("DETAILED METRICS:")
        print(f"  Ground truth cells (test): {metrics['test_ground_truth_cells']}")
        print(f"  Predicted cells (test): {metrics['test_predicted_cells']}")
        print(f"  True positives: {metrics['true_positives']}")
        print(f"  False positives: {metrics['false_positives']}")
        print(f"  False negatives: {metrics['false_negatives']}")
        print(f"  mAP@50:95: {metrics['map50_95']:.3f}")
        print()
        
        # Files created
        print("FILES CREATED:")
        print(f"  Model: {self.trained_model_path}")
        print(f"  Evaluation: detector/evaluation/evaluation_results.json")
        print(f"  Metrics plot: detector/evaluation/metrics_summary.png")
        print(f"  Contact sheet: {self.contact_sheet_path}")
        print()
        
        # Final verdict
        print("="*80)
        if len(criteria_failed) == 0:
            print("✓ DETECTOR PASSED QUANTITATIVE VALIDATION")
            print()
            print("NEXT STEPS:")
            print("  1. Review contact sheet for visual validation")
            print(f"     Open: {self.contact_sheet_path}")
            print("  2. Verify crops contain actual individual cells")
            print("  3. If visual validation passes, deploy to production")
            print("  4. Replace heuristic detector in streamlit_app.py")
        else:
            print("✗ DETECTOR FAILED QUANTITATIVE VALIDATION")
            print()
            print("RECOMMENDATIONS:")
            if recall < 0.80:
                print("  - Increase training epochs")
                print("  - Use larger model size (try 's' or 'm')")
                print("  - Check ground-truth annotations for completeness")
            if precision < 0.70:
                print("  - Increase confidence threshold")
                print("  - Add more training data")
                print("  - Check for annotation errors (false positives)")
            print("  - Retrain and re-evaluate before deployment")
        print("="*80)
    
    def run(self):
        """Run complete pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Prepare dataset
            self.step_1_prepare_dataset()
            
            # Step 2: Train detector
            self.step_2_train_detector()
            
            # Step 3: Evaluate
            self.step_3_evaluate_detector()
            
            # Step 4: Visualize
            self.step_4_visualize_predictions()
            
            # Step 5: Final report
            self.step_5_final_report()
            
            elapsed = time.time() - start_time
            print(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"PIPELINE FAILED")
            print(f"{'='*80}\n")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run complete detector training pipeline")
    
    parser.add_argument("--model-size", type=str, default="n", 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLO model size (default: n)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU threshold for evaluation (default: 0.5)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and use existing model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to existing model (required if --skip-training)")
    
    args = parser.parse_args()
    
    # Validate args
    if args.skip_training and not args.model:
        parser.error("--model is required when --skip-training is used")
    
    # Run pipeline
    runner = DetectorPipelineRunner(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        skip_training=args.skip_training,
        model_path=args.model,
    )
    
    runner.run()


if __name__ == "__main__":
    main()
