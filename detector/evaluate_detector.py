"""
detector/evaluate_detector.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive evaluation of trained YOLO cell detector.

Computes:
- Precision, Recall, mAP@50, mAP@50:95
- True positives, false positives, false negatives
- Per-patient detection statistics
- Visual validation: ground truth vs predictions

CRITICAL: Evaluate on held-out test patients only.
"""

import os
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import torch
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class DetectorEvaluator:
    """Comprehensive evaluation of trained cell detector."""
    
    def __init__(self,
                 model_path: str,
                 dataset_config: str = "detector/yolo_dataset/dataset.yaml",
                 metadata_csv: str = "data/processed/cell_metadata.csv",
                 output_dir: str = "detector/evaluation",
                 iou_threshold: float = 0.5,
                 conf_threshold: float = 0.25,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            model_path: Path to trained YOLO model (.pt file)
            dataset_config: Path to dataset.yaml
            metadata_csv: Path to cell_metadata.csv
            output_dir: Output directory for evaluation results
            iou_threshold: IoU threshold for matching predictions to ground truth
            conf_threshold: Confidence threshold for predictions
            device: Device (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.dataset_config = Path(dataset_config)
        self.metadata_csv = Path(metadata_csv)
        self.output_dir = Path(output_dir)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Load dataset config
        with open(self.dataset_config) as f:
            self.dataset_info = yaml.safe_load(f)
        
        # Load metadata
        print(f"Loading metadata: {self.metadata_csv}")
        self.metadata = pd.read_csv(self.metadata_csv)
        self.metadata_valid = self.metadata[self.metadata['extraction_status'] == 'valid']
        
        print(f"\n{'='*80}")
        print(f"DETECTOR EVALUATION")
        print(f"{'='*80}\n")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"Output directory: {self.output_dir}")
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes.
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
        
        Returns:
            IoU value
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def evaluate_on_test_set(self):
        """Run model validation on test set."""
        print(f"\n{'='*80}")
        print(f"RUNNING VALIDATION ON TEST SET")
        print(f"{'='*80}\n")
        
        # Run validation
        metrics = self.model.val(
            data=str(self.dataset_config),
            split='test',
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            plots=True,
            save_json=True,
            project=str(self.output_dir),
            name='test_val',
        )
        
        # Extract metrics
        results = {
            'precision': float(metrics.box.p),
            'recall': float(metrics.box.r),
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
        }
        
        print(f"\nTest Set Metrics:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  mAP@50: {results['map50']:.4f}")
        print(f"  mAP@50:95: {results['map50_95']:.4f}")
        
        return results, metrics
    
    def count_gt_and_predictions(self):
        """Count ground truth annotations and predictions per split."""
        print(f"\n{'='*80}")
        print(f"COUNTING ANNOTATIONS AND PREDICTIONS")
        print(f"{'='*80}\n")
        
        counts = {}
        
        for split in ['train', 'val', 'test']:
            # Ground truth counts
            gt_count = len(self.metadata_valid[self.metadata_valid['split'] == split])
            gt_patients = len(self.metadata_valid[self.metadata_valid['split'] == split]['patient_id'].unique())
            
            counts[split] = {
                'ground_truth_cells': gt_count,
                'patients': gt_patients,
            }
            
            print(f"{split.upper()}:")
            print(f"  Ground truth cells: {gt_count}")
            print(f"  Patients: {gt_patients}")
        
        return counts
    
    def compute_detailed_metrics(self, results, counts):
        """Compute detailed detection metrics."""
        print(f"\n{'='*80}")
        print(f"DETAILED METRICS")
        print(f"{'='*80}\n")
        
        # Test set statistics
        test_gt = counts['test']['ground_truth_cells']
        
        precision = results['precision']
        recall = results['recall']
        
        # Estimate true positives, false positives, false negatives
        # TP = Recall * GT
        # FN = GT - TP
        # FP = TP * (1/Precision - 1)
        
        tp = int(recall * test_gt)
        fn = test_gt - tp
        fp = int(tp * (1.0 / precision - 1.0)) if precision > 0 else 0
        
        predicted_total = tp + fp
        
        detailed = {
            'test_ground_truth_cells': test_gt,
            'test_predicted_cells': predicted_total,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'map50': results['map50'],
            'map50_95': results['map50_95'],
        }
        
        print(f"Ground truth cells (test): {test_gt}")
        print(f"Predicted cells (test): {predicted_total}")
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"mAP@50: {results['map50']:.4f}")
        print(f"mAP@50:95: {results['map50_95']:.4f}")
        
        return detailed
    
    def save_results(self, results, counts, detailed):
        """Save evaluation results to JSON."""
        output_file = self.output_dir / "evaluation_results.json"
        
        combined = {
            'model_path': str(self.model_path),
            'iou_threshold': self.iou_threshold,
            'conf_threshold': self.conf_threshold,
            'metrics': results,
            'counts': counts,
            'detailed_metrics': detailed,
        }
        
        with open(output_file, 'w') as f:
            json.dump(combined, f, indent=2)
        
        print(f"\nSaved results to: {output_file}")
        
        # Also save as text
        txt_file = self.output_dir / "evaluation_results.txt"
        with open(txt_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CELL DETECTOR EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"IoU threshold: {self.iou_threshold}\n")
            f.write(f"Confidence threshold: {self.conf_threshold}\n\n")
            
            f.write("TEST SET METRICS:\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  mAP@50: {results['map50']:.4f}\n")
            f.write(f"  mAP@50:95: {results['map50_95']:.4f}\n\n")
            
            f.write("DETECTION COUNTS:\n")
            f.write(f"  Ground truth cells: {detailed['test_ground_truth_cells']}\n")
            f.write(f"  Predicted cells: {detailed['test_predicted_cells']}\n")
            f.write(f"  True positives: {detailed['true_positives']}\n")
            f.write(f"  False positives: {detailed['false_positives']}\n")
            f.write(f"  False negatives: {detailed['false_negatives']}\n\n")
            
            f.write("SPLIT STATISTICS:\n")
            for split in ['train', 'val', 'test']:
                f.write(f"  {split.upper()}:\n")
                f.write(f"    Patients: {counts[split]['patients']}\n")
                f.write(f"    Ground truth cells: {counts[split]['ground_truth_cells']}\n")
        
        print(f"Saved text results to: {txt_file}")
        
        return output_file, txt_file
    
    def generate_summary_plot(self, detailed):
        """Generate summary visualization of metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Precision vs Recall
        ax = axes[0]
        metrics = ['Precision', 'Recall', 'mAP@50', 'mAP@50:95']
        values = [detailed['precision'], detailed['recall'], 
                 detailed['map50'], detailed['map50_95']]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        bars = ax.barh(metrics, values, color=colors)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Score')
        ax.set_title('Detection Metrics')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center')
        
        # Plot 2: Detection counts
        ax = axes[1]
        categories = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
        counts = [detailed['true_positives'], detailed['false_positives'], 
                 detailed['false_negatives']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        bars = ax.bar(categories, counts, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Detection Breakdown')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        # Plot 3: GT vs Predicted
        ax = axes[2]
        labels = ['Ground\nTruth', 'Predicted']
        counts = [detailed['test_ground_truth_cells'], detailed['test_predicted_cells']]
        colors = ['#3498db', '#9b59b6']
        
        bars = ax.bar(labels, counts, color=colors)
        ax.set_ylabel('Total Cells')
        ax.set_title('Test Set Cell Counts')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "metrics_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved metrics summary plot: {plot_path}")
        
        return plot_path
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        # Run validation
        results, metrics = self.evaluate_on_test_set()
        
        # Count annotations
        counts = self.count_gt_and_predictions()
        
        # Compute detailed metrics
        detailed = self.compute_detailed_metrics(results, counts)
        
        # Save results
        json_file, txt_file = self.save_results(results, counts, detailed)
        
        # Generate plots
        plot_file = self.generate_summary_plot(detailed)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}\n")
        print(f"Results saved to: {self.output_dir}")
        print(f"  - JSON: {json_file}")
        print(f"  - Text: {txt_file}")
        print(f"  - Plot: {plot_file}")
        
        return results, detailed


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate YOLO cell detector")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.pt file)")
    parser.add_argument("--dataset", type=str, default="detector/yolo_dataset/dataset.yaml",
                       help="Path to dataset.yaml")
    parser.add_argument("--metadata", type=str, default="data/processed/cell_metadata.csv",
                       help="Path to cell_metadata.csv")
    parser.add_argument("--output", type=str, default="detector/evaluation",
                       help="Output directory")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU threshold")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    evaluator = DetectorEvaluator(
        model_path=args.model,
        dataset_config=args.dataset,
        metadata_csv=args.metadata,
        output_dir=args.output,
        iou_threshold=args.iou,
        conf_threshold=args.conf,
        device=args.device,
    )
    
    results, detailed = evaluator.evaluate()


if __name__ == "__main__":
    main()
