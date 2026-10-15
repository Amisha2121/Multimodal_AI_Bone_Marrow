"""
detector/train_detector.py
─────────────────────────────────────────────────────────────────────────────
Train YOLOv8 cell detection model on ground-truth annotations.

ARCHITECTURE:
  Input: Microscopy image
  Output: Cell bounding boxes (single class detection)
  
CRITICAL:
  - Use ONLY ground-truth bounding boxes (NOT heuristic predictions)
  - Preserve patient-level splits
  - Single detection class: "cell"
  - DO NOT add morphology classification to detector
"""

import os
import sys
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class CellDetectorTrainer:
    """Trains YOLO cell detector on ground-truth annotations."""
    
    def __init__(self,
                 dataset_config: str = "detector/yolo_dataset/dataset.yaml",
                 model_size: str = "n",  # n, s, m, l, x
                 epochs: int = 100,
                 batch_size: int = 16,
                 img_size: int = 640,
                 project: str = "detector/runs/train",
                 name: str = "cell_detector",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            dataset_config: Path to YOLO dataset.yaml
            model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            project: Project directory for results
            name: Experiment name
            device: Device to use (cuda/cpu)
        """
        self.dataset_config = Path(dataset_config)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.project = project
        self.name = name
        self.device = device
        
        # Validate dataset config exists
        if not self.dataset_config.exists():
            raise FileNotFoundError(
                f"Dataset config not found: {self.dataset_config}\n"
                f"Run prepare_dataset.py first to create the dataset."
            )
        
        # Load dataset config
        with open(self.dataset_config) as f:
            self.dataset_info = yaml.safe_load(f)
        
        print(f"\n{'='*80}")
        print(f"YOLO CELL DETECTOR TRAINING")
        print(f"{'='*80}\n")
        print(f"Dataset config: {self.dataset_config}")
        print(f"Model size: YOLOv8{self.model_size}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"\nDataset statistics:")
        print(f"  Classes: {self.dataset_info['nc']} ({self.dataset_info['names']})")
        print(f"  Train images: {self.dataset_info.get('train_images', 'N/A')}")
        print(f"  Val images: {self.dataset_info.get('val_images', 'N/A')}")
        print(f"  Test images: {self.dataset_info.get('test_images', 'N/A')}")
        print(f"  Train annotations: {self.dataset_info.get('train_annotations', 'N/A')}")
        print(f"  Val annotations: {self.dataset_info.get('val_annotations', 'N/A')}")
        print(f"  Test annotations: {self.dataset_info.get('test_annotations', 'N/A')}")
    
    def train(self):
        """Train the YOLO detector."""
        print(f"\n{'='*80}")
        print(f"STARTING TRAINING")
        print(f"{'='*80}\n")
        
        # Load pretrained YOLOv8 model
        model_name = f"yolov8{self.model_size}.pt"
        print(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)
        
        # Train the model
        results = model.train(
            data=str(self.dataset_config),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=self.project,
            name=self.name,
            device=self.device,
            
            # Training settings
            patience=20,  # Early stopping patience
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            
            # Optimization
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Other
            verbose=True,
            plots=True,
            val=True,
        )
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}\n")
        
        # Get best model path
        best_model_path = Path(self.project) / self.name / "weights" / "best.pt"
        last_model_path = Path(self.project) / self.name / "weights" / "last.pt"
        
        print(f"Best model: {best_model_path}")
        print(f"Last model: {last_model_path}")
        print(f"\nResults directory: {Path(self.project) / self.name}")
        
        return results, best_model_path
    
    def validate(self, model_path: Path):
        """Validate the trained model."""
        print(f"\n{'='*80}")
        print(f"VALIDATION ON TEST SET")
        print(f"{'='*80}\n")
        
        model = YOLO(str(model_path))
        
        # Validate on test set
        metrics = model.val(
            data=str(self.dataset_config),
            split='test',
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device,
            plots=True,
            save_json=True,
        )
        
        print(f"\n{'='*80}")
        print(f"TEST SET METRICS")
        print(f"{'='*80}\n")
        
        # Print metrics
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
        print(f"mAP@50: {metrics.box.map50:.4f}")
        print(f"mAP@50:95: {metrics.box.map:.4f}")
        
        return metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO cell detector")
    parser.add_argument("--dataset", type=str, default="detector/yolo_dataset/dataset.yaml",
                       help="Path to dataset.yaml")
    parser.add_argument("--model-size", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--project", type=str, default="detector/runs/train",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="cell_detector",
                       help="Experiment name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip validation after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CellDetectorTrainer(
        dataset_config=args.dataset,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    
    # Train
    results, best_model_path = trainer.train()
    
    # Validate on test set
    if not args.skip_validation:
        metrics = trainer.validate(best_model_path)
        
        print(f"\n{'='*80}")
        print(f"TRAINING AND VALIDATION COMPLETE")
        print(f"{'='*80}\n")
        print(f"Best model: {best_model_path}")
        print(f"Ready for evaluation and inference.")


if __name__ == "__main__":
    main()
