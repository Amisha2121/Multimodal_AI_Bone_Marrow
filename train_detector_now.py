#!/usr/bin/env python3
"""
Quick training script - starts YOLOv8 training immediately without prompts.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

# Configuration
MODEL_SIZE = "s"  # Small model (good balance of speed and accuracy)
EPOCHS = 150
BATCH_SIZE = 16
IMG_SIZE = 1024
DATASET_CONFIG = "detector/yolo_dataset/dataset.yaml"
PROJECT_NAME = "detector/runs/train"
RUN_NAME = "cell_detector_pediatric"

print("="*80)
print("STARTING YOLOV8 CELL DETECTOR TRAINING")
print("="*80)
print(f"\nConfiguration:")
print(f"  Model: YOLOv8-{MODEL_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Dataset: {DATASET_CONFIG}")
print(f"  Training on: Pediatric bone marrow (14,563 annotations)")
print(f"\nExpected training time: 1.5-3 hours (GPU) or 12-24 hours (CPU)")
print("="*80 + "\n")

# Verify dataset exists
if not Path(DATASET_CONFIG).exists():
    print(f"ERROR: Dataset config not found at {DATASET_CONFIG}")
    print("Run: python detector/prepare_dataset.py first")
    sys.exit(1)

# Initialize YOLO model
print(f"Initializing YOLOv8{MODEL_SIZE} model...")
model = YOLO(f"yolov8{MODEL_SIZE}.pt")  # Load pretrained weights

# Start training
print(f"\n🚀 Starting training (this will take 1-3 hours)...")
print("Monitor progress below. Training will save checkpoints automatically.\n")

try:
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        patience=20,  # Early stopping if no improvement
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        device="0" if sys.platform != "darwin" else "mps",  # Use GPU if available
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,  # Box loss weight
        cls=0.5,  # Classification loss weight
        dfl=1.5,  # Distribution focal loss weight
        plots=True,  # Save training plots
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest model saved to: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"Training results: {PROJECT_NAME}/{RUN_NAME}/results.png")
    print(f"\nFinal metrics:")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print(f"  mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    print("\n📊 Next steps:")
    print("1. Evaluate on test set:")
    print(f"   python detector/evaluate_detector.py --model {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print("\n2. Visualize predictions:")
    print(f"   python detector/visualize_predictions.py --model {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print("\n3. Deploy to production:")
    print(f"   cp {PROJECT_NAME}/{RUN_NAME}/weights/best.pt models/detector_yolov8s_pediatric.pt")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user.")
    print("Partial model saved. You can resume training later.")
except Exception as e:
    print(f"\n\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
