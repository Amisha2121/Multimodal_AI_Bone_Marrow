"""
detector/__init__.py
─────────────────────────────────────────────────────────────────────────────
Cell Object Detector Module

Provides YOLO-based cell detection for bone marrow microscopy images.

Training Pipeline:
    1. prepare_dataset.py   - Convert ground-truth annotations to YOLO format
    2. train_detector.py    - Train YOLOv8 detector
    3. evaluate_detector.py - Compute metrics on test set
    4. visualize_predictions.py - Generate 50-crop contact sheet
    5. infer_detector.py    - Production inference pipeline

Quick Start:
    >>> python detector/run_full_pipeline.py --model-size n --epochs 100

See:
    - detector/README.md for comprehensive documentation
    - detector/QUICKSTART.md for quick start guide
"""

__version__ = "1.0.0"
__author__ = "Bone Marrow AI Team"

# Module components
__all__ = [
    "prepare_dataset",
    "train_detector", 
    "evaluate_detector",
    "visualize_predictions",
    "infer_detector",
]
