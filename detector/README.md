# Cell Object Detector Training Pipeline

## Overview

This module trains a **YOLO-based object detector** to detect individual cells in bone marrow microscopy images. The detector outputs bounding boxes only; morphology classification remains a separate downstream task using the existing ViT classifier.

## Critical Requirements ✅

- ✅ Uses **ground-truth consensus annotations** (20,852 cells)
- ✅ Preserves **patient-level splits** (79 train / 17 val / 17 test)
- ✅ **Zero patient overlap** across splits (verified programmatically)
- ✅ **Single detection class**: "cell" (no morphology classes in detector)
- ✅ Coordinates from **original WSI/DICOM annotations**
- ✅ Does **NOT** use heuristic detector outputs as labels

## Pipeline Architecture

```
Uploaded Image
     ↓
Trained YOLO Cell Detector
     ↓
Individual Cell Bounding Boxes
     ↓
224×224 Cell Crops
     ↓
Existing ViT-Small/16 + LoRA
     ↓
Morphology Classification
```

## Project Data

| Split | Patients | Valid Annotations |
|-------|----------|-------------------|
| Train | 79       | 14,563           |
| Val   | 17       | 2,951            |
| Test  | 17       | 3,338            |
| **Total** | **113** | **20,852** |

**Source:** `data/processed/cell_metadata.csv`

**Patient Manifest:** `data/patient_manifest.csv`

## Dataset Preparation

### 1. Prepare YOLO Dataset

```bash
python detector/prepare_dataset.py
```

**What it does:**
- Loads ground-truth annotations from `cell_metadata.csv`
- Filters for `extraction_status == 'valid'`
- Groups annotations by source image (patient + series UID)
- Converts bounding boxes to YOLO format: `(class_id, x_center, y_center, width, height)` normalized [0, 1]
- Creates train/val/test splits preserving patient assignments
- Verifies zero patient overlap across splits
- Outputs YOLO dataset structure:
  ```
  detector/yolo_dataset/
  ├── dataset.yaml
  ├── metadata_summary.txt
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
  ```

**Verification:**
- Prints train/val/test patient counts
- Asserts intersection(train, val) == ∅
- Asserts intersection(train, test) == ∅
- Asserts intersection(val, test) == ∅

### 2. Train Detector

```bash
python detector/train_detector.py \
  --model-size n \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640
```

**Arguments:**
- `--model-size`: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--device`: cuda or cpu (auto-detected)

**What it does:**
- Loads pretrained YOLOv8 checkpoint
- Fine-tunes on bone marrow cell detection dataset
- Single detection class: "cell" (class_id = 0)
- Applies augmentation (HSV, rotation, flip, mosaic)
- Uses AdamW optimizer with warmup
- Early stopping with patience=20
- Saves best and last checkpoints

**Output:**
```
detector/runs/train/cell_detector/
├── weights/
│   ├── best.pt        # Best model (use this)
│   └── last.pt        # Last epoch
├── results.png        # Training curves
├── confusion_matrix.png
├── PR_curve.png
└── ...
```

### 3. Evaluate Detector

```bash
python detector/evaluate_detector.py \
  --model detector/runs/train/cell_detector/weights/best.pt \
  --conf 0.25 \
  --iou 0.5
```

**Metrics Computed:**
- ✅ **Precision**: What fraction of predictions are correct?
- ✅ **Recall**: What fraction of ground-truth cells are detected? **(CRITICAL)**
- ✅ **mAP@50**: Mean Average Precision at IoU=0.50
- ✅ **mAP@50:95**: Mean Average Precision at IoU=0.50:0.95
- ✅ **True Positives**: Correctly detected cells
- ✅ **False Positives**: Incorrect detections (background, artifacts)
- ✅ **False Negatives**: Missed cells

**Output:**
```
detector/evaluation/
├── evaluation_results.json     # Metrics in JSON
├── evaluation_results.txt      # Human-readable summary
└── metrics_summary.png         # Visualization
```

**Recall is Critical:**
The previous heuristic detector had **low recall** (missed most cells). The trained detector must achieve high recall to be useful.

### 4. Visual Validation (MANDATORY)

```bash
python detector/visualize_predictions.py \
  --model detector/runs/train/cell_detector/weights/best.pt
```

**MANDATORY OUTPUT:**

#### 50-Crop Contact Sheet

A grid of **50 randomly sampled predicted cell crops** with:
- Cell ID
- Detection confidence
- Bounding box dimensions (width × height)

**Visual Acceptance Criteria:**

Each crop must contain **ONE actual individual cell**.

**REJECT if crops contain:**
- ❌ Background only
- ❌ Staining artifacts
- ❌ Tiny texture fragments
- ❌ Image edges
- ❌ Partial/unrelated regions
- ❌ Multiple cells merged into one box

**Location:**
```
detector/visualizations/
└── contact_sheet_50_crops.png  # REVIEW THIS FILE
```

**DO NOT proceed to deployment unless visual validation passes.**

### 5. Inference on New Images

```bash
python detector/infer_detector.py \
  --detector detector/runs/train/cell_detector/weights/best.pt \
  --vit path/to/vit_model.pth \
  --image path/to/image.png
```

**Pipeline:**
1. Load image
2. Run trained YOLO detector → bounding boxes
3. Crop each detected cell → 224×224
4. Apply ViT preprocessing (SAME as training):
   - Resize to 224×224 (bicubic interpolation)
   - Convert to tensor
   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
5. Run ViT-Small/16 + LoRA → morphology predictions
6. Aggregate cell counts → patient-level assessment

**Critical Preprocessing Verification:**
- ✅ RGB order (not BGR)
- ✅ Resize interpolation method (bicubic)
- ✅ Normalization values (ImageNet stats)
- ✅ Tensor shape: (batch, 3, 224, 224)
- ✅ Class index mapping matches training

## Success Criteria

The detector is **READY FOR DEPLOYMENT** only if:

1. ✅ Trained using real ground-truth annotations
2. ✅ Patient splits preserved (79/17/17)
3. ✅ Test patients completely unseen during training
4. ✅ **Recall > 0.80** (detects at least 80% of cells)
5. ✅ **Precision > 0.70** (at least 70% of predictions are valid)
6. ✅ Ground-truth vs predicted boxes visualized
7. ✅ **50-crop contact sheet visually inspected**
8. ✅ **Crops contain actual individual cells** (not artifacts/fragments)
9. ✅ Duplicate/overlapping detections controlled (NMS applied)
10. ✅ ViT preprocessing verified and preserved

## File Structure

```
detector/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── prepare_dataset.py             # Convert annotations to YOLO format
├── train_detector.py              # Train YOLO detector
├── evaluate_detector.py           # Compute metrics on test set
├── visualize_predictions.py       # Visual validation (50-crop contact sheet)
├── infer_detector.py              # Full inference pipeline
├── yolo_dataset/                  # Generated dataset
│   ├── dataset.yaml
│   ├── metadata_summary.txt
│   ├── train/
│   ├── val/
│   └── test/
├── runs/                          # Training runs
│   └── train/
│       └── cell_detector/
│           └── weights/
│               └── best.pt        # Trained model
├── evaluation/                    # Evaluation results
│   ├── evaluation_results.json
│   ├── evaluation_results.txt
│   └── metrics_summary.png
└── visualizations/                # Visual validation
    └── contact_sheet_50_crops.png # MANDATORY REVIEW
```

## Common Issues

### 1. "No cells detected"
- **Cause**: Confidence threshold too high
- **Fix**: Lower `--conf` threshold (try 0.1 - 0.25)

### 2. "Too many false positives"
- **Cause**: Confidence threshold too low, or detector not trained enough
- **Fix**: Increase `--conf` threshold or train longer

### 3. "Low recall"
- **Cause**: Detector missing cells
- **Fix**: 
  - Train longer (increase epochs)
  - Use larger model size (try 's' or 'm')
  - Increase augmentation
  - Check if ground-truth annotations are correct

### 4. "Crops contain background/artifacts"
- **Cause**: False positive detections
- **Fix**: Increase confidence threshold, retrain with more data

### 5. "Multiple cells in one box"
- **Cause**: NMS threshold too high or cells too close
- **Fix**: Decrease `--iou` threshold for NMS

## Coordinate Systems (IMPORTANT)

### Ground Truth Coordinates
- **Source**: DICOM consensus annotations
- **Format**: Absolute pixel coordinates (x1, y1, x2, y2)
- **Reference**: Original WSI coordinate space
- **Stored in**: `cell_metadata.csv` columns: `x1`, `y1`, `x2`, `y2`

### YOLO Format
- **Format**: (class_id, x_center, y_center, width, height)
- **Normalized**: All values in [0, 1] relative to image dimensions
- **Conversion**: Applied in `prepare_dataset.py`

### Important Notes
- ⚠️ Do NOT mix coordinate systems
- ⚠️ Ground truth uses original WSI coordinates (may be large numbers)
- ⚠️ YOLO training uses normalized coordinates [0, 1]
- ⚠️ Inference outputs absolute pixel coordinates
- ⚠️ Document any coordinate transformations explicitly

## DO NOT

- ❌ Use heuristic detector outputs as training labels
- ❌ Use predicted boxes to generate training data
- ❌ Allow patient overlap across train/val/test
- ❌ Add morphology classes to the detector
- ❌ Claim success based on mAP alone (visual validation required)
- ❌ Modify ViT preprocessing without retraining ViT
- ❌ Deploy detector without inspecting 50-crop contact sheet

## Next Steps After Validation

1. ✅ Detector passes visual validation
2. ✅ Replace heuristic detector in `app/streamlit_app.py`
3. ✅ Update inference pipeline to use trained detector
4. ✅ Verify ViT preprocessing matches training
5. ✅ Test on fresh patient images
6. ✅ Compare morphology distributions: old vs new detector
7. ✅ Ensure agent uses consistent classification counts

## References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Cell Metadata**: `data/processed/cell_metadata.csv`
- **Patient Manifest**: `data/patient_manifest.csv`
- **ViT Training Config**: `configs/default.yaml`

## Support

For issues or questions:
1. Check this README
2. Verify patient splits are preserved
3. Inspect 50-crop contact sheet
4. Ensure coordinates are from ground truth (not heuristic detector)
