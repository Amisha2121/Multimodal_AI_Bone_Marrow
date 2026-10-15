# Quick Start Guide: Cell Detector Training

## Prerequisites

```bash
# Install dependencies
pip install -r detector/requirements.txt
```

## Option 1: Run Full Pipeline (Recommended)

```bash
# Run complete pipeline: prepare data → train → evaluate → visualize
python detector/run_full_pipeline.py --model-size n --epochs 100
```

**What happens:**
1. Prepares YOLO dataset from ground-truth annotations
2. Trains YOLOv8-nano detector (100 epochs)
3. Evaluates on held-out test set
4. Generates 50-crop contact sheet for visual validation
5. Reports final metrics and readiness

**Expected time:** 1-3 hours depending on hardware

## Option 2: Step-by-Step

### Step 1: Prepare Dataset

```bash
python detector/prepare_dataset.py
```

**Output:** `detector/yolo_dataset/`

### Step 2: Train Detector

```bash
python detector/train_detector.py \
  --model-size n \
  --epochs 100 \
  --batch-size 16
```

**Output:** `detector/runs/train/cell_detector/weights/best.pt`

### Step 3: Evaluate

```bash
python detector/evaluate_detector.py \
  --model detector/runs/train/cell_detector/weights/best.pt
```

**Output:** `detector/evaluation/evaluation_results.json`

### Step 4: Visualize (MANDATORY)

```bash
python detector/visualize_predictions.py \
  --model detector/runs/train/cell_detector/weights/best.pt
```

**Output:** `detector/visualizations/contact_sheet_50_crops.png`

⚠️ **OPEN AND INSPECT THIS FILE BEFORE DEPLOYMENT**

## Model Size Selection

| Size | Parameters | Speed | Accuracy | Recommended For |
|------|-----------|-------|----------|-----------------|
| n    | ~3M       | Fast  | Good     | Testing, rapid iteration |
| s    | ~11M      | Medium| Better   | Production (recommended) |
| m    | ~25M      | Slow  | Best     | High accuracy required |
| l    | ~43M      | Slower| Excellent| Research, maximum performance |
| x    | ~68M      | Slowest| Best   | Overkill for this task |

**Recommendation:** Start with `n` for testing, use `s` or `m` for production.

## Expected Metrics (Target)

| Metric | Target | Meaning |
|--------|--------|---------|
| Recall | ≥ 0.80 | Detects at least 80% of cells |
| Precision | ≥ 0.70 | At least 70% of predictions are valid |
| mAP@50 | ≥ 0.75 | Overall detection quality |

## Visual Validation Checklist

Open: `detector/visualizations/contact_sheet_50_crops.png`

For each crop:
- [ ] Contains ONE actual cell
- [ ] Cell is centered in crop
- [ ] Cell boundaries are visible
- [ ] Not background only
- [ ] Not staining artifact
- [ ] Not image edge
- [ ] Not multiple merged cells

**PASS:** If ≥90% of crops (45+/50) meet criteria
**FAIL:** If <90% meet criteria → retrain or adjust parameters

## Troubleshooting

### Low Recall (<0.80)
```bash
# Try: Train longer
python detector/run_full_pipeline.py --epochs 200

# Or: Use larger model
python detector/run_full_pipeline.py --model-size s --epochs 150
```

### Low Precision (<0.70)
```bash
# Try: Higher confidence threshold
python detector/evaluate_detector.py --model <model.pt> --conf 0.35
```

### Contact Sheet Shows Artifacts
```bash
# Increase confidence threshold in inference
python detector/infer_detector.py --detector <model.pt> --conf 0.35
```

## Next Steps After Successful Training

1. ✅ Trained model: `detector/runs/train/cell_detector/weights/best.pt`
2. ✅ Metrics: Recall ≥ 0.80, Precision ≥ 0.70
3. ✅ Visual validation: 50-crop contact sheet inspected
4. ✅ Deploy to production:
   - Replace heuristic detector in `app/streamlit_app.py`
   - Update inference pipeline to use trained detector
   - Test on fresh patient images

## Example: Full Training Run

```bash
# Start fresh training
python detector/run_full_pipeline.py \
  --model-size s \
  --epochs 150 \
  --batch-size 16 \
  --img-size 640 \
  --conf 0.25 \
  --iou 0.5

# Expected output:
# [Dataset prepared: 14,563 train, 2,951 val, 3,338 test annotations]
# [Training: 150 epochs, ~2 hours on GPU]
# [Evaluation: Recall=0.85, Precision=0.78, mAP@50=0.82]
# [Visual validation: contact_sheet_50_crops.png generated]
# [Result: DETECTOR PASSED QUANTITATIVE VALIDATION]
```

## GPU vs CPU

**GPU (Recommended):**
- Training time: 1-3 hours
- Inference: Real-time

**CPU:**
- Training time: 12-24 hours
- Inference: Slow but usable

Force CPU:
```bash
python detector/train_detector.py --device cpu
```

## Re-using Existing Model

If you already have a trained model:

```bash
python detector/run_full_pipeline.py \
  --skip-training \
  --model detector/runs/train/cell_detector/weights/best.pt
```

This runs evaluation and visualization only.

## Questions?

See: `detector/README.md` for detailed documentation.
