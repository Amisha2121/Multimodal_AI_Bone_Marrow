# Calibration Quick Start

**⚡ 5-Minute Guide to Temperature Scaling**

---

## What is Calibration?

Makes your model's confidence scores **reliable**:
- 90% confidence → Actually correct 90% of the time ✅
- Without calibration → 90% confidence might be correct only 70% ⚠️

---

## Quick Commands

### 1. After Training (Automatic)

```bash
# Calibration runs automatically
python vision/train.py
```

Output: `models/checkpoints/calibration/` folder

### 2. Standalone Calibration

```bash
# Run on existing checkpoint
python scripts/calibrate_model.py --checkpoint models/checkpoints/best_vit_lora_morphology.pt
```

### 3. Use Calibrated Model

```python
from vision.calibration import CalibratedBoneMarrowViT

model = CalibratedBoneMarrowViT.from_checkpoint(
    model_path="models/checkpoints/best_vit_lora_morphology.pt",
    temp_path="models/calibration/temperature_layer.pt",
    device="cuda"
)

# Get calibrated probabilities
probs = model.predict_proba(images)  # ✓ Calibrated
```

---

## Check Results

```bash
# View report
cat models/calibration/calibration_report.json

# Key metrics:
#   - temperature: Should be 1.0-2.5
#   - ece_reduction_pct: Should be 40-70%
```

---

## Interpret Output

```json
{
  "temperature": 1.45,
  "before": { "ece": 0.084 },
  "after": { "ece": 0.032 },
  "improvement": { "ece_reduction_pct": 61.9 }
}
```

✅ **Good**: ECE reduced by 62%  
✅ **Good**: Temperature reasonable (1.45)  
✅ **Good**: Final ECE < 0.05 (excellent calibration)

---

## When to Recalibrate

- ✅ After every model training
- ✅ When using model on new dataset
- ✅ If ECE > 0.10 on validation set
- ❌ NOT needed if same model + same data

---

## Full Guide

See: `docs/CALIBRATION_GUIDE.md` (300+ lines)

**Quick test**: `python tests/test_calibration.py`
