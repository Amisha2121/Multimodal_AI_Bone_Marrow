# Temperature Scaling Calibration Guide

## Overview

**Temperature scaling** is a post-hoc calibration method that improves the reliability of model confidence scores. For clinical AI systems, calibrated confidence is critical: a prediction with 90% confidence should be correct 90% of the time.

### Why Calibration Matters for Bone Marrow Analysis

1. **Clinical Trust**: Pathologists need reliable confidence scores to decide when AI predictions are trustworthy
2. **Downstream Decision-Making**: The Critic agent uses confidence scores to approve/reject reports
3. **FDA Approval**: Medical device submissions require calibration evidence
4. **Safety**: Prevents overconfident wrong predictions that could mislead diagnosis

### Literature Foundation

- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks" - foundational paper
- **Minderer et al. (2021)**: "Revisiting the Calibration of Modern Neural Networks"
- **Naeini et al. (2015)**: Bayesian Binning Into Quantiles (BBQ)

Reference: Section 7.1 of LITERATURE_SURVEY_CONTEXT.md

---

## How Temperature Scaling Works

### The Problem
Modern neural networks (including Vision Transformers) tend to be **overconfident** - they produce high softmax probabilities even when wrong.

### The Solution
Temperature scaling adjusts the softmax temperature to calibrate probabilities:

```
p_calibrated = softmax(logits / T)
```

where `T > 1` reduces confidence (flattens distribution), `T < 1` increases confidence (sharpens distribution).

### Key Properties
- **Preserves accuracy**: `argmax` remains unchanged
- **Single parameter**: Only `T` is learned (fast optimization)
- **Post-hoc**: Applied after training, no retraining needed
- **Convex optimization**: Guaranteed convergence

---

## Metrics

### Expected Calibration Error (ECE)

Measures average difference between confidence and accuracy:

```
ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
```

**Interpretation**:
- ECE = 0.00: Perfect calibration
- ECE < 0.05: Excellent calibration
- ECE < 0.10: Good calibration
- ECE > 0.15: Poor calibration

### Maximum Calibration Error (MCE)

Maximum calibration error across all bins (worst-case metric).

### Reliability Diagram

Visual representation of calibration:
- **X-axis**: Confidence bins
- **Y-axis**: Accuracy in each bin
- **Diagonal line**: Perfect calibration
- **Gap**: Red vertical lines show miscalibration

---

## Usage

### 1. Automatic Calibration During Training

By default, calibration runs automatically after training:

```bash
python vision/train.py --config configs/default.yaml
```

Output:
- `models/checkpoints/calibration/temperature_layer.pt` - Optimized temperature
- `models/checkpoints/calibration/calibration_report.json` - Metrics
- `models/checkpoints/calibration/reliability_*.png` - Diagrams

To skip calibration:
```bash
python vision/train.py --skip-calibration
```

### 2. Standalone Calibration Script

Run calibration on an already-trained model:

```bash
python scripts/calibrate_model.py \
    --checkpoint models/checkpoints/best_vit_lora_morphology.pt \
    --output-dir models/calibration \
    --n-bins 15
```

### 3. Using Calibrated Model for Inference

**Option A: Wrapper Class (Recommended)**

```python
from vision.calibration import CalibratedBoneMarrowViT

# Load calibrated model
model = CalibratedBoneMarrowViT.from_checkpoint(
    model_path="models/checkpoints/best_vit_lora_morphology.pt",
    temp_path="models/calibration/temperature_layer.pt",
    device="cuda"
)

# Get calibrated probabilities
with torch.no_grad():
    probs = model.predict_proba(images)  # Calibrated
    preds = torch.argmax(probs, dim=1)
```

**Option B: Manual Application**

```python
from vision.model import BoneMarrowViT
from vision.calibration import TemperatureScalingLayer

# Load base model
model = BoneMarrowViT.load_checkpoint("models/checkpoints/best_vit_lora_morphology.pt")

# Load temperature
temp_layer = TemperatureScalingLayer()
temp_layer.load_state_dict(torch.load("models/calibration/temperature_layer.pt"))

# Inference
logits = model(images)
calibrated_logits = temp_layer(logits)
probs = torch.softmax(calibrated_logits, dim=1)
```

---

## Integration with Pipeline

### Update Cell Classification Pipeline

Modify `pipeline/cell_detector.py` to use calibrated probabilities:

```python
# Before
from vision.model import BoneMarrowViT
model = BoneMarrowViT.load_checkpoint(model_path)

# After
from vision.calibration import CalibratedBoneMarrowViT
model = CalibratedBoneMarrowViT.from_checkpoint(
    model_path=model_path,
    temp_path=model_path.parent / "calibration" / "temperature_layer.pt"
)
```

### Update Confidence Thresholding

Calibrated probabilities are more reliable, so you may want to adjust thresholds:

```python
# Before calibration (conservative)
HIGH_CONFIDENCE_THRESHOLD = 0.70

# After calibration (can be more trusting)
HIGH_CONFIDENCE_THRESHOLD = 0.60
```

---

## Interpreting Results

### Example Output

```
CALIBRATION SUMMARY
======================================================================
Temperature: 1.4523
Samples:     3,842
Bins:        15

Metrics Comparison:
  Metric               Before      After     Change
  ----------------------------------------------------------
  Accuracy             0.8214     0.8214     0.0000
  ECE                  0.0842     0.0321    -0.0521
  MCE                  0.1523     0.0687    -0.0836
  Avg Confidence       0.8912     0.7543    -0.1369

ECE Reduction: 61.9%
======================================================================
```

**Interpretation**:
- **Temperature = 1.45**: Model was overconfident (T > 1 reduces confidence)
- **Accuracy unchanged**: Calibration doesn't affect predictions, only probabilities
- **ECE reduced 62%**: Major improvement in calibration (0.084 → 0.032)
- **Avg confidence dropped**: From 89% to 75% (more realistic)

### Reliability Diagrams

**Before Calibration**:
- Bars above diagonal → overconfident
- Large red gaps → poor calibration

**After Calibration**:
- Bars closer to diagonal → well-calibrated
- Small red gaps → good calibration

---

## Clinical Impact

### Improved Decision-Making

**Scenario**: Blast cell classification

**Before Calibration**:
- Model predicts "Blast" with 92% confidence
- Actually correct only 78% of the time
- Overconfidence → false sense of certainty

**After Calibration**:
- Model predicts "Blast" with 78% confidence
- Actually correct 78% of the time
- Calibrated → reliable uncertainty quantification

### Critic Agent Integration

The Critic agent uses confidence scores for approval decisions:

```python
# Before: Overly trusting of high confidence
if max_confidence > 0.85:
    approve_report()

# After: Calibrated confidence is reliable
if max_confidence > 0.75:  # Lower threshold, but more meaningful
    approve_report()
```

---

## Troubleshooting

### Issue: ECE Increases After Calibration

**Cause**: Validation set too small or noisy.

**Solution**:
- Use larger validation set (>1000 samples)
- Check for data quality issues
- Try different number of bins (10, 15, 20)

### Issue: Temperature Near 1.0

**Cause**: Model already well-calibrated (good!).

**Solution**: No action needed. Keep temperature layer for consistency.

### Issue: Temperature > 3.0 or < 0.5

**Cause**: Model is severely miscalibrated or validation set is unrepresentative.

**Solution**:
- Review training process (overfitting?)
- Check validation set distribution
- Consider retraining with label smoothing

---

## Advanced Topics

### Bin Selection

- **10 bins**: Smoother, less sensitive to noise
- **15 bins** (default): Good balance
- **20 bins**: More granular, requires larger validation set

### Multi-Class Calibration

Temperature scaling is **class-agnostic** - single T for all classes. For per-class calibration, consider:
- **Vector Scaling**: T per class (more parameters)
- **Matrix Scaling**: Full affine transformation

### Confidence Thresholding

After calibration, use ECE to set confidence thresholds:

```python
# High confidence (likely correct)
if confidence > (1.0 - ece):
    auto_approve()

# Low confidence (likely wrong)
elif confidence < 0.5:
    flag_for_review()
```

---

## References

1. Guo et al. (2017): "On Calibration of Modern Neural Networks"
   - https://arxiv.org/abs/1706.04599

2. Minderer et al. (2021): "Revisiting the Calibration of Modern Neural Networks"
   - https://arxiv.org/abs/2106.07998

3. Naeini et al. (2015): "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
   - https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf

4. Vaicenavicius et al. (2019): "Evaluating Model Calibration in Classification"
   - https://arxiv.org/abs/1902.06977

---

## Checklist for Clinical Deployment

- [ ] ECE < 0.05 on validation set
- [ ] ECE < 0.08 on held-out test set
- [ ] Reliability diagram shows bars near diagonal
- [ ] Per-class calibration checked (no class severely miscalibrated)
- [ ] Temperature reasonable (0.8 - 2.5 range)
- [ ] Pipeline updated to use calibrated probabilities
- [ ] Critic agent thresholds adjusted for calibrated confidence
- [ ] Documentation updated with calibration details

---

## Next Steps

1. ✅ **Run calibration** on your trained model
2. ✅ **Review reliability diagrams** to verify improvement
3. ⬜ **Update pipeline** to use `CalibratedBoneMarrowViT`
4. ⬜ **Adjust Critic thresholds** based on calibrated confidence
5. ⬜ **Document temperature** in model cards and reports
6. ⬜ **Re-evaluate end-to-end pipeline** with calibrated probabilities
7. ⬜ **Prepare FDA submission materials** (if pursuing approval)

For questions or issues, see: `docs/LITERATURE_SURVEY_CONTEXT.md` Section 7.1
