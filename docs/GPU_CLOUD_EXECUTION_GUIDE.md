# Cloud GPU Execution Guide: BoneMarrowWSI-PediatricLeukemia Vision Baseline

This guide details the procedure for executing the complete extraction, verification, training, and evaluation pipeline on a cloud GPU instance (Google Colab, Kaggle, AWS EC2, GCP, Lambda Labs, or RunPod).

---

## 1. Environment Specifications

* **Operating System:** Linux (Ubuntu 22.04 LTS recommended)
* **GPU:** NVIDIA GPU with $\ge 12\text{ GB}$ VRAM (T4, V100, A10G, A100, RTX 3090/4090)
* **Python Version:** `3.11` or `3.12` (avoid Python 3.14 until ecosystem build tools match)
* **PyTorch:** $\ge 2.2.0$ with CUDA runtime support (`torch.cuda.is_available() == True`)

---

## 2. One-Line Setup & Execution

From the cloned repository root:

```bash
chmod +x scripts/setup_and_run_gpu.sh
./scripts/setup_and_run_gpu.sh
```

Or using the Python orchestrator:

```bash
python scripts/run_all.py --start-step 2
```

---

## 3. Step-by-Step Execution Sequence

### Step 1: Verify GPU & CUDA
```bash
nvidia-smi

python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA!')
assert torch.cuda.is_available(), 'CUDA is required for GPU training!'
"
```

### Step 2: Install Verified Dependencies
```bash
pip install --upgrade pip
pip install wsidicom upath s3fs timm peft omegaconf pydicom pandas scikit-learn matplotlib seaborn tqdm pillow
```

### Step 3: Verify Anonymous IDC S3 Access
```bash
python3 -c "
import upath, json
from wsidicom import WsiDicom

with open('data/processed/patient_wsi_s3_map.json') as f:
    s3_map = json.load(f)

test_pid = 'A8F15EDA80C50ADB0E71943ADC8015CF'
s3_url = s3_map[test_pid]['s3_url']
print(f'Checking {test_pid} at {s3_url}...')
slide = WsiDicom.open(upath.UPath(s3_url, anon=True))
print(f'[OK] Slide opened anonymously! Levels: {len(slide.levels)}, Base resolution: {slide.size}')
slide.close()
"
```

### Step 4: Re-extract All 20,852 Real Crops
Streams 224x224 RGB crops directly from IDC S3 using bounding box coordinates and 20% margin padding:
```bash
python3 scripts/step2_extract_all_crops.py
```

### Step 5: Verify Dataset Integrity (Mandatory Gate)
Strict audit that verifies 0 missing crops, valid 224x224 RGB dimensions, and zero patient split leakage:
```bash
python3 scripts/step3_verify_crops.py
```
> **CRITICAL GATE:** Do not proceed to Step 6 unless Step 5 reports 0 errors and exits with code 0.

### Step 6: Train ViT-Small/16 + LoRA Classifier
Trains the single-cell morphology classifier on verified real crops with class-imbalance weighting and patient-stratified validation:
```bash
python3 scripts/step4_train_cell_classifier.py
```

### Step 7: Cell Evaluation & Attention Rollout XAI
Evaluates held-out test patients and computes Abnar & Zuidema (2020) Attention Rollout heatmaps:
```bash
python3 scripts/step5_evaluate_and_xai.py
```

### Step 8: Patient 4-Class Baseline & Results Documentation
Aggregates predicted cell probabilities to patient level, trains logistic regression baseline, evaluates on 17 test patients, flags low-evidence cases (<50 cells), and writes `docs/EXPERIMENT_002_RESULTS.md`:
```bash
python3 scripts/step6_patient_baseline.py
```

---

## 4. Expected Final Outputs

1. **Cell Crops:** `data/processed/cell_crops/{train,val,test}/{patient_id}/*.png` (20,852 real images)
2. **Model Weights:** `models/checkpoints/best_vit_lora_morphology.pt`
3. **Training Log:** `models/checkpoints/train_log.jsonl`
4. **Evaluation Metrics:** `models/evaluation/cell_classification_report.csv`
5. **XAI Rollouts:** `models/evaluation/xai/xai_*.png`
6. **Confusion Matrices:**
   - `models/evaluation/cell_confusion_matrix.png`
   - `models/evaluation/patient_confusion_matrix.png`
7. **Audit Report:** `docs/CROP_RESTORATION_AUDIT.md`
8. **Final Experiment Report:** `docs/EXPERIMENT_002_RESULTS.md`
