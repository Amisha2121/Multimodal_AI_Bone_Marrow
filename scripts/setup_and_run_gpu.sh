#!/bin/bash
# ==============================================================================
# BoneMarrowWSI-PediatricLeukemia: GPU End-to-End Pipeline
# Reproducible execution on Linux / Colab / Kaggle / AWS EC2 / RunPod
# Python: 3.11 or 3.12 (Recommended) with CUDA-enabled PyTorch
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo " 1. SYSTEM & GPU ENVIRONMENT VERIFICATION"
echo "========================================================================"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "[ERROR] nvidia-smi not found. Ensure an NVIDIA GPU is attached."
    exit 1
fi

python3 -c "
import sys
print(f'Python version: {sys.version}')
assert sys.version_info >= (3, 10), 'Python 3.10+ required'
assert sys.version_info < (3, 14), 'Python 3.11 or 3.12 strongly recommended'

import torch
print(f'PyTorch version: {torch.__version__}')
if not torch.cuda.is_available():
    print('[ERROR] CUDA is not available in PyTorch. Halting.')
    sys.exit(1)
print(f'CUDA available: True')
print(f'Device count:   {torch.cuda.device_count()}')
print(f'GPU device:     {torch.cuda.get_device_name(0)}')
print(f'CUDA arch list: {torch.cuda.get_arch_list() if hasattr(torch.cuda, \"get_arch_list\") else \"N/A\"}')
"

echo "========================================================================"
echo " 2. INSTALLING VERIFIED DEPENDENCIES"
echo "========================================================================"

pip install --upgrade pip
pip install \
    wsidicom \
    upath \
    s3fs \
    timm \
    peft \
    omegaconf \
    pydicom \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    pillow

echo "========================================================================"
echo " 3. VERIFYING ANONYMOUS IDC S3 ACCESS"
echo "========================================================================"

python3 -c "
import upath
from wsidicom import WsiDicom
import json

s3_map_path = 'data/processed/patient_wsi_s3_map.json'
with open(s3_map_path) as f:
    s3_map = json.load(f)

test_pid = 'A8F15EDA80C50ADB0E71943ADC8015CF'
s3_url = s3_map[test_pid]['s3_url']
print(f'Testing anonymous IDC S3 connectivity for patient {test_pid}...')
print(f'URL: {s3_url}')

path = upath.UPath(s3_url, anon=True)
slide = WsiDicom.open(path)
print(f'SUCCESS: Slide opened via anonymous S3. Levels: {len(slide.levels)}, Size: {slide.size}')
slide.close()
"

echo "========================================================================"
echo " 4. STEP 2: RE-EXTRACT ALL 20,852 REAL CROPS FROM IDC S3"
echo "========================================================================"

python3 scripts/step2_extract_all_crops.py

echo "========================================================================"
echo " 5. STEP 3: STRICT DATASET INTEGRITY VERIFICATION"
echo "========================================================================"

python3 scripts/step3_verify_crops.py

echo "========================================================================"
echo " 6. STEP 4: TRAIN VIT-SMALL/16 + LORA CELL CLASSIFIER"
echo "========================================================================"

python3 scripts/step4_train_cell_classifier.py

echo "========================================================================"
echo " 7. STEP 5: CELL-LEVEL EVALUATION & ATTENTION ROLLOUT XAI"
echo "========================================================================"

python3 scripts/step5_evaluate_and_xai.py

echo "========================================================================"
echo " 8. STEP 6: PATIENT-LEVEL 4-CLASS BASELINE & DOC GENERATION"
echo "========================================================================"

python3 scripts/step6_patient_baseline.py

echo "========================================================================"
echo " PIPELINE COMPLETE! Verification & Experiment Results:"
echo "   - docs/CROP_RESTORATION_AUDIT.md"
echo "   - docs/EXPERIMENT_002_RESULTS.md"
echo "========================================================================"
