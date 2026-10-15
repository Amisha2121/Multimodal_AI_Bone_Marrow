# ==============================================================================
# BoneMarrowWSI-PediatricLeukemia: Windows PowerShell GPU Pipeline Runner
# Compatible with Windows PowerShell 5.1 and PowerShell 7+
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " 1. SYSTEM & GPU ENVIRONMENT VERIFICATION" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi
} else {
    Write-Host "[WARNING] nvidia-smi command not found." -ForegroundColor Yellow
}

python -c "
import sys, torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: PyTorch is currently CPU-only.')
    print('To install CUDA PyTorch on Windows:')
    print('  pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128')
"

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host " 2. VERIFYING ANONYMOUS IDC S3 CONNECTIVITY" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

python -c "
import upath, json
from wsidicom import WsiDicom

with open('data/processed/patient_wsi_s3_map.json') as f:
    s3_map = json.load(f)

test_pid = 'A8F15EDA80C50ADB0E71943ADC8015CF'
s3_url = s3_map[test_pid]['s3_url']
print(f'Testing S3 connectivity for {test_pid}...')
slide = WsiDicom.open(upath.UPath(s3_url, anon=True))
print(f'[OK] Slide opened anonymously! Levels: {len(slide.levels)}, Size: {slide.size}')
slide.close()
"

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host " 3. EXECUTING MASTER PIPELINE (Steps 2 through 6)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

python scripts/run_all.py --start-step 2

Write-Host "`n========================================================================" -ForegroundColor Green
Write-Host " PIPELINE FINISHED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
