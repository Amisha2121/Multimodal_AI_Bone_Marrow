@echo off
REM ==============================================================================
REM BoneMarrowWSI-PediatricLeukemia: Windows GPU Pipeline Runner
REM Compatible with Windows Command Prompt (cmd.exe) and PowerShell
REM ==============================================================================

echo ========================================================================
echo  1. GPU & CUDA VERIFICATION
echo ========================================================================

where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not in PATH. Checking PyTorch directly...
) else (
    nvidia-smi
)

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Install CUDA build with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)')"

echo.
echo ========================================================================
echo  2. VERIFYING ANONYMOUS IDC S3 CONNECTIVITY
echo ========================================================================

python -c "import upath, json; from wsidicom import WsiDicom; s3_map = json.load(open('data/processed/patient_wsi_s3_map.json')); test_pid = 'A8F15EDA80C50ADB0E71943ADC8015CF'; s3_url = s3_map[test_pid]['s3_url']; print('Connecting to S3:', s3_url); slide = WsiDicom.open(upath.UPath(s3_url, anon=True)); print('[OK] Slide opened anonymously! Levels:', len(slide.levels), 'Size:', slide.size); slide.close()"
if %errorlevel% neq 0 (
    echo [ERROR] S3 connection test failed. Halting.
    exit /b 1
)

echo.
echo ========================================================================
echo  3. RUNNING MASTER PIPELINE (Steps 2 through 6)
echo ========================================================================

python scripts/run_all.py --start-step 2
if %errorlevel% neq 0 (
    echo [ERROR] Pipeline encountered an error.
    exit /b %errorlevel%
)

echo.
echo ========================================================================
echo  ALL STEPS COMPLETE!
echo ========================================================================
pause
