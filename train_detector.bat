@echo off
REM train_detector.bat
REM Quick launcher for detector training on Windows
REM
REM Usage:
REM   train_detector.bat              (default: YOLOv8s, 100 epochs)
REM   train_detector.bat n 50         (YOLOv8n, 50 epochs - fast testing)
REM   train_detector.bat m 150        (YOLOv8m, 150 epochs - high quality)

echo ================================================================================
echo CELL DETECTOR TRAINING
echo ================================================================================
echo.

REM Default parameters
set MODEL_SIZE=s
set EPOCHS=100

REM Override if arguments provided
if not "%1"=="" set MODEL_SIZE=%1
if not "%2"=="" set EPOCHS=%2

echo Configuration:
echo   Model size: %MODEL_SIZE%
echo   Epochs: %EPOCHS%
echo.
echo Press Ctrl+C to cancel, or any other key to continue...
pause > nul

echo.
echo Starting training...
echo.

python detector\run_full_pipeline.py --model-size %MODEL_SIZE% --epochs %EPOCHS%

echo.
echo ================================================================================
echo TRAINING COMPLETE
echo ================================================================================
echo.
echo Review results:
echo   1. Metrics: detector\evaluation\evaluation_results.txt
echo   2. Contact sheet: detector\visualizations\contact_sheet_50_crops.png
echo.
echo Next: Inspect the 50-crop contact sheet before deployment!
echo.
pause
