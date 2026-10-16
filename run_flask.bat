@echo off
echo ========================================
echo Starting Bone Marrow Analysis Flask App
echo ========================================
echo.
echo Make sure you have activated your virtual environment first!
echo.
cd /d "%~dp0"
python app\flask_app.py
pause
