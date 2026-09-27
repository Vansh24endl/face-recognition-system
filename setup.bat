@echo off
echo ========================================================
echo    Face Recognition System - Windows Setup Script
echo ========================================================
echo.

IF NOT EXIST venv (
    echo Creating virtual environment 'venv'...
    python -m venv venv
) ELSE (
    echo Virtual environment 'venv' already exists.
)

echo.
echo Installing requirements inside venv...
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Creating required project directories...
if not exist dataset mkdir dataset
if not exist trainer mkdir trainer
if not exist data mkdir data

echo.
echo Setup completed successfully!
echo To run the application, run: .\run.bat
echo ========================================================
pause
