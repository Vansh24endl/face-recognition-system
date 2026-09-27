@echo off
echo ========================================================
echo    Starting Face Recognition System (Streamlit)
echo ========================================================
echo.
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe -m streamlit run app.py
) else (
    echo [ERROR] Virtual environment 'venv' not found. Please run setup.bat first!
)
pause
