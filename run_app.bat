@echo off
REM Start HealthBandhu V (Windows). Activate your Conda environment first if you use one.
cd /d "%~dp0"
python -m streamlit run App.py
pause
