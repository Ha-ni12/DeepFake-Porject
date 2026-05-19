@echo off
REM ── AI Deepfake Interaction System — Startup Script ──
REM Run this from the project root: start_server.bat

cd /d "%~dp0"

echo =============================================================
echo  AI-Based Deepfake Interaction System  ^|  CENG 384
echo =============================================================
echo.

REM Use the venv Python explicitly so missing packages don't silently kill the window
set "PYEXE=%~dp0venv311\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] venv311 not found at "%PYEXE%".
    echo Create it with:  python -m venv venv311  ^&^&  venv311\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM 1. Generate placeholder templates if they don't exist
if not exist "backend\dsp_models\templates\profile_1.jpg" (
    echo [SETUP] Generating placeholder face templates...
    "%PYEXE%" generate_templates.py
    echo.
)

REM 2. Start FastAPI backend
echo [SERVER] Starting backend on http://127.0.0.1:8000
echo [SERVER] First launch downloads XTTS v2 (~1.9 GB) — be patient.
echo [SERVER] Press Ctrl+C to stop.
echo.
"%PYEXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo.
echo [SERVER] Process exited with code %errorlevel%.
pause
