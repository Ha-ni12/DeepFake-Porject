@echo off
REM ── AI Deepfake Interaction System — Startup Script ──
REM Run this from the project root: start_server.bat

cd /d "%~dp0"

REM Force UTF-8 throughout Python so Unicode characters in model output
REM (arrows, checkmarks etc.) don't crash on Windows cp1252 console encoding.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set PYTHONPATH=%~dp0

REM Kill any stale server already on port 8000 (prevents "address in use" crash)
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

echo =============================================================
echo  AI-Based Deepfake Interaction System  ^|  CENG 384
echo =============================================================
echo.

REM Use project-local venv Python; bootstrap automatically on first run
set "PYEXE=%~dp0venv311\Scripts\python.exe"

set "NEEDS_SETUP=0"
if not exist "%PYEXE%" (
    echo [SETUP] venv311 not found.
    set "NEEDS_SETUP=1"
) else (
    "%PYEXE%" --version >nul 2>&1
    if errorlevel 1 (
        echo [SETUP] Existing venv311 appears broken due to path mismatch or missing base Python.
        echo [SETUP] Attempting to repair/recreate the virtual environment...
        set "NEEDS_SETUP=1"
    )
)

if "%NEEDS_SETUP%"=="1" (
    REM --- Find a Python 3.11 executable (kept separate from its args) ---
    set "PYEXEC="

    REM 1. py launcher with explicit version (preferred on Windows)
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXEC=py"
        set "PYVER=-3.11"
        goto :found_py
    )

    REM 2. python3.11 (common on direct installs and Scoop)
    python3.11 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXEC=python3.11"
        set "PYVER="
        goto :found_py
    )

    REM 3. python (works if 3.11 was added to PATH and is the default)
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXEC=python"
        set "PYVER="
        goto :found_py
    )

    REM 4. Common default install path
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYEXEC=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        set "PYVER="
        goto :found_py
    )

    echo.
    echo [ERROR] Python 3.11 was not found on this machine.
    echo [ERROR] Install it from https://www.python.org/downloads/release/python-3119/
    echo [ERROR] During install, CHECK "Add Python to PATH".
    echo [ERROR] Then run this file again.
    echo.
    pause
    exit /b 1

    :found_py
    echo [SETUP] Using Python: %PYEXEC% %PYVER%
    "%PYEXEC%" %PYVER% -m venv venv311
    if errorlevel 1 (
        echo [ERROR] Failed to setup venv311. Make sure Python 3.11 is installed.
        pause
        exit /b 1
    )

    echo [SETUP] Checking and installing dependencies from requirements.txt...
    "%PYEXE%" -m pip install --upgrade pip
    "%PYEXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )

    echo [SETUP] Environment setup complete.
    echo.
)

"%PYEXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Existing venv311 appears broken.
    echo [ERROR] Delete the venv311 folder and run this file again.
    pause
    exit /b 1
)

REM 1. Generate placeholder templates if they don't exist
if not exist "backend\dsp_models\templates\profile_1.jpg" (
    echo [SETUP] Generating placeholder face templates...
    "%PYEXE%" generate_templates.py
    echo.
)

REM 2. Download missing ML models (skipped if already present)
if not exist "backend\dsp_models\simswap_256.onnx" (
    echo [SETUP] Downloading ML models simswap_256, crossface, codeformer, blendswap...
    "%PYEXE%" download_models.py
    echo.
) else if not exist "backend\dsp_models\crossface_simswap.onnx" (
    echo [SETUP] Downloading crossface_simswap.onnx...
    "%PYEXE%" download_models.py
    echo.
) else if not exist "backend\dsp_models\codeformer.onnx" (
    echo [SETUP] Downloading codeformer.onnx...
    "%PYEXE%" download_models.py
    echo.
)

REM 3. Start FastAPI backend
echo [SERVER] Starting backend on http://127.0.0.1:8000
echo [SERVER] First launch downloads F5-TTS model (~1.3 GB) be patient :).
echo [SERVER] Press Ctrl+C to stop.
echo.
"%PYEXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo.
echo [SERVER] Process exited with code %errorlevel%.
pause
