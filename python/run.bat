@echo off
:: ============================================================
:: run.bat  — CloudTracker Python launcher
:: ============================================================
:: Installs dependencies (if missing) and starts the application.
:: Place this file in the python/ subdirectory alongside main.py.

echo === CloudTracker Python Launcher ===
cd /d "%~dp0"

:: Try to locate python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found on PATH.
    echo Please install Python 3.9+ from https://python.org and add it to PATH.
    pause
    exit /b 1
)

:: Install / upgrade dependencies quietly
echo Installing / verifying dependencies...
python -m pip install -r requirements.txt --quiet --upgrade

:: Launch the application
echo Starting CloudTracker...
python main.py

pause
