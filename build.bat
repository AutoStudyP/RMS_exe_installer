@echo off
REM ============================================================
REM  SNR Meter - Windows .exe Builder
REM  Usage: Double-click this file OR run from cmd.exe
REM ============================================================

echo.
echo  SNR Meter - Build Script
echo  ========================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

REM Install dependencies
echo [1/3] Installing dependencies...
pip install PyQt5 sounddevice numpy pyinstaller

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/3] Building .exe (this may take 1-2 minutes)...
pyinstaller build_exe.spec --clean --noconfirm

if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo.
echo  Output location: dist\SNR_Meter.exe
echo  File size:       (see above)
echo.
echo  You can copy SNR_Meter.exe ANYWHERE - no install needed!
echo  Just double-click to run.
echo.

REM Open the dist folder
explorer dist

pause
