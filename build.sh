#!/usr/bin/env bash
# ============================================================
#  SNR Meter - macOS/Linux .exe-equivalent Builder
# ============================================================
set -e

echo ""
echo "  SNR Meter - Build Script (macOS/Linux)"
echo "  ======================================="
echo ""

# Install deps
echo "[1/3] Installing dependencies..."
pip install PyQt5 sounddevice numpy pyinstaller

echo ""
echo "[2/3] Building binary..."
pyinstaller build_exe.spec --clean --noconfirm

echo ""
echo "[3/3] Build complete!"
echo ""
echo "  Output: dist/SNR_Meter"
echo "  Run:    ./dist/SNR_Meter"
echo ""
