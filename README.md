# 📡 SNR Meter

> **Signal-to-Noise Ratio Analyzer** — Measure and visualize your microphone's SNR in seconds.

[![Version](https://img.shields.io/badge/version-0.0.0-blue)](https://github.com/AutoStudyP/RMS_exe_installer/releases)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()

SNR Meter is a **desktop application** that measures the Signal-to-Noise Ratio of your audio environment through **two-stage microphone recording** — first capturing your signal, then the background noise. It provides real-time waveform display, frequency spectrum overlay, and an analog-style SNR gauge.

---

## Features

| Feature | Description |
|---------|-------------|
| **2-Stage Recording** | Record Signal → Record Noise → Auto SNR calculation |
| **Live Waveform** | 30fps real-time scrolling waveform during recording |
| **VU Meter** | Peak-hold level meter for monitoring input level |
| **SNR Arc Gauge** | Analog-style gauge (0 – 50+ dB) with quality color coding |
| **Spectrum Overlay** | Signal (blue) vs Noise (red) FFT frequency comparison |
| **Quality Grading** | Automatic: Excellent / Good / Acceptable / Poor / Very Poor |
| **Device Selection** | Choose any connected microphone from the dropdown |
| **Portable .exe** | Single-file executable — no installation required |

---

## SNR Quality Reference

| SNR | Grade | Typical Use |
|-----|-------|-------------|
| ≥ 40 dB | 🟢 **Excellent** | Broadcast / studio quality |
| 25 – 40 dB | 🟢 **Good** | Phone calls, video conferencing |
| 15 – 25 dB | 🟡 **Acceptable** | Casual recording |
| 5 – 15 dB | 🟠 **Poor** | Heavy background noise |
| < 5 dB | 🔴 **Very Poor** | Signal barely audible |

---

## Installation

### Option 1 — pip (Recommended for Python users)

```bash
pip install snr-meter
snr-meter
```

### Option 2 — Windows .exe (No Python required)

Download **SNR_Meter.exe** from the [Releases](https://github.com/AutoStudyP/RMS_exe_installer/releases) page and double-click to run. No installation needed — copy anywhere.

### Option 3 — Run from source

```bash
git clone https://github.com/AutoStudyP/RMS_exe_installer.git
cd RMS_exe_installer
pip install -r requirements.txt
python run_app.py
```

---

## Build .exe Yourself

### Windows

```bat
pip install -r requirements.txt
build.bat
```

Output: `dist/SNR_Meter.exe` — standalone, portable executable.

### macOS / Linux

```bash
pip install -r requirements.txt
chmod +x build.sh && ./build.sh
```

Output: `dist/SNR_Meter`

---

## How to Use

### Step 1 — Record Signal
1. Select your microphone from the **Input Device** dropdown
2. Click **⏺ Record Signal**
3. Produce the audio you want to measure (speech, music, test tone…)
4. Click **⏹ Stop**

### Step 2 — Record Noise
1. Click **⏺ Record Noise**
2. Stay silent — let the microphone capture only background noise
3. Click **⏹ Stop**

### Result
- SNR gauge updates immediately in dB
- Waveforms and spectrum are displayed side-by-side
- You can re-record either channel without resetting

---

## Project Structure

```
RMS_exe_installer/
├── snr_meter/
│   ├── __init__.py
│   ├── audio_engine.py     # Recording engine + SNR calculation
│   ├── widgets.py          # Custom PyQt5 widgets (waveform, spectrum, gauge)
│   └── main_window.py      # Main application window & state machine
├── run_app.py              # Entry point (source execution)
├── build_exe.spec          # PyInstaller configuration
├── build.bat               # Windows one-click .exe builder
├── build.sh                # macOS/Linux binary builder
├── pyproject.toml          # pip package configuration
└── requirements.txt        # Python dependencies
```

---

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `PyQt5` | GUI framework | ≥ 5.15.0 |
| `sounddevice` | Microphone recording | ≥ 0.4.6 |
| `numpy` | Signal processing & SNR math | ≥ 1.21.0 |
| `pyinstaller` | Build .exe (optional) | ≥ 5.0.0 |

---

## Technical Notes

**SNR Calculation:**
```
SNR (dB) = 10 × log₁₀( P_signal / P_noise )

where P = mean(samples²)  — average power
```

**Spectrum:** FFT with Hanning window, 4096-point DFT

**Sample Rate:** 44,100 Hz, mono, float32

---

## License

MIT © [AutoStudyP](https://github.com/AutoStudyP)
