# 📡 SNR Meter — Signal-to-Noise Ratio Analyzer

Dark-themed PyQt5 desktop app that measures SNR via **2-stage microphone recording**.

## 사용 방법 (Usage)

### Step 1 — Signal 녹음
- 마이크 선택 후 **⏺ Record Signal** 클릭
- 측정할 소리(목소리, 음악, 테스트톤 등) 발생
- **⏹ Stop** 클릭

### Step 2 — Noise 녹음
- **⏺ Record Noise** 클릭
- 배경 노이즈만 녹음 (조용히 있거나 팬 소음만)
- **⏹ Stop** 클릭

### 결과
- SNR 게이지 표시 (dB)
- Signal/Noise 파형 비교
- 주파수 스펙트럼 오버레이
- 품질 등급: Excellent / Good / Acceptable / Poor / Very Poor

---

## 빌드 방법

### Windows (.exe 단독 실행파일)
```bat
pip install -r requirements.txt
build.bat
```
→ `dist/SNR_Meter.exe` 생성 — 어디든 복사해서 실행 가능!

### macOS / Linux
```bash
pip install -r requirements.txt
chmod +x build.sh && ./build.sh
```
→ `dist/SNR_Meter` 실행파일 생성

---

## 직접 실행 (개발용)
```bash
pip install -r requirements.txt
python run_app.py
```

---

## SNR 품질 기준

| SNR (dB) | 등급 | 설명 |
|----------|------|------|
| ≥ 40 dB  | Excellent 🟢 | 방송 품질 |
| 25–40 dB | Good 🟢 | 전화 통화 품질 |
| 15–25 dB | Acceptable 🟡 | 허용 가능 |
| 5–15 dB  | Poor 🟠 | 노이즈가 심함 |
| < 5 dB   | Very Poor 🔴 | 신호 거의 안 들림 |

---

## 파일 구조

```
SNR_Ratio/
├── snr_meter/
│   ├── __init__.py
│   ├── audio_engine.py    # 녹음 + SNR 계산
│   ├── widgets.py         # 커스텀 Qt 위젯 (파형, 스펙트럼, 게이지)
│   └── main_window.py     # 메인 앱 윈도우
├── run_app.py             # 실행 진입점
├── build_exe.spec         # PyInstaller 스펙
├── build.bat              # Windows 빌드 스크립트
├── build.sh               # macOS/Linux 빌드 스크립트
└── requirements.txt
```

## 의존성

- `PyQt5` — GUI 프레임워크
- `sounddevice` — 오디오 녹음
- `numpy` — 신호 처리 / SNR 계산
- `pyinstaller` — .exe 빌드용 (개발 시에만)
