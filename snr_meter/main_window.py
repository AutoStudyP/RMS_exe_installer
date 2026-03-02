"""
SNR Meter - Main Application Window
2-Stage Recording: Signal → Noise → SNR Result
"""

import sys
import numpy as np
from enum import Enum, auto
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QProgressBar,
    QGroupBox,
    QSplitter,
    QFrame,
    QStatusBar,
    QSizePolicy,
    QSpacerItem,
    QGridLayout,
    QTextEdit,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QColor, QFont, QPalette, QIcon

from .audio_engine import AudioRecorder, AudioData, SNRResult
from .widgets import (
    WaveformWidget,
    SpectrumWidget,
    SNRGaugeWidget,
    VUMeterWidget,
    BG_DARK,
    BG_PANEL,
    BG_WIDGET,
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_GREEN,
    ACCENT_GOLD,
    TEXT_PRIMARY,
    TEXT_MUTED,
    GRID_COLOR,
)


# ──────────────────────────────────────────────
#  App States
# ──────────────────────────────────────────────
class AppState(Enum):
    IDLE = auto()  # Initial state
    REC_SIGNAL = auto()  # Recording signal
    SIGNAL_DONE = auto()  # Signal recorded, ready for noise
    REC_NOISE = auto()  # Recording noise
    RESULT = auto()  # Showing result


# ──────────────────────────────────────────────
#  Stylesheet
# ──────────────────────────────────────────────
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: 'Segoe UI', Arial, sans-serif;
}

QGroupBox {
    border: 1px solid #30363D;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 4px;
    font-size: 11px;
    font-weight: bold;
    color: #7D8590;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QPushButton {
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 13px;
    font-weight: bold;
    border: none;
}
QPushButton#btn_record_signal {
    background-color: #1F6FEB;
    color: #FFFFFF;
}
QPushButton#btn_record_signal:hover { background-color: #388BFD; }
QPushButton#btn_record_signal:pressed { background-color: #0D5BD8; }
QPushButton#btn_record_signal:disabled { background-color: #21262D; color: #484F58; }

QPushButton#btn_record_noise {
    background-color: #B62324;
    color: #FFFFFF;
}
QPushButton#btn_record_noise:hover { background-color: #CF3030; }
QPushButton#btn_record_noise:pressed { background-color: #8A1A1A; }
QPushButton#btn_record_noise:disabled { background-color: #21262D; color: #484F58; }

QPushButton#btn_reset {
    background-color: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
}
QPushButton#btn_reset:hover { background-color: #30363D; }

QPushButton#btn_stop {
    background-color: #6E7681;
    color: #FFFFFF;
}
QPushButton#btn_stop:hover { background-color: #8B949E; }

QComboBox {
    background-color: #1C2128;
    border: 1px solid #30363D;
    border-radius: 4px;
    padding: 4px 8px;
    color: #E6EDF3;
    font-size: 11px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #161B22;
    border: 1px solid #30363D;
    selection-background-color: #1F6FEB;
    color: #E6EDF3;
}

QProgressBar {
    background-color: #21262D;
    border-radius: 4px;
    text-align: center;
    color: #E6EDF3;
    font-size: 10px;
}
QProgressBar::chunk {
    border-radius: 4px;
    background-color: #1F6FEB;
}

QStatusBar {
    background-color: #161B22;
    color: #7D8590;
    border-top: 1px solid #30363D;
    font-size: 11px;
}

QLabel#status_label {
    font-size: 13px;
    font-weight: bold;
    padding: 6px 12px;
    border-radius: 4px;
}

QTextEdit {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 4px;
    color: #E6EDF3;
    font-family: 'Consolas', monospace;
    font-size: 10px;
}
"""


# ──────────────────────────────────────────────
#  Styled Separator
# ──────────────────────────────────────────────
def make_separator(vertical=False):
    line = QFrame()
    line.setFrameShape(QFrame.VLine if vertical else QFrame.HLine)
    line.setStyleSheet("background: #30363D;")
    line.setFixedWidth(1) if vertical else line.setFixedHeight(1)
    return line


def make_label(text, size=11, bold=False, color="#E6EDF3"):
    lbl = QLabel(text)
    font = QFont("Segoe UI", size)
    if bold:
        font.setBold(True)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color};")
    return lbl


# ──────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────
class SNRMeterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.state = AppState.IDLE
        self.signal_data: Optional[AudioData] = None
        self.noise_data: Optional[AudioData] = None
        self.snr_result: Optional[SNRResult] = None
        self._rec_elapsed = 0

        self._setup_ui()
        self._setup_timers()
        self._refresh_devices()
        self._set_state(AppState.IDLE)

    # ── UI Setup ──────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("SNR Meter  •  Signal-to-Noise Ratio Analyzer")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 780)
        self.setStyleSheet(STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 8, 12, 8)
        root_layout.setSpacing(8)

        # ── Top bar
        root_layout.addWidget(self._build_top_bar())
        root_layout.addWidget(make_separator())

        # ── Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #30363D; width: 2px; }")

        # Left: controls + VU
        left_panel = self._build_left_panel()
        left_panel.setMaximumWidth(300)
        splitter.addWidget(left_panel)

        # Right: visualizations
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, stretch=1)

        # ── Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready — Select a microphone and record Signal first."
        )

    def _build_top_bar(self):
        bar = QWidget()
        bar.setFixedHeight(50)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 0, 4, 0)

        # Title
        title = make_label("📡 SNR Meter", 16, bold=True)
        layout.addWidget(title)
        layout.addSpacing(20)

        # Status chip
        self.lbl_state = QLabel("IDLE")
        self.lbl_state.setObjectName("status_label")
        self.lbl_state.setAlignment(Qt.AlignCenter)
        self.lbl_state.setFixedWidth(180)
        layout.addWidget(self.lbl_state)

        layout.addStretch()

        # Elapsed timer
        self.lbl_timer = make_label("00:00.0", 13, color="#7D8590")
        self.lbl_timer.setFont(QFont("Consolas", 13))
        layout.addWidget(self.lbl_timer)

        return bar

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # ── Device Selection
        dev_group = QGroupBox("Input Device")
        dev_layout = QVBoxLayout(dev_group)
        self.combo_device = QComboBox()
        dev_layout.addWidget(self.combo_device)
        btn_refresh = QPushButton("↻ Refresh")
        btn_refresh.setObjectName("btn_reset")
        btn_refresh.setFixedHeight(28)
        btn_refresh.clicked.connect(self._refresh_devices)
        dev_layout.addWidget(btn_refresh)
        layout.addWidget(dev_group)

        # ── Step 1: Record Signal
        sig_group = QGroupBox("Step 1 — Signal Recording")
        sig_layout = QVBoxLayout(sig_group)
        hint1 = make_label(
            "Record your audio signal\n(voice, music, test tone...)", 9, color="#7D8590"
        )
        hint1.setWordWrap(True)
        sig_layout.addWidget(hint1)
        self.btn_signal = QPushButton("⏺  Record Signal")
        self.btn_signal.setObjectName("btn_record_signal")
        self.btn_signal.setFixedHeight(44)
        self.btn_signal.clicked.connect(self._on_signal_clicked)
        sig_layout.addWidget(self.btn_signal)
        self.lbl_signal_info = make_label("Not recorded", 9, color="#7D8590")
        sig_layout.addWidget(self.lbl_signal_info)
        layout.addWidget(sig_group)

        # ── Step 2: Record Noise
        noise_group = QGroupBox("Step 2 — Noise Recording")
        noise_layout = QVBoxLayout(noise_group)
        hint2 = make_label(
            "Record background noise\n(room silence, fan hum...)", 9, color="#7D8590"
        )
        hint2.setWordWrap(True)
        noise_layout.addWidget(hint2)
        self.btn_noise = QPushButton("⏺  Record Noise")
        self.btn_noise.setObjectName("btn_record_noise")
        self.btn_noise.setFixedHeight(44)
        self.btn_noise.clicked.connect(self._on_noise_clicked)
        noise_layout.addWidget(self.btn_noise)
        self.lbl_noise_info = make_label("Not recorded", 9, color="#7D8590")
        noise_layout.addWidget(self.lbl_noise_info)
        layout.addWidget(noise_group)

        # ── Stop / Reset
        ctrl_layout = QHBoxLayout()
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        ctrl_layout.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setFixedHeight(36)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        ctrl_layout.addWidget(self.btn_reset)
        layout.addLayout(ctrl_layout)

        # ── VU Meter
        vu_group = QGroupBox("Live Level")
        vu_layout = QHBoxLayout(vu_group)
        vu_layout.setContentsMargins(8, 8, 8, 8)
        self.vu_meter = VUMeterWidget()
        self.vu_meter.setMinimumHeight(100)
        vu_layout.addWidget(self.vu_meter)

        vu_labels = QVBoxLayout()
        for db_text in ["0 dB", "-6", "-12", "-20", "-40", "-∞"]:
            lbl = make_label(db_text, 7, color="#7D8590")
            lbl.setAlignment(Qt.AlignRight)
            vu_labels.addWidget(lbl)
        vu_layout.addLayout(vu_labels)
        layout.addWidget(vu_group)

        layout.addStretch()

        # ── Stats box
        stats_group = QGroupBox("Measurements")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setHorizontalSpacing(8)
        stats_layout.setVerticalSpacing(4)

        self._stat_labels = {}
        stat_keys = [
            ("signal_rms", "Signal RMS"),
            ("signal_peak", "Signal Peak"),
            ("noise_rms", "Noise RMS"),
            ("noise_peak", "Noise Peak"),
            ("snr_db", "SNR"),
            ("quality", "Quality"),
        ]
        for row, (key, display) in enumerate(stat_keys):
            lbl_key = make_label(display + ":", 9, color="#7D8590")
            lbl_val = make_label("—", 9, bold=True)
            stats_layout.addWidget(lbl_key, row, 0)
            stats_layout.addWidget(lbl_val, row, 1)
            self._stat_labels[key] = lbl_val
        layout.addWidget(stats_group)

        return panel

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        # ── Top row: Gauge + Waveforms
        top_row = QHBoxLayout()

        # SNR Gauge
        gauge_group = QGroupBox("SNR Result")
        gauge_layout = QVBoxLayout(gauge_group)
        self.gauge = SNRGaugeWidget()
        self.gauge.setMinimumSize(220, 220)
        gauge_layout.addWidget(self.gauge)
        gauge_group.setFixedWidth(280)
        top_row.addWidget(gauge_group)

        # Waveforms (stacked)
        wave_group = QGroupBox("Waveforms")
        wave_layout = QVBoxLayout(wave_group)

        wave_layout.addWidget(make_label("Signal", 9, color="#58A6FF"))
        self.wave_signal = WaveformWidget(color=ACCENT_BLUE, label="SIGNAL")
        self.wave_signal.setMinimumHeight(90)
        wave_layout.addWidget(self.wave_signal)

        wave_layout.addWidget(make_label("Noise", 9, color="#F85149"))
        self.wave_noise = WaveformWidget(color=ACCENT_RED, label="NOISE")
        self.wave_noise.setMinimumHeight(90)
        wave_layout.addWidget(self.wave_noise)

        top_row.addWidget(wave_group, stretch=1)
        layout.addLayout(top_row)

        # ── Bottom: Spectrum
        spec_group = QGroupBox("Frequency Spectrum  (Blue = Signal  |  Red = Noise)")
        spec_layout = QVBoxLayout(spec_group)
        self.spectrum = SpectrumWidget()
        self.spectrum.setMinimumHeight(160)
        spec_layout.addWidget(self.spectrum)
        layout.addWidget(spec_group, stretch=1)

        return panel

    # ── Timers ────────────────────────────────
    def _setup_timers(self):
        # UI refresh @ 30fps
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._tick_ui)
        self._ui_timer.setInterval(33)

        # Recording elapsed counter
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._elapsed_timer.setInterval(100)

    def _tick_elapsed(self):
        self._rec_elapsed += 100
        ms = self._rec_elapsed
        mins = ms // 60000
        secs = (ms % 60000) / 1000
        self.lbl_timer.setText(f"{mins:02d}:{secs:04.1f}")

    def _tick_ui(self):
        if self.recorder.is_recording:
            level = self.recorder.get_live_level()
            self.vu_meter.set_level(level)
            waveform = self.recorder.get_live_waveform()

            if self.state == AppState.REC_SIGNAL:
                self.wave_signal.set_data(waveform)
            elif self.state == AppState.REC_NOISE:
                self.wave_noise.set_data(waveform)
        else:
            self.vu_meter.set_level(0)

    # ── Device ────────────────────────────────
    def _refresh_devices(self):
        self.combo_device.clear()
        devices = self.recorder.get_input_devices()
        for dev in devices:
            self.combo_device.addItem(f"[{dev['index']}] {dev['name']}", dev["index"])
        if not devices:
            self.combo_device.addItem("No input devices found", -1)

    def _get_selected_device(self) -> Optional[int]:
        idx = self.combo_device.currentData()
        return idx if idx is not None and idx >= 0 else None

    # ── State Machine ─────────────────────────
    def _set_state(self, new_state: AppState):
        self.state = new_state

        # Button enablement
        self.btn_signal.setEnabled(
            new_state in (AppState.IDLE, AppState.SIGNAL_DONE, AppState.RESULT)
        )
        self.btn_noise.setEnabled(new_state == AppState.SIGNAL_DONE)
        self.btn_stop.setEnabled(new_state in (AppState.REC_SIGNAL, AppState.REC_NOISE))
        self.btn_reset.setEnabled(
            new_state != AppState.IDLE or bool(self.signal_data or self.noise_data)
        )

        # Status chip
        state_styles = {
            AppState.IDLE: ("IDLE", "#30363D", "#7D8590"),
            AppState.REC_SIGNAL: ("● RECORDING SIGNAL", "#0D419D", "#58A6FF"),
            AppState.SIGNAL_DONE: ("SIGNAL ✓", "#1A3D20", "#3FB950"),
            AppState.REC_NOISE: ("● RECORDING NOISE", "#3D1212", "#F85149"),
            AppState.RESULT: ("SNR CALCULATED ✓", "#2D2007", "#E3B341"),
        }
        label, bg, fg = state_styles[new_state]
        self.lbl_state.setText(label)
        self.lbl_state.setStyleSheet(
            f"color: {fg}; background: {bg}; border-radius: 4px; "
            f"padding: 4px 10px; font-weight: bold; font-size: 11px;"
        )

    # ── Button Handlers ───────────────────────
    def _on_signal_clicked(self):
        if self.state in (AppState.IDLE, AppState.SIGNAL_DONE, AppState.RESULT):
            # Start recording signal
            device = self._get_selected_device()
            self.recorder.start(device_index=device)
            self._rec_elapsed = 0
            self._elapsed_timer.start()
            self._ui_timer.start()
            self._set_state(AppState.REC_SIGNAL)
            self.status_bar.showMessage("Recording SIGNAL... Click ⏹ Stop when done.")
            self.wave_signal.clear()

    def _on_noise_clicked(self):
        if self.state == AppState.SIGNAL_DONE:
            device = self._get_selected_device()
            self.recorder.start(device_index=device)
            self._rec_elapsed = 0
            self._elapsed_timer.start()
            self._ui_timer.start()
            self._set_state(AppState.REC_NOISE)
            self.status_bar.showMessage("Recording NOISE... Click ⏹ Stop when done.")
            self.wave_noise.clear()

    def _on_stop_clicked(self):
        if self.state == AppState.REC_SIGNAL:
            self._elapsed_timer.stop()
            self._ui_timer.stop()
            self.signal_data = self.recorder.stop("signal")
            self._update_waveform_display(self.signal_data, is_signal=True)
            self.lbl_signal_info.setText(
                f"✓ {self.signal_data.duration:.1f}s  |  {self.signal_data.rms_db:.1f} dBFS"
            )
            self.lbl_signal_info.setStyleSheet("color: #3FB950;")
            self._set_state(AppState.SIGNAL_DONE)
            self.status_bar.showMessage("Signal recorded! Now record the Noise.")

        elif self.state == AppState.REC_NOISE:
            self._elapsed_timer.stop()
            self._ui_timer.stop()
            self.noise_data = self.recorder.stop("noise")
            self._update_waveform_display(self.noise_data, is_signal=False)
            self.lbl_noise_info.setText(
                f"✓ {self.noise_data.duration:.1f}s  |  {self.noise_data.rms_db:.1f} dBFS"
            )
            self.lbl_noise_info.setStyleSheet("color: #F85149;")
            self._calculate_snr()

    def _on_reset_clicked(self):
        if self.recorder.is_recording:
            self.recorder.stop("reset")
            self._elapsed_timer.stop()
            self._ui_timer.stop()

        self.signal_data = None
        self.noise_data = None
        self.snr_result = None
        self.lbl_signal_info.setText("Not recorded")
        self.lbl_signal_info.setStyleSheet("color: #7D8590;")
        self.lbl_noise_info.setText("Not recorded")
        self.lbl_noise_info.setStyleSheet("color: #7D8590;")
        self.lbl_timer.setText("00:00.0")
        self.wave_signal.clear()
        self.wave_noise.clear()
        self.spectrum.clear()
        self.gauge.clear()
        self._clear_stats()
        self._set_state(AppState.IDLE)
        self.status_bar.showMessage("Reset complete. Ready to record.")

    # ── Visualization ─────────────────────────
    def _update_waveform_display(self, data: AudioData, is_signal: bool):
        n = min(len(data.samples), 4096)
        chunk = data.samples[:n] if n > 0 else np.zeros(1024)
        if is_signal:
            self.wave_signal.set_data(chunk)
        else:
            self.wave_noise.set_data(chunk)

        # Update spectrum
        if len(data.samples) > 0:
            freqs, db = data.get_spectrum(n_fft=4096)
            if is_signal:
                self.spectrum.set_signal(freqs, db)
            else:
                self.spectrum.set_noise(freqs, db)

    def _calculate_snr(self):
        if not self.signal_data or not self.noise_data:
            return

        self.snr_result = SNRResult(signal=self.signal_data, noise=self.noise_data)
        snr = self.snr_result.snr_db
        label = self.snr_result.quality_label
        color = self.snr_result.quality_color

        # Update gauge
        self.gauge.set_result(snr, label, color)

        # Update stats
        self._update_stats()

        self._set_state(AppState.RESULT)
        self.status_bar.showMessage(
            f"SNR = {snr:.2f} dB  ({label})  — You can re-record either channel to refine."
        )

    def _update_stats(self):
        if not self.snr_result:
            return
        r = self.snr_result
        stats = {
            "signal_rms": f"{r.signal.rms_db:.2f} dBFS",
            "signal_peak": f"{r.signal.peak_db:.2f} dBFS",
            "noise_rms": f"{r.noise.rms_db:.2f} dBFS",
            "noise_peak": f"{r.noise.peak_db:.2f} dBFS",
            "snr_db": f"{r.snr_db:.2f} dB",
            "quality": r.quality_label,
        }
        quality_colors = {
            "Excellent": "#00E676",
            "Good": "#69F0AE",
            "Acceptable": "#FFEB3B",
            "Poor": "#FF9800",
            "Very Poor": "#F44336",
        }
        for key, val in stats.items():
            lbl = self._stat_labels.get(key)
            if lbl:
                lbl.setText(val)
                if key == "quality":
                    c = quality_colors.get(val, "#E6EDF3")
                    lbl.setStyleSheet(f"color: {c}; font-weight: bold;")
                elif key == "snr_db":
                    lbl.setStyleSheet("color: #E3B341; font-weight: bold;")
                else:
                    lbl.setStyleSheet("color: #E6EDF3;")

    def _clear_stats(self):
        for lbl in self._stat_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet("color: #E6EDF3;")

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop("close")
        self._ui_timer.stop()
        self._elapsed_timer.stop()
        event.accept()


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SNR Meter")
    app.setOrganizationName("AudioTools")

    # Dark palette base
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0D1117"))
    palette.setColor(QPalette.WindowText, QColor("#E6EDF3"))
    palette.setColor(QPalette.Base, QColor("#161B22"))
    palette.setColor(QPalette.AlternateBase, QColor("#1C2128"))
    palette.setColor(QPalette.ToolTipBase, QColor("#E6EDF3"))
    palette.setColor(QPalette.ToolTipText, QColor("#E6EDF3"))
    palette.setColor(QPalette.Text, QColor("#E6EDF3"))
    palette.setColor(QPalette.Button, QColor("#21262D"))
    palette.setColor(QPalette.ButtonText, QColor("#E6EDF3"))
    palette.setColor(QPalette.Highlight, QColor("#1F6FEB"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    win = SNRMeterWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
