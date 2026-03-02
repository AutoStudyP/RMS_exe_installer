"""
Custom PyQt5 Widgets for SNR Meter visualization
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QLinearGradient,
    QFont,
    QPainterPath,
    QRadialGradient,
)


# ──────────────────────────────────────────────
#  Color Palette (Dark Theme)
# ──────────────────────────────────────────────
BG_DARK = QColor("#0D1117")
BG_PANEL = QColor("#161B22")
BG_WIDGET = QColor("#1C2128")
ACCENT_BLUE = QColor("#58A6FF")
ACCENT_GREEN = QColor("#3FB950")
ACCENT_RED = QColor("#F85149")
ACCENT_GOLD = QColor("#E3B341")
TEXT_PRIMARY = QColor("#E6EDF3")
TEXT_MUTED = QColor("#7D8590")
GRID_COLOR = QColor("#30363D")


def lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    r = int(c1.red() + (c2.red() - c1.red()) * t)
    g = int(c1.green() + (c2.green() - c1.green()) * t)
    b = int(c1.blue() + (c2.blue() - c1.blue()) * t)
    return QColor(r, g, b)


# ──────────────────────────────────────────────
#  Live Waveform Widget
# ──────────────────────────────────────────────
class WaveformWidget(QWidget):
    """Real-time scrolling waveform display"""

    def __init__(self, color: QColor = ACCENT_BLUE, label: str = "", parent=None):
        super().__init__(parent)
        self.color = color
        self.label = label
        self._data = np.zeros(1024, dtype=np.float32)
        self.setMinimumSize(300, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: transparent;")

    def set_data(self, data: np.ndarray):
        self._data = data.copy()
        self.update()

    def clear(self):
        self._data = np.zeros(1024, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        # Background
        painter.fillRect(0, 0, w, h, BG_WIDGET)

        # Border
        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Grid lines
        painter.setPen(QPen(GRID_COLOR, 1, Qt.DotLine))
        for i in range(1, 4):
            y = int(h * i / 4)
            painter.drawLine(0, y, w, y)

        # Center line
        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawLine(0, int(cy), w, int(cy))

        # Label
        if self.label:
            painter.setPen(TEXT_MUTED)
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(8, 16, self.label)

        # Waveform
        data = self._data
        if len(data) == 0:
            return

        n = len(data)
        amplitude = (h / 2) * 0.85

        # Gradient fill
        gradient = QLinearGradient(0, 0, 0, h)
        col_t = QColor(self.color)
        col_t.setAlpha(120)
        col_b = QColor(self.color)
        col_b.setAlpha(20)
        gradient.setColorAt(0, col_t)
        gradient.setColorAt(1, col_b)

        path_top = QPainterPath()
        path_bottom = QPainterPath()

        for i in range(n):
            x = w * i / n
            y_top = cy - data[i] * amplitude
            y_bot = cy + data[i] * amplitude
            if i == 0:
                path_top.moveTo(x, y_top)
                path_bottom.moveTo(x, y_bot)
            else:
                path_top.lineTo(x, y_top)
                path_bottom.lineTo(x, y_bot)

        # Fill between top and bottom
        fill_path = QPainterPath(path_top)
        for i in range(n - 1, -1, -1):
            x = w * i / n
            y_bot = cy + data[i] * amplitude
            fill_path.lineTo(x, y_bot)
        fill_path.closeSubpath()

        painter.fillPath(fill_path, QBrush(gradient))

        # Draw line
        pen = QPen(self.color, 1.5)
        painter.setPen(pen)
        painter.drawPath(path_top)

        painter.end()


# ──────────────────────────────────────────────
#  Spectrum Widget
# ──────────────────────────────────────────────
class SpectrumWidget(QWidget):
    """Frequency spectrum bar display"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal_db: np.ndarray = None
        self._noise_db: np.ndarray = None
        self._freqs: np.ndarray = None
        self.setMinimumSize(300, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_signal(self, freqs: np.ndarray, db: np.ndarray):
        self._freqs = freqs
        self._signal_db = db
        self.update()

    def set_noise(self, freqs: np.ndarray, db: np.ndarray):
        self._freqs = freqs
        self._noise_db = db
        self.update()

    def clear(self):
        self._signal_db = None
        self._noise_db = None
        self._freqs = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, BG_WIDGET)
        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Axes labels
        pad_left, pad_bot, pad_top, pad_right = 40, 25, 10, 10
        plot_w = w - pad_left - pad_right
        plot_h = h - pad_bot - pad_top

        # Grid
        db_min, db_max = -100, 0
        painter.setPen(QPen(GRID_COLOR, 1, Qt.DotLine))
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(TEXT_MUTED)
        for db_val in range(db_min, db_max + 1, 20):
            y = pad_top + plot_h * (1 - (db_val - db_min) / (db_max - db_min))
            painter.setPen(QPen(GRID_COLOR, 1, Qt.DotLine))
            painter.drawLine(pad_left, int(y), pad_left + plot_w, int(y))
            painter.setPen(TEXT_MUTED)
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(2, int(y) + 4, f"{db_val}")

        # Frequency labels (log scale hint)
        freq_labels = [100, 500, 1000, 5000, 10000, 20000]
        if self._freqs is not None and len(self._freqs) > 1:
            f_max = self._freqs[-1]
            for fl in freq_labels:
                if fl > f_max:
                    continue
                x = pad_left + plot_w * (fl / f_max)
                painter.setPen(QPen(GRID_COLOR, 1, Qt.DotLine))
                painter.drawLine(int(x), pad_top, int(x), pad_top + plot_h)
                painter.setPen(TEXT_MUTED)
                lbl = f"{fl // 1000}k" if fl >= 1000 else str(fl)
                painter.drawText(int(x) - 10, h - 5, lbl)

        # Plot spectra
        def draw_spectrum(db_arr, color, alpha=200):
            if db_arr is None or self._freqs is None:
                return
            n = min(len(db_arr), len(self._freqs))
            f_max = self._freqs[-1] if self._freqs[-1] > 0 else 1

            path = QPainterPath()
            started = False
            for i in range(n):
                x = pad_left + plot_w * (self._freqs[i] / f_max)
                db_clamp = max(db_min, min(db_max, db_arr[i]))
                y = pad_top + plot_h * (1 - (db_clamp - db_min) / (db_max - db_min))
                if not started:
                    path.moveTo(x, y)
                    started = True
                else:
                    path.lineTo(x, y)

            c = QColor(color)
            c.setAlpha(alpha)
            pen = QPen(c, 1.5)
            painter.setPen(pen)
            painter.drawPath(path)

            # Fill under
            fill = QPainterPath(path)
            fill.lineTo(pad_left + plot_w, pad_top + plot_h)
            fill.lineTo(pad_left, pad_top + plot_h)
            fill.closeSubpath()
            fc = QColor(color)
            fc.setAlpha(30)
            painter.fillPath(fill, QBrush(fc))

        draw_spectrum(self._signal_db, ACCENT_BLUE)
        draw_spectrum(self._noise_db, ACCENT_RED, alpha=160)

        # Legend
        painter.setPen(QPen(ACCENT_BLUE, 2))
        painter.drawLine(pad_left + 5, 12, pad_left + 20, 12)
        painter.setPen(TEXT_PRIMARY)
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(pad_left + 24, 16, "Signal")

        painter.setPen(QPen(ACCENT_RED, 2))
        painter.drawLine(pad_left + 70, 12, pad_left + 85, 12)
        painter.setPen(TEXT_PRIMARY)
        painter.drawText(pad_left + 89, 16, "Noise")

        painter.end()


# ──────────────────────────────────────────────
#  SNR Gauge Widget
# ──────────────────────────────────────────────
class SNRGaugeWidget(QWidget):
    """Analog-style arc gauge for SNR dB display"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snr_db: float = 0.0
        self._quality_label: str = ""
        self._quality_color: QColor = ACCENT_GREEN
        self._has_result = False
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_result(self, snr_db: float, quality_label: str, quality_color: str):
        self._snr_db = snr_db
        self._quality_label = quality_label
        self._quality_color = QColor(quality_color)
        self._has_result = True
        self.update()

    def clear(self):
        self._has_result = False
        self._snr_db = 0.0
        self._quality_label = ""
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, BG_WIDGET)

        side = min(w, h) - 20
        cx = w / 2
        cy = h / 2 + 10

        radius = side / 2
        arc_rect = QRectF(cx - radius, cy - radius, side, side)

        # Arc range: -220° to 40° (total 260°)
        START_DEG = -220
        SPAN_DEG = 260

        # Background arc (track)
        pen = QPen(GRID_COLOR, 12, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(arc_rect, int(START_DEG * 16), int(-SPAN_DEG * 16))

        if self._has_result:
            # SNR clamp: 0..50 dB → 0..1
            t = max(0.0, min(1.0, self._snr_db / 50.0))
            angle_span = SPAN_DEG * t

            # Colored arc gradient by quality
            c_start = ACCENT_RED
            c_mid = ACCENT_GOLD
            c_end = ACCENT_GREEN
            if t < 0.5:
                arc_color = lerp_color(c_start, c_mid, t * 2)
            else:
                arc_color = lerp_color(c_mid, c_end, (t - 0.5) * 2)

            pen2 = QPen(arc_color, 12, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen2)
            painter.drawArc(arc_rect, int(START_DEG * 16), int(-angle_span * 16))

            # Needle
            import math

            needle_angle_deg = START_DEG + SPAN_DEG * t
            needle_angle_rad = math.radians(needle_angle_deg)
            needle_len = radius - 18
            nx = cx + needle_len * math.cos(needle_angle_rad)
            ny = cy - needle_len * math.sin(needle_angle_rad)

            needle_pen = QPen(QColor("#FFFFFF"), 2.5, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(needle_pen)
            painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

            # Center dot
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, cy), 6, 6)

            # SNR Value
            painter.setPen(self._quality_color)
            font = QFont("Consolas", 22, QFont.Bold)
            painter.setFont(font)
            snr_text = f"{self._snr_db:.1f}"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(snr_text)
            painter.drawText(int(cx - tw / 2), int(cy + 5), snr_text)

            # dB unit
            painter.setPen(TEXT_MUTED)
            painter.setFont(QFont("Consolas", 10))
            painter.drawText(int(cx - 10), int(cy + 22), "dB")

            # Quality label
            painter.setPen(self._quality_color)
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            fm2 = painter.fontMetrics()
            tw2 = fm2.horizontalAdvance(self._quality_label)
            painter.drawText(int(cx - tw2 / 2), int(cy + 46), self._quality_label)

        else:
            # Placeholder
            painter.setPen(TEXT_MUTED)
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(
                arc_rect.toRect(),
                Qt.AlignCenter,
                "Record signal\n& noise\nto measure SNR",
            )

        # Scale ticks
        import math

        tick_labels = {0: "0", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50+"}
        for db_val, lbl in tick_labels.items():
            t_tick = db_val / 50.0
            angle_deg = START_DEG + SPAN_DEG * t_tick
            angle_rad = math.radians(angle_deg)
            r_outer = radius - 2
            r_inner = radius - 16
            x1 = cx + r_outer * math.cos(angle_rad)
            y1 = cy - r_outer * math.sin(angle_rad)
            x2 = cx + r_inner * math.cos(angle_rad)
            y2 = cy - r_inner * math.sin(angle_rad)
            painter.setPen(QPen(TEXT_MUTED, 1))
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Label
            r_label = radius + 12
            lx = cx + r_label * math.cos(angle_rad)
            ly = cy - r_label * math.sin(angle_rad)
            painter.setPen(TEXT_MUTED)
            painter.setFont(QFont("Consolas", 7))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(lbl)
            painter.drawText(int(lx - tw / 2), int(ly + 4), lbl)

        painter.end()


# ──────────────────────────────────────────────
#  VU Meter Widget
# ──────────────────────────────────────────────
class VUMeterWidget(QWidget):
    """Vertical VU meter for live level monitoring"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level: float = 0.0
        self._peak: float = 0.0
        self._peak_timer = 0
        self.setFixedWidth(30)
        self.setMinimumHeight(120)

    def set_level(self, level: float):
        """level: 0.0 ~ 1.0"""
        self._level = max(0.0, min(1.0, level))
        if self._level > self._peak:
            self._peak = self._level
            self._peak_timer = 30
        else:
            if self._peak_timer > 0:
                self._peak_timer -= 1
            else:
                self._peak = max(self._level, self._peak - 0.01)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, BG_WIDGET)

        bar_h = int(h * self._level)
        bar_y = h - bar_h

        # Gradient bar
        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0.0, ACCENT_RED)
        gradient.setColorAt(0.3, ACCENT_GOLD)
        gradient.setColorAt(1.0, ACCENT_GREEN)

        painter.fillRect(4, bar_y, w - 8, bar_h, QBrush(gradient))

        # Peak hold
        peak_y = int(h * (1 - self._peak))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawLine(4, peak_y, w - 4, peak_y)

        # Border
        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        painter.end()
