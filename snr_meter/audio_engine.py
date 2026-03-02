"""
Audio Engine - Recording and SNR Calculation Module
"""

import numpy as np
import sounddevice as sd
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable


SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.float32


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    label: str  # "signal" or "noise"

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        if len(self.samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.samples**2)))

    @property
    def rms_db(self) -> float:
        rms = self.rms
        if rms <= 0:
            return -np.inf
        return 20 * np.log10(rms)

    @property
    def peak_db(self) -> float:
        if len(self.samples) == 0:
            return -np.inf
        peak = np.max(np.abs(self.samples))
        if peak <= 0:
            return -np.inf
        return 20 * np.log10(peak)

    def get_spectrum(self, n_fft: int = 2048) -> tuple:
        """Returns (frequencies, magnitude_db)"""
        if len(self.samples) < n_fft:
            n_fft = len(self.samples)
        window = np.hanning(n_fft)
        chunk = self.samples[:n_fft] * window
        fft = np.fft.rfft(chunk)
        magnitude = np.abs(fft)
        magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)
        return freqs, magnitude_db


@dataclass
class SNRResult:
    signal: AudioData
    noise: AudioData

    @property
    def signal_power(self) -> float:
        return float(np.mean(self.signal.samples**2))

    @property
    def noise_power(self) -> float:
        return float(np.mean(self.noise.samples**2))

    @property
    def snr_linear(self) -> float:
        np_val = self.noise_power
        if np_val <= 0:
            return float("inf")
        return self.signal_power / np_val

    @property
    def snr_db(self) -> float:
        snr = self.snr_linear
        if snr <= 0:
            return -np.inf
        if snr == float("inf"):
            return float("inf")
        return 10 * np.log10(snr)

    @property
    def quality_label(self) -> str:
        snr = self.snr_db
        if snr == float("inf") or snr >= 40:
            return "Excellent"
        elif snr >= 25:
            return "Good"
        elif snr >= 15:
            return "Acceptable"
        elif snr >= 5:
            return "Poor"
        else:
            return "Very Poor"

    @property
    def quality_color(self) -> str:
        label = self.quality_label
        colors = {
            "Excellent": "#00E676",
            "Good": "#69F0AE",
            "Acceptable": "#FFEB3B",
            "Poor": "#FF9800",
            "Very Poor": "#F44336",
        }
        return colors.get(label, "#FFFFFF")


class AudioRecorder:
    """
    Thread-safe audio recorder with live callback support.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: List[np.ndarray] = []
        self._is_recording = False
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self.on_chunk: Optional[Callable[[np.ndarray], None]] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def get_input_devices(self) -> List[dict]:
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append({"index": i, "name": dev["name"]})
        return result

    def start(self, device_index: Optional[int] = None):
        if self._is_recording:
            return
        with self._lock:
            self._frames = []
            self._is_recording = True

        def callback(indata, frames, time, status):
            if self._is_recording:
                chunk = indata[:, 0].copy()
                with self._lock:
                    self._frames.append(chunk)
                if self.on_chunk:
                    self.on_chunk(chunk)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=DTYPE,
            device=device_index,
            callback=callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self, label: str) -> AudioData:
        if not self._is_recording:
            return AudioData(np.array([], dtype=DTYPE), self.sample_rate, label)

        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if self._frames:
                samples = np.concatenate(self._frames).astype(DTYPE)
            else:
                samples = np.array([], dtype=DTYPE)

        return AudioData(samples=samples, sample_rate=self.sample_rate, label=label)

    def get_live_level(self) -> float:
        """Returns current RMS level (0~1) for VU meter"""
        with self._lock:
            if not self._frames:
                return 0.0
            recent = self._frames[-4:] if len(self._frames) >= 4 else self._frames
            chunk = np.concatenate(recent)
        rms = float(np.sqrt(np.mean(chunk**2)))
        return min(rms * 10, 1.0)  # normalize

    def get_live_waveform(self, n_samples: int = 2048) -> np.ndarray:
        """Returns last n_samples for live waveform display"""
        with self._lock:
            if not self._frames:
                return np.zeros(n_samples, dtype=DTYPE)
            all_data = np.concatenate(self._frames)

        if len(all_data) >= n_samples:
            return all_data[-n_samples:]
        else:
            padded = np.zeros(n_samples, dtype=DTYPE)
            padded[-len(all_data) :] = all_data
            return padded
