"""
Audio Engine - Recording and SNR Calculation Module

Includes:
  AudioRecorder     — manual 2-stage recorder
  AutoVADRecorder   — single-pass recorder that auto-classifies
                      speech (Signal) vs silence (Noise) frames
                      using short-time RMS energy gating.
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


# ──────────────────────────────────────────────
#  Auto VAD Recorder
# ──────────────────────────────────────────────
class AutoVADRecorder:
    """
    Single-pass recorder that automatically separates
    speech/signal frames from background-noise frames
    using a short-time RMS energy threshold.

    Algorithm
    ---------
    * Blocksize = 1024 samples (~23 ms @ 44100 Hz)
    * For every block, compute RMS energy.
    * Maintain a rolling baseline (median of quiet blocks)
      that adapts to the room over time.
    * If block_rms > baseline * SPEECH_RATIO  → signal frame
    * Else                                    → noise frame
    * SNR is re-computed every UPDATE_INTERVAL signal+noise frames.
    """

    SPEECH_RATIO   = 3.0   # block must be 3× louder than baseline to count as signal
    BLOCKSIZE      = 1024
    UPDATE_INTERVAL = 20   # recompute SNR every N new frames total
    BASELINE_WINDOW = 80   # rolling window (frames) for noise floor estimation

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 on_snr_update: Optional[Callable[['SNRResult', np.ndarray, np.ndarray], None]] = None,
                 on_chunk: Optional[Callable[[np.ndarray, bool], None]] = None):
        self.sample_rate = sample_rate
        self.on_snr_update = on_snr_update   # callback(SNRResult, signal_frames, noise_frames)
        self.on_chunk = on_chunk              # callback(chunk, is_signal: bool)

        self._signal_frames: List[np.ndarray] = []
        self._noise_frames:  List[np.ndarray] = []
        self._rms_history:   List[float] = []   # rolling RMS of all blocks
        self._frame_counter  = 0
        self._is_running     = False
        self._lock           = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self.last_snr: Optional[SNRResult] = None
        self.last_block_is_signal = False
        self._last_raw: np.ndarray = np.zeros(self.BLOCKSIZE, dtype=DTYPE)

    # ── public API ──────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, device_index: Optional[int] = None):
        if self._is_running:
            return
        with self._lock:
            self._signal_frames.clear()
            self._noise_frames.clear()
            self._rms_history.clear()
            self._frame_counter = 0
            self._is_running = True
            self.last_snr = None

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device_index,
            callback=self._audio_callback,
            blocksize=self.BLOCKSIZE,
        )
        self._stream.start()

    def stop(self):
        self._is_running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def reset(self):
        """Clear accumulated frames without stopping."""
        with self._lock:
            self._signal_frames.clear()
            self._noise_frames.clear()
            self._rms_history.clear()
            self._frame_counter = 0
            self.last_snr = None

    def get_frame_counts(self) -> tuple:
        """Returns (n_signal_frames, n_noise_frames)"""
        with self._lock:
            return len(self._signal_frames), len(self._noise_frames)

    def get_signal_duration(self) -> float:
        with self._lock:
            return len(self._signal_frames) * self.BLOCKSIZE / self.sample_rate

    def get_noise_duration(self) -> float:
        with self._lock:
            return len(self._noise_frames) * self.BLOCKSIZE / self.sample_rate

    def get_live_level(self) -> float:
        """Latest RMS 0..1 for VU meter."""
        with self._lock:
            if not self._rms_history:
                return 0.0
            return min(self._rms_history[-1] * 10, 1.0)

    def get_live_waveform(self, n_samples: int = 2048) -> np.ndarray:
        """Most recent n_samples of raw audio for waveform display."""
        with self._lock:
            all_frames = self._signal_frames + self._noise_frames
            if not all_frames:
                return np.zeros(n_samples, dtype=DTYPE)
            # use last frames sorted by order — we keep insertion order separately
            raw = self._last_raw
        if len(raw) >= n_samples:
            return raw[-n_samples:]
        out = np.zeros(n_samples, dtype=DTYPE)
        out[-len(raw):] = raw
        return out

    # ── internal ────────────────────────────────────────────────────────
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._is_running:
            return
        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        with self._lock:
            self._rms_history.append(rms)
            if len(self._rms_history) > self.BASELINE_WINDOW:
                self._rms_history.pop(0)

            # Adaptive noise floor: median of the quietest 50% of recent blocks
            if len(self._rms_history) >= 4:
                sorted_rms = sorted(self._rms_history)
                baseline = float(np.median(sorted_rms[: len(sorted_rms) // 2 + 1]))
            else:
                baseline = rms  # not enough history yet

            threshold = max(baseline * self.SPEECH_RATIO, 1e-4)
            is_signal = rms > threshold
            self.last_block_is_signal = is_signal
            self._last_raw = chunk   # store latest chunk for waveform

            if is_signal:
                self._signal_frames.append(chunk)
            else:
                self._noise_frames.append(chunk)

            self._frame_counter += 1
            should_update = (
                self._frame_counter % self.UPDATE_INTERVAL == 0
                and len(self._signal_frames) >= 4
                and len(self._noise_frames) >= 4
            )

            if should_update:
                sig = np.concatenate(self._signal_frames).astype(DTYPE)
                noi = np.concatenate(self._noise_frames).astype(DTYPE)
                signal_data = AudioData(sig, self.sample_rate, 'signal')
                noise_data  = AudioData(noi, self.sample_rate, 'noise')
                result = SNRResult(signal=signal_data, noise=noise_data)
                self.last_snr = result
                sig_snap = sig.copy()
                noi_snap = noi.copy()
            else:
                result = None
                sig_snap = None
                noi_snap = None

        # fire callbacks outside the lock
        if self.on_chunk:
            self.on_chunk(chunk, is_signal)
        if result and self.on_snr_update:
            self.on_snr_update(result, sig_snap, noi_snap)  # type: ignore
