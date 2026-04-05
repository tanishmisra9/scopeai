"""
ScopeAI feature extraction utilities.

Extracts exactly the five model features expected by the trained classifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

FEATURES = [
    "freq_hz",
    "amplitude_rel",
    "jitter_ms",
    "spectral_spread",
    "zero_crossing_rate",
]


def _as_1d_float64(samples: np.ndarray) -> np.ndarray:
    """Convert input samples into a 1D float64 NumPy array."""
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    return arr


def extract_features(samples: np.ndarray, sample_rate: int = 44100) -> dict[str, float]:
    """
    Extract the 5 ScopeAI features from a waveform.

    Returns keys:
    - freq_hz
    - amplitude_rel
    - jitter_ms
    - spectral_spread
    - zero_crossing_rate
    """
    x = _as_1d_float64(samples)
    if x.size < 4 or sample_rate <= 0:
        return {
            "freq_hz": 0.0,
            "amplitude_rel": 0.0,
            "jitter_ms": 0.0,
            "spectral_spread": 0.0,
            "zero_crossing_rate": 0.0,
        }

    centered = x - float(np.mean(x))
    n = centered.size

    # Frequency domain
    fft_vals = np.fft.rfft(centered)
    mags = np.abs(fft_vals)
    power = mags**2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))

    # Fundamental frequency from dominant non-DC bin.
    if mags.size <= 1:
        freq_hz = 0.0
    else:
        signal_threshold = float(np.percentile(mags[1:], 95) * 0.2)
        if float(np.max(mags[1:])) <= max(signal_threshold, 1e-12):
            freq_hz = 0.0
        else:
            peak_bin = int(np.argmax(mags[1:]) + 1)
            freq_hz = float(peak_bin * sample_rate / n)

    # Relative amplitude must match collect_data.py training extraction.
    amplitude_rel = float(np.clip(np.ptp(x) / 2.0, 0.0, 1.0))

    # Jitter must match collect_data.py peak selection and interval conversion.
    peak_threshold = float(np.mean(centered) + 0.2 * np.std(centered))
    peaks, _ = find_peaks(centered, height=peak_threshold)
    intervals = np.diff(peaks) if peaks.size >= 2 else np.array([], dtype=np.float64)
    if intervals.size > 0:
        intervals_ms = intervals.astype(np.float64) * 1000.0 / float(sample_rate)
        jitter_ms = float(np.std(intervals_ms))
    else:
        jitter_ms = 0.0

    # Spectral spread must be centered on dominant frequency (freq_hz).
    power_sum = float(np.sum(power))
    if power_sum <= 1e-20 or freqs.size == 0:
        spectral_spread = 0.0
    else:
        spectral_spread = float(
            np.sqrt(np.sum(power * (freqs - freq_hz) ** 2) / power_sum)
        )

    duration_s = x.size / float(sample_rate)
    zero_crossings = int(np.sum((centered[:-1] * centered[1:]) < 0.0))
    zero_crossing_rate = float(zero_crossings / duration_s) if duration_s > 0 else 0.0

    return {
        "freq_hz": freq_hz,
        "amplitude_rel": amplitude_rel,
        "jitter_ms": jitter_ms,
        "spectral_spread": spectral_spread,
        "zero_crossing_rate": zero_crossing_rate,
    }


def features_to_vector(feature_dict: dict[str, Any]) -> list[float]:
    """Convert feature dictionary into model-ready list in canonical order."""
    return [float(feature_dict.get(name, 0.0)) for name in FEATURES]


def format_metrics_display(feature_dict: dict[str, Any]) -> dict[str, str]:
    """Format feature values for human-readable dashboard display."""
    freq_hz = float(feature_dict.get("freq_hz", 0.0))
    amplitude_rel = float(feature_dict.get("amplitude_rel", 0.0))
    jitter_ms = float(feature_dict.get("jitter_ms", 0.0))
    spectral_spread = float(feature_dict.get("spectral_spread", 0.0))
    zcr = float(feature_dict.get("zero_crossing_rate", 0.0))
    return {
        "freq_hz": f"{freq_hz:.2f} Hz",
        "amplitude_rel": f"{amplitude_rel:.4f}",
        "jitter_ms": f"{jitter_ms:.2f} ms",
        "spectral_spread": f"{spectral_spread:.1f}",
        "zero_crossing_rate": f"{zcr:.0f}",
    }
