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

    # Relative amplitude. For 0-5V signals: amp=(p2p/2)/5 gives 0.0-0.5 nominal region.
    half_p2p = float((np.max(x) - np.min(x)) / 2.0)
    amplitude_rel = float(max(0.0, half_p2p / 5.0))

    # Jitter from inter-peak intervals.
    prominence = max(1e-6, float(np.std(centered)) * 0.25)
    min_distance = max(1, int(sample_rate / max(freq_hz, 1.0) * 0.4))
    peaks, _ = find_peaks(centered, prominence=prominence, distance=min_distance)
    if peaks.size >= 3:
        periods_sec = np.diff(peaks).astype(np.float64) / float(sample_rate)
        jitter_ms = float(np.std(periods_sec) * 1000.0)
    else:
        jitter_ms = 0.0

    # Spectral spread around spectral centroid.
    power_sum = float(np.sum(power))
    if power_sum <= 1e-20 or freqs.size == 0:
        spectral_spread = 0.0
    else:
        weights = power / power_sum
        centroid = float(np.sum(weights * freqs))
        spectral_spread = float(np.sqrt(np.sum(weights * (freqs - centroid) ** 2)))

    # Zero-crossing "rate" aligned with training: return raw count per window.
    zero_crossing_count = int(np.sum(np.diff(np.sign(centered)) != 0))

    return {
        "freq_hz": freq_hz,
        "amplitude_rel": amplitude_rel,
        "jitter_ms": jitter_ms,
        "spectral_spread": spectral_spread,
        "zero_crossing_rate": float(zero_crossing_count),
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
