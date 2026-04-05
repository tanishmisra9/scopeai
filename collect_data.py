"""
ScopeAI data collection tool — Analog Discovery 2 version.

Usage examples:
    py collect_data.py --label "mode_A__nominal" --n_samples 20 --duration 10
    py collect_data.py --label "mode_A__R_too_high" --n_samples 20 --duration 10
    py collect_data.py --label "mode_A__cap_missing" --n_samples 20 --duration 10
"""

import argparse
import csv
import ctypes
import datetime
import os
import sys
import time

import numpy as np
from scipy.signal import find_peaks

if sys.platform == "win32":
    _dwf = ctypes.cdll.dwf
elif sys.platform == "darwin":
    _dwf = ctypes.cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    _dwf = ctypes.cdll.LoadLibrary("libdwf.so")

_ACQMODE_RECORD = ctypes.c_int(3)
_STATE_DONE     = 2
_HDWF_NONE      = ctypes.c_int(0)

SAMPLE_RATE   = 10_000
VOLTAGE_RANGE = 5.0

CSV_COLUMNS = [
    "timestamp", "label", "circuit_mode", "fault_class",
    "freq_hz", "amplitude_rel", "jitter_ms",
    "spectral_spread", "zero_crossing_rate",
]
CIRCUIT_MODES = {"mode_A", "mode_B", "mode_C"}
FAULT_CLASSES = {
    "nominal", "cap_missing", "R_too_high",
    "R_too_low", "clipping", "no_oscillation", "chatter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScopeAI AD2 data collector.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--n_samples", type=int, default=None)
    return parser.parse_args()


def validate_label(label: str) -> tuple[str, str]:
    parts = label.split("__")
    if len(parts) != 2:
        raise ValueError(f'Expected "mode__fault", got "{label}"')
    circuit_mode, fault_class = parts
    if circuit_mode not in CIRCUIT_MODES:
        raise ValueError(f'Bad circuit_mode "{circuit_mode}". Options: {sorted(CIRCUIT_MODES)}')
    if fault_class not in FAULT_CLASSES:
        raise ValueError(f'Bad fault_class "{fault_class}". Options: {sorted(FAULT_CLASSES)}')
    return circuit_mode, fault_class


def open_device() -> ctypes.c_int:
    print("\n--- ScopeAI AD2 Setup ---")
    print("1. AD2 CH1+ (orange)       -> pin 3 of 555")
    print("2. AD2 CH1- (orange/white) -> breadboard GND rail")
    print("3. AD2 GND  (black)        -> breadboard GND rail")
    print("4. Circuit powered by 5V adapter")
    reply = input("\nType 'yes' to confirm and start: ").strip().lower()
    if reply not in {"yes", "y"}:
        raise SystemExit("Cancelled.")
    hdwf = ctypes.c_int()
    _dwf.FDwfDeviceOpen(ctypes.c_int(-1), ctypes.byref(hdwf))
    if hdwf.value == _HDWF_NONE.value:
        errmsg = ctypes.create_string_buffer(512)
        _dwf.FDwfGetLastErrorMsg(errmsg)
        raise SystemExit(f"Could not open AD2.\n{errmsg.value.decode()}")
    print("AD2 connected.\n")
    return hdwf


def close_device(hdwf: ctypes.c_int) -> None:
    _dwf.FDwfDeviceClose(hdwf)


def record_sample(duration: float, sr: int, hdwf: ctypes.c_int) -> np.ndarray:
    print("\nStabilize circuit. Recording starts in:")
    for sec in range(3, 0, -1):
        print(f"  {sec}...")
        time.sleep(1)

    n_samples = int(duration * sr)
    _dwf.FDwfAnalogInChannelEnableSet(hdwf, ctypes.c_int(0), ctypes.c_int(1))
    _dwf.FDwfAnalogInChannelRangeSet(hdwf, ctypes.c_int(0), ctypes.c_double(VOLTAGE_RANGE))
    _dwf.FDwfAnalogInFrequencySet(hdwf, ctypes.c_double(float(sr)))
    _dwf.FDwfAnalogInRecordLengthSet(hdwf, ctypes.c_double(duration))
    _dwf.FDwfAnalogInAcquisitionModeSet(hdwf, _ACQMODE_RECORD)
    _dwf.FDwfAnalogInConfigure(hdwf, ctypes.c_int(0), ctypes.c_int(1))

    print(f"Recording for {duration:.0f} seconds...")
    collected = []
    samples_remaining = n_samples

    while samples_remaining > 0:
        sts = ctypes.c_byte()
        _dwf.FDwfAnalogInStatus(hdwf, ctypes.c_int(1), ctypes.byref(sts))
        available = ctypes.c_int()
        lost      = ctypes.c_int()
        corrupt   = ctypes.c_int()
        _dwf.FDwfAnalogInStatusRecord(hdwf, ctypes.byref(available), ctypes.byref(lost), ctypes.byref(corrupt))
        chunk = min(available.value, samples_remaining)
        if chunk > 0:
            buf = (ctypes.c_double * chunk)()
            _dwf.FDwfAnalogInStatusData(hdwf, ctypes.c_int(0), buf, ctypes.c_int(chunk))
            collected.append(np.frombuffer(buf, dtype=np.float64).copy())
            samples_remaining -= chunk
        if sts.value == _STATE_DONE and available.value == 0:
            break

    audio = np.concatenate(collected) if collected else np.zeros(n_samples)
    return audio[:n_samples].astype(np.float64)


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Extract features — duty_cycle removed, unreliable for slow signals."""
    if audio.size == 0:
        return {k: 0.0 for k in ["freq_hz","amplitude_rel","jitter_ms","spectral_spread","zero_crossing_rate"]}

    centered   = audio - float(np.mean(audio))
    duration_s = audio.size / float(sr) if sr > 0 else 0.0

    fft_vals = np.fft.rfft(centered)
    power    = np.abs(fft_vals) ** 2
    freqs    = np.fft.rfftfreq(centered.size, d=1.0 / sr)

    peak_idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    freq_hz  = float(freqs[peak_idx]) if freqs.size > peak_idx else 0.0

    power_sum = float(np.sum(power))
    spectral_spread = float(np.sqrt(np.sum(power * (freqs - freq_hz) ** 2) / power_sum)) if power_sum > 0.0 else 0.0

    amplitude_rel = float(np.clip(np.ptp(audio) / VOLTAGE_RANGE, 0.0, 1.0))

    peak_threshold = float(np.mean(centered) + 0.2 * np.std(centered))
    peaks, _  = find_peaks(centered, height=peak_threshold)
    intervals = np.diff(peaks) if peaks.size >= 2 else np.array([], dtype=np.float64)
    jitter_ms = float(np.std(intervals * 1000.0 / float(sr))) if intervals.size > 0 else 0.0

    zero_crossings     = int(np.sum((centered[:-1] * centered[1:]) < 0.0))
    zero_crossing_rate = float(zero_crossings / duration_s) if duration_s > 0 else 0.0

    return {
        "freq_hz":            freq_hz,
        "amplitude_rel":      amplitude_rel,
        "jitter_ms":          jitter_ms,
        "spectral_spread":    spectral_spread,
        "zero_crossing_rate": zero_crossing_rate,
    }


def data_paths() -> tuple[str, str]:
    root     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    csv_path = os.path.join(data_dir, "training_data.csv")
    return data_dir, csv_path


def load_label_counts(csv_path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lbl = row.get("label", "").strip()
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def append_sample(csv_path, label, circuit_mode, fault_class, features) -> None:
    file_exists = os.path.exists(csv_path)
    row = {
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
        "label":        label,
        "circuit_mode": circuit_mode,
        "fault_class":  fault_class,
        **{k: f"{features[k]:.6f}" for k in ["freq_hz","amplitude_rel","jitter_ms","spectral_spread","zero_crossing_rate"]},
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_features_table(features: dict[str, float]) -> None:
    print("\nExtracted features:")
    print("+--------------------+--------------+")
    print("| Feature            | Value        |")
    print("+--------------------+--------------+")
    for key in ["freq_hz","amplitude_rel","jitter_ms","spectral_spread","zero_crossing_rate"]:
        print(f"| {key:<18} | {features[key]:>12.4f} |")
    print("+--------------------+--------------+")


def print_label_counts(counts: dict[str, int]) -> None:
    print("\nLabel counts:")
    print("+---------------------------+--------+")
    print("| Label                     | Count  |")
    print("+---------------------------+--------+")
    for label in sorted(counts):
        print(f"| {label:<25} | {counts[label]:>6} |")
    print("+---------------------------+--------+")


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0.")
    if args.n_samples is not None and args.n_samples <= 0:
        raise SystemExit("--n_samples must be a positive integer.")

    try:
        circuit_mode, fault_class = validate_label(args.label)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    hdwf = open_device()

    data_dir, csv_path = data_paths()
    os.makedirs(data_dir, exist_ok=True)
    counts = load_label_counts(csv_path)

    batch_mode       = args.n_samples is not None
    saved_in_session = 0

    try:
        while True:
            audio    = record_sample(duration=args.duration, sr=SAMPLE_RATE, hdwf=hdwf)
            features = extract_features(audio=audio, sr=SAMPLE_RATE)
            print_features_table(features)

            if batch_mode:
                save_now    = True
                should_quit = False
            else:
                choice = input("\nSave this sample? (y/n/q to quit): ").strip().lower()
                if choice == "q":
                    should_quit = True
                    save_now    = False
                elif choice == "y":
                    should_quit = False
                    save_now    = True
                elif choice == "n":
                    should_quit = False
                    save_now    = False
                    print("Discarded. Re-recording.")
                else:
                    should_quit = False
                    save_now    = False
                    print("Enter y, n, or q.")

            if should_quit:
                print_label_counts(counts)
                return

            if not save_now:
                continue

            append_sample(csv_path, args.label, circuit_mode, fault_class, features)
            counts[args.label] = counts.get(args.label, 0) + 1
            saved_in_session  += 1
            print_label_counts(counts)

            if batch_mode and saved_in_session >= (args.n_samples or 0):
                print(f"\nDone. Collected {saved_in_session} samples.")
                print_label_counts(counts)
                return

    finally:
        close_device(hdwf)
        print("AD2 closed.")


if __name__ == "__main__":
    main()
