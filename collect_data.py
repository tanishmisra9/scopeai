"""
ScopeAI data collection tool.

Records labeled audio samples via laptop audio-in, extracts signal features,
and appends rows to a CSV for model training.

Usage examples:
python collect_data.py --label "mode_A__nominal"
python collect_data.py --label "mode_A__R_too_high" --n_samples 40
python collect_data.py --label "mode_B__cap_missing" --duration 3
"""

import argparse
import csv
import datetime
import os

import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks


SAMPLE_RATE = 44100
CSV_COLUMNS = [
    "timestamp",
    "label",
    "circuit_mode",
    "fault_class",
    "freq_hz",
    "duty_cycle_pct",
    "amplitude_rel",
    "jitter_ms",
    "spectral_spread",
    "zero_crossing_rate",
]
CIRCUIT_MODES = {"mode_A", "mode_B", "mode_C"}
FAULT_CLASSES = {
    "nominal",
    "cap_missing",
    "R_too_high",
    "R_too_low",
    "clipping",
    "no_oscillation",
    "chatter",
}
DEFAULT_DEVICE_SENTINEL = -1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for collection options."""
    parser = argparse.ArgumentParser(description="ScopeAI labeled audio data collector.")
    parser.add_argument(
        "--label",
        required=True,
        help='Combined label in the format "{circuit_mode}__{fault_class}".',
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Recording duration in seconds per sample (default: 2.0).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Batch mode: auto-collect and save N samples without prompts.",
    )
    return parser.parse_args()


def validate_label(label: str) -> tuple[str, str]:
    """Validate label format and allowed values, returning its components."""
    parts = label.split("__")
    if len(parts) != 2:
        raise ValueError(
            'Invalid --label format. Expected "{circuit_mode}__{fault_class}", '
            f'got "{label}".'
        )

    circuit_mode, fault_class = parts
    if circuit_mode not in CIRCUIT_MODES:
        raise ValueError(
            f'Invalid circuit_mode "{circuit_mode}". Allowed: {sorted(CIRCUIT_MODES)}'
        )
    if fault_class not in FAULT_CLASSES:
        raise ValueError(
            f'Invalid fault_class "{fault_class}". Allowed: {sorted(FAULT_CLASSES)}'
        )
    return circuit_mode, fault_class


def confirm_safety() -> None:
    """Show startup safety warning and require explicit user confirmation."""
    print("VERIFY voltage divider is in place. Max ~150mV peak into audio-in.")
    reply = input("Type 'yes' to confirm and continue: ").strip().lower()
    if reply not in {"yes", "y"}:
        raise SystemExit("Safety confirmation not provided. Exiting.")


def select_input_device() -> int:
    """List available input devices and return selected index or default sentinel."""
    devices = sd.query_devices()
    input_indices = []

    print("\nAvailable input devices:")
    for idx, dev in enumerate(devices):
        max_input_channels = int(dev.get("max_input_channels", 0))
        if max_input_channels > 0:
            input_indices.append(idx)
            name = dev.get("name", f"device_{idx}")
            print(f"  [{idx}] {name} (inputs: {max_input_channels})")

    if not input_indices:
        raise RuntimeError("No audio input devices found.")

    raw = input("\nSelect input device index (Enter for system default): ").strip()
    if raw == "":
        return DEFAULT_DEVICE_SENTINEL
    if not raw.isdigit():
        raise ValueError(f'Invalid device index "{raw}". Expected an integer.')

    selected = int(raw)
    if selected not in input_indices:
        raise ValueError(
            f"Device index {selected} is invalid or not input-capable. "
            f"Choose from: {input_indices}"
        )
    return selected


def record_sample(duration: float, sr: int, device: int) -> np.ndarray:
    """Record one mono sample with a 3-second countdown."""
    print("\nStabilize circuit. Recording starts in:")
    for sec in range(3, 0, -1):
        print(f"{sec}...")

    n_frames = int(duration * sr)
    rec_device = None if device == DEFAULT_DEVICE_SENTINEL else device
    print("Recording...")
    audio = sd.rec(
        frames=n_frames,
        samplerate=sr,
        channels=1,
        blocking=True,
        device=rec_device,
    )
    audio_1d = np.asarray(audio).reshape(-1).astype(np.float64)
    return audio_1d


def extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Extract ScopeAI training features from a raw mono audio waveform."""
    if audio.size == 0:
        return {
            "freq_hz": 0.0,
            "duty_cycle_pct": 0.0,
            "amplitude_rel": 0.0,
            "jitter_ms": 0.0,
            "spectral_spread": 0.0,
            "zero_crossing_rate": 0.0,
        }

    centered = audio - float(np.mean(audio))
    duration_s = audio.size / float(sr) if sr > 0 else 0.0

    fft_vals = np.fft.rfft(centered)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sr)

    if power.size > 1:
        peak_idx = int(np.argmax(power[1:]) + 1)
    elif power.size == 1:
        peak_idx = 0
    else:
        peak_idx = 0
    freq_hz = float(freqs[peak_idx]) if freqs.size > peak_idx else 0.0

    power_sum = float(np.sum(power))
    if power_sum > 0.0 and freqs.size > 0:
        spectral_spread = float(
            np.sqrt(np.sum(power * (freqs - freq_hz) ** 2) / power_sum)
        )
    else:
        spectral_spread = 0.0

    amplitude_rel = float(np.clip(np.ptp(audio) / 2.0, 0.0, 1.0))

    peak_threshold = float(np.mean(centered) + 0.2 * np.std(centered))
    peaks, _ = find_peaks(centered, height=peak_threshold)
    intervals = np.diff(peaks) if peaks.size >= 2 else np.array([], dtype=np.float64)

    if intervals.size > 0:
        intervals_ms = intervals.astype(np.float64) * 1000.0 / float(sr)
        jitter_ms = float(np.std(intervals_ms))
    else:
        jitter_ms = 0.0

    signal_mean = float(np.mean(audio))
    if intervals.size > 0:
        period = int(max(1, round(float(np.median(intervals)))))
        above_ratios = []
        for start in peaks[:-1]:
            stop = int(start + period)
            if stop <= audio.size:
                segment = audio[int(start) : stop]
                if segment.size > 0:
                    above_ratios.append(float(np.mean(segment > signal_mean)))
        if above_ratios:
            duty_cycle_pct = float(100.0 * np.mean(above_ratios))
        else:
            duty_cycle_pct = float(100.0 * np.mean(audio > signal_mean))
    else:
        duty_cycle_pct = float(100.0 * np.mean(audio > signal_mean))

    zero_crossings = int(np.sum((centered[:-1] * centered[1:]) < 0.0))
    zero_crossing_rate = float(zero_crossings / duration_s) if duration_s > 0 else 0.0

    return {
        "freq_hz": freq_hz,
        "duty_cycle_pct": duty_cycle_pct,
        "amplitude_rel": amplitude_rel,
        "jitter_ms": jitter_ms,
        "spectral_spread": spectral_spread,
        "zero_crossing_rate": zero_crossing_rate,
    }


def data_paths() -> tuple[str, str]:
    """Return absolute paths for data directory and training CSV file."""
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    csv_path = os.path.join(data_dir, "training_data.csv")
    return data_dir, csv_path


def load_label_counts(csv_path: str) -> dict[str, int]:
    """Load existing label counts from CSV if present."""
    counts: dict[str, int] = {}
    if not os.path.exists(csv_path):
        return counts

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def append_sample(
    csv_path: str,
    label: str,
    circuit_mode: str,
    fault_class: str,
    features: dict[str, float],
) -> None:
    """Append one labeled sample with extracted features to the training CSV."""
    file_exists = os.path.exists(csv_path)
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "circuit_mode": circuit_mode,
        "fault_class": fault_class,
        "freq_hz": f"{features['freq_hz']:.6f}",
        "duty_cycle_pct": f"{features['duty_cycle_pct']:.6f}",
        "amplitude_rel": f"{features['amplitude_rel']:.6f}",
        "jitter_ms": f"{features['jitter_ms']:.6f}",
        "spectral_spread": f"{features['spectral_spread']:.6f}",
        "zero_crossing_rate": f"{features['zero_crossing_rate']:.6f}",
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_features_table(features: dict[str, float]) -> None:
    """Print extracted features in a readable table."""
    print("\nExtracted features:")
    print("+--------------------+--------------+")
    print("| Feature            | Value        |")
    print("+--------------------+--------------+")
    for key in (
        "freq_hz",
        "duty_cycle_pct",
        "amplitude_rel",
        "jitter_ms",
        "spectral_spread",
        "zero_crossing_rate",
    ):
        print(f"| {key:<18} | {features[key]:>12.6f} |")
    print("+--------------------+--------------+")


def print_label_counts(counts: dict[str, int]) -> None:
    """Print a label-to-count summary table."""
    print("\nLabel counts:")
    print("+---------------------------+--------+")
    print("| Label                     | Count  |")
    print("+---------------------------+--------+")
    for label in sorted(counts):
        print(f"| {label:<25} | {counts[label]:>6} |")
    print("+---------------------------+--------+")


def main() -> None:
    """Run the ScopeAI collection workflow."""
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0.")
    if args.n_samples is not None and args.n_samples <= 0:
        raise SystemExit("--n_samples must be a positive integer.")

    try:
        circuit_mode, fault_class = validate_label(args.label)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    confirm_safety()
    try:
        device = select_input_device()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    data_dir, csv_path = data_paths()
    os.makedirs(data_dir, exist_ok=True)
    counts = load_label_counts(csv_path)

    batch_mode = args.n_samples is not None
    saved_in_session = 0

    while True:
        audio = record_sample(duration=args.duration, sr=SAMPLE_RATE, device=device)
        features = extract_features(audio=audio, sr=SAMPLE_RATE)
        print_features_table(features)

        if batch_mode:
            save_now = True
            should_quit = False
        else:
            choice = input("\nSave this sample? (y/n/q to quit): ").strip().lower()
            if choice == "q":
                should_quit = True
                save_now = False
            elif choice == "y":
                should_quit = False
                save_now = True
            elif choice == "n":
                should_quit = False
                save_now = False
                print("Sample discarded. Re-recording same label.")
            else:
                should_quit = False
                save_now = False
                print("Invalid choice. Please enter y, n, or q.")

        if should_quit:
            print_label_counts(counts)
            return

        if not save_now:
            continue

        append_sample(
            csv_path=csv_path,
            label=args.label,
            circuit_mode=circuit_mode,
            fault_class=fault_class,
            features=features,
        )
        counts[args.label] = counts.get(args.label, 0) + 1
        saved_in_session += 1
        print_label_counts(counts)

        if batch_mode and args.n_samples is not None and saved_in_session >= args.n_samples:
            print(f"\nCollected {saved_in_session} samples in batch mode. Exiting.")
            print_label_counts(counts)
            return


if __name__ == "__main__":
    main()
