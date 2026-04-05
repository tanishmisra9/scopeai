"""
ScopeAI diagnosis and AI chat module.

Provides:
- auto_diagnose(): cached one-shot dashboard explanation
- chat(): multi-turn GPT chat with hardware tool calling
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import joblib
import numpy as np
from openai import OpenAI
from scipy.signal import find_peaks

import capture
from features import FEATURES, extract_features, features_to_vector

AUTO_DIAG_SYSTEM_PROMPT = """You are ScopeAI, an expert hardware debugging assistant for introductory electronics students.
You analyze oscilloscope signals from student circuits and explain faults in clear, beginner-friendly language.

When given a fault diagnosis and signal metrics:
1. Explain what the fault means in plain English
2. Describe what's physically happening in the circuit
3. Give specific, actionable fix instructions (e.g., "swap the 10kΩ resistor at R2 for a 2.2kΩ")
4. If nominal, confirm the circuit is working and briefly explain why the metrics look healthy

Keep responses concise — 3-5 sentences. The circuit is a NE555P astable oscillator on a breadboard."""

_CHAT_COOLDOWN_SEC = 2.0
_last_openai_call_ts = 0.0
_auto_diag_cache: dict[tuple[str, str], str] = {}
_last_capture: tuple[np.ndarray, int] | None = None
_last_capture_features: dict[str, float] | None = None
_client: OpenAI | None = None
_model_bundle: dict[str, Any] | None = None


def _now() -> float:
    """Return current wall-clock timestamp in seconds."""
    return time.time()


def _enforce_cooldown() -> None:
    """Rate-limit OpenAI calls to avoid bursty chat traffic."""
    global _last_openai_call_ts
    elapsed = _now() - _last_openai_call_ts
    if elapsed < _CHAT_COOLDOWN_SEC:
        time.sleep(_CHAT_COOLDOWN_SEC - elapsed)
    _last_openai_call_ts = _now()


def _get_openai_client() -> OpenAI | None:
    """Create and cache an OpenAI client if credentials are available."""
    global _client
    if _client is not None:
        return _client
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        _client = OpenAI()
    except Exception:
        _client = None
    return _client


def _resolve_artifact_path(filename: str) -> str | None:
    """Resolve model artifact path using models/ first, then project root fallback."""
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "models", filename),
        os.path.join(root, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_model_bundle() -> dict[str, Any]:
    """Load classifier metadata and encoders; gracefully handle missing artifacts."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    bundle: dict[str, Any] = {
        "classifier": None,
        "label_encoder": None,
        "meta": None,
        "available": False,
    }
    clf_path = _resolve_artifact_path("fault_classifier.pkl")
    le_path = _resolve_artifact_path("label_encoder.pkl")
    meta_path = _resolve_artifact_path("model_meta.json")

    if meta_path:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                bundle["meta"] = json.load(f)
        except Exception:
            bundle["meta"] = None

    if clf_path and le_path:
        try:
            bundle["classifier"] = joblib.load(clf_path)
            bundle["label_encoder"] = joblib.load(le_path)
            bundle["available"] = True
        except Exception:
            bundle["available"] = False

    _model_bundle = bundle
    return bundle


def _heuristic_fault(metrics: dict[str, float]) -> tuple[str, float]:
    """Fallback fault classifier when model artifacts are unavailable."""
    freq = float(metrics.get("freq_hz", 0.0))
    amp = float(metrics.get("amplitude_rel", 0.0))
    jitter = float(metrics.get("jitter_ms", 0.0))
    spread = float(metrics.get("spectral_spread", 0.0))
    zc = float(metrics.get("zero_crossing_rate", 0.0))

    if amp < 0.03 and freq < 1.0:
        return "no_oscillation", 0.62
    if freq < 8.0:
        return "R_too_high", 0.58
    if freq > 52.0:
        return "R_too_low", 0.60
    if spread > 1400.0 and zc > 500.0:
        return "chatter", 0.56
    if spread > 1200.0 and amp > 0.35:
        return "clipping", 0.57
    if jitter > 12.0 and amp < 0.12:
        return "cap_missing", 0.55
    return "nominal", 0.61


def _predict_from_metrics(circuit_mode: str, metrics: dict[str, float]) -> tuple[str, str, float]:
    """Predict combined label with confidence from model or fallback heuristic."""
    bundle = _load_model_bundle()
    if bundle["available"]:
        try:
            vec = np.array(features_to_vector(metrics), dtype=np.float64).reshape(1, -1)
            clf = bundle["classifier"]
            le = bundle["label_encoder"]
            pred_enc = int(clf.predict(vec)[0])
            pred_label = str(le.inverse_transform([pred_enc])[0])
            confidence = 0.0
            if hasattr(clf, "predict_proba"):
                confidence = float(np.max(clf.predict_proba(vec)[0]))
            if "__" in pred_label:
                mode, fault = pred_label.split("__", 1)
                return mode, fault, confidence
        except Exception:
            pass
    fault, confidence = _heuristic_fault(metrics)
    return circuit_mode, fault, confidence


def auto_diagnose(circuit_mode: str, fault_class: str, confidence: float, metrics: dict[str, Any]) -> str:
    """Quick one-shot diagnosis for dashboard auto-loop with class-change cache."""
    cache_key = (circuit_mode, fault_class)
    if cache_key in _auto_diag_cache:
        return _auto_diag_cache[cache_key]

    client = _get_openai_client()
    if client is None:
        fallback = (
            f"ML indicates {fault_class} in {circuit_mode} at {confidence:.0%} confidence. "
            "OpenAI API unavailable, so this is a local fallback explanation. "
            "Check power rails, 555 orientation, timing resistor/capacitor values, and ground continuity."
        )
        _auto_diag_cache[cache_key] = fallback
        return fallback

    prompt = (
        f"Circuit: {circuit_mode} (NE555P astable oscillator)\n"
        f"ML Diagnosis: {fault_class} (confidence: {confidence:.0%})\n"
        "Signal Metrics:\n"
        f"  - Frequency: {float(metrics.get('freq_hz', 0.0)):.1f} Hz\n"
        f"  - Amplitude: {float(metrics.get('amplitude_rel', 0.0)):.4f} (relative)\n"
        f"  - Jitter: {float(metrics.get('jitter_ms', 0.0)):.2f} ms\n"
        f"  - Spectral Spread: {float(metrics.get('spectral_spread', 0.0)):.1f}\n"
        f"  - Zero-Crossing Rate: {float(metrics.get('zero_crossing_rate', 0.0)):.1f}\n\n"
        "Explain what's wrong and how to fix it."
    )

    try:
        _enforce_cooldown()
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": AUTO_DIAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            text = "Diagnosis unavailable at the moment. Please re-capture and retry."
    except Exception:
        text = (
            f"ML indicates {fault_class} in {circuit_mode} at {confidence:.0%} confidence. "
            "Could not reach OpenAI API right now. Verify component values and recapture."
        )
    _auto_diag_cache[cache_key] = text
    return text


def _chat_system_prompt(
    current_metrics: dict[str, Any] | None,
    current_diagnosis: tuple[str, str, float] | None,
) -> str:
    """Build contextual system prompt for interactive chat."""
    mode, fault, conf = current_diagnosis or ("mode_A", "unknown", 0.0)
    metrics = current_metrics or {}
    return (
        "You are ScopeAI, an AI lab partner for electronics students. "
        "You can see and interact with the student's circuit through an Analog Discovery 2 oscilloscope.\n\n"
        "You have tools to:\n"
        "- Capture new signal readings from the circuit\n"
        "- Analyze waveform characteristics in detail\n"
        "- Adjust the power supply voltage\n\n"
        "Current circuit state:\n"
        f"- Circuit mode: {mode}\n"
        f"- Last diagnosis: {fault} ({conf:.0%} confidence)\n"
        f"- Last metrics: freq={float(metrics.get('freq_hz', 0.0)):.2f}Hz, "
        f"amplitude={float(metrics.get('amplitude_rel', 0.0)):.4f}, "
        f"jitter={float(metrics.get('jitter_ms', 0.0)):.2f}ms\n\n"
        "Be conversational, helpful, and educational. When a student asks a question, "
        "use your tools proactively if it would help answer their question.\n\n"
        "Keep responses concise and beginner-friendly. You're a patient TA, not a textbook."
    )


def _tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI tool schema definitions for chat tool-calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "capture_signal",
                "description": "Capture a new signal snapshot and run feature extraction + prediction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_sec": {"type": "number", "default": 0.5, "minimum": 0.1, "maximum": 5.0}
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_waveform",
                "description": "Run deeper analysis on the most recent capture.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "enum": ["frequency", "jitter", "amplitude", "noise", "harmonics"],
                        }
                    },
                    "required": ["focus"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_voltage",
                "description": "Set AD2 supply voltage (0-5V).",
                "parameters": {
                    "type": "object",
                    "properties": {"voltage": {"type": "number", "minimum": 0.0, "maximum": 5.0}},
                    "required": ["voltage"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_device_status",
                "description": "Get current AD2/simulation status and configuration.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def _tool_capture_signal(args: dict[str, Any], current_diagnosis: tuple[str, str, float] | None) -> str:
    """Capture signal and return metrics plus classifier prediction."""
    global _last_capture, _last_capture_features
    duration = float(args.get("duration_sec", 0.5))
    samples, sr = capture.capture_snapshot(duration_sec=duration)
    metrics = extract_features(samples, sample_rate=sr)
    _last_capture = (samples, sr)
    _last_capture_features = metrics
    circuit_mode = current_diagnosis[0] if current_diagnosis else "mode_A"
    mode, fault, conf = _predict_from_metrics(circuit_mode=circuit_mode, metrics=metrics)
    return (
        f"Captured {samples.size} samples at {sr} Hz.\n"
        f"Prediction: {mode}__{fault} ({conf:.0%} confidence)\n"
        f"Metrics: freq={metrics['freq_hz']:.2f}Hz, amp={metrics['amplitude_rel']:.4f}, "
        f"jitter={metrics['jitter_ms']:.2f}ms, spread={metrics['spectral_spread']:.1f}, "
        f"zcr={metrics['zero_crossing_rate']:.0f}"
    )


def _tool_analyze_waveform(args: dict[str, Any]) -> str:
    """Analyze latest captured waveform according to requested focus area."""
    if _last_capture is None:
        return "No waveform captured yet. Run capture_signal first."
    focus = str(args.get("focus", "")).strip().lower()
    samples, sr = _last_capture
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    centered = x - float(np.mean(x))
    n = centered.size
    if n < 8:
        return "Waveform too short to analyze."

    if focus == "frequency":
        mags = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        top = np.argsort(mags[1:])[-3:] + 1 if mags.size > 3 else np.array([0])
        peaks = ", ".join(f"{freqs[i]:.2f}Hz({mags[i]:.2f})" for i in reversed(top))
        return f"Dominant frequency components: {peaks}"

    if focus == "harmonics":
        mags = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        idx = np.argsort(mags[1:])[-5:] + 1 if mags.size > 6 else np.arange(1, mags.size)
        lines = [f"{freqs[i]:.2f} Hz (mag={mags[i]:.2f})" for i in reversed(idx)]
        return "Top harmonic peaks:\n" + "\n".join(f"- {line}" for line in lines)

    if focus == "jitter":
        peaks, _ = find_peaks(centered, prominence=max(1e-6, np.std(centered) * 0.2))
        if peaks.size < 3:
            return "Not enough peaks to estimate jitter reliably."
        intervals_ms = np.diff(peaks) * 1000.0 / float(sr)
        return (
            f"Jitter stats: mean period={np.mean(intervals_ms):.2f}ms, "
            f"std={np.std(intervals_ms):.2f}ms, min={np.min(intervals_ms):.2f}ms, "
            f"max={np.max(intervals_ms):.2f}ms."
        )

    if focus == "amplitude":
        p2p = float(np.ptp(x))
        rms = float(np.sqrt(np.mean(centered**2)))
        return f"Amplitude analysis: p2p={p2p:.3f}V, half-p2p={p2p/2:.3f}V, rms={rms:.3f}V."

    if focus == "noise":
        mags = np.abs(np.fft.rfft(centered))
        if mags.size < 4:
            return "Insufficient FFT bins to estimate noise."
        peak = float(np.max(mags[1:]))
        floor = float(np.median(mags[1:]) + 1e-9)
        snr_db = 20.0 * np.log10(max(peak, 1e-9) / floor)
        return f"Estimated spectral noise floor={floor:.4f}, peak={peak:.4f}, approx SNR={snr_db:.2f} dB."

    return "Unknown focus. Use one of: frequency, jitter, amplitude, noise, harmonics."


def _tool_set_voltage(args: dict[str, Any]) -> str:
    """Set AD2 supply voltage with safety clamp to 0-5V."""
    voltage = float(args.get("voltage", 0.0))
    return capture.set_supply_voltage(voltage)


def _tool_get_device_status() -> str:
    """Get AD2/simulation status as a compact JSON string."""
    info = capture.get_device_info()
    return json.dumps(info, indent=2)


def _execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    current_diagnosis: tuple[str, str, float] | None,
) -> str:
    """Dispatch tool calls and return textual tool results."""
    try:
        if tool_name == "capture_signal":
            return _tool_capture_signal(tool_args, current_diagnosis=current_diagnosis)
        if tool_name == "analyze_waveform":
            return _tool_analyze_waveform(tool_args)
        if tool_name == "set_voltage":
            return _tool_set_voltage(tool_args)
        if tool_name == "get_device_status":
            return _tool_get_device_status()
        return f"Unknown tool: {tool_name}"
    except Exception as exc:
        return f"Tool error ({tool_name}): {exc}"


def chat(
    user_message: str,
    conversation_history: list[dict[str, Any]],
    current_metrics: dict[str, Any] | None = None,
    current_diagnosis: tuple[str, str, float] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Multi-turn chat with tool access.

    Returns:
        (response_text, updated_history)
    """
    client = _get_openai_client()
    if client is None:
        text = (
            "OpenAI API unavailable right now. I can still show live metrics and ML predictions; "
            "set OPENAI_API_KEY to enable interactive AI chat."
        )
        updated = conversation_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": text}]
        return text, updated

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _chat_system_prompt(current_metrics, current_diagnosis)}
    ]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    tools = _tool_schemas()
    final_text = "I ran into an issue generating a response."

    try:
        for _ in range(5):
            _enforce_cooldown()
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    args = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                    tool_result = _execute_tool(tc.function.name, args, current_diagnosis)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": tool_result,
                        }
                    )
                continue

            final_text = (msg.content or "").strip()
            if not final_text:
                final_text = "I completed the request but got an empty response. Please try again."
            break
    except Exception:
        final_text = (
            "I couldn't reach the AI service right now. You can still use ScopeAI live capture and metrics."
        )

    updated_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ]
    return final_text, updated_history
