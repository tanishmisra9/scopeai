"""
ScopeAI diagnosis and AI chat module.
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

AUTO_DIAG_SYSTEM_PROMPT = """You are ScopeAI, a hardware debugging assistant for a NE555P astable oscillator circuit.

The circuit uses a potentiometer as R2. Expected frequency ranges:
- nominal: ~1-2.5 Hz (pot in middle)
- R_too_high: ~0.5 Hz (pot at maximum resistance, signal too slow)
- R_too_low: ~5-8 Hz (pot at minimum resistance, signal too fast)
- cap_missing: ~0 Hz, near-zero amplitude (timing capacitor removed)
- no_oscillation: ~0 Hz, near-zero amplitude (output disconnected)

ALWAYS give specific, actionable fix instructions in 2-3 sentences max.
- R_too_high: tell them to turn the pot toward minimum or swap a smaller resistor
- R_too_low: tell them to turn the pot toward maximum or swap a larger resistor
- cap_missing: tell them to reinsert the timing capacitor between pin 6 and GND
- no_oscillation: tell them to check the wire from pin 3 to the probe
- nominal: confirm circuit is healthy and frequency is in expected range"""


_CHAT_SYSTEM_PROMPT = """You are ScopeAI, an AI lab partner for a NE555P astable oscillator circuit on a breadboard.
You have access to live oscilloscope data via an Analog Discovery 2.

The circuit uses a potentiometer as R2. Expected frequency ranges:
- nominal: ~1-2.5 Hz
- R_too_high: ~0.5 Hz (pot cranked to max, decrease resistance)
- R_too_low: ~5-8 Hz (pot at min, increase resistance)
- cap_missing: ~0 Hz, near-zero amplitude
- no_oscillation: ~0 Hz, near-zero amplitude

When the student asks about their circuit:
1. Use the capture_signal tool to get a fresh 6-second reading
2. Look at the frequency and amplitude
3. Give a specific 2-sentence diagnosis and fix instruction

Be direct. Say exactly what resistor to change or what to check. Never be vague."""

_CHAT_COOLDOWN_SEC = 2.0
_last_openai_call_ts = 0.0
_auto_diag_cache: dict[tuple[str, str], str] = {}
_last_capture: tuple[np.ndarray, int] | None = None
_last_capture_features: dict[str, float] | None = None
_client: OpenAI | None = None
_model_bundle: dict[str, Any] | None = None


def _now() -> float:
    return time.time()


def _enforce_cooldown() -> None:
    global _last_openai_call_ts
    elapsed = _now() - _last_openai_call_ts
    if elapsed < _CHAT_COOLDOWN_SEC:
        time.sleep(_CHAT_COOLDOWN_SEC - elapsed)
    _last_openai_call_ts = _now()


def _get_openai_client() -> OpenAI | None:
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
    root = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(root, "models", filename), os.path.join(root, filename)):
        if os.path.exists(path):
            return path
    return None


def _load_model_bundle() -> dict[str, Any]:
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
    freq = float(metrics.get("freq_hz", 0.0))
    amp = float(metrics.get("amplitude_rel", 0.0))
    if amp < 0.03:
        return "no_oscillation", 0.62
    if freq < 0.8:
        return "R_too_high", 0.58
    if freq > 4.0:
        return "R_too_low", 0.60
    return "nominal", 0.61


def _predict_from_metrics(circuit_mode: str, metrics: dict[str, float]) -> tuple[str, str, float]:
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
    cache_key = (circuit_mode, fault_class)
    if cache_key in _auto_diag_cache:
        return _auto_diag_cache[cache_key]

    client = _get_openai_client()
    if client is None:
        fallback = (
            f"ML indicates {fault_class} at {confidence:.0%} confidence. "
            "OpenAI API unavailable — check power rails, 555 orientation, timing resistor/capacitor values."
        )
        _auto_diag_cache[cache_key] = fallback
        return fallback

    prompt = (
        f"NE555P astable oscillator diagnosis:\n"
        f"ML Fault: {fault_class} ({confidence:.0%} confidence)\n"
        f"Frequency: {float(metrics.get('freq_hz', 0.0)):.2f} Hz\n"
        f"Amplitude: {float(metrics.get('amplitude_rel', 0.0)):.4f}\n"
        f"Jitter: {float(metrics.get('jitter_ms', 0.0)):.2f} ms\n"
        f"Give a specific 2-sentence diagnosis and fix instruction."
    )

    try:
        _enforce_cooldown()
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role": "system", "content": AUTO_DIAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            text = "Diagnosis unavailable. Please re-capture and retry."
    except Exception:
        text = f"ML indicates {fault_class} at {confidence:.0%} confidence. Verify component values and recapture."

    _auto_diag_cache[cache_key] = text
    return text


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "capture_signal",
                "description": "Capture a fresh 6-second signal reading from the circuit and classify it.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_device_status",
                "description": "Get current AD2 connection status and supply voltage.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def _tool_capture_signal(current_diagnosis: tuple[str, str, float] | None) -> str:
    global _last_capture, _last_capture_features
    # Always use 6 seconds to match training data
    samples, sr = capture.capture(duration_sec=5.0)
    metrics = extract_features(samples, sample_rate=sr)
    _last_capture = (samples, sr)
    _last_capture_features = metrics
    circuit_mode = current_diagnosis[0] if current_diagnosis else "mode_A"
    mode, fault, conf = _predict_from_metrics(circuit_mode=circuit_mode, metrics=metrics)
    return (
        f"Fresh capture complete.\n"
        f"Prediction: {fault} ({conf:.0%} confidence)\n"
        f"Frequency: {metrics['freq_hz']:.2f} Hz\n"
        f"Amplitude: {metrics['amplitude_rel']:.4f}\n"
        f"Jitter: {metrics['jitter_ms']:.2f} ms"
    )


def _tool_get_device_status() -> str:
    info = capture.get_device_info()
    return json.dumps(info, indent=2)


def _execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    current_diagnosis: tuple[str, str, float] | None,
) -> str:
    try:
        if tool_name == "capture_signal":
            return _tool_capture_signal(current_diagnosis=current_diagnosis)
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
    client = _get_openai_client()
    if client is None:
        text = "OpenAI API unavailable. Set OPENAI_API_KEY to enable AI chat."
        updated = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": text},
        ]
        return text, updated

    mode, fault, conf = current_diagnosis or ("mode_A", "unknown", 0.0)
    metrics = current_metrics or {}

    system = (
        f"{_CHAT_SYSTEM_PROMPT}\n\n"
        f"Current state: fault={fault} ({conf:.0%}), "
        f"freq={float(metrics.get('freq_hz', 0.0)):.2f}Hz, "
        f"amplitude={float(metrics.get('amplitude_rel', 0.0)):.4f}"
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    tools = _tool_schemas()
    final_text = "I ran into an issue generating a response."

    try:
        for _ in range(5):
            _enforce_cooldown()
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append({
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
                })
                for tc in msg.tool_calls:
                    args = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                    tool_result = _execute_tool(tc.function.name, args, current_diagnosis)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": tool_result,
                    })
                continue

            final_text = (msg.content or "").strip()
            if not final_text:
                final_text = "Empty response. Please try again."
            break
    except Exception as exc:
        final_text = f"Could not reach AI service: {exc}"

    updated_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ]
    return final_text, updated_history
