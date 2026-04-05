"""
ScopeAI Streamlit app: live dashboard + AI chat.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import capture
import diagnose
from features import FEATURES, extract_features, features_to_vector, format_metrics_display

st.set_page_config(page_title="ScopeAI", layout="wide")


def _resolve_artifact_path(filename: str) -> str | None:
    """Resolve artifact path from models/ first, then project root."""
    root = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(root, "models", filename), os.path.join(root, filename)):
        if os.path.exists(path):
            return path
    return None


def _load_model_assets() -> tuple[Any, Any, dict[str, Any], str | None]:
    """Load classifier, label encoder, and metadata with fallback paths."""
    clf = None
    label_encoder = None
    meta: dict[str, Any] = {"features": FEATURES}
    err: str | None = None

    meta_path = _resolve_artifact_path("model_meta.json")
    clf_path = _resolve_artifact_path("fault_classifier.pkl")
    le_path = _resolve_artifact_path("label_encoder.pkl")

    if meta_path:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            err = f"Could not read metadata: {exc}"

    if clf_path and le_path:
        try:
            clf = joblib.load(clf_path)
            label_encoder = joblib.load(le_path)
        except Exception as exc:
            err = f"Could not load model artifacts: {exc}"
    else:
        if err is None:
            err = "Model artifacts not found; running heuristic fallback."

    return clf, label_encoder, meta, err


def _heuristic_predict(circuit_mode: str, feat: dict[str, float]) -> tuple[str, str, float]:
    """Fallback prediction path when classifier artifacts are unavailable."""
    freq = float(feat.get("freq_hz", 0.0))
    amp = float(feat.get("amplitude_rel", 0.0))
    jitter = float(feat.get("jitter_ms", 0.0))
    spread = float(feat.get("spectral_spread", 0.0))
    zcr = float(feat.get("zero_crossing_rate", 0.0))
    if amp < 0.03 and freq < 1.0:
        return circuit_mode, "no_oscillation", 0.62
    if freq < 8.0:
        return circuit_mode, "R_too_high", 0.58
    if freq > 52.0:
        return circuit_mode, "R_too_low", 0.60
    if spread > 1200.0 and zcr > 350.0:
        return circuit_mode, "cap_missing", 0.55
    if jitter > 9.0:
        return circuit_mode, "chatter", 0.54
    return circuit_mode, "nominal", 0.64


def _predict(circuit_mode: str, feat: dict[str, float]) -> tuple[str, str, float]:
    """Predict mode/fault/confidence from model if available, otherwise heuristic."""
    clf = st.session_state.classifier
    le = st.session_state.label_encoder
    if clf is None or le is None:
        return _heuristic_predict(circuit_mode, feat)
    try:
        vec = np.array(features_to_vector(feat), dtype=np.float64).reshape(1, -1)
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
    return _heuristic_predict(circuit_mode, feat)


def _init_session_state() -> None:
    """Initialize required Streamlit session state fields."""
    defaults = {
        "conversation_history": [],
        "last_waveform": None,
        "last_features": None,
        "last_prediction": None,
        "last_diagnosis_text": "",
        "previous_features": None,
        "capture_device": None,
        "simulation_mode": False,
        "classifier": None,
        "label_encoder": None,
        "model_meta": {"features": FEATURES},
        "model_error": None,
        "last_diag_key": None,
        "last_sample_ts": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.classifier is None and st.session_state.label_encoder is None:
        clf, le, meta, err = _load_model_assets()
        st.session_state.classifier = clf
        st.session_state.label_encoder = le
        st.session_state.model_meta = meta
        st.session_state.model_error = err

    st.session_state.capture_device = capture.get_device_info()
    st.session_state.simulation_mode = bool(st.session_state.capture_device.get("simulation_mode", False))


def _run_sample_cycle(circuit_mode: str, duration_sec: float = 1.0) -> None:
    """Capture waveform, extract features, classify, and refresh diagnosis text."""
    samples, sr = capture.capture(duration_sec=duration_sec)
    feat = extract_features(samples, sample_rate=sr)
    mode, fault, confidence = _predict(circuit_mode, feat)
    diag_key = (mode, fault)

    st.session_state.previous_features = st.session_state.last_features
    st.session_state.last_waveform = (samples, sr)
    st.session_state.last_features = feat
    st.session_state.last_prediction = (mode, fault, confidence)

    if st.session_state.last_diag_key != diag_key:
        st.session_state.last_diagnosis_text = diagnose.auto_diagnose(
            circuit_mode=mode,
            fault_class=fault,
            confidence=confidence,
            metrics=feat,
        )
        st.session_state.last_diag_key = diag_key
    st.session_state.last_sample_ts = time.time()
    st.session_state.capture_device = capture.get_device_info()
    st.session_state.simulation_mode = bool(st.session_state.capture_device.get("simulation_mode", False))


def _metric_delta(key: str) -> str | None:
    """Compute display delta string for metric cards."""
    prev = st.session_state.previous_features
    curr = st.session_state.last_features
    if not prev or not curr:
        return None
    dv = float(curr.get(key, 0.0)) - float(prev.get(key, 0.0))
    if abs(dv) < 1e-9:
        return "0"
    if key == "freq_hz":
        return f"{dv:+.2f} Hz"
    if key == "amplitude_rel":
        return f"{dv:+.4f}"
    if key == "jitter_ms":
        return f"{dv:+.2f} ms"
    if key == "spectral_spread":
        return f"{dv:+.1f}"
    return f"{dv:+.0f}"


def _render_waveform() -> None:
    """Render the waveform chart focused on the last ~50ms."""
    st.subheader("Live Signal - CH1")
    if not st.session_state.last_waveform:
        st.info("No capture yet.")
        return
    samples, sr = st.session_state.last_waveform
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    win = max(64, int(sr * 0.050))
    arr = arr[-win:]
    t_ms = np.arange(arr.size) / float(sr) * 1000.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=arr,
            mode="lines",
            line={"color": "#00D4FF", "width": 2},
            name="CH1",
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (V)",
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics() -> None:
    """Render the feature metric cards."""
    st.subheader("Signal Metrics")
    feat = st.session_state.last_features
    if not feat:
        st.info("Waiting for first sample.")
        return
    display = format_metrics_display(feat)
    cols = st.columns(5)
    labels = [
        ("freq_hz", "Frequency (Hz)"),
        ("amplitude_rel", "Amplitude"),
        ("jitter_ms", "Jitter (ms)"),
        ("spectral_spread", "Spectral Spread"),
        ("zero_crossing_rate", "Zero-Crossing Rate"),
    ]
    for c, (key, title) in zip(cols, labels):
        c.metric(label=title, value=display[key], delta=_metric_delta(key))


def _render_diagnosis() -> None:
    """Render diagnosis status badge, confidence, and explanation text."""
    st.subheader("ML Diagnosis")
    pred = st.session_state.last_prediction
    if not pred:
        st.info("No prediction yet.")
        return
    mode, fault, confidence = pred
    if fault == "nominal":
        st.success("Circuit OK - Nominal")
    else:
        st.error(f"Fault Detected: {fault}")

    st.progress(min(1.0, max(0.0, float(confidence))), text=f"Confidence: {confidence:.0%}")
    st.caption(f"Mode: {mode}")
    st.write(st.session_state.last_diagnosis_text or "No diagnosis text available.")


def _render_chat_panel() -> None:
    """Render right-side chat UI and handle chat submissions."""
    st.subheader("AI Chat")
    if not st.session_state.conversation_history:
        st.caption("Try: What's wrong with my circuit?")
        st.caption("Try: Can you capture another reading?")
        st.caption("Try: Explain what jitter means")
        st.caption("Try: Try running at 3.3V")

    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask ScopeAI about your circuit...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("ScopeAI is thinking..."):
            response, updated_history = diagnose.chat(
                user_message=user_input,
                conversation_history=st.session_state.conversation_history,
                current_metrics=st.session_state.last_features,
                current_diagnosis=st.session_state.last_prediction,
            )
            st.session_state.conversation_history = updated_history
        with st.chat_message("assistant"):
            st.write(response)


def main() -> None:
    """Run Streamlit dashboard app."""
    _init_session_state()
    st.title("ScopeAI")

    if st.session_state.model_error:
        st.warning(st.session_state.model_error)

    with st.sidebar:
        st.header("Controls")
        circuit_mode = st.selectbox("Circuit Mode", options=["mode_A", "mode_B", "mode_C"], index=0)
        auto_sample = st.toggle("Auto-Sample", value=True)
        interval_sec = st.slider("Sample Interval (sec)", min_value=1, max_value=10, value=3)
        capture_now = st.button("Capture Now")

        info = st.session_state.capture_device or {}
        if info.get("simulation_mode"):
            st.markdown("🟡 **Simulation Mode**")
        elif info.get("connected"):
            st.markdown("🟢 **AD2 Connected**")
        else:
            st.markdown("🔴 **Disconnected**")

        st.caption(f"Supply: {float(info.get('supply_voltage', 0.0)):.2f}V")

        if st.session_state.simulation_mode:
            st.divider()
            st.subheader("Simulation Controls")
            sim_fault = st.selectbox(
                "Simulated Fault",
                options=["nominal", "R_too_high", "R_too_low", "cap_missing", "no_oscillation", "chatter"],
                index=0,
            )
            if st.button("Apply Simulated Fault"):
                try:
                    msg = capture.set_simulated_fault(sim_fault)
                    st.success(msg)
                except Exception as exc:
                    st.error(str(exc))

    # Trigger manual capture before rendering panels.
    if capture_now or st.session_state.last_features is None:
        _run_sample_cycle(circuit_mode=circuit_mode)

    left_col, right_col = st.columns([3, 2], vertical_alignment="top")

    with left_col:
        _render_waveform()
        _render_metrics()
        _render_diagnosis()

    with right_col:
        _render_chat_panel()

    if auto_sample:
        time.sleep(interval_sec)
        st.rerun()


if __name__ == "__main__":
    main()
