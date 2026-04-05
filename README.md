# ScopeAI

> **Turn any laptop into an AI-powered oscilloscope. Plug in a circuit, get a diagnosis.**

Built at **Catapult Hacks 2026** — ML@Purdue × The Founders Club | April 3–5, 2026 | WALC, Purdue University

---

## What It Does

When a circuit doesn't work, there are no error messages. No stack traces. No debugger — just a breadboard and guesswork. Professional oscilloscopes cost $200–$2,000 and require significant training. ScopeAI eliminates the black box.

Connect a **Digilent Analog Discovery 2 (AD2)** to your laptop via USB. ScopeAI:

1. **Captures** electrical signals at 44.1 kHz via the AD2's built-in oscilloscope
2. **Extracts** 5 physics-grounded features: frequency, amplitude, jitter, spectral spread, zero-crossing rate
3. **Classifies** circuit mode and fault type using a pre-trained RandomForest model (100% holdout accuracy across 13 fault classes)
4. **Explains** what's wrong in plain English via GPT-4o, grounded in your actual signal data — not generic advice

**Example:** Swap a timing resistor for a 10× larger value. The waveform changes. The ML model immediately flags `R_too_high` at 66% confidence. GPT-4o explains: *"Your oscillator slowed from 200 Hz to 47 Hz. This matches a timing resistor that's ~10× too large. Swap R2 for a 2.2kΩ resistor."*

---

## Architecture

```
[Physical Circuit]
      │  (AD2 USB)
[Analog Discovery 2 — 44.1 kHz ADC + 5V programmable power supply]
      │
[capture.py — WaveForms SDK via ctypes, simulation fallback]
      │
[features.py — FFT, find_peaks, zero-crossing rate, spectral spread]
      │
[ML Classifier — scikit-learn RandomForest, pre-trained, bundled .pkl]
      │
[diagnose.py — GPT-4o auto-diagnosis + interactive chat with hardware tools]
      │
[app.py — Streamlit live dashboard + AI chat panel]
```

### Three Circuit Modes

| Mode | Circuit | Data Source |
|------|---------|-------------|
| **Mode A** | NE555P Astable Oscillator | Real (AD2 captures) |
| **Mode B** | RC Step/Decay | Synthetic |
| **Mode C** | LM339N Comparator | Synthetic |

### Fault Classes (13 total)

`nominal`, `R_too_high`, `R_too_low`, `cap_missing`, `no_oscillation` (Modes A & B) + `chatter`, `clipping` (Mode C)

---

## ML Model

| Metric | Value |
|--------|-------|
| Model | RandomForestClassifier (200 estimators) |
| Training samples | 520 (200 real + 320 synthetic) |
| Classes | 13 |
| CV accuracy (5-fold stratified) | **100%** |
| Holdout test accuracy | **100%** |

**Feature importances:**

| Feature | Importance |
|---------|-----------|
| `spectral_spread` | 31.3% |
| `freq_hz` | 21.0% |
| `zero_crossing_rate` | 18.5% |
| `amplitude_rel` | 14.9% |
| `jitter_ms` | 14.4% |

---

## Hardware Setup

### Required
- **Digilent Analog Discovery 2** (USB oscilloscope + programmable power supply)
- NE555P timer IC, LM339N comparator, resistors, electrolytic capacitors, trimmer potentiometer

### Signal Path
```
NE555P Pin 3 (Output) ──► AD2 CH1+ (probe)
                           AD2 CH1- ──► GND
AD2 V+ (5.0V)         ──► Breadboard power rail
AD2 GND               ──► Breadboard ground rail
AD2 USB               ──► Laptop
```

At startup, the app enables the AD2 power supply programmatically — no external bench supply needed.

---

## Installation

### 1. System dependency (required for real hardware)

Install [Digilent WaveForms](https://digilent.com/reference/software/waveforms/waveforms-3/start) to get the WaveForms SDK (`libdwf`). This is not a pip package.

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. OpenAI API key

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Running ScopeAI

### With AD2 connected

```bash
streamlit run app.py
```

### Simulation mode (no hardware needed)

```bash
SCOPEAI_SIMULATE=1 streamlit run app.py
```

In simulation mode, the app generates synthetic waveforms tuned to match real training data distributions. All features — ML classification, LLM diagnosis, AI chat — still work.

---

## Dashboard

Two-column layout:

**Left — Live Dashboard (auto-refreshing)**
- Real-time waveform (Plotly, last ~50ms of signal)
- 5 signal metric cards with delta from previous reading
- ML diagnosis badge (green / red) with confidence bar and GPT-4o explanation

**Right — AI Chat**
- Multi-turn conversation with GPT-4o
- The LLM has four hardware tools it can call:
  - `capture_signal` — trigger a new AD2 capture and run the full pipeline
  - `analyze_waveform` — deeper analysis: harmonics, jitter stats, SNR
  - `set_voltage` — adjust the AD2 supply voltage (0–5V)
  - `get_device_status` — return AD2 connection info and current config

**Sidebar controls:** circuit mode selector, auto-sample toggle (1–10s interval), manual capture button, simulation fault selector

---

## Data Collection

To collect new training samples with real hardware:

```bash
# Single sample
python collect_data.py --label "mode_A__nominal"

# Batch of 40 samples
python collect_data.py --label "mode_A__R_too_high" --n_samples 40
```

Features are appended to `data/training_data.csv`. The script includes a safety confirmation prompt, 3-second countdown before each recording, and a live feature summary after each capture.

---

## Retraining the Model

```bash
python scopeai_pipeline.py
```

Outputs `fault_classifier.pkl`, `label_encoder.pkl`, and `model_meta.json` into `models/`. Copy the `.pkl` and `model_meta.json` to the project root as well (the dashboard loads from both locations).

---

## File Structure

```
scopeai/
├── app.py                  Streamlit live dashboard + AI chat
├── capture.py              AD2 signal acquisition + simulation fallback
├── features.py             Feature extraction (5 features)
├── diagnose.py             GPT-4o auto-diagnosis + chat with tool calling
├── collect_data.py         CLI data collection tool
├── scopeai_pipeline.py     ML training pipeline
├── requirements.txt
├── fault_classifier.pkl    Trained RandomForest (root copy)
├── model_meta.json         Model metadata (root copy)
└── models/
    ├── fault_classifier.pkl
    ├── label_encoder.pkl
    ├── fault_encoder.pkl
    ├── mode_encoder.pkl
    ├── model_meta.json
    └── training_report.txt
```

---

## Simulation vs. Real Hardware

| Feature | Simulation | Real AD2 |
|---------|-----------|----------|
| Waveform capture | Synthetic (Gaussian profiles) | 44.1 kHz live signal |
| Power supply control | No-op | 0–5V programmable via AD2 |
| ML classification | ✅ | ✅ |
| GPT-4o diagnosis | ✅ | ✅ |
| AI chat + tool calling | ✅ | ✅ |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| AD2 not detected at demo | Simulation mode auto-activates with warning banner |
| ML accuracy drops on new real signals | Heuristic rule-based fallback built into `app.py` and `diagnose.py` |
| Feature scale mismatch (train vs. live) | `features.py` uses identical extraction logic to `collect_data.py`; verified end-to-end in simulation |
| LLM API rate limits or outage | Auto-diagnosis cached by fault class; local fallback text when API is down |
| scikit-learn version mismatch | Model trained with 1.8.0; run `pip install --upgrade scikit-learn` |

---

## What's Next

- Expand fault coverage: solder bridges, broken traces, component tolerance outliers, power supply noise
- Circuit schematic upload — give the LLM structural context alongside signal data
- Mobile app with local model inference (no oscilloscope or external API required)
- Manufacturing QA use case: electrical signature testing on production lines

---

## Built With

Python · scikit-learn · GPT-4o · Streamlit · Plotly · Digilent WaveForms SDK · NE555P · LM339N

---

*ScopeAI: because your breadboard deserves a debugger too.*
