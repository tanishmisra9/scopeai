# ScopeAI — Definitive Project MVP & Hackathon Outline
### Catapult Hacks 2026 | ML@Purdue × The Founders Club | April 3–5, 2026 | WALC, Purdue University

---

## 1. The One-Line Pitch

> **ScopeAI turns any laptop into an AI-powered oscilloscope — plug in a circuit, get a diagnosis.**

---

## 2. The Problem

When a circuit doesn't work, there are no error messages. No stack traces. No debugger. You stare at a breadboard and guess: *Is the resistor backwards? Is the capacitor dead? Is the 555 timer fried?*

Professional oscilloscopes cost $200–$2,000 and require significant training. Students in introductory circuits labs — including thousands at Purdue — are left completely in the dark.

**ScopeAI eliminates the black box.**

---

## 3. The Solution

ScopeAI uses a standard **3.5mm audio cable** as a data bridge, feeding live electrical signals from a physical circuit directly into a Python-powered ML + LLM pipeline. The laptop's existing sound card becomes a 44.1 kHz sampling instrument. The AI becomes a senior electrical engineer sitting next to you.

### Three Layers of Intelligence

| Layer | What It Does | Demo Moment |
|-------|-------------|-------------|
| **Signal Parser** | Captures raw voltage via audio-in, runs FFT, extracts frequency/duty cycle/noise | Live waveform appears on screen instantly |
| **ML Classifier** | Pre-trained model identifies circuit mode and fault class from signal features | "Fault detected: capacitor disconnected" |
| **LLM Consultant** | Converts raw metrics into plain-English diagnosis and fix instructions | "Your frequency dropped 40%. This matches a 10× resistor increase in your 555 timer path." |

---

## 4. Hardware Architecture (From the Kit)

All hardware comes from the **Eureka Leap Purdue Dream Ice Lab kit**.

### Circuit Modes (Three Demo States)

**Mode A — NE555P Astable Oscillator (Primary Demo)**
- Components: NE555P timer, resistors from pack, electrolytic caps (2.2µF, 10µF, 100µF)
- Output: Clean square wave at a tunable frequency (200 Hz – 5 kHz range)
- Use 3386P trimmer resistor (10K) as a tunable timing element AND as a voltage safety divider before audio-in
- Fault injections: swap resistor values, disconnect timing cap, short resistor

**Mode B — RC Step/Decay (Exponential Curves)**
- Components: Resistors from pack + mylar film caps (0.1µF, 0.47µF) or electrolytic caps
- Output: Exponential charge/discharge curves
- AI task: infer RC time constant, identify wrong component value

**Mode C — LM339N Comparator Thresholding**
- Components: LM339N quad comparator, LM324N op-amp, resistors, trimmer
- Output: Clean digital transitions — or noisy chatter when threshold is near noise floor
- Fault injections: hysteresis too small → chatter; reference voltage drift

### Signal Path (Safety-Critical)
```
Circuit Output → 3386P Trimmer (voltage divider) → 3.5mm stereo cable → Laptop audio-in
```
- The 3386P trimmer acts as a **safety valve** — never allow more than ~100–200 mV peak into the sound card
- Use banana plug leads to probe the breadboard, connecting to the 3.5mm cable via the TRS jack (STX-3120-3B) in the kit
- Verify voltage with a known-safe signal before connecting

---

## 5. Software Architecture

### Full Stack
```
[Physical Circuit]
      ↓ (3.5mm audio cable)
[Sound Card — 44.1kHz ADC]
      ↓
[Python Acquisition — sounddevice + NumPy]
      ↓
[Signal Feature Engine — SciPy FFT, find_peaks, zero-crossing rate]
      ↓
[ML Classifier — scikit-learn (pre-trained, bundled model)]
      ↓
[LLM Reasoning Loop — OpenAI GPT-4o or Databricks]
      ↓
[Frontend Dashboard — Streamlit (MVP) or React]
```

### Module Breakdown

#### Module 1: Signal Acquisition (`capture.py`)
- Library: `sounddevice`
- Records N-second chunks into NumPy arrays at 44,100 Hz
- Outputs: raw time-series array + sample rate

#### Module 2: Feature Extraction (`features.py`)
- **FFT**: `numpy.fft.rfft` → fundamental frequency (Hz), harmonic content
- **Peak detection**: `scipy.signal.find_peaks` → duty cycle, period jitter
- **Time-domain**: rise/fall time estimation, amplitude envelope
- **Audio features** (optional, surprisingly effective): zero-crossing rate via `librosa`
- Output: feature dict `{freq_hz, duty_cycle_pct, amplitude_rel, jitter_ms, spectral_spread}`

#### Module 3: ML Classifier (`classifier.py`)
- Framework: `scikit-learn` (RandomForest or SVM — both train fast, are explainable)
- **Labels to train**:
  - Circuit mode: `{mode_A, mode_B, mode_C}`
  - Fault class: `{nominal, cap_missing, R_too_high, R_too_low, clipping, no_oscillation, chatter}`
- **Data collection plan** (can be done in 2–3 hours):
  - 30–50 recordings per class × 8 classes = ~400 samples
  - Record known-good and fault-injected states for each mode
  - Save as CSV: `[freq, duty, amplitude, jitter, spectral_spread, label]`
- Output: predicted `(circuit_mode, fault_class)` + confidence score
- **Save model with `joblib`** — bundle the `.pkl` with the repo

#### Module 4: LLM Reasoning Loop (`diagnose.py`)
- API: OpenAI GPT-4o (sponsor: OpenAI is a Catapult partner) or Databricks (also a sponsor)
- System prompt primes the model as a "hardware debugging expert"
- User message includes: circuit mode, detected fault class, raw feature metrics, user-described schematic
- Output: plain-English explanation + recommended fix steps
- Keep a conversation history for multi-turn debugging sessions

**Example prompt injection:**
```
System: You are ScopeAI, an expert hardware debugger for intro-level electronics students.
User: Circuit: 555 astable. Detected fault: R_too_high (confidence 94%).
      Metrics: freq=47Hz (baseline 200Hz), duty=52%, jitter=high.
      Explain what's wrong and how to fix it.
```

**Example response:**
> "Your oscillator slowed to about 47 Hz — roughly 4× slower than baseline. This is consistent with a timing resistor being 4× too large. Check R1 and R2 in your 555 circuit: you may have accidentally grabbed a 10kΩ from the pack where you needed 2.2kΩ. Swap it and re-run the analysis."

#### Module 5: Frontend (`app.py` — Streamlit for MVP)
- **Panel 1**: Live waveform plot (matplotlib + streamlit auto-refresh)
- **Panel 2**: Metrics dashboard (frequency, duty cycle, amplitude, jitter)
- **Panel 3**: ML diagnosis badge (circuit mode + fault class + confidence bar)
- **Panel 4**: AI chat window (LLM explanation + fix instructions)
- Optional: schematic upload or dropdown for circuit context

---

## 6. The 36-Hour Build Schedule

| Time Block | What Gets Built | Who Owns It |
|------------|----------------|-------------|
| **Hour 0–2** | Hardware setup: breadboard all three circuit modes, verify signal with headphones, verify safe voltage levels | Hardware lead |
| **Hour 2–5** | `capture.py` + `features.py`: get a live FFT running on screen | Signal/DSP lead |
| **Hour 5–9** | Data collection: record 400 labeled samples across all modes + faults | All hands |
| **Hour 9–14** | Train + validate ML classifier; target >85% accuracy | ML lead |
| **Hour 14–18** | `diagnose.py`: wire up LLM API, write system prompt, test explanations | AI/LLM lead |
| **Hour 18–24** | Streamlit dashboard: connect all modules, live demo flow works end-to-end | Frontend lead |
| **Hour 24–30** | Polish, edge cases, fault-injection rehearsal, record DevPost demo video | All |
| **Hour 30–36** | Finals pitch prep, slide deck, business framing, Q&A practice | All |

---

## 7. Award Category Strategy

ScopeAI is positioned to compete in **four** award categories simultaneously:

| Category | Prize | Why ScopeAI Qualifies |
|----------|-------|-----------------------|
| **Best Use of Hardware** | Raspberry Pi 5 Kit | Audio-jack-as-oscilloscope is a textbook hardware hack; real circuits, real signals |
| **Best ML Project** | RCAC GPU-Hours | Pre-trained classifier with labeled dataset, measurable accuracy, clear ML pipeline |
| **Most Promising Startup** | Bose SoundLink Micro | QA-as-a-Service pitch, education market, scalable to manufacturing |
| **Overall Winner** | MacBook Neo | Hits all four rubric pillars: creativity, technical difficulty, design, usefulness |

---

## 8. The Demo Script (Finals Pitch)

**[0:00–0:30] Hook**
> "Every software engineer has a debugger. Hardware engineers have nothing but guesswork."

**[0:30–1:30] Build the circuit live**
- Plug in the 555 astable oscillator. Show the waveform appear on screen in real time.
- Show the AI metrics panel: "200 Hz, 48% duty cycle, low jitter. Nominal."

**[1:30–2:30] Break something**
- Swap the timing resistor for a 10× larger value from the resistor pack.
- Waveform changes visibly. Metrics update. ML model fires: *"Fault: R_too_high — 92% confidence."*
- LLM explanation populates: *"Your frequency dropped 4×. This matches a timing resistor that's ~10× too large…"*

**[2:30–3:30] The Pitch**
- Market: Intro circuits labs (Purdue alone runs hundreds of ECE 201 sections/year)
- Business model: SaaS for makerspaces, universities, and eventually factory QA lines
- Moat: proprietary signal dataset + fine-tuned fault classifier

**[3:30–4:00] Close**
> "ScopeAI: because your breadboard deserves a debugger too."

---

## 9. Market & Business Case

### Immediate Market: Education
- Target: Intro circuits courses at universities (ECE 201, EE 101 equivalents)
- Pain: 1 TA per 30 students; hardware debugging is a bottleneck
- Offer: ScopeAI as a lab software tool — $10/student/semester or institutional license

### Medium-Term: Makers & Hobbyists
- 27M+ Arduino/Raspberry Pi users globally who don't own oscilloscopes
- Distribution: sell via Adafruit, SparkFun, or as a standalone app

### Long-Term: Manufacturing QA
- "Electrical signature" QA for assembled products on production lines
- Replaces manual spot-checking; AI flags anomalies in real time
- B2B SaaS: usage-based pricing per unit tested

### One-Line Startup Framing
> **ScopeAI is GitHub Copilot for hardware — an AI that reads your circuit and tells you what's broken.**

---

## 10. Judging Rubric Alignment

| Rubric Pillar | ScopeAI's Answer |
|---------------|-----------------|
| **Creativity** | Repurposing a 3.5mm audio jack as a scientific instrument is an elegant, unexpected hack |
| **Technical Difficulty** | Combines hardware engineering + DSP (FFT) + supervised ML + LLM prompt engineering + real-time UI |
| **Design** | High contrast: messy breadboard → clean AI dashboard. Live fault-injection makes it viscerally compelling |
| **Usefulness** | Solves a real, immediate pain for millions of students and makers globally |

---

## 11. Kit Component Usage Map

| Component | Role in ScopeAI |
|-----------|----------------|
| 2× NE555P | Astable oscillator (Mode A) — primary signal source |
| LM324N | Op-amp buffer/amplifier for signal conditioning |
| LM339N | Comparator circuit (Mode C) — chatter fault demo |
| 3386P Trimmers (×6) | Voltage divider (safety) + frequency tuning |
| Resistor Pack | Fault injection (swap values to change frequency) |
| Mylar caps (0.1µF, 0.47µF) | RC time constant circuit (Mode B) |
| Electrolytic caps (2.2µF–100µF) | 555 timing capacitors |
| 3.5mm Stereo Cable (6ft) | The core "data bridge" from circuit to laptop |
| TRS Jack (STX-3120-3B) | Breadboard-side connector for clean signal tap |
| Banana plug leads | Probing breadboard nodes safely |
| Breadboard (3220 tie-point) | All circuit prototyping |
| LEDs (R/G/Y) | Visual fault indicator on hardware side |
| Tactile switches | Manual mode switching on the breadboard |
| 5V power adapter | Clean power for all ICs |

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Audio-in damages laptop | Trimmer voltage divider limits input; test with multimeter before connecting |
| ML accuracy too low | Fall back to rule-based thresholds (frequency buckets) — still demo-able |
| LLM API rate limits | Cache common fault explanations locally; use Databricks as backup |
| No clean signal | LM386 audio amp in kit can boost weak signals before audio-in |
| Frontend not ready | Streamlit app is 50 lines — prioritize this over React polish |

---

*Built for Catapult Hacks 2026 — ML@Purdue × The Founders Club*
*April 3–5, 2026 | Wilmeth Active Learning Center, Purdue University*
