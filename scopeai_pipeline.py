#!/usr/bin/env python3
"""
ScopeAI Pipeline — Catapult Hacks 2026
=======================================
Run this script to:
  1. Load & validate the collected training data
  2. (Optional) Generate synthetic data for Mode B & Mode C if not collected
  3. Train the ML fault classifier (RandomForest)
  4. Evaluate with cross-validation + confusion matrix
  5. Export the trained model as a .pkl file for use in the Streamlit app

Usage:
  python scopeai_pipeline.py

Outputs:
  models/fault_classifier.pkl    — trained scikit-learn model
  models/label_encoder.pkl       — fitted LabelEncoder for fault classes
  models/mode_encoder.pkl        — fitted LabelEncoder for circuit modes
  models/training_report.txt     — accuracy, classification report, feature importances
  data/training_data_augmented.csv — full dataset (real + synthetic if generated)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FEATURES = ["freq_hz", "amplitude_rel", "jitter_ms", "spectral_spread", "zero_crossing_rate"]
RANDOM_STATE = 42

# How many synthetic samples per class for Mode B / Mode C
SYNTH_SAMPLES_PER_CLASS = 40


def load_real_data() -> pd.DataFrame:
    """Load the teammate-collected training data."""
    path = DATA_DIR / "training_data.csv"
    if not path.exists():
        # Try concatenating individual mode files
        csvs = sorted(DATA_DIR.glob("mode_*__*.csv"))
        if not csvs:
            print("ERROR: No training data found in data/ directory.")
            sys.exit(1)
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    else:
        df = pd.read_csv(path)

    print(f"Loaded {len(df)} real samples")
    print(f"  Circuit modes: {df['circuit_mode'].unique().tolist()}")
    print(f"  Fault classes: {df['fault_class'].unique().tolist()}")
    print(f"  Samples per fault class:")
    for cls, count in df["fault_class"].value_counts().items():
        print(f"    {cls}: {count}")
    return df


def generate_synthetic_data() -> pd.DataFrame:
    """
    Generate synthetic training data for Mode B (RC decay) and Mode C (comparator).
    Based on the signal characteristics described in the project outline.
    Uses Gaussian noise around expected feature centroids.

    This is a fallback — real data is always preferred.
    """
    np.random.seed(RANDOM_STATE)
    records = []
    n = SYNTH_SAMPLES_PER_CLASS

    # ── Mode B: RC Step/Decay ──
    # Characteristics: exponential charge/discharge curves
    # Features reflect time-constant-based behavior, not oscillation

    mode_b_profiles = {
        "nominal": {
            "freq_hz": (1.0, 0.1),         # ~1 Hz refresh rate for step input
            "amplitude_rel": (0.35, 0.02),  # clean voltage swing
            "jitter_ms": (1.5, 0.3),        # low jitter
            "spectral_spread": (150.0, 15.0),  # moderate — dominated by low freq
            "zero_crossing_rate": (1.5, 0.3),  # slow transitions
        },
        "cap_missing": {
            "freq_hz": (1.0, 0.1),
            "amplitude_rel": (0.40, 0.01),  # instant voltage jump, no decay
            "jitter_ms": (0.5, 0.2),        # very stable (just a wire)
            "spectral_spread": (2500.0, 200.0),  # broadband — step function
            "zero_crossing_rate": (1.0, 0.2),
        },
        "R_too_high": {
            "freq_hz": (1.0, 0.1),
            "amplitude_rel": (0.15, 0.03),  # slow charge, low amplitude in window
            "jitter_ms": (8.0, 2.0),        # sluggish response
            "spectral_spread": (50.0, 10.0),  # very low freq content
            "zero_crossing_rate": (0.5, 0.2),
        },
        "R_too_low": {
            "freq_hz": (1.0, 0.1),
            "amplitude_rel": (0.39, 0.01),  # fast charge, nearly full swing
            "jitter_ms": (0.8, 0.2),
            "spectral_spread": (300.0, 30.0),  # wider bandwidth
            "zero_crossing_rate": (2.0, 0.5),
        },
    }

    for fault, params in mode_b_profiles.items():
        for _ in range(n):
            row = {
                "timestamp": datetime.now().isoformat(),
                "label": f"mode_B__{fault}",
                "circuit_mode": "mode_B",
                "fault_class": fault,
            }
            for feat in FEATURES:
                mean, std = params[feat]
                row[feat] = max(0.0, np.random.normal(mean, std))
            records.append(row)

    # ── Mode C: LM339N Comparator Thresholding ──
    # Characteristics: clean digital transitions or noisy chatter

    mode_c_profiles = {
        "nominal": {
            "freq_hz": (5.0, 0.5),          # comparator switching rate
            "amplitude_rel": (0.38, 0.01),   # rail-to-rail output
            "jitter_ms": (0.3, 0.1),         # clean switching
            "spectral_spread": (200.0, 20.0),
            "zero_crossing_rate": (10.0, 1.5),
        },
        "chatter": {
            "freq_hz": (50.0, 10.0),         # rapid oscillation near threshold
            "amplitude_rel": (0.30, 0.05),   # partial swings
            "jitter_ms": (5.0, 1.5),         # very jittery
            "spectral_spread": (800.0, 100.0),  # broadband noise
            "zero_crossing_rate": (500.0, 80.0),  # tons of crossings
        },
        "no_oscillation": {
            "freq_hz": (0.0, 0.01),          # stuck output
            "amplitude_rel": (0.01, 0.005),  # near-zero signal variation
            "jitter_ms": (0.1, 0.05),
            "spectral_spread": (10.0, 3.0),
            "zero_crossing_rate": (0.1, 0.05),
        },
        "clipping": {
            "freq_hz": (5.0, 0.5),
            "amplitude_rel": (0.40, 0.002),  # hard at supply rail
            "jitter_ms": (0.5, 0.15),
            "spectral_spread": (1500.0, 150.0),  # harmonics from clipping
            "zero_crossing_rate": (10.0, 1.5),
        },
    }

    for fault, params in mode_c_profiles.items():
        for _ in range(n):
            row = {
                "timestamp": datetime.now().isoformat(),
                "label": f"mode_C__{fault}",
                "circuit_mode": "mode_C",
                "fault_class": fault,
            }
            for feat in FEATURES:
                mean, std = params[feat]
                row[feat] = max(0.0, np.random.normal(mean, std))
            records.append(row)

    synth_df = pd.DataFrame(records)
    print(f"\nGenerated {len(synth_df)} synthetic samples")
    print(f"  Mode B classes: {synth_df[synth_df['circuit_mode']=='mode_B']['fault_class'].unique().tolist()}")
    print(f"  Mode C classes: {synth_df[synth_df['circuit_mode']=='mode_C']['fault_class'].unique().tolist()}")
    return synth_df


def train_and_evaluate(df: pd.DataFrame):
    """Train the classifier, evaluate, and save artifacts."""
    MODEL_DIR.mkdir(exist_ok=True)

    X = df[FEATURES].values
    y_fault = df["fault_class"].values
    y_mode = df["circuit_mode"].values

    # Encode labels
    le_fault = LabelEncoder().fit(y_fault)
    le_mode = LabelEncoder().fit(y_mode)
    y_fault_enc = le_fault.transform(y_fault)
    y_mode_enc = le_mode.transform(y_mode)

    # Combined label for multi-output classification
    # We'll use (circuit_mode, fault_class) as a composite label
    y_combined = df["label"].values
    le_combined = LabelEncoder().fit(y_combined)
    y_combined_enc = le_combined.transform(y_combined)

    # ── Cross-Validation ──
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold stratified)")
    print("="*60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Try multiple classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=3,
            random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=RANDOM_STATE
        ),
    }

    best_name, best_score, best_clf = None, 0, None
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y_combined_enc, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        print(f"  {name:25s}  accuracy = {mean_acc:.4f} (+/- {scores.std():.4f})")
        if mean_acc > best_score:
            best_name, best_score, best_clf = name, mean_acc, clf

    print(f"\n  Best model: {best_name} ({best_score:.4f})")

    # ── Train/Test Split for detailed report ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_combined_enc, test_size=0.2, stratify=y_combined_enc, random_state=RANDOM_STATE
    )
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n  Holdout test accuracy: {test_acc:.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=le_combined.classes_,
        zero_division=0
    )
    print("\n  Classification Report:")
    for line in report.split("\n"):
        print(f"    {line}")

    cm = confusion_matrix(y_test, y_pred)
    print("\n  Confusion Matrix:")
    print(f"    Labels: {le_combined.classes_.tolist()}")
    for row in cm:
        print(f"    {row}")

    # ── Final model: retrain on ALL data ──
    print("\n" + "="*60)
    print("FINAL MODEL — trained on all data")
    print("="*60)
    best_clf.fit(X, y_combined_enc)

    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        print("\n  Feature Importances:")
        for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"    {feat:25s}  {imp:.4f}  {bar}")

    # ── Save artifacts ──
    joblib.dump(best_clf, MODEL_DIR / "fault_classifier.pkl")
    joblib.dump(le_combined, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(le_fault, MODEL_DIR / "fault_encoder.pkl")
    joblib.dump(le_mode, MODEL_DIR / "mode_encoder.pkl")

    # Save feature list for the app to reference
    meta = {
        "features": FEATURES,
        "combined_classes": le_combined.classes_.tolist(),
        "fault_classes": le_fault.classes_.tolist(),
        "mode_classes": le_mode.classes_.tolist(),
        "best_model": best_name,
        "cv_accuracy": float(best_score),
        "test_accuracy": float(test_acc),
        "n_samples": len(df),
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save report
    with open(MODEL_DIR / "training_report.txt", "w") as f:
        f.write(f"ScopeAI Fault Classifier — Training Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: {best_name}\n")
        f.write(f"CV Accuracy: {best_score:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Total Samples: {len(df)}\n\n")
        f.write(f"Classification Report:\n{report}\n\n")
        f.write(f"Feature Importances:\n")
        if hasattr(best_clf, "feature_importances_"):
            for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
                f.write(f"  {feat}: {imp:.4f}\n")

    print(f"\n  Saved to {MODEL_DIR}/:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(f"    {p.name} ({p.stat().st_size:,} bytes)")

    return best_clf, le_combined, meta


def main():
    print("="*60)
    print("  ScopeAI — ML Pipeline")
    print("  Catapult Hacks 2026")
    print("="*60)

    # Step 1: Load real data
    print("\n[1/4] Loading collected data...")
    real_df = load_real_data()

    # Step 2: Check if we need synthetic data
    modes_present = set(real_df["circuit_mode"].unique())
    modes_needed = {"mode_A", "mode_B", "mode_C"}
    missing_modes = modes_needed - modes_present

    if missing_modes:
        print(f"\n[2/4] Missing circuit modes: {missing_modes}")
        print("      Generating synthetic data as fallback...")
        synth_df = generate_synthetic_data()
        # Only keep synthetic data for missing modes
        synth_df = synth_df[synth_df["circuit_mode"].isin(missing_modes)]
        full_df = pd.concat([real_df, synth_df], ignore_index=True)
    else:
        print("\n[2/4] All circuit modes present — no synthetic data needed")
        full_df = real_df

    # Save augmented dataset
    full_df.to_csv(DATA_DIR / "training_data_augmented.csv", index=False)
    print(f"\n      Full dataset: {len(full_df)} samples")
    print(f"      Saved to {DATA_DIR / 'training_data_augmented.csv'}")

    # Step 3: Train & evaluate
    print("\n[3/4] Training classifier...")
    clf, le, meta = train_and_evaluate(full_df)

    # Step 4: Quick inference demo
    print("\n[4/4] Inference demo...")
    print("      Simulating a live signal prediction:\n")

    # Grab a random sample from each class for demo
    for label in sorted(full_df["label"].unique()):
        sample = full_df[full_df["label"] == label].iloc[0]
        x = sample[FEATURES].values.reshape(1, -1)
        pred_enc = clf.predict(x)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        proba = clf.predict_proba(x)[0]
        confidence = proba.max() * 100

        mode, fault = pred_label.split("__")
        print(f"      Input: {label}")
        print(f"      → Predicted: mode={mode}, fault={fault} ({confidence:.1f}% confidence)")
        print()

    print("="*60)
    print("  Pipeline complete! Next steps:")
    print("  1. Copy models/ directory into your Streamlit app repo")
    print("  2. Wire up capture.py → features.py → classifier prediction")
    print("  3. Feed results into diagnose.py (LLM reasoning loop)")
    print("  4. Run: streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
