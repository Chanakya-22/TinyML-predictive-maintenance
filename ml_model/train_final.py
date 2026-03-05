import numpy as np
import pandas as pd
import joblib
import os
from scipy.fft import fft, fftfreq
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# FAULT CLASS DEFINITIONS
# ============================================================
# 0 = HEALTHY
# 1 = BEARING_INNER_RACE  (BPFI harmonics + sidebands, high kurtosis)
# 2 = ROTOR_UNBALANCE     (dominant 1X frequency, low kurtosis)
# 3 = MISALIGNMENT        (dominant 2X frequency, axial vibration)
# 4 = LOOSENESS           (sub-harmonic 0.5X + broadband noise)

FAULT_NAMES = {
    0: "HEALTHY",
    1: "BEARING_INNER_RACE",
    2: "ROTOR_UNBALANCE",
    3: "MISALIGNMENT",
    4: "LOOSENESS"
}

# ============================================================
# PHYSICS-BASED SIGNAL GENERATOR
# ============================================================
SAMPLE_RATE = 20000  # 20 kHz — standard for bearing diagnostics
DURATION    = 0.5    # 0.5 second window per sample
SHAFT_FREQ  = 30     # Hz — 1800 RPM / 60 = 30 Hz

# Bearing geometry constants (SKF 6205-2RS typical values)
NUM_BALLS       = 9
BALL_DIAMETER   = 7.94   # mm
PITCH_DIAMETER  = 38.5   # mm
CONTACT_ANGLE   = 0      # degrees (radial bearing)

# Characteristic defect frequencies (multiples of shaft freq)
BPFI = SHAFT_FREQ * (NUM_BALLS / 2) * (1 + (BALL_DIAMETER / PITCH_DIAMETER))  # ≈ 162 Hz
BPFO = SHAFT_FREQ * (NUM_BALLS / 2) * (1 - (BALL_DIAMETER / PITCH_DIAMETER))  # ≈ 108 Hz
BSF  = SHAFT_FREQ * (PITCH_DIAMETER / (2 * BALL_DIAMETER)) * (1 - (BALL_DIAMETER / PITCH_DIAMETER)**2)  # ≈ 71 Hz


def generate_signal(fault_type=0, severity=1.0, sample_rate=SAMPLE_RATE, duration=DURATION):
    """
    Generate a realistic vibration signal for a given fault type and severity.
    severity: 0.0 (incipient) → 1.0 (advanced fault)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Always-present shaft rotation component
    base = 0.05 * np.sin(2 * np.pi * SHAFT_FREQ * t)

    # ── HEALTHY ──────────────────────────────────────────────────────────────
    if fault_type == 0:
        noise_level = np.random.uniform(0.02, 0.07)
        # 25% chance of "noisy healthy" to prevent overconfidence
        if np.random.random() < 0.25:
            noise_level = np.random.uniform(0.08, 0.14)
        return base + np.random.normal(0, noise_level, len(t))

    # ── BEARING INNER RACE FAULT ──────────────────────────────────────────────
    elif fault_type == 1:
        # Physics: periodic impacts at BPFI, modulated by shaft rotation
        # Results in high kurtosis, sidebands around BPFI
        amp = severity * np.random.uniform(0.05, 0.12)
        # Primary BPFI harmonic
        impact = amp * np.sin(2 * np.pi * BPFI * t)
        # First sideband (BPFI ± shaft_freq) — key diagnostic signature
        sb1 = (amp * 0.4) * np.sin(2 * np.pi * (BPFI + SHAFT_FREQ) * t)
        sb2 = (amp * 0.4) * np.sin(2 * np.pi * (BPFI - SHAFT_FREQ) * t)
        # 2nd harmonic
        h2  = (amp * 0.25) * np.sin(2 * np.pi * (2 * BPFI) * t)
        noise = np.random.normal(0, 0.04, len(t))
        # Incipient fault simulation — 20% chance of barely-detectable fault
        if severity < 0.3:
            amp *= 0.2
        return base + impact + sb1 + sb2 + h2 + noise

    # ── ROTOR UNBALANCE ───────────────────────────────────────────────────────
    elif fault_type == 2:
        # Physics: pure 1X forced vibration — low kurtosis, high RMS
        amp = severity * np.random.uniform(0.15, 0.35)
        unbalance = amp * np.sin(2 * np.pi * SHAFT_FREQ * t + np.random.uniform(0, 2*np.pi))
        # Very small harmonics (nearly pure sinusoid)
        h2 = (amp * 0.05) * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t)
        noise = np.random.normal(0, 0.02, len(t))
        return base + unbalance + h2 + noise

    # ── SHAFT MISALIGNMENT ────────────────────────────────────────────────────
    elif fault_type == 3:
        # Physics: dominant 2X, significant axial component, some 1X
        amp_2x = severity * np.random.uniform(0.10, 0.22)
        amp_1x = amp_2x * np.random.uniform(0.3, 0.6)
        misalign_2x = amp_2x * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t)
        misalign_1x = amp_1x * np.sin(2 * np.pi * SHAFT_FREQ * t + np.pi / 3)
        # Some 3X as well for angular misalignment
        h3 = (amp_2x * 0.3) * np.sin(2 * np.pi * 3 * SHAFT_FREQ * t)
        noise = np.random.normal(0, 0.03, len(t))
        return base + misalign_1x + misalign_2x + h3 + noise

    # ── MECHANICAL LOOSENESS ──────────────────────────────────────────────────
    elif fault_type == 4:
        # Physics: sub-harmonics (0.5X), broadband noise floor, chaotic response
        amp_sub = severity * np.random.uniform(0.06, 0.14)
        sub_harm = amp_sub * np.sin(2 * np.pi * 0.5 * SHAFT_FREQ * t)
        # Multiple shaft harmonics (truncation of nonlinear response)
        harmonics = sum(
            (amp_sub / k) * np.sin(2 * np.pi * k * SHAFT_FREQ * t)
            for k in range(1, 6)
        )
        # High broadband noise floor (chaotic looseness)
        broadband = np.random.normal(0, severity * 0.10, len(t))
        return base + sub_harm + harmonics + broadband

    return base + np.random.normal(0, 0.03, len(t))


# ============================================================
# FEATURE EXTRACTION  (7 features per sample)
# ============================================================

def extract_features(signal, sample_rate=SAMPLE_RATE):
    """
    Extract time-domain and frequency-domain features.
    Feature vector: [RMS, Kurtosis, Crest_Factor, Sub_Sync_Energy, Sync_Energy, High_Freq_Energy, Spectral_Kurtosis]
    """
    # ── Time Domain ──────────────────────────────────────────
    rms     = np.sqrt(np.mean(signal ** 2))
    peak    = np.max(np.abs(signal))
    kurt    = float(pd.Series(signal).kurtosis())
    crest   = peak / (rms + 1e-9)

    # ── Frequency Domain ─────────────────────────────────────
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N // 2] / N
    freqs    = fftfreq(N, 1 / sample_rate)[:N // 2]

    # Frequency band energy (key discriminative features)
    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(fft_vals[mask] ** 2))

    sub_sync_energy  = band_energy(5,  25)    # Sub-synchronous: looseness
    sync_energy      = band_energy(25, 75)    # Synchronous: unbalance / misalign
    high_freq_energy = band_energy(75, 500)   # High freq: bearing defects

    # Spectral kurtosis (broadband randomness indicator — looseness diagnostic)
    psd = fft_vals ** 2
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spec_kurt = float(pd.Series(psd_norm).kurtosis())

    return [rms, kurt, crest, sub_sync_energy, sync_energy, high_freq_energy, spec_kurt]


FEATURE_NAMES = ['RMS', 'Kurtosis', 'Crest_Factor', 'Sub_Sync_Energy', 'Sync_Energy', 'High_Freq_Energy', 'Spectral_Kurtosis']


# ============================================================
# DATA GENERATION
# ============================================================

def generate_dataset(samples_per_class=600):
    print("\n  Generating physics-based multi-class dataset...")
    X, y = [], []

    for fault_type in range(5):
        name = FAULT_NAMES[fault_type]
        print(f"    [{fault_type}] {name}: {samples_per_class} samples")
        for _ in range(samples_per_class):
            # Random severity — uniform for healthy, weighted towards higher for faults
            if fault_type == 0:
                severity = np.random.uniform(0.0, 0.3)
            else:
                # 20% incipient (low severity) — real-world class imbalance
                severity = np.random.uniform(0.1, 0.3) if np.random.random() < 0.20 else np.random.uniform(0.4, 1.0)

            sig = generate_signal(fault_type=fault_type, severity=severity)
            X.append(extract_features(sig))
            y.append(fault_type)

    return np.array(X), np.array(y)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_pipeline():
    print("=" * 60)
    print("  INDUSTRIAL PREDICTIVE MAINTENANCE — MODEL TRAINING")
    print("  Multi-Class Fault Classifier + Anomaly Health Scorer")
    print("=" * 60)

    # ── 1. Generate Dataset ───────────────────────────────────
    X, y = generate_dataset(samples_per_class=600)
    print(f"\n  Total samples: {len(X)}  |  Features: {X.shape[1]}  |  Classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ── 2. Train Multi-Class GBM Classifier ──────────────────
    print("\n  Training Gradient Boosting Classifier (multi-class)...")
    clf = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        min_samples_leaf=8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    preds   = clf.predict(X_test)
    acc     = accuracy_score(y_test, preds)
    print(f"\n  ✅ Multi-Class Accuracy: {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=[FAULT_NAMES[i] for i in range(5)]))

    # ── 3. Train Isolation Forest (Health Scorer) ─────────────
    # Trained ONLY on healthy data — gives anomaly score for any input
    print("  Training Isolation Forest (anomaly health scorer)...")
    X_healthy = X[y == 0]
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_healthy)
    print("  ✅ Isolation Forest trained on healthy baseline.")

    # ── 4. Confusion Matrix Plot ──────────────────────────────
    print("\n  Generating evaluation plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0b0e14')
    for ax in axes:
        ax.set_facecolor('#151a23')

    cm = confusion_matrix(y_test, preds)
    class_labels = [FAULT_NAMES[i] for i in range(5)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[0], linewidths=0.5, linecolor='#1e293b')
    axes[0].set_title(f'Confusion Matrix  (Acc: {acc*100:.1f}%)', color='white', pad=15, fontsize=13)
    axes[0].set_ylabel('Actual', color='#94a3b8')
    axes[0].set_xlabel('Predicted', color='#94a3b8')
    axes[0].tick_params(colors='#94a3b8')
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8, color='#94a3b8')
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=8, color='#94a3b8')

    # Feature Importance plot
    importances = clf.feature_importances_
    sorted_idx  = np.argsort(importances)
    axes[1].barh([FEATURE_NAMES[i] for i in sorted_idx], importances[sorted_idx],
                 color=['#0ea5e9' if i > 3 else '#f59e0b' for i in sorted_idx])
    axes[1].set_title('Feature Importance', color='white', pad=15, fontsize=13)
    axes[1].tick_params(colors='#94a3b8')
    axes[1].set_facecolor('#151a23')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#1e293b')

    plt.tight_layout()
    os.makedirs('ml_model', exist_ok=True)
    plt.savefig('ml_model/training_report.png', dpi=150, bbox_inches='tight', facecolor='#0b0e14')
    print("  📊 Saved: ml_model/training_report.png")

    # ── 5. Save Models ────────────────────────────────────────
    joblib.dump(clf, 'ml_model/sota_model_final.pkl')
    joblib.dump(iso, 'ml_model/health_scorer.pkl')
    joblib.dump(FEATURE_NAMES, 'ml_model/feature_names.pkl')

    print("\n  💾 Models saved:")
    print("     ml_model/sota_model_final.pkl  (multi-class fault classifier)")
    print("     ml_model/health_scorer.pkl     (isolation forest health scorer)")
    print("\n" + "=" * 60)
    print("  Training complete. Run: python dashboard/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()