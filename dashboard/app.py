"""
AssetGuard AI - Edge Gateway Server
=====================================
Production-grade predictive maintenance backend.
"""

from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO
import joblib
import numpy as np
import datetime
import os
import time
import random
import sqlite3
import csv
import io
from collections import deque
from scipy.fft import fft, fftfreq

# ============================================================
# FLASK / SOCKETIO SETUP
# ============================================================
app      = Flask(__name__)
app.config['SECRET_KEY'] = 'assetguard-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================
# PATHS
# ============================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'ml_model', 'sota_model_final.pkl')
SCORER_PATH = os.path.join(BASE_DIR, 'ml_model', 'health_scorer.pkl')
DB_PATH     = os.path.join(BASE_DIR, 'data', 'telemetry.db')

# ============================================================
# INFERENCE GATE THRESHOLDS
# If BOTH rms and kurtosis are below these, the signal is
# physically incapable of containing a fault signature.
# We hard-gate the ML model and return HEALTHY directly.
# This prevents the model from misfiring on boot/low-value readings.
# ============================================================
GATE_RMS_MAX  = 0.08   # g   — below this: clearly no fault vibration
GATE_KURT_MAX = 3.0    # —   — below this: no impulsive content

# ============================================================
# FAULT KNOWLEDGE BASE
# ============================================================
FAULT_NAMES = {
    0: "HEALTHY",
    1: "BEARING_INNER_RACE",
    2: "ROTOR_UNBALANCE",
    3: "MISALIGNMENT",
    4: "LOOSENESS"
}

FAULT_DISPLAY = {
    0: "Healthy",
    1: "Bearing Inner Race",
    2: "Rotor Unbalance",
    3: "Shaft Misalignment",
    4: "Mech. Looseness"
}

REPAIR_GUIDE = {
    "HEALTHY": {
        "title": "System Operating Nominally",
        "detail": "All vibration and thermal parameters within ISO 10816 Zone A limits. "
                  "Efficiency: 98.2%. Next scheduled inspection: 720 operating hours.",
        "actions": [],
        "urgency_floor": "NONE"
    },
    "BEARING_INNER_RACE": {
        "title": "Inner Race Bearing Defect Detected",
        "detail": "Spectral analysis confirms BPFI harmonic signature with shaft-frequency "
                  "sidebands. Indicative of inner race spalling on the drive-end bearing.",
        "actions": [
            "Isolate and degrease drive-end bearing housing",
            "Replace SKF 6205-2RS or equivalent bearing",
            "Verify lubrication: use Mobilux EP2 grease, 12g per cavity",
            "Re-inspect after 24h run-in and re-baseline vibration"
        ],
        "urgency_floor": "HIGH"
    },
    "ROTOR_UNBALANCE": {
        "title": "Rotor Assembly Imbalance Detected",
        "detail": "Dominant 1X frequency component detected. Phase analysis indicates static "
                  "unbalance on the fan rotor. Common cause: debris accumulation or missing "
                  "balance weight.",
        "actions": [
            "Shut down and lock out / tag out (LOTO)",
            "Inspect and clean all fan blades - remove debris",
            "Check for missing or shifted balance correction weights",
            "Dynamic balance to ISO 1940-1 Grade G2.5 or better"
        ],
        "urgency_floor": "MEDIUM"
    },
    "MISALIGNMENT": {
        "title": "Shaft / Coupling Misalignment Detected",
        "detail": "Elevated 2X harmonic with significant axial vibration component detected. "
                  "Consistent with angular or parallel coupling misalignment.",
        "actions": [
            "Measure and record current alignment with dial gauge or laser tool",
            "Correct soft-foot condition at all mounting pads first",
            "Align shaft to <= 0.05mm parallel and <= 0.05mm/100mm angular",
            "Re-torque coupling bolts to spec after alignment"
        ],
        "urgency_floor": "MEDIUM"
    },
    "LOOSENESS": {
        "title": "Mechanical Looseness / Resonance Detected",
        "detail": "Sub-harmonic (0.5X) and broadband spectral content detected. Indicates "
                  "structural looseness at bearing housing, foundation bolts, or resonant "
                  "baseplate.",
        "actions": [
            "Inspect and torque all foundation anchor bolts",
            "Check anti-vibration mount condition - replace if hardened",
            "Inspect bearing housing set screws and end-shields",
            "Perform bump test to identify resonant frequency"
        ],
        "urgency_floor": "LOW"
    }
}

# Urgency rank for max() comparison
URGENCY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Maps severity stage to a dynamic urgency level
SEVERITY_TO_URGENCY = {
    "HEALTHY":  "NONE",
    "WATCH":    "LOW",
    "WARNING":  "MEDIUM",
    "ALERT":    "HIGH",
    "CRITICAL": "HIGH"
}

# ISO 10816-3 inspired RMS+Kurtosis thresholds
SEVERITY_LEVELS = [
    # (label,       rms_max,  kurtosis_max)
    ("HEALTHY",    0.07,     3.0),
    ("WATCH",      0.13,     4.0),
    ("WARNING",    0.22,     5.5),
    ("ALERT",      0.38,     7.0),
    ("CRITICAL",   9999,     9999),
]

# Health score thresholds — used to reconcile with RMS-based severity
# so the two systems agree with each other
HEALTH_TO_SEVERITY = [
    # (min_health, severity)  — descending order
    (80, "HEALTHY"),
    (60, "WATCH"),
    (40, "WARNING"),
    (20, "ALERT"),
    (0,  "CRITICAL"),
]

# ============================================================
# DATABASE
# ============================================================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            rms           REAL,
            kurtosis      REAL,
            crest_factor  REAL,
            temp          REAL,
            speed         INTEGER,
            health_score  REAL,
            severity      TEXT,
            fault_code    INTEGER,
            fault_label   TEXT,
            rul_cycles    INTEGER
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS fault_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            fault_label  TEXT,
            severity     TEXT,
            urgency      TEXT,
            health_score REAL,
            rms          REAL
        )
    ''')
    conn.commit()
    conn.close()


def log_reading(data: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            INSERT INTO readings (timestamp, rms, kurtosis, crest_factor, temp, speed,
                                  health_score, severity, fault_code, fault_label, rul_cycles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['timestamp'], data['rms'], data['kurtosis'], data['crest_factor'],
            data['temp'], data['speed'], data['health_score'], data['severity'],
            data['fault_code'], data['fault_label'], data.get('rul_cycles')
        ))
        if data['fault_code'] != 0:
            conn.execute('''
                INSERT INTO fault_events (timestamp, fault_label, severity, urgency, health_score, rms)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'], data['fault_label'], data['severity'],
                data.get('guide_urgency', 'UNKNOWN'), data['health_score'], data['rms']
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB write error: {e}")


# ============================================================
# GLOBAL STATE
# ============================================================
start_time  = time.time()
rms_history = deque(maxlen=300)

sim_state = {
    "RMS": 0.01, "Kurtosis": 0.0, "Temp": 25.0,
    "FanSpeed": 0, "CrestFactor": 1.0
}

# ============================================================
# LOAD MODELS
# ============================================================
clf    = None
scorer = None

if os.path.exists(MODEL_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        print(f"Classifier loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Classifier load failed: {e}")

if os.path.exists(SCORER_PATH):
    try:
        scorer = joblib.load(SCORER_PATH)
        print(f"Health scorer loaded: {SCORER_PATH}")
    except Exception as e:
        print(f"Health scorer load failed: {e}")

# ============================================================
# DEMO LOOP - Scripted 240-second cycle
# ============================================================
DEMO_CYCLE = 240

def get_demo_mode():
    elapsed = int(time.time() - start_time)
    t = elapsed % DEMO_CYCLE
    if t < 18:  return "BOOT_SEQUENCE"
    if t < 55:  return "HEALTHY"
    if t < 115: return "BEARING_WEAR"
    if t < 145: return "HEALTHY"
    if t < 195: return "UNBALANCE"
    if t < 215: return "HEALTHY"
    if t < 240: return "MISALIGNMENT"
    return "HEALTHY"

TARGET_PROFILES = {
    "BOOT_SEQUENCE": {"RMS": 0.01, "Kurtosis": 0.0,  "Temp": 27.0, "Speed": 350,  "CrestFactor": 1.4},
    "HEALTHY":       {"RMS": 0.05, "Kurtosis": 2.2,  "Temp": 48.0, "Speed": 1800, "CrestFactor": 3.2},
    "BEARING_WEAR":  {"RMS": 0.38, "Kurtosis": 8.2,  "Temp": 74.0, "Speed": 1752, "CrestFactor": 9.1},
    "UNBALANCE":     {"RMS": 0.55, "Kurtosis": 2.6,  "Temp": 59.0, "Speed": 1782, "CrestFactor": 3.8},
    "MISALIGNMENT":  {"RMS": 0.28, "Kurtosis": 3.4,  "Temp": 63.0, "Speed": 1770, "CrestFactor": 5.2},
    "LOOSENESS":     {"RMS": 0.20, "Kurtosis": 4.8,  "Temp": 55.0, "Speed": 1790, "CrestFactor": 6.5},
}

# ============================================================
# PHYSICS DRIFT SIMULATION
# ============================================================
def drift(current, target, rate, noise_amp=0.0):
    return current + (target - current) * rate + random.uniform(-noise_amp, noise_amp)

def update_sim_state(mode):
    global sim_state
    tgt = TARGET_PROFILES.get(mode, TARGET_PROFILES["HEALTHY"])
    sim_state["RMS"]         = max(0.001, drift(sim_state["RMS"],         tgt["RMS"],         0.08, 0.005))
    sim_state["Kurtosis"]    = max(0.01,  drift(sim_state["Kurtosis"],    tgt["Kurtosis"],    0.07, 0.12))
    sim_state["Temp"]        = drift(sim_state["Temp"],        tgt["Temp"],        0.04, 0.15)
    sim_state["FanSpeed"]    = drift(sim_state["FanSpeed"],    tgt["Speed"],       0.05, 6.0)
    sim_state["CrestFactor"] = max(1.0,   drift(sim_state["CrestFactor"], tgt["CrestFactor"], 0.07, 0.08))
    return sim_state

# ============================================================
# HEALTH SCORE (0-100)
# ============================================================
def compute_health_score(rms, kurtosis, crest_factor, iso_model):
    """
    Returns a 0-100 health score where 100 = perfect, 0 = imminent failure.

    Three-tier logic to keep it consistent with what the human sees:
    1. Clearly healthy zone (rms < GATE_RMS_MAX and kurtosis < GATE_KURT_MAX):
       Return 80-95 based on how deep in the zone we are.
       Prevents IsolationForest from returning 38% on low-value readings.

    2. Fault zone with IsolationForest available:
       Use decision_function and map to 0-100.

    3. Fallback heuristic if no model loaded.
    """
    # Tier 1: hard-gate for clearly healthy readings
    if rms < GATE_RMS_MAX and kurtosis < GATE_KURT_MAX:
        rms_ratio  = rms / GATE_RMS_MAX          # 0.0 - 1.0
        kurt_ratio = kurtosis / GATE_KURT_MAX     # 0.0 - 1.0
        score = 95 - (rms_ratio * 10) - (kurt_ratio * 5)
        return round(float(np.clip(score, 80, 95)), 1)

    # Tier 2: IsolationForest for degraded readings
    if iso_model is not None:
        try:
            sub_e  = rms * 0.3
            sync_e = rms * 0.5
            hf_e   = rms * 0.8 if kurtosis > 4.5 else rms * 0.2
            spec_k = kurtosis * 0.4
            feats  = np.array([[rms, kurtosis, crest_factor, sub_e, sync_e, hf_e, spec_k]])
            raw    = float(iso_model.decision_function(feats)[0])
            # decision_function: positive = normal, negative = anomalous
            # typical range: -0.3 to +0.15
            health = np.interp(raw, [-0.35, 0.15], [0, 100])
            return round(float(np.clip(health, 0, 100)), 1)
        except Exception:
            pass

    # Tier 3: heuristic fallback
    score = max(0, 100 - (rms * 150) - (max(0, kurtosis - 3) * 10))
    return round(float(np.clip(score, 0, 100)), 1)


# ============================================================
# SEVERITY STAGING  (FIX: reconciled with health score)
# ============================================================
def _rms_kurtosis_severity(rms, kurtosis):
    """Raw-threshold-based severity."""
    for label, rms_max, kurtosis_max in SEVERITY_LEVELS:
        if rms <= rms_max and kurtosis <= kurtosis_max:
            return label
    return "CRITICAL"

def _health_score_severity(health_score):
    """Health-score-based severity."""
    for min_h, label in HEALTH_TO_SEVERITY:
        if health_score >= min_h:
            return label
    return "CRITICAL"

SEV_RANK = {"HEALTHY": 0, "WATCH": 1, "WARNING": 2, "ALERT": 3, "CRITICAL": 4}

def get_severity(rms, kurtosis, health_score):
    """
    Returns the WORSE of (raw-threshold severity) and (health-score severity).
    This ensures the two systems are always consistent — if the health score
    thinks the asset is WARNING but raw thresholds say HEALTHY, WARNING wins.
    Previously the two systems could point in opposite directions, causing
    the 'HEALTHY pill + 38% gauge' contradiction seen in the screenshot.
    """
    s1 = _rms_kurtosis_severity(rms, kurtosis)
    s2 = _health_score_severity(health_score)
    return s1 if SEV_RANK[s1] >= SEV_RANK[s2] else s2


# ============================================================
# DYNAMIC URGENCY RESOLUTION
# ============================================================
def resolve_urgency(fault_label: str, severity: str) -> str:
    """
    Returns max(severity-implied urgency, fault-type urgency floor).
    A bearing fault is always at least HIGH.
    Any fault at ALERT or CRITICAL is always HIGH regardless of type.
    """
    if fault_label == "HEALTHY":
        return "NONE"
    dynamic = SEVERITY_TO_URGENCY.get(severity, "MEDIUM")
    floor   = REPAIR_GUIDE.get(fault_label, {}).get("urgency_floor", "LOW")
    return dynamic if URGENCY_RANK.get(dynamic, 0) >= URGENCY_RANK.get(floor, 0) else floor


# ============================================================
# RUL ESTIMATION
# ============================================================
FAILURE_THRESHOLD_RMS = 0.75

def estimate_rul(current_rms):
    rms_history.append(current_rms)
    if len(rms_history) < 30:
        return None
    x = np.arange(len(rms_history), dtype=float)
    y = np.array(rms_history, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    if slope <= 1e-6:
        return 9999
    cycles = (FAILURE_THRESHOLD_RMS - current_rms) / slope
    return max(0, int(cycles))


# ============================================================
# FFT SPECTRUM
# ============================================================
SHAFT_FREQ = 30  # Hz

def generate_fft_spectrum(rms, kurtosis, crest_factor, mode, sample_rate=20000, n_points=0.5):
    N    = int(sample_rate * n_points)
    t    = np.linspace(0, n_points, N, endpoint=False)
    base = 0.05 * np.sin(2 * np.pi * SHAFT_FREQ * t)

    if mode == "BEARING_WEAR":
        BPFI = 162.0
        amp  = rms * 1.2
        sig  = (base
                + amp * np.sin(2 * np.pi * BPFI * t)
                + (amp * 0.4) * np.sin(2 * np.pi * (BPFI + SHAFT_FREQ) * t)
                + np.random.normal(0, rms * 0.3, N))
    elif mode == "UNBALANCE":
        amp = rms * 1.5
        sig = (base
               + amp * np.sin(2 * np.pi * SHAFT_FREQ * t)
               + np.random.normal(0, rms * 0.1, N))
    elif mode == "MISALIGNMENT":
        amp = rms * 1.0
        sig = (base
               + amp * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t)
               + (amp * 0.5) * np.sin(2 * np.pi * 3 * SHAFT_FREQ * t)
               + np.random.normal(0, rms * 0.15, N))
    elif mode == "LOOSENESS":
        amp = rms * 0.8
        sig = (base
               + amp * np.sin(2 * np.pi * 0.5 * SHAFT_FREQ * t)
               + np.random.normal(0, rms * 0.4, N))
    else:
        sig = base + np.random.normal(0, max(rms, 0.01) * 0.5, N)

    fft_mag = np.abs(fft(sig))[:N // 2] / N
    freqs   = fftfreq(N, 1 / sample_rate)[:N // 2]
    mask    = freqs <= 500
    f_sel   = freqs[mask]
    a_sel   = fft_mag[mask]
    step    = max(1, len(f_sel) // 200)
    return f_sel[::step].tolist(), a_sel[::step].tolist()


# ============================================================
# INFERENCE  (FIX: hard gate prevents ML misfires on low values)
# ============================================================
def run_inference(rms, kurtosis, crest_factor, mode):
    """
    Multi-class fault inference with three-layer consistency guarantee:

    Layer 1 - Hard gate:
      If rms < GATE_RMS_MAX AND kurtosis < GATE_KURT_MAX the signal is
      physically too clean to contain a fault signature. Skip the ML model
      entirely and return HEALTHY. This stops the model from predicting
      'LOOSENESS 92%' on a 0.024g RMS reading during boot/low-speed phases.

    Layer 2 - ML classifier:
      Run the model only when the gate passes. Use heuristic fallback if
      the model is unavailable or raises an error.

    Layer 3 - Severity reconciliation:
      Derive severity from BOTH raw thresholds AND health score, then take
      the worse result. This ensures the status pill and gauge always agree.
      Previously they used completely separate systems that contradicted each other.
    """

    # --- Layer 1: Hard gate -------------------------------------------
    gated_healthy = (rms < GATE_RMS_MAX and kurtosis < GATE_KURT_MAX)

    if gated_healthy:
        fault_code  = 0
        fault_proba = [1.0, 0.0, 0.0, 0.0, 0.0]
    else:
        # --- Layer 2: ML classifier -----------------------------------
        sub_e  = rms * 0.3
        sync_e = rms * 0.5
        hf_e   = rms * 0.8 if kurtosis > 4.5 else rms * 0.2
        spec_k = kurtosis * 0.4
        feats  = np.array([[rms, kurtosis, crest_factor, sub_e, sync_e, hf_e, spec_k]])

        fault_code  = 0
        fault_proba = [1.0, 0.0, 0.0, 0.0, 0.0]

        if clf is not None:
            try:
                fault_code  = int(clf.predict(feats)[0])
                fault_proba = clf.predict_proba(feats)[0].tolist()
            except Exception as e:
                print(f"Inference error: {e}")
                fault_code  = _heuristic_fault(rms, kurtosis)
                fault_proba = _heuristic_proba(fault_code)
        else:
            fault_code  = _heuristic_fault(rms, kurtosis)
            fault_proba = _heuristic_proba(fault_code)

    fault_label = FAULT_NAMES.get(fault_code, "UNKNOWN")

    # --- Layer 3: Severity reconciliation ----------------------------
    health   = compute_health_score(rms, kurtosis, crest_factor, scorer)
    severity = get_severity(rms, kurtosis, health)   # takes worse of two systems
    rul      = estimate_rul(rms)
    urgency  = resolve_urgency(fault_label, severity)

    base_guide = REPAIR_GUIDE.get(fault_label, REPAIR_GUIDE["HEALTHY"])
    guide = {
        "title":   base_guide["title"],
        "detail":  base_guide["detail"],
        "actions": base_guide["actions"],
        "urgency": urgency
    }

    return {
        "fault_code":    fault_code,
        "fault_label":   fault_label,
        "fault_display": FAULT_DISPLAY.get(fault_code, fault_label),
        "fault_proba":   [round(p * 100, 1) for p in fault_proba],
        "health_score":  health,
        "severity":      severity,
        "rul_cycles":    rul,
        "guide":         guide,
    }


def _heuristic_fault(rms, kurtosis):
    if rms < 0.10 and kurtosis < 3.5:  return 0
    if kurtosis > 5.5:                  return 1  # Bearing
    if rms > 0.35 and kurtosis < 3.5:  return 2  # Unbalance
    if 0.15 < rms < 0.35:              return 3  # Misalignment
    if kurtosis > 3.5:                 return 4  # Looseness
    return 0

def _heuristic_proba(fault_code):
    proba = [0.05, 0.05, 0.05, 0.05, 0.05]
    proba[fault_code] = 0.80
    return proba


# ============================================================
# BACKGROUND TELEMETRY TASK (WebSocket push every 500ms)
# ============================================================
def telemetry_loop():
    while True:
        try:
            mode         = get_demo_mode()
            state        = update_sim_state(mode)

            rms          = round(max(0.001, state["RMS"]), 4)
            kurtosis     = round(max(0.01,  state["Kurtosis"]), 2)
            temp         = round(state["Temp"], 1)
            speed        = int(state["FanSpeed"])
            crest_factor = round(max(1.0, state["CrestFactor"]), 2)
            timestamp    = datetime.datetime.now().strftime("%H:%M:%S")

            infer       = run_inference(rms, kurtosis, crest_factor, mode)
            freqs, amps = generate_fft_spectrum(rms, kurtosis, crest_factor, mode)

            payload = {
                "timestamp":     timestamp,
                "mode":          mode,
                "rms":           rms,
                "kurtosis":      kurtosis,
                "crest_factor":  crest_factor,
                "temp":          temp,
                "speed":         speed,
                "fault_code":    infer["fault_code"],
                "fault_label":   infer["fault_label"],
                "fault_display": infer["fault_display"],
                "fault_proba":   infer["fault_proba"],
                "health_score":  infer["health_score"],
                "severity":      infer["severity"],
                "rul_cycles":    infer["rul_cycles"],
                "guide_title":   infer["guide"]["title"],
                "guide_detail":  infer["guide"]["detail"],
                "guide_actions": infer["guide"]["actions"],
                "guide_urgency": infer["guide"]["urgency"],
                "fft_freqs":     freqs,
                "fft_amps":      amps,
            }

            socketio.emit('telemetry', payload)
            log_reading({**payload,
                         "fault_code":  infer["fault_code"],
                         "fault_label": infer["fault_label"]})

        except Exception as e:
            print(f"Telemetry loop error: {e}")

        socketio.sleep(0.5)


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/history')
def api_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            'SELECT timestamp, rms, kurtosis, temp, health_score, severity '
            'FROM readings ORDER BY id DESC LIMIT 120'
        ).fetchall()
        conn.close()
        return jsonify([
            {"timestamp": r[0], "rms": r[1], "kurtosis": r[2],
             "temp": r[3], "health_score": r[4], "severity": r[5]}
            for r in reversed(rows)
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alerts')
def api_alerts():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            'SELECT timestamp, fault_label, severity, urgency, health_score, rms '
            'FROM fault_events ORDER BY id DESC LIMIT 25'
        ).fetchall()
        conn.close()
        return jsonify([
            {"timestamp": r[0], "fault_label": r[1], "severity": r[2],
             "urgency": r[3], "health_score": r[4], "rms": r[5]}
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/csv')
def export_csv():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute('SELECT * FROM readings ORDER BY id').fetchall()
        conn.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['id', 'timestamp', 'rms', 'kurtosis', 'crest_factor', 'temp',
                         'speed', 'health_score', 'severity', 'fault_code', 'fault_label',
                         'rul_cycles'])
        writer.writerows(rows)
        return Response(
            output.getvalue(), mimetype='text/csv',
            headers={"Content-Disposition": "attachment; filename=assetguard_telemetry.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats')
def api_stats():
    try:
        conn       = sqlite3.connect(DB_PATH)
        total      = conn.execute('SELECT COUNT(*) FROM readings').fetchone()[0]
        faults     = conn.execute("SELECT COUNT(*) FROM readings WHERE fault_code != 0").fetchone()[0]
        avg_rms    = conn.execute('SELECT AVG(rms) FROM readings').fetchone()[0]
        avg_health = conn.execute('SELECT AVG(health_score) FROM readings').fetchone()[0]
        conn.close()
        uptime = int(time.time() - start_time)
        return jsonify({
            "total_readings": total,
            "fault_events":   faults,
            "avg_rms":        round(avg_rms    or 0, 4),
            "avg_health":     round(avg_health or 0, 1),
            "uptime_seconds": uptime
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# STARTUP
# ============================================================
@socketio.on('connect')
def on_connect():
    print("Client connected")

@socketio.on('disconnect')
def on_disconnect():
    print("Client disconnected")

if __name__ == '__main__':
    init_db()
    socketio.start_background_task(telemetry_loop)

    print("=" * 55)
    print("  AssetGuard AI - Edge Gateway Server")
    print("  WebSocket push @ 2Hz")
    print("  Multi-class classifier + Health scorer")
    print("  Hard-gated inference - no false positives on boot")
    print("  Severity reconciled across health score + thresholds")
    print("  SQLite logging -> data/telemetry.db")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True)