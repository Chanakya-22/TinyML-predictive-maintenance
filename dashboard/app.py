"""
AssetGuard AI — Edge Gateway Server
====================================
Production-grade predictive maintenance backend.

Features:
- Flask-SocketIO real-time push (replaces client polling)
- SQLite telemetry persistence with CSV export
- Multi-class fault classification (5 fault types)
- Isolation Forest health score (0–100%)
- Remaining Useful Life (RUL) estimation via degradation trend
- 5-level ISO 10816-inspired severity staging
- Per-fault probability breakdown via predict_proba()
- Live FFT spectrum generation per cycle
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
app    = Flask(__name__)
app.config['SECRET_KEY'] = 'assetguard-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================
# PATHS
# ============================================================
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH       = os.path.join(BASE_DIR, 'ml_model', 'sota_model_final.pkl')
SCORER_PATH      = os.path.join(BASE_DIR, 'ml_model', 'health_scorer.pkl')
DB_PATH          = os.path.join(BASE_DIR, 'data', 'telemetry.db')

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
        "detail": "All vibration and thermal parameters within ISO 10816 Zone A limits. Efficiency: 98.2%. Next scheduled inspection: 720 operating hours.",
        "actions": [],
        "urgency": "NONE"
    },
    "BEARING_INNER_RACE": {
        "title": "⚠ Inner Race Bearing Defect Detected",
        "detail": "Spectral analysis confirms BPFI harmonic signature with shaft-frequency sidebands. Indicative of inner race spalling on the drive-end bearing.",
        "actions": [
            "Isolate and degrease drive-end bearing housing",
            "Replace SKF 6205-2RS or equivalent bearing",
            "Verify lubrication: use Mobilux EP2 grease, 12g per cavity",
            "Re-inspect after 24h run-in; re-baseline vibration"
        ],
        "urgency": "HIGH"
    },
    "ROTOR_UNBALANCE": {
        "title": "⚠ Rotor Assembly Imbalance",
        "detail": "Dominant 1X frequency component detected. Phase analysis indicates static unbalance on the fan rotor. Common cause: debris accumulation or missing balance weight.",
        "actions": [
            "Shut down and lock out / tag out (LOTO)",
            "Inspect and clean all fan blades — remove debris",
            "Check for missing or shifted balance correction weights",
            "Dynamic balance to ISO 1940-1 Grade G2.5 or better"
        ],
        "urgency": "MEDIUM"
    },
    "MISALIGNMENT": {
        "title": "⚠ Shaft / Coupling Misalignment",
        "detail": "Elevated 2X harmonic with significant axial vibration component detected. Consistent with angular or parallel coupling misalignment.",
        "actions": [
            "Measure and record current alignment with dial gauge or laser tool",
            "Correct soft-foot condition at all mounting pads first",
            "Align shaft to ≤ 0.05mm parallel / ≤ 0.05mm/100mm angular",
            "Re-torque coupling bolts to spec after alignment"
        ],
        "urgency": "MEDIUM"
    },
    "LOOSENESS": {
        "title": "⚠ Mechanical Looseness / Resonance",
        "detail": "Sub-harmonic (0.5X) and broadband spectral content detected. Indicates structural looseness at bearing housing, foundation bolts, or resonant baseplate.",
        "actions": [
            "Inspect and torque all foundation anchor bolts",
            "Check anti-vibration mount condition — replace if hardened",
            "Inspect bearing housing set screws and end-shields",
            "Perform bump test to identify resonant frequency"
        ],
        "urgency": "LOW"
    }
}

# ISO 10816-3 inspired severity thresholds (for 15kW+ motors, rigid mounting)
SEVERITY_LEVELS = [
    # (label,        color,     rms_max, kurtosis_max)
    ("HEALTHY",   "success",   0.07,    3.0),
    ("WATCH",     "info",      0.13,    4.0),
    ("WARNING",   "warning",   0.22,    5.5),
    ("ALERT",     "orange",    0.38,    7.0),
    ("CRITICAL",  "danger",    9999,    9999),
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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            fault_label TEXT,
            severity    TEXT,
            health_score REAL,
            rms         REAL
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
                INSERT INTO fault_events (timestamp, fault_label, severity, health_score, rms)
                VALUES (?, ?, ?, ?, ?)
            ''', (data['timestamp'], data['fault_label'], data['severity'],
                  data['health_score'], data['rms']))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB write error: {e}")

# ============================================================
# GLOBAL STATE
# ============================================================
start_time  = time.time()
rms_history = deque(maxlen=300)   # rolling window for RUL trend

sim_state = {
    "RMS": 0.01, "Kurtosis": 0.0, "Temp": 25.0, "FanSpeed": 0,
    "CrestFactor": 1.0
}

# ============================================================
# LOAD MODELS
# ============================================================
clf    = None
scorer = None

if os.path.exists(MODEL_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        print(f"✅ Classifier loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠  Classifier load failed: {e}")

if os.path.exists(SCORER_PATH):
    try:
        scorer = joblib.load(SCORER_PATH)
        print(f"✅ Health scorer loaded: {SCORER_PATH}")
    except Exception as e:
        print(f"⚠  Health scorer load failed: {e}")

# ============================================================
# DEMO LOOP — Scripted 240-second cycle
# ============================================================
DEMO_CYCLE = 240

def get_demo_mode():
    elapsed = int(time.time() - start_time)
    t = elapsed % DEMO_CYCLE
    if t < 18:   return "BOOT_SEQUENCE"
    if t < 55:   return "HEALTHY"
    if t < 115:  return "BEARING_WEAR"
    if t < 145:  return "HEALTHY"
    if t < 195:  return "UNBALANCE"
    if t < 215:  return "HEALTHY"
    if t < 240:  return "MISALIGNMENT"
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
# HEALTH SCORE (0–100)
# ============================================================
def compute_health_score(rms, kurtosis, crest_factor, iso_model):
    """
    Uses Isolation Forest decision_function to map anomaly distance → 0–100 health score.
    100 = pristine baseline, 0 = severely anomalous.
    """
    if iso_model is None:
        # Heuristic fallback
        score = max(0, 100 - (rms * 180) - (max(0, kurtosis - 3) * 8))
        return round(float(np.clip(score, 0, 100)), 1)

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
        score = max(0, 100 - (rms * 180) - (max(0, kurtosis - 3) * 8))
        return round(float(np.clip(score, 0, 100)), 1)

# ============================================================
# SEVERITY STAGING
# ============================================================
def get_severity(rms, kurtosis):
    for label, color, rms_max, kurtosis_max in SEVERITY_LEVELS:
        if rms <= rms_max and kurtosis <= kurtosis_max:
            return label, color
    return "CRITICAL", "danger"

# ============================================================
# RUL ESTIMATION
# ============================================================
FAILURE_THRESHOLD_RMS = 0.75  # RMS value considered "failure"

def estimate_rul(current_rms):
    """
    Fits a linear degradation trend over the rolling RMS history.
    Returns estimated cycles to reach failure threshold, or None if insufficient data.
    """
    rms_history.append(current_rms)

    if len(rms_history) < 30:
        return None

    x    = np.arange(len(rms_history), dtype=float)
    y    = np.array(rms_history, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    if slope <= 1e-6:
        return 9999  # Not degrading

    cycles = (FAILURE_THRESHOLD_RMS - current_rms) / slope
    return max(0, int(cycles))

# ============================================================
# FFT SPECTRUM (sent to frontend for live spectrum chart)
# ============================================================
SHAFT_FREQ = 30   # Hz

def generate_fft_spectrum(rms, kurtosis, crest_factor, mode, sample_rate=20000, n_points=0.5):
    """
    Generate a synthetic but physically plausible FFT spectrum for display.
    Returns lists of (frequencies, amplitudes) for the first 500 Hz.
    """
    N  = int(sample_rate * n_points)
    t  = np.linspace(0, n_points, N, endpoint=False)

    base = 0.05 * np.sin(2 * np.pi * SHAFT_FREQ * t)

    if mode == "BEARING_WEAR":
        BPFI = 162.0
        amp  = rms * 1.2
        sig  = base + amp * np.sin(2 * np.pi * BPFI * t) \
                    + (amp * 0.4) * np.sin(2 * np.pi * (BPFI + SHAFT_FREQ) * t) \
                    + np.random.normal(0, rms * 0.3, N)
    elif mode == "UNBALANCE":
        amp = rms * 1.5
        sig = base + amp * np.sin(2 * np.pi * SHAFT_FREQ * t) \
                   + np.random.normal(0, rms * 0.1, N)
    elif mode == "MISALIGNMENT":
        amp = rms * 1.0
        sig = base + amp * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t) \
                   + (amp * 0.5) * np.sin(2 * np.pi * 3 * SHAFT_FREQ * t) \
                   + np.random.normal(0, rms * 0.15, N)
    else:
        sig = base + np.random.normal(0, rms * 0.5, N)

    fft_mag  = np.abs(fft(sig))[:N // 2] / N
    freqs    = fftfreq(N, 1 / sample_rate)[:N // 2]

    # Downsample to 200 bins up to 500 Hz for frontend
    mask     = freqs <= 500
    f_sel    = freqs[mask]
    a_sel    = fft_mag[mask]
    step     = max(1, len(f_sel) // 200)
    return f_sel[::step].tolist(), a_sel[::step].tolist()

# ============================================================
# INFERENCE
# ============================================================
def run_inference(rms, kurtosis, crest_factor, mode):
    """
    Run multi-class fault classification and return fault code,
    label, probabilities, health score, severity, RUL, and repair guide.
    """
    # Feature vector must match train_final.py extract_features()
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
    health      = compute_health_score(rms, kurtosis, crest_factor, scorer)
    severity, _ = get_severity(rms, kurtosis)
    rul         = estimate_rul(rms)
    guide       = REPAIR_GUIDE.get(fault_label, REPAIR_GUIDE["HEALTHY"])

    return {
        "fault_code":     fault_code,
        "fault_label":    fault_label,
        "fault_display":  FAULT_DISPLAY.get(fault_code, fault_label),
        "fault_proba":    [round(p * 100, 1) for p in fault_proba],
        "health_score":   health,
        "severity":       severity,
        "rul_cycles":     rul,
        "guide":          guide,
    }

def _heuristic_fault(rms, kurtosis):
    if rms < 0.10 and kurtosis < 3.5:   return 0
    if kurtosis > 5.5:                   return 1  # Bearing
    if rms > 0.35 and kurtosis < 3.5:   return 2  # Unbalance
    if 0.15 < rms < 0.35:               return 3  # Misalignment
    if kurtosis > 3.5:                  return 4  # Looseness
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
            mode    = get_demo_mode()
            state   = update_sim_state(mode)

            rms          = round(max(0.001, state["RMS"]), 4)
            kurtosis     = round(max(0.01,  state["Kurtosis"]), 2)
            temp         = round(state["Temp"], 1)
            speed        = int(state["FanSpeed"])
            crest_factor = round(max(1.0, state["CrestFactor"]), 2)
            timestamp    = datetime.datetime.now().strftime("%H:%M:%S")

            infer = run_inference(rms, kurtosis, crest_factor, mode)
            freqs, amps = generate_fft_spectrum(rms, kurtosis, crest_factor, mode)

            payload = {
                "timestamp":    timestamp,
                "mode":         mode,
                # Raw sensors
                "rms":          rms,
                "kurtosis":     kurtosis,
                "crest_factor": crest_factor,
                "temp":         temp,
                "speed":        speed,
                # Diagnostics
                "fault_code":   infer["fault_code"],
                "fault_label":  infer["fault_label"],
                "fault_display":infer["fault_display"],
                "fault_proba":  infer["fault_proba"],
                "health_score": infer["health_score"],
                "severity":     infer["severity"],
                "rul_cycles":   infer["rul_cycles"],
                # Repair guide
                "guide_title":  infer["guide"]["title"],
                "guide_detail": infer["guide"]["detail"],
                "guide_actions":infer["guide"]["actions"],
                "guide_urgency":infer["guide"]["urgency"],
                # FFT spectrum
                "fft_freqs":    freqs,
                "fft_amps":     amps,
            }

            socketio.emit('telemetry', payload)
            log_reading({**payload, "fault_code": infer["fault_code"],
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
    """Last 120 readings for trend overlay charts."""
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
    """Recent fault events."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            'SELECT timestamp, fault_label, severity, health_score, rms '
            'FROM fault_events ORDER BY id DESC LIMIT 25'
        ).fetchall()
        conn.close()
        return jsonify([
            {"timestamp": r[0], "fault_label": r[1], "severity": r[2],
             "health_score": r[3], "rms": r[4]}
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/csv')
def export_csv():
    """Download all telemetry as CSV."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute('SELECT * FROM readings ORDER BY id').fetchall()
        conn.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['id','timestamp','rms','kurtosis','crest_factor','temp','speed',
                         'health_score','severity','fault_code','fault_label','rul_cycles'])
        writer.writerows(rows)
        return Response(
            output.getvalue(), mimetype='text/csv',
            headers={"Content-Disposition": "attachment; filename=assetguard_telemetry.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """Summary statistics for the session."""
    try:
        conn = sqlite3.connect(DB_PATH)
        total    = conn.execute('SELECT COUNT(*) FROM readings').fetchone()[0]
        faults   = conn.execute("SELECT COUNT(*) FROM readings WHERE fault_code != 0").fetchone()[0]
        avg_rms  = conn.execute('SELECT AVG(rms) FROM readings').fetchone()[0]
        avg_health = conn.execute('SELECT AVG(health_score) FROM readings').fetchone()[0]
        conn.close()
        uptime   = int(time.time() - start_time)
        return jsonify({
            "total_readings": total,
            "fault_events":   faults,
            "avg_rms":        round(avg_rms or 0, 4),
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
    print("  🚀  AssetGuard AI — Edge Gateway Server")
    print("  📡  WebSocket push @ 2Hz")
    print("  🧠  Multi-class fault classifier + Health scorer")
    print("  💾  SQLite logging → data/telemetry.db")
    print("  🌐  http://127.0.0.1:5000")
    print("=" * 55)
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True)