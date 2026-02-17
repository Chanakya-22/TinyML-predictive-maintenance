from flask import Flask, render_template, jsonify
import joblib
import numpy as np
import random
import datetime
import os

app = Flask(__name__)

# --- LOAD MODEL ---
# Points to ../ml_model/sota_model_final.pkl relative to this file
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_model', 'sota_model_final.pkl')
print(f"🔍 Loading AI Model from: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Model NOT found. Error: {e}")
    print("   Please run 'ml_model/train_sota_final.py' first!")
    model = None

# Global state for sensor drift simulation
# We use this to make the values change slowly over time (Drift) rather than jumping randomly
sim_state = {
    "mode": 0, # 0 = Healthy, 1 = Faulty
    "RMS": 0.05,
    "Kurtosis": 2.0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry')
def telemetry():
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    global sim_state

    # --- 1. REALISTIC SENSOR DRIFT LOGIC ---
    # Real machines don't jump from 0 to 100 instantly. They drift.
    
    # 5% chance to switch machine state (Stable demo)
    if random.random() > 0.95:
        sim_state["mode"] = 1 - sim_state["mode"]

    # Set targets based on mode
    # Faulty mode targets higher RMS and significantly higher Kurtosis (impacts)
    target_rms = 0.05 if sim_state["mode"] == 0 else 0.35
    target_kurt = 2.0 if sim_state["mode"] == 0 else 6.0

    # Drift towards target (Linear interpolation with noise)
    # This creates the "slow rise" effect seen in real failures
    sim_state["RMS"] += (target_rms - sim_state["RMS"]) * 0.1 + random.uniform(-0.01, 0.01)
    sim_state["Kurtosis"] += (target_kurt - sim_state["Kurtosis"]) * 0.1 + random.uniform(-0.2, 0.2)
    
    # Clamp values to realistic positive ranges
    rms = max(0.01, sim_state["RMS"])
    kurtosis = max(1.0, sim_state["Kurtosis"])
    
    # Derive other features to maintain physical consistency
    # Peak is usually related to RMS * Crest Factor
    peak = rms * 1.414 + random.uniform(0.0, 0.05)
    # High frequency energy correlates with impacts (Kurtosis)
    energy = (rms * 10) + (kurtosis * 0.5) + random.uniform(0.0, 1.0)

    # --- 2. AI INFERENCE ---
    pred_label = "UNKNOWN"
    pred_code = 0
    detailed_status = "Initializing..."
    
    if model:
        # Features must match training order: [RMS, Kurtosis, Peak, FFT_Energy]
        features = np.array([[rms, kurtosis, peak, energy]])
        try:
            pred_code = int(model.predict(features)[0])
        except:
            pred_code = 0 # Fallback
        
        if pred_code == 1:
            pred_label = "CRITICAL FAULT"
            # --- DEPLOYABILITY FEATURE: EXPLAINABLE FAULTS ---
            # This logic explains "Why" the fault happened, showing the "Cons" of the current state
            if rms > 0.25 and kurtosis > 4.5:
                detailed_status = f"High Energy Impact & Vibration ({rms:.2f}G, K={kurtosis:.1f})"
            elif rms > 0.25:
                detailed_status = f"Excessive Vibration Level ({rms:.2f}G)"
            elif kurtosis > 4.0:
                detailed_status = f"Early Bearing Spalling Detected (K={kurtosis:.1f})"
            else:
                detailed_status = "Irregular Frequency Resonance"
        else:
            pred_label = "OPTIMAL"
            # Even in healthy state, show minor warnings if noise is high (Realistic)
            if rms > 0.08:
                detailed_status = "Running Smoothly (High Load)"
            else:
                detailed_status = "Running Smoothly"
    
    # Console Log for Demo Presentation
    icon = "🔴" if pred_code == 1 else "🟢"
    print(f"[{timestamp}] {icon} STATUS: {pred_label} | REASON: {detailed_status}")

    return jsonify({
        'timestamp': timestamp,
        'rms': round(rms, 4),
        'kurtosis': round(kurtosis, 2),
        'status': pred_label,
        'status_code': pred_code,
        'reason': detailed_status 
    })

if __name__ == '__main__':
    print("🚀 SOTA Industrial Server Running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)