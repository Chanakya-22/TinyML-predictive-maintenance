from flask import Flask, render_template, jsonify
import joblib
import numpy as np
import random
import datetime
import os
import time

app = Flask(__name__)

# --- CONFIGURATION ---
# Points to ../ml_model/sota_model_final.pkl
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_model', 'sota_model_final.pkl')

# Global State Variables
last_state_change = time.time()
current_mode = "BOOT_SEQUENCE"  # Start with a realistic boot-up phase
boot_start_time = time.time()

# Simulation State (Physics Engine)
sim_state = {
    "RMS": 0.00,
    "Kurtosis": 0.0,
    "Temp": 25.0,  # Ambient temp
    "FanSpeed": 0  # RPM
}

# Expert System Knowledge Base
REPAIR_GUIDE = {
    "BOOT": "System initializing... Calibrating sensors.",
    "OPTIMAL": "System operating within normal parameters. Efficiency: 98%.",
    "CRITICAL FAULT": "CRITICAL ANOMALY DETECTED. IMMEDIATE ACTION REQUIRED.",
    "BEARING_FAIL": "⚠️ DIAGNOSIS: Inner Race Bearing Spalling.\n🔧 REPAIR: Replace Drive-End (DE) Bearing. Inspect lubrication for contamination.",
    "UNBALANCE": "⚠️ DIAGNOSIS: Rotor Assembly Imbalance.\n🔧 REPAIR: Clean fan blades to remove debris buildup. Check balance weights.",
    "HIGH_VIBE": "⚠️ DIAGNOSIS: Loose Mechanical Mounting.\n🔧 REPAIR: Tighten foundation bolts to 45Nm torque. Check isolation pads."
}

# --- HELPER FUNCTIONS ---
def load_ai_model():
    """Robust model loader with error handling."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ AI Model Loaded: {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Using Simulation Mode.")
        return None

model = load_ai_model()

def get_target_state(mode):
    """Returns physics targets based on the current operating mode."""
    if mode == "HEALTHY":
        return {"RMS": 0.05, "Kurtosis": 2.2, "Temp": 48.0, "Speed": 1800}
    elif mode == "BEARING_WEAR":
        return {"RMS": 0.38, "Kurtosis": 7.5, "Temp": 72.0, "Speed": 1750} # Friction slows it down
    elif mode == "UNBALANCE":
        return {"RMS": 0.55, "Kurtosis": 2.8, "Temp": 58.0, "Speed": 1780}
    elif mode == "BOOT_SEQUENCE":
        return {"RMS": 0.02, "Kurtosis": 1.5, "Temp": 35.0, "Speed": 500}
    return {"RMS": 0.05, "Kurtosis": 2.0, "Temp": 45.0, "Speed": 1800}

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry')
def telemetry():
    global sim_state, current_mode, last_state_change, boot_start_time
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # 1. STATE MACHINE LOGIC
    now = time.time()
    
    # Handle Boot Sequence (First 10 seconds)
    if current_mode == "BOOT_SEQUENCE":
        if now - boot_start_time > 10:
            current_mode = "HEALTHY"
            last_state_change = now
    
    # Normal Operation Logic (Switch every 60 seconds)
    elif now - last_state_change > 60:
        # 30% chance of fault, 70% chance of healthy
        if random.random() > 0.70:
            scenarios = ["BEARING_WEAR", "UNBALANCE"]
            current_mode = random.choice(scenarios)
        else:
            current_mode = "HEALTHY"
        last_state_change = now

    # 2. PHYSICS SIMULATION (Drift & Noise)
    targets = get_target_state(current_mode)
    
    # Linear interpolation for smooth transitions (Drift)
    # Factor 0.05 = slow drift, 0.1 = fast drift
    sim_state["RMS"] += (targets["RMS"] - sim_state["RMS"]) * 0.05 + random.uniform(-0.005, 0.005)
    sim_state["Kurtosis"] += (targets["Kurtosis"] - sim_state["Kurtosis"]) * 0.05 + random.uniform(-0.1, 0.1)
    sim_state["Temp"] += (targets["Temp"] - sim_state["Temp"]) * 0.02 + random.uniform(-0.1, 0.1)
    sim_state["FanSpeed"] += (targets["Speed"] - sim_state["FanSpeed"]) * 0.05 + random.uniform(-5, 5)

    # Clamp values to realistic positive ranges
    rms = max(0.00, sim_state["RMS"])
    kurtosis = max(1.0, sim_state["Kurtosis"])
    temp = sim_state["Temp"]
    speed = int(sim_state["FanSpeed"])

    # Derived spectral features for the AI
    peak = rms * 1.414 
    energy = (rms * 10) + (kurtosis * 0.5)

    # 3. AI INFERENCE ENGINE
    pred_label = "UNKNOWN"
    pred_code = 0
    bot_recommendation = "System Analyzing..."
    
    if current_mode == "BOOT_SEQUENCE":
        pred_label = "INITIALIZING"
        bot_recommendation = REPAIR_GUIDE["BOOT"]
    elif model:
        try:
            # Predict using the loaded SOTA model
            features = np.array([[rms, kurtosis, peak, energy]])
            pred_code = int(model.predict(features)[0])
            
            if pred_code == 1:
                pred_label = "CRITICAL FAULT"
                # Expert System Logic (Rule-Based)
                if kurtosis > 4.5:
                    bot_recommendation = REPAIR_GUIDE["BEARING_FAIL"]
                elif rms > 0.30 and kurtosis < 3.5:
                    bot_recommendation = REPAIR_GUIDE["UNBALANCE"]
                else:
                    bot_recommendation = REPAIR_GUIDE["HIGH_VIBE"]
            else:
                pred_label = "OPTIMAL"
                bot_recommendation = REPAIR_GUIDE["OPTIMAL"]
                
        except Exception as e:
            print(f"Inference Error: {e}")
            pred_label = "ERROR"

    # 4. JSON RESPONSE
    return jsonify({
        'timestamp': timestamp,
        'rms': round(rms, 4),
        'kurtosis': round(kurtosis, 2),
        'temp': round(temp, 1),
        'speed': speed,
        'status': pred_label,
        'status_code': pred_code,
        'recommendation': bot_recommendation,
        'mode_debug': current_mode # For debugging if needed
    })

if __name__ == '__main__':
    print("-------------------------------------------------------")
    print("   🚀 SOTA INDUSTRIAL SERVER v2.5 (Professional Build)")
    print("   Running on: http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop")
    print("-------------------------------------------------------")
    app.run(debug=True, port=5000)