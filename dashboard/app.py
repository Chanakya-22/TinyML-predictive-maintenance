from flask import Flask, render_template, jsonify
import joblib
import numpy as np
import random
import datetime
import os

app = Flask(__name__)

# --- LOAD MODEL ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_model', 'sota_model_final.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Model NOT found. Error: {e}")
    model = None

# --- APPLIANCE SIMULATION STATE ---
# We are simulating an Industrial HVAC Unit (Fan + Compressor)
sim_state = {
    "mode": "HEALTHY", # Options: HEALTHY, BEARING_WEAR, FILTER_CLOG, UNBALANCE
    "RMS": 0.05,       # Vibration Level
    "Kurtosis": 2.0,   # Impact Factor
    "Temp": 45.0       # Temperature (New Feature!)
}

# --- HELPER BOT KNOWLEDGE BASE ---
# This dictionary maps fault codes to specific repair instructions
REPAIR_GUIDE = {
    "OPTIMAL": "System is running optimally. No action required.",
    "CRITICAL FAULT": "Immediate inspection required.",
    "BEARING_FAIL": "⚠️ DETECTED: Bearing Inner Race Spalling.\n🔧 ACTION: Schedule replacement of Drive End (DE) Bearing. Check lubrication schedule.",
    "UNBALANCE": "⚠️ DETECTED: Rotor Imbalance.\n🔧 ACTION: Clean fan blades to remove dust buildup. Check alignment weights.",
    "HIGH_VIBE": "⚠️ DETECTED: General High Vibration.\n🔧 ACTION: Tighten mounting bolts. Check for loose foundation."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry')
def telemetry():
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    global sim_state

    # --- 1. REALISTIC FAILURE MODES SIMULATION ---
    # Randomly switch scenarios every ~20 seconds (for demo speed)
    if random.random() > 0.98: 
        scenarios = ["HEALTHY", "HEALTHY", "BEARING_WEAR", "UNBALANCE"]
        sim_state["mode"] = random.choice(scenarios)

    # Physics Logic for each Mode
    if sim_state["mode"] == "HEALTHY":
        target_rms = 0.05
        target_kurt = 2.0
        target_temp = 45.0 + random.uniform(-1, 1)
    elif sim_state["mode"] == "BEARING_WEAR":
        target_rms = 0.35      # High vibration
        target_kurt = 6.5      # Very high impacts (clicking)
        target_temp = 65.0     # Overheating due to friction
    elif sim_state["mode"] == "UNBALANCE":
        target_rms = 0.45      # Very high vibration
        target_kurt = 2.5      # Low impacts (wobbling, not clicking)
        target_temp = 55.0     # Slight heat rise

    # Drift variables towards targets (Simulates inertia)
    sim_state["RMS"] += (target_rms - sim_state["RMS"]) * 0.1 + random.uniform(-0.01, 0.01)
    sim_state["Kurtosis"] += (target_kurt - sim_state["Kurtosis"]) * 0.1 + random.uniform(-0.1, 0.1)
    sim_state["Temp"] += (target_temp - sim_state["Temp"]) * 0.05 + random.uniform(-0.2, 0.2)

    # Clamp to safe physics limits
    rms = max(0.01, sim_state["RMS"])
    kurtosis = max(1.0, sim_state["Kurtosis"])
    temp = sim_state["Temp"]
    
    # Derived features for the AI Model
    peak = rms * 1.414 
    energy = (rms * 10) + (kurtosis * 0.5)

    # --- 2. AI INFERENCE ---
    pred_label = "UNKNOWN"
    pred_code = 0
    bot_recommendation = "Analyzing..."
    
    if model:
        features = np.array([[rms, kurtosis, peak, energy]])
        try:
            pred_code = int(model.predict(features)[0])
        except:
            pred_code = 0 
        
        if pred_code == 1:
            pred_label = "CRITICAL FAULT"
            # --- 3. HELPER BOT LOGIC (RULE-BASED EXPERT SYSTEM) ---
            # The AI says "It's Broken". The Rule Engine says "WHY and HOW TO FIX".
            
            if kurtosis > 4.5:
                # High impacts = Bearing failure
                bot_recommendation = REPAIR_GUIDE["BEARING_FAIL"]
            elif rms > 0.30 and kurtosis < 3.0:
                # High vibe, low impact = Unbalance/Wobble
                bot_recommendation = REPAIR_GUIDE["UNBALANCE"]
            else:
                bot_recommendation = REPAIR_GUIDE["HIGH_VIBE"]
        else:
            pred_label = "OPTIMAL"
            bot_recommendation = REPAIR_GUIDE["OPTIMAL"]
    
    return jsonify({
        'timestamp': timestamp,
        'rms': round(rms, 4),
        'kurtosis': round(kurtosis, 2),
        'temp': round(temp, 1),
        'status': pred_label,
        'status_code': pred_code,
        'recommendation': bot_recommendation
    })

if __name__ == '__main__':
    print("🚀 Smart HVAC Monitor Running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)