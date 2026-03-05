"""
AssetGuard AI — IoT Virtual Device Simulator
==============================================
Simulates a physical edge node equipped with:
  - MPU6050: 3-axis accelerometer (X=radial, Y=axial, Z=tangential)
  - DS18B20 × 2: bearing temp + ambient temp sensors
  - INA219: current clamp (detects electrical signature of faults)
  - Tachometer: shaft RPM via optical encoder

Fault injection is scripted in a realistic degradation profile:
  1. Healthy baseline
  2. Incipient bearing fault (low amplitude, building)
  3. Developed bearing fault (clear BPFI signature)
  4. Recovery (simulated maintenance intervention)
  5. Rotor unbalance
  ...

Uploads to ThingSpeak (or a local endpoint) via HTTP POST.
"""

import time
import random
import math
import datetime
import sys
import requests
import json

# ─── Secure key import ────────────────────────────────────────
try:
    from secrets import THINGSPEAK_WRITE_API_KEY, LOCAL_ENDPOINT
except ImportError:
    print("[WARN] secrets.py not found. Using demo mode (no upload).")
    THINGSPEAK_WRITE_API_KEY = None
    LOCAL_ENDPOINT = "http://127.0.0.1:5001/api/ingest"  # Optional local endpoint

# ─── Configuration ────────────────────────────────────────────
THINGSPEAK_URL  = "https://api.thingspeak.com/update"
UPLOAD_INTERVAL = 16      # seconds (ThingSpeak free tier: ≥15s)
SAMPLE_RATE     = 20000   # Hz — simulated sensor sample rate
SHAFT_FREQ      = 30      # Hz — 1800 RPM nominal

# Bearing characteristic frequencies (SKF 6205-2RS @ 1800 RPM)
BPFI = 162.0  # Ball Pass Frequency Inner race (Hz)
BPFO = 108.0  # Ball Pass Frequency Outer race (Hz)
BSF  =  71.0  # Ball Spin Frequency (Hz)

# Fault injection profile — total cycle: 480 seconds
FAULT_PROFILE = [
    # (start_sec, end_sec, fault_type, severity_start, severity_end, label)
    (0,    60,  "HEALTHY",          0.0,  0.0,  "Baseline healthy"),
    (60,   90,  "BEARING_INCIPIENT",0.1,  0.25, "Bearing fault — incipient"),
    (90,   180, "BEARING_DEVELOPED",0.3,  0.9,  "Bearing fault — developing"),
    (180,  220, "HEALTHY",          0.0,  0.0,  "Post-maintenance recovery"),
    (220,  300, "UNBALANCE",        0.5,  0.8,  "Rotor unbalance"),
    (300,  330, "HEALTHY",          0.0,  0.0,  "Healthy baseline"),
    (330,  400, "MISALIGNMENT",     0.4,  0.7,  "Shaft misalignment"),
    (400,  440, "HEALTHY",          0.0,  0.0,  "Healthy baseline"),
    (440,  480, "LOOSENESS",        0.3,  0.65, "Mechanical looseness"),
]
PROFILE_CYCLE = 480


# ─── Physics helpers ──────────────────────────────────────────

def _rms(signal):
    return math.sqrt(sum(x * x for x in signal) / len(signal))

def _kurtosis(signal):
    n    = len(signal)
    mean = sum(signal) / n
    var  = sum((x - mean) ** 2 for x in signal) / n
    if var < 1e-12:
        return 0.0
    std4 = var ** 2
    k    = sum((x - mean) ** 4 for x in signal) / (n * std4)
    return round(k - 3, 3)  # Excess kurtosis


def generate_vibration_signal(fault_type: str, severity: float, axis: str = 'radial') -> list:
    """
    Generate a short (50ms) synthetic vibration signal for one accelerometer axis.
    Returns a list of float samples.
    """
    n  = 1000  # 1000 samples @ 20kHz = 50ms window
    t  = [i / SAMPLE_RATE for i in range(n)]

    # Base shaft rotation
    base = [0.05 * math.sin(2 * math.pi * SHAFT_FREQ * ti) for ti in t]

    if fault_type in ("HEALTHY",):
        noise_amp = random.uniform(0.02, 0.06)
        return [base[i] + random.gauss(0, noise_amp) for i in range(n)]

    elif fault_type in ("BEARING_INCIPIENT", "BEARING_DEVELOPED"):
        amp  = severity * random.uniform(0.05, 0.12)
        # BPFI impact train + sidebands
        sig  = [
            base[i]
            + amp * math.sin(2 * math.pi * BPFI * t[i])
            + amp * 0.4 * math.sin(2 * math.pi * (BPFI + SHAFT_FREQ) * t[i])
            + amp * 0.4 * math.sin(2 * math.pi * (BPFI - SHAFT_FREQ) * t[i])
            + amp * 0.25 * math.sin(2 * math.pi * 2 * BPFI * t[i])
            + random.gauss(0, 0.04)
            for i in range(n)
        ]
        return sig

    elif fault_type == "UNBALANCE":
        amp = severity * random.uniform(0.18, 0.35)
        phase = random.uniform(0, 2 * math.pi)
        # Axial axis shows less unbalance
        scale = 0.3 if axis == 'axial' else 1.0
        sig = [
            base[i] + amp * scale * math.sin(2 * math.pi * SHAFT_FREQ * t[i] + phase)
            + random.gauss(0, 0.015)
            for i in range(n)
        ]
        return sig

    elif fault_type == "MISALIGNMENT":
        amp2x = severity * random.uniform(0.10, 0.20)
        # Axial axis shows high 2X for angular misalignment
        scale = 1.5 if axis == 'axial' else 1.0
        sig = [
            base[i]
            + amp2x * scale * math.sin(2 * math.pi * 2 * SHAFT_FREQ * t[i])
            + amp2x * 0.4   * math.sin(2 * math.pi * 3 * SHAFT_FREQ * t[i])
            + random.gauss(0, 0.03)
            for i in range(n)
        ]
        return sig

    elif fault_type == "LOOSENESS":
        amp = severity * random.uniform(0.06, 0.14)
        sub = [amp * math.sin(2 * math.pi * 0.5 * SHAFT_FREQ * t[i]) for i in range(n)]
        sig = [base[i] + sub[i] + random.gauss(0, severity * 0.10) for i in range(n)]
        return sig

    return base


def get_current_draw(fault_type: str, severity: float) -> float:
    """
    Estimate motor current draw (Amps).
    Unbalance and misalignment increase current draw;
    bearing faults show a small but measurable increase.
    """
    nominal = 4.20  # Amps at 1800 RPM / nominal load
    if fault_type == "UNBALANCE":
        delta = severity * random.uniform(1.2, 2.0)
    elif fault_type == "MISALIGNMENT":
        delta = severity * random.uniform(0.8, 1.4)
    elif fault_type in ("BEARING_INCIPIENT", "BEARING_DEVELOPED"):
        delta = severity * random.uniform(0.2, 0.5)
    elif fault_type == "LOOSENESS":
        delta = severity * random.uniform(0.3, 0.8)
    else:
        delta = 0.0
    return round(nominal + delta + random.gauss(0, 0.06), 3)


def get_temperatures(fault_type: str, severity: float):
    """
    Returns (bearing_temp_C, ambient_temp_C).
    Bearing faults heat up the bearing housing.
    """
    ambient  = round(24.0 + random.gauss(0, 0.4), 1)
    baseline = 48.0

    if fault_type in ("BEARING_INCIPIENT", "BEARING_DEVELOPED"):
        bearing_temp = baseline + severity * random.uniform(18, 30)
    elif fault_type == "UNBALANCE":
        bearing_temp = baseline + severity * random.uniform(8, 14)
    elif fault_type == "MISALIGNMENT":
        bearing_temp = baseline + severity * random.uniform(10, 18)
    elif fault_type == "LOOSENESS":
        bearing_temp = baseline + severity * random.uniform(5, 12)
    else:
        bearing_temp = baseline + random.gauss(0, 1.0)

    return round(bearing_temp, 1), ambient


def get_shaft_speed(fault_type: str, severity: float) -> int:
    """
    Shaft speed in RPM. Faults cause slight speed reduction and jitter.
    """
    nominal = 1800
    if fault_type in ("BEARING_INCIPIENT", "BEARING_DEVELOPED"):
        drop = severity * random.uniform(20, 60)
    elif fault_type == "UNBALANCE":
        drop = severity * random.uniform(10, 30)
    else:
        drop = severity * random.uniform(0, 15)
    return int(nominal - drop + random.gauss(0, 4))


# ─── Profile resolver ─────────────────────────────────────────

def get_current_profile(elapsed: float):
    """Returns (fault_type, severity, label) for the current elapsed time."""
    t = elapsed % PROFILE_CYCLE
    for start, end, fault, sev_start, sev_end, label in FAULT_PROFILE:
        if start <= t < end:
            # Linearly interpolate severity within window
            progress = (t - start) / max(end - start, 1)
            severity = sev_start + progress * (sev_end - sev_start)
            return fault, round(severity, 3), label
    return "HEALTHY", 0.0, "Baseline healthy"


# ─── Upload ───────────────────────────────────────────────────

def upload_thingspeak(payload: dict):
    """Upload key metrics to ThingSpeak (up to 8 fields)."""
    if not THINGSPEAK_WRITE_API_KEY:
        return
    ts_payload = {
        "api_key": THINGSPEAK_WRITE_API_KEY,
        "field1":  payload["rms_radial"],
        "field2":  payload["rms_axial"],
        "field3":  payload["kurtosis"],
        "field4":  payload["bearing_temp"],
        "field5":  payload["current_amps"],
        "field6":  payload["shaft_rpm"],
        "field7":  1 if payload["fault_type"] != "HEALTHY" else 0,
        "field8":  payload["severity"],
    }
    try:
        resp = requests.get(THINGSPEAK_URL, params=ts_payload, timeout=10)
        return resp.text.strip()
    except requests.RequestException as e:
        print(f"[ThingSpeak] Upload failed: {e}")
        return None


def upload_local(payload: dict):
    """Optional: POST full payload to local AssetGuard server."""
    try:
        requests.post(LOCAL_ENDPOINT, json=payload, timeout=3)
    except Exception:
        pass  # Local endpoint is optional


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🏭  AssetGuard AI — IoT Virtual Device")
    print("  📡  Simulating: MPU6050 + DS18B20 × 2 + INA219")
    print("  🔁  Fault injection cycle: 480s")
    print(f"  ☁️   ThingSpeak: {'ENABLED' if THINGSPEAK_WRITE_API_KEY else 'DISABLED (demo mode)'}")
    print("=" * 55 + "\n")

    sim_start = time.time()

    try:
        while True:
            elapsed   = time.time() - sim_start
            fault_type, severity, label = get_current_profile(elapsed)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # ── Generate multi-axis signals ──────────────────
            sig_x = generate_vibration_signal(fault_type, severity, axis='radial')
            sig_y = generate_vibration_signal(fault_type, severity, axis='axial')
            sig_z = generate_vibration_signal(fault_type, severity, axis='tangential')

            rms_x  = round(_rms(sig_x), 4)
            rms_y  = round(_rms(sig_y), 4)
            rms_z  = round(_rms(sig_z), 4)
            kurt_x = _kurtosis(sig_x)
            peak_x = round(max(abs(v) for v in sig_x), 4)
            crest  = round(peak_x / (rms_x + 1e-9), 2)

            bearing_temp, ambient_temp = get_temperatures(fault_type, severity)
            current                    = get_current_draw(fault_type, severity)
            rpm                        = get_shaft_speed(fault_type, severity)

            payload = {
                "timestamp":    timestamp,
                "fault_type":   fault_type,
                "severity":     severity,
                "label":        label,
                "rms_radial":   rms_x,
                "rms_axial":    rms_y,
                "rms_tangential": rms_z,
                "kurtosis":     kurt_x,
                "crest_factor": crest,
                "bearing_temp": bearing_temp,
                "ambient_temp": ambient_temp,
                "current_amps": current,
                "shaft_rpm":    rpm,
            }

            # ── Console output ───────────────────────────────
            fault_icon = "✅" if fault_type == "HEALTHY" else "⚠️ "
            print(f"[{timestamp}] {fault_icon}  {label:<35}  "
                  f"RMS={rms_x:.3f}g  Kurt={kurt_x:+.2f}  "
                  f"Temp={bearing_temp}°C  {current:.2f}A  {rpm}RPM  "
                  f"[sev={severity:.2f}]")

            # ── Upload ───────────────────────────────────────
            entry_id = upload_thingspeak(payload)
            if entry_id:
                print(f"           ☁️  ThingSpeak entry #{entry_id}")

            upload_local(payload)

            print(f"           (next reading in {UPLOAD_INTERVAL}s)\n")
            time.sleep(UPLOAD_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n🛑  Simulation stopped by user.")
        elapsed_total = int(time.time() - sim_start)
        print(f"    Total runtime: {elapsed_total}s | Readings sent: {elapsed_total // UPLOAD_INTERVAL}")


if __name__ == "__main__":
    main()