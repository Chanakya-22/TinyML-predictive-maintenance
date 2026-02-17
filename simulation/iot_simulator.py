import time
import random
import requests
import datetime
import sys

# Import the API key securely
try:
    from secrets import THINGSPEAK_WRITE_API_KEY
except ImportError:
    print("Error: secrets.py not found. Please create it and add your API Key.")
    sys.exit(1)

# --- CONFIGURATION ---
THINGSPEAK_URL = "https://api.thingspeak.com/update"
INTERVAL = 16  # Seconds (ThingSpeak free tier limit is 15s)

def get_sensor_data():
    """
    Simulates reading data from an MPU6050 Accelerometer.
    Returns: (vibration_value, status_code)
    """
    # Simulate a 20% chance of a machine fault
    is_faulty = random.random() > 0.80

    if is_faulty:
        # Generate high vibration data (Fault)
        # Random float between 3.5 and 8.0
        vibration = round(random.uniform(3.5, 8.0), 2)
        status = 1  # 1 represents "Anomaly"
        print(f"⚠️  SIMULATING FAULT | Vibration: {vibration}G")
    else:
        # Generate low vibration data (Normal)
        # Random float between 0.1 and 1.2
        vibration = round(random.uniform(0.1, 1.2), 2)
        status = 0  # 0 represents "Normal"
        print(f"✅  System Normal    | Vibration: {vibration}G")
    
    return vibration, status

def upload_to_cloud(vibration, status):
    """
    Sends the payload to ThingSpeak via HTTP POST.
    """
    payload = {
        "api_key": THINGSPEAK_WRITE_API_KEY,
        "field1": vibration,
        "field2": status
    }

    try:
        response = requests.get(THINGSPEAK_URL, params=payload)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if response.status_code == 200:
            print(f"[{timestamp}] ☁️  Data uploaded successfully! Entry ID: {response.text}")
        else:
            print(f"[{timestamp}] ❌ Error uploading: {response.status_code}")
            
    except Exception as e:
        print(f"Network Error: {e}")

def main():
    print("==========================================")
    print("   IoT VIRTUAL DEVICE: MOTOR MONITOR")
    print("   Target: ThingSpeak Cloud")
    print("==========================================\n")
    
    try:
        while True:
            vib, stat = get_sensor_data()
            upload_to_cloud(vib, stat)
            
            # Countdown for next update
            print(f"   (Waiting {INTERVAL}s for next reading...)\n")
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation stopped by user.")

if __name__ == "__main__":
    main()