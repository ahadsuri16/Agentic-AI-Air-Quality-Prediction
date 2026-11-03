import time
import random
import requests
import sys

API_URL = "http://localhost:8000"

def wait_for_backend():
    print("Connecting to backend API service...")
    while True:
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            if r.status_code == 200:
                print("Backend service confirmed online! Starting IoT sensor telemetry loop.")
                break
        except Exception:
            pass
        time.sleep(2)

def generate_telemetry(inject_anomaly=False):
    if inject_anomaly:
        print("[!] Injecting simulated industrial particulate anomaly...")
        pm25 = random.uniform(140.0, 260.0)
        pm10 = pm25 * random.uniform(1.4, 2.2)
        no2 = random.uniform(60.0, 110.0)
    else:
        pm25 = random.uniform(15.0, 55.0)
        pm10 = pm25 * random.uniform(1.2, 1.8)
        no2 = random.uniform(15.0, 45.0)

    packet = {
        "pm25": round(pm25, 2),
        "pm10": round(pm10, 2),
        "no2": round(no2, 2),
        "so2": round(random.uniform(3.0, 25.0), 2),
        "co": round(random.uniform(0.4, 2.5), 2),
        "o3": round(random.uniform(15.0, 65.0), 2),
        "temperature": round(random.uniform(22.0, 36.0), 1),
        "humidity": round(random.uniform(30.0, 75.0), 1),
        "wind_speed": round(random.uniform(1.0, 8.0), 1),
        "pressure": round(random.uniform(1005.0, 1020.0), 1)
    }
    return packet

def run_simulator(interval=3.0):
    wait_for_backend()
    step = 0
    while True:
        step += 1
        # Inject anomaly every 12 cycles
        anomaly = (step % 12 == 0)
        payload = generate_telemetry(inject_anomaly=anomaly)

        try:
            res = requests.post(f"{API_URL}/api/sensor_data", json=payload, timeout=2)
            print(f"Cycle {step:04d} -> Telemetry transmitted | PM2.5: {payload['pm25']} | PM10: {payload['pm10']} | Response: {res.status_code}")
        except Exception as e:
            print(f"Failed to transmit telemetry: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    run_simulator()
