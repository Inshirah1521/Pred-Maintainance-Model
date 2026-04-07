from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import pytz

app = FastAPI(title="Thermal Failure Prediction API")

# -----------------------------
# CORS (FlutterFlow compatible)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ML model
# -----------------------------
model = joblib.load("thermal_failure_model_clean.joblib")

# -----------------------------
# SQLite Database Setup
# -----------------------------
conn = sqlite3.connect("sensor_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    temp REAL,
    change REAL,
    prediction INTEGER,
    probability REAL
)
""")
conn.commit()

# -----------------------------
# Demo video mode: scripted temperature path
# -----------------------------
DEMO_TEMPS = [
    24.02, 24.08, 24.14, 24.20, 24.27, 24.35, 24.44, 24.56,
    24.71, 24.90, 25.18, 25.62, 26.31, 27.45, 29.12, 30.48,
    31.33
]

demo_step_index = 0

# -----------------------------
# Helper: current SG timestamp
# -----------------------------
def get_sg_timestamp():
    sg_tz = pytz.timezone("Asia/Singapore")
    now = datetime.now(sg_tz)
    return now.strftime("%d-%m-%Y, %H:%M:%S")

# -----------------------------
# Helper: get previous temp
# -----------------------------
def get_previous_temp():
    cursor.execute("""
    SELECT temp
    FROM sensor_data
    ORDER BY id DESC
    LIMIT 1
    """)
    row = cursor.fetchone()
    return float(row[0]) if row else None

# -----------------------------
# Helper: run model + save row
# -----------------------------
def run_prediction_and_save(temp: float, change: float):
    X = np.array([[temp, change]])

    prediction = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        failure_probability = float(proba[1])
    else:
        failure_probability = None

    timestamp = get_sg_timestamp()

    cursor.execute("""
    INSERT INTO sensor_data (timestamp, temp, change, prediction, probability)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, temp, change, prediction, failure_probability))
    conn.commit()

    return {
        "timestamp": timestamp,
        "Temp_Sensor_1": f"{temp:.2f}",
        "Temp_Sensor_1_Change": f"{change:.2f}",
        "prediction": prediction,
        "probability": failure_probability
    }

# -----------------------------
# Helper: scripted demo record
# -----------------------------
def generate_scripted_record():
    global demo_step_index

    prev_temp = get_previous_temp()

    if demo_step_index < len(DEMO_TEMPS):
        temp = DEMO_TEMPS[demo_step_index]
        demo_step_index += 1
    else:
        temp = 31.33

    if prev_temp is None:
        change = 0.00
    else:
        change = round(temp - prev_temp, 2)

    return run_prediction_and_save(temp, change)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -----------------------------
# Prediction endpoint (manual / sensor input)
# -----------------------------
@app.post("/predict")
def predict(sensor_data: dict):
    if "Temp_Sensor_1" not in sensor_data or "Temp_Sensor_1_Change" not in sensor_data:
        return {"error": "Missing required sensor fields"}

    temp = round(float(sensor_data["Temp_Sensor_1"]), 2)
    change = round(float(sensor_data["Temp_Sensor_1_Change"]), 2)

    return run_prediction_and_save(temp, change)

# -----------------------------
# Latest data endpoint (scripted demo mode)
# -----------------------------
@app.get("/latest")
def get_latest():
    return generate_scripted_record()

# -----------------------------
# Reset demo sequence
# -----------------------------
@app.post("/reset-demo")
def reset_demo():
    global demo_step_index
    demo_step_index = 0

    cursor.execute("DELETE FROM sensor_data")
    conn.commit()

    return {"message": "Demo sequence reset successfully"}

# -----------------------------
# Get full history
# -----------------------------
@app.get("/history")
def get_history():
    cursor.execute("""
    SELECT timestamp, temp, change, prediction, probability
    FROM sensor_data
    ORDER BY id DESC
    """)
    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "timestamp": row[0],
            "Temp_Sensor_1": f"{row[1]:.2f}",
            "Temp_Sensor_1_Change": f"{row[2]:.2f}",
            "prediction": row[3],
            "probability": row[4]
        })

    return data

# -----------------------------
# Optional: limit history
# -----------------------------
@app.get("/history/{limit}")
def get_limited_history(limit: int):
    cursor.execute("""
    SELECT timestamp, temp, change, prediction, probability
    FROM sensor_data
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "timestamp": row[0],
            "Temp_Sensor_1": f"{row[1]:.2f}",
            "Temp_Sensor_1_Change": f"{row[2]:.2f}",
            "prediction": row[3],
            "probability": row[4]
        })

    return data