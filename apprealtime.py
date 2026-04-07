from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import pytz
import random

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
# Config for demo temperature drift
# -----------------------------
MIN_TEMP = 23.8
MAX_TEMP = 25.2
MAX_STEP = 0.12   # max change per refresh, adjust if you want slower/faster drift

# -----------------------------
# Helper: current SG timestamp
# -----------------------------
def get_sg_timestamp():
    sg_tz = pytz.timezone("Asia/Singapore")
    now = datetime.now(sg_tz)
    return now.strftime("%d-%m-%Y, %H:%M:%S")

# -----------------------------
# Helper: get previous saved row
# -----------------------------
def get_previous_row():
    cursor.execute("""
    SELECT temp, change, prediction, probability, timestamp
    FROM sensor_data
    ORDER BY id DESC
    LIMIT 1
    """)
    return cursor.fetchone()

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
# Helper: generate drifting temp
# -----------------------------
def generate_drifting_record():
    prev_row = get_previous_row()

    if prev_row is None:
        # Start somewhere safely inside the range
        temp = round(random.uniform(24.2, 24.8), 2)
        change = 0.00
    else:
        prev_temp = float(prev_row[0])

        # Small drift step, positive or negative
        step = round(random.uniform(-MAX_STEP, MAX_STEP), 2)
        new_temp = prev_temp + step

        # Clamp to min/max range
        new_temp = max(MIN_TEMP, min(MAX_TEMP, new_temp))
        new_temp = round(new_temp, 2)

        change = round(new_temp - prev_temp, 2)
        temp = new_temp

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
# Latest data endpoint (demo mode)
# Generates a fresh drifting reading each time
# -----------------------------
@app.get("/latest")
def get_latest():
    return generate_drifting_record()

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
    