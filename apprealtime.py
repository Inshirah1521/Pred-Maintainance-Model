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
    probability REAL,
    is_demo INTEGER DEFAULT 0
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
def run_prediction_and_save(temp: float, change: float, is_demo: int = 0):
    X = np.array([[temp, change]])

    prediction = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        failure_probability = float(proba[1])
    else:
        failure_probability = 0.0  # default 0 if not supported

    timestamp = get_sg_timestamp()

    cursor.execute("""
    INSERT INTO sensor_data (timestamp, temp, change, prediction, probability, is_demo)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, temp, change, prediction, failure_probability, is_demo))
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

    change = 0.0 if prev_temp is None else round(temp - prev_temp, 2)

    return run_prediction_and_save(temp, change, is_demo=1)  # mark as demo

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

    return run_prediction_and_save(temp, change, is_demo=0)  # real sensor input

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

    cursor.execute("DELETE FROM sensor_data WHERE is_demo=1")  # delete demo only
    conn.commit()

    return {"message": "Demo sequence reset successfully"}

# -----------------------------
# Get full history (real data only)
# -----------------------------
@app.get("/history")
def get_history(limit: int = None):
    if limit:
        cursor.execute("""
        SELECT timestamp, temp, change, prediction, probability
        FROM sensor_data
        WHERE is_demo=0
        ORDER BY id DESC
        LIMIT ?
        """, (limit,))
    else:
        cursor.execute("""
        SELECT timestamp, temp, change, prediction, probability
        FROM sensor_data
        WHERE is_demo=0
        ORDER BY id DESC
        """)

    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "timestamp": row[0],
            "Temp_Sensor_1": f"{float(row[1]):.2f}",
            "Temp_Sensor_1_Change": f"{float(row[2]):.2f}",
            "prediction": row[3],
            "probability": float(row[4]) if row[4] is not None else 0.0
        })

    return data