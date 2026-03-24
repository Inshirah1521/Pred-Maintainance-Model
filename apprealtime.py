from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import sqlite3
from datetime import datetime

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
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -----------------------------
# Prediction endpoint (ESP32)
# -----------------------------
@app.post("/predict")
def predict(sensor_data: dict):

    # Validate input
    if "Temp_Sensor_1" not in sensor_data or "Temp_Sensor_1_Change" not in sensor_data:
        return {"error": "Missing required sensor fields"}

    temp = float(sensor_data["Temp_Sensor_1"])
    change = float(sensor_data["Temp_Sensor_1_Change"])

    X = np.array([[temp, change]])

    # ---- Prediction ----
    prediction = int(model.predict(X)[0])

    # ---- Probability ----
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        failure_probability = float(proba[1])
    else:
        failure_probability = None

    # ---- Timestamp ----
    timestamp = datetime.utcnow().isoformat()

    # ---- Save to SQLite ----
    cursor.execute("""
    INSERT INTO sensor_data (timestamp, temp, change, prediction, probability)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, temp, change, prediction, failure_probability))
    conn.commit()

    return {
        "timestamp": timestamp,
        "prediction": prediction,
        "probability": failure_probability
    }

# -----------------------------
# Get latest data (FlutterFlow)
# -----------------------------
@app.get("/latest")
def get_latest():
    cursor.execute("""
    SELECT timestamp, temp, change, prediction, probability
    FROM sensor_data
    ORDER BY id DESC
    LIMIT 1
    """)
    row = cursor.fetchone()

    if row:
        return {
            "timestamp": row[0],
            "Temp_Sensor_1": row[1],
            "Temp_Sensor_1_Change": row[2],
            "prediction": row[3],
            "probability": row[4]
        }
    else:
        return {"message": "No data yet"}

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
            "Temp_Sensor_1": row[1],
            "Temp_Sensor_1_Change": row[2],
            "prediction": row[3],
            "probability": row[4]
        })

    return data

# -----------------------------
# Optional: limit history (better for FlutterFlow)
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
            "Temp_Sensor_1": row[1],
            "Temp_Sensor_1_Change": row[2],
            "prediction": row[3],
            "probability": row[4]
        })

    return data