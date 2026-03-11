from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Thermal Failure Prediction API")

# -----------------------------
# CORS (to make it FlutterFlow compatible)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Store latest data (in-memory)
# -----------------------------
latest_data = {}

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -----------------------------
# Load ML model
# -----------------------------
model = joblib.load("thermal_failure_model_clean.joblib")

# -----------------------------
# Prediction endpoint (ESP32)
# -----------------------------
@app.post("/predict")
def predict(sensor_data: dict):
    global latest_data

    # Validate input
    if "Temp_Sensor_1" not in sensor_data or "Temp_Sensor_1_Change" not in sensor_data:
        return {"error": "Missing required sensor fields"}

    temp = float(sensor_data["Temp_Sensor_1"])
    change = float(sensor_data["Temp_Sensor_1_Change"])

    X = np.array([[temp, change]])

    # ---- Prediction ----
    prediction = int(model.predict(X)[0])

    # ---- Probability (confidence) ----
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        failure_probability = float(proba[1])   # class = 1
    else:
        failure_probability = None

    # ---- Save latest values ----
    latest_data = {
        "Temp_Sensor_1": temp,
        "Temp_Sensor_1_Change": change,
        "prediction": prediction,
        "probability": failure_probability
    }

    return {
        "prediction": prediction,
        "probability": failure_probability
    }

# -----------------------------
# Latest data endpoint (FlutterFlow)
# -----------------------------
@app.get("/latest")
def get_latest():
    return {
        "Temp_Sensor_1": latest_data.get("Temp_Sensor_1"),
        "Temp_Sensor_1_Change": latest_data.get("Temp_Sensor_1_Change"),
        "prediction": latest_data.get("prediction"),
        "probability": latest_data.get("probability")
    }
