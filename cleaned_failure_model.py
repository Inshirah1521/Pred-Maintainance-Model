import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Li_S_Battery_Thermal_Failure_Dataset.csv")

try:
    if not os.path.exists(data_path):
         raise FileNotFoundError(f"File {data_path} not found.")
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully from {data_path}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error loading dataset: {e}")

# -----------------------------
# 2️⃣ Preprocessing
# -----------------------------

# Sort by timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Feature Engineering: Rate of Change for sensors
sensor_cols = ['Temp_Sensor_1']
for sensor in sensor_cols:
    df[f'{sensor}_Change'] = df[sensor].diff().fillna(0)

# Create target: Failure in next 15 minutes
prediction_window = 15
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=prediction_window)
df['Failure_Next_15min'] = df['Failure_Label'].rolling(window=indexer).max()
df = df.dropna(subset=['Failure_Next_15min'])

# -----------------------------
# 3️⃣ Keep only relevant features
# -----------------------------
keep_cols = sensor_cols + [f"{s}_Change" for s in sensor_cols] + ['Failure_Next_15min']
df_clean = df[keep_cols]

X = df_clean.drop(columns=['Failure_Next_15min'])
y = df_clean['Failure_Next_15min']

print("Clean feature columns:", list(X.columns))

# -----------------------------
# 4️⃣ Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5️⃣ Build Pipeline & Train
# -----------------------------
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # fill any missing values
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training model...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# -----------------------------
# 6️⃣ Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 7️⃣ Test prediction function
# -----------------------------
# Prediction Testing 
def test_prediction(pipeline, sample_data):
    """
    Takes a single row of data (as DataFrame), predicts failure probability,
    and outputs a user-friendly alert message.
    """
    prediction = pipeline.predict(sample_data)[0]
    probability = pipeline.predict_proba(sample_data)[0][1]
    
    print("--- Diagnostic Report ---")
    print("Current Sensor Readings:")
    for col in sample_data.columns:
        print(f"  {col}: {sample_data[col].values[0]}")
    
    print("\nModel Analysis:")
    if prediction == 1:
        print(f"⚠️  ALERT: High Risk of Thermal Failure in next 15 minutes! (Probability: {probability:.2%})")
    else:
        print(f"✅  Status Normal. No immediate failure predicted. (Probability of failure: {probability:.2%})")

# Sample a few cases from the test set for demonstration
print("Test Case 1 (Random Sample):")
sample_row = X_test.sample(1, random_state=42)
test_prediction(pipeline, sample_row)

print("\n" + "="*30 + "\n")

# Try to find a positive case in test set to show the alert
positive_cases = X_test[y_test == 1]
if not positive_cases.empty:
    print("Test Case 2 (Known Future Failure):")
    positive_sample = positive_cases.sample(1, random_state=123)
    test_prediction(pipeline, positive_sample)
else:
    print("No positive failure cases in test set to display.")

# -----------------------------
# 8️⃣ Save the clean model
# -----------------------------
joblib.dump(pipeline, "thermal_failure_model_clean.joblib")
print("Clean model saved as 'thermal_failure_model_clean.joblib'.")
