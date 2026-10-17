import joblib
import os

MODEL_FILE = "risk_model.pkl"

if not os.path.exists(MODEL_FILE):
    print("risk_model.pkl not found.")
else:
    model = joblib.load(MODEL_FILE)
    print("EduPredict ML model loaded successfully!")