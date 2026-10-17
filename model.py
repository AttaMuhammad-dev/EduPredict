import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

try:
    import config
    MODEL_FILE = getattr(config, 'MODEL_FILE', 'risk_model.pkl')
    SCALER_FILE = getattr(config, 'SCALER_FILE', 'scaler.pkl')
    DEMAND_MODEL_FILE = getattr(config, 'COURSE_DEMAND_MODEL_FILE', 'demand_model.pkl')
except ImportError:
    MODEL_FILE = 'risk_model.pkl'
    SCALER_FILE = 'scaler.pkl'
    DEMAND_MODEL_FILE = 'demand_model.pkl'

GRADE_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'grade_model.pkl')
GRADE_SCALER_FILE = os.path.join(os.path.dirname(__file__), 'grade_scaler.pkl')
DEMAND_SCALER_FILE = os.path.join(os.path.dirname(__file__), 'demand_scaler.pkl')
MODEL_METADATA = os.path.join(os.path.dirname(__file__), 'model_metadata.json')


# ============================================================
# SYNTHETIC DATA GENERATORS FOR TRAINING
# ============================================================

def generate_student_data(n=3000):
    """Generates synthetic student performance records reflecting realistic institutional trends."""
    np.random.seed(42)
    attendance = np.random.normal(76, 14, n).clip(30, 100)
    lms_activity = np.random.normal(68, 18, n).clip(15, 100)
    previous_performance = np.random.normal(72, 16, n).clip(25, 100)

    marks = (0.45 * attendance + 0.35 * previous_performance + 0.20 * lms_activity)
    marks += np.random.normal(0, 4, n)
    marks = marks.clip(20, 100)

    risk_indices = (
        (100.0 - attendance) * 0.35 +
        (100.0 - marks) * 0.35 +
        (100.0 - lms_activity) * 0.15 +
        (100.0 - previous_performance) * 0.15
    )
    risk_indices += np.random.normal(0, 3, n)
    risk_indices = risk_indices.clip(0, 100)

    risk = []
    for r in risk_indices:
        if r >= 48:
            risk.append('High')
        elif r >= 26:
            risk.append('Medium')
        else:
            risk.append('Low')

    return pd.DataFrame({
        'attendance': attendance.round(1),
        'marks': marks.round(1),
        'lms_activity': lms_activity.round(1),
        'previous_performance': previous_performance.round(1),
        'risk': risk
    })


def generate_course_demand_data(n=1200):
    """Generates synthetic elective course enrollment and capacity metrics."""
    np.random.seed(42)
    enrolled_pool = np.random.randint(100, 800, size=n)
    faculty_capacity = np.random.randint(2, 10, size=n)
    interest_score = np.random.uniform(30.0, 99.0, size=n)
    semester_level = np.random.randint(1, 9, size=n)

    demand_seats = (enrolled_pool * (interest_score / 100.0) * 0.65) + (faculty_capacity * 12)
    demand_seats = demand_seats.clip(20, 500).round()

    return pd.DataFrame({
        'enrolled_pool': enrolled_pool,
        'faculty_capacity': faculty_capacity,
        'interest_score': interest_score.round(1),
        'semester_level': semester_level,
        'demand_seats': demand_seats
    })


# ============================================================
# MODEL TRAINING & PERSISTENCE (THE PREDICTIVE TRIAD)
# ============================================================

def train_and_save_model():
    """Trains all 3 models: Risk Classifier, Grade Regressor, and Course Demand Estimator."""
    df_students = generate_student_data(3500)

    # 1. Train Risk Classifier
    X_risk = df_students[['attendance', 'marks', 'lms_activity', 'previous_performance']]
    y_risk = df_students['risk']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)

    risk_scaler = StandardScaler()
    X_train_r_scaled = risk_scaler.fit_transform(X_train_r)
    X_test_r_scaled = risk_scaler.transform(X_test_r)

    risk_model = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        class_weight='balanced'
    )
    risk_model.fit(X_train_r_scaled, y_train_r)
    risk_acc = accuracy_score(y_test_r, risk_model.predict(X_test_r_scaled))

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(risk_model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(risk_scaler, f)

    # 2. Train Continuous Grade Forecaster
    X_grade = df_students[['attendance', 'lms_activity', 'previous_performance']]
    y_grade = df_students['marks']
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_grade, y_grade, test_size=0.2, random_state=42)

    grade_scaler = StandardScaler()
    X_train_g_scaled = grade_scaler.fit_transform(X_train_g)
    X_test_g_scaled = grade_scaler.transform(X_test_g)

    grade_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
    grade_model.fit(X_train_g_scaled, y_train_g)
    grade_r2 = r2_score(y_test_g, grade_model.predict(X_test_g_scaled))

    with open(GRADE_MODEL_FILE, 'wb') as f:
        pickle.dump(grade_model, f)
    with open(GRADE_SCALER_FILE, 'wb') as f:
        pickle.dump(grade_scaler, f)

    # 3. Train Course Demand Forecaster
    df_demand = generate_course_demand_data(1500)
    X_demand = df_demand[['enrolled_pool', 'faculty_capacity', 'interest_score', 'semester_level']]
    y_demand = df_demand['demand_seats']
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)

    demand_scaler = StandardScaler()
    X_train_d_scaled = demand_scaler.fit_transform(X_train_d)
    X_test_d_scaled = demand_scaler.transform(X_test_d)

    demand_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    demand_model.fit(X_train_d_scaled, y_train_d)
    demand_r2 = r2_score(y_test_d, demand_model.predict(X_test_d_scaled))

    with open(DEMAND_MODEL_FILE, 'wb') as f:
        pickle.dump(demand_model, f)
    with open(DEMAND_SCALER_FILE, 'wb') as f:
        pickle.dump(demand_scaler, f)

    metadata = {
        'risk_model_accuracy': round(float(risk_acc), 4),
        'grade_model_r2': round(float(grade_r2), 4),
        'demand_model_r2': round(float(demand_r2), 4),
        'features': ['attendance', 'marks', 'lms_activity', 'previous_performance'],
        'classes': ['Low', 'Medium', 'High'],
        'trained_at': datetime.now().isoformat()
    }
    with open(MODEL_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[ML PIPELINE READY] Risk Model Accuracy: {risk_acc:.4f} | Grade R2: {grade_r2:.4f} | Demand R2: {demand_r2:.4f}")
    return risk_acc


def load_model():
    """Loads the student risk classifier and its precomputed standard scaler."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        train_and_save_model()

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def load_grade_model():
    """Loads continuous grade regression model and scaler."""
    if not os.path.exists(GRADE_MODEL_FILE) or not os.path.exists(GRADE_SCALER_FILE):
        train_and_save_model()

    with open(GRADE_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(GRADE_SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def load_demand_model():
    """Loads course demand model and scaler."""
    if not os.path.exists(DEMAND_MODEL_FILE) or not os.path.exists(DEMAND_SCALER_FILE):
        train_and_save_model()

    with open(DEMAND_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(DEMAND_SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# ============================================================
# INFERENCE PIPELINE FUNCTIONS
# ============================================================

def predict_risk(attendance, marks, lms_activity, previous_performance):
    """Evaluates retention risk level, projected grade, and remedial directives for a student."""
    model, scaler = load_model()

    features = pd.DataFrame([[float(attendance), float(marks), float(lms_activity), float(previous_performance)]], columns=['attendance', 'marks', 'lms_activity', 'previous_performance'])
    features_scaled = scaler.transform(features)

    risk = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(max(probabilities))

    # Calculate calibrated risk index (0 - 100)
    risk_score = (
        (100.0 - float(attendance)) * 0.35 +
        (100.0 - float(marks)) * 0.35 +
        (100.0 - float(lms_activity)) * 0.15 +
        (100.0 - float(previous_performance)) * 0.15
    )
    risk_score = round(max(0.0, min(100.0, risk_score)), 2)

    # Determine projected letter grade
    m_val = float(marks)
    if m_val >= 80.0:
        predicted_grade = "A"
    elif m_val >= 65.0:
        predicted_grade = "B"
    elif m_val >= 50.0:
        predicted_grade = "C"
    else:
        predicted_grade = "F"

    risk_level_map = {
        'Low': 'Low Risk (Nominal)',
        'Medium': 'Medium Risk (Watchlist)',
        'High': 'High Risk (Critical Intervention)'
    }

    recommendation_map = {
        'Low': 'Learner demonstrates nominal engagement and retention indicators. Continue standard tracking.',
        'Medium': 'Early degradation signs detected. Recommended: automated check-in reminder and peer tutoring allocation.',
        'High': 'Critical retention hazard. Immediate advisor email alert escalation and scheduled remedial counseling required.'
    }

    return {
        'risk': risk,
        'risk_level': risk_level_map.get(risk, risk),
        'confidence': round(confidence, 4),
        'risk_score': risk_score,
        'risk_probability': round(float(risk_score / 100.0), 3),
        'predicted_grade': predicted_grade,
        'recommendation': recommendation_map.get(risk, 'Standard academic monitoring recommended.'),
        'attendance': float(attendance),
        'marks': float(marks),
        'lms_activity': float(lms_activity),
        'previous_performance': float(previous_performance)
    }


def predict_grade(attendance, lms_activity, previous_performance):
    """Predicts expected continuous examination score."""
    model, scaler = load_grade_model()
    features = pd.DataFrame([[float(attendance), float(lms_activity), float(previous_performance)]], columns=['attendance', 'lms_activity', 'previous_performance'])
    features_scaled = scaler.transform(features)
    predicted_score = round(float(model.predict(features_scaled)[0]), 1)

    if predicted_score >= 80:
        letter = 'A'
    elif predicted_score >= 65:
        letter = 'B'
    elif predicted_score >= 50:
        letter = 'C'
    else:
        letter = 'F'

    return {
        'predicted_marks': predicted_score,
        'predicted_grade': letter
    }


def predict_course_demand(enrolled_pool, faculty_capacity, interest_score, semester_level=1):
    """Projects expected student seat enrollment demand for elective/course modules."""
    model, scaler = load_demand_model()
    features = np.array([[float(enrolled_pool), float(faculty_capacity), float(interest_score), float(semester_level)]])
    features_scaled = scaler.transform(features)
    seats = int(round(float(model.predict(features_scaled)[0])))

    return {
        'projected_seats_demand': max(10, seats),
        'saturation_rate': round(min(100.0, (seats / max(1, faculty_capacity * 25)) * 100.0), 1)
    }


if __name__ == "__main__":
    train_and_save_model()