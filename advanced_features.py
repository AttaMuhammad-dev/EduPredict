import os
import json
import statistics
from datetime import datetime
from db import (
    db, users_col, educational_records_col, alerts_col,
    system_logs_col
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


def _to_dict(doc):
    if doc is None:
        return None
    d = dict(doc)
    if "_id" in d:
        d["id"] = d.pop("_id")
    return d


def _to_dicts(docs):
    return [_to_dict(d) for d in docs]


def init_database():
    print("MongoDB collections ready (advanced_features)")


def save_record(student_id, student_name, attendance, marks, lms_activity, previous_performance):
    average = (attendance + marks + lms_activity + previous_performance) / 4
    if average >= 75:
        risk = "Low"
    elif average >= 60:
        risk = "Medium"
    else:
        risk = "High"

    educational_records_col.insert_one({
        "student_id": str(student_id),
        "student_name": student_name,
        "attendance": attendance,
        "marks": marks,
        "lms_activity": lms_activity,
        "previous_performance": previous_performance,
        "risk": risk,
        "created_at": datetime.now().isoformat()
    })
    return risk


def get_records():
    records = list(educational_records_col.find().sort("_id", -1))
    return _to_dicts(records)


def detect_anomalies(records):
    anomalies = []
    for record in records:
        reasons = []
        attendance = record.get("attendance", 0)
        marks = record.get("marks", 0)
        lms = record.get("lms_activity", 0)
        previous = record.get("previous_performance", 0)
        if attendance < 40:
            reasons.append("Very low attendance")
        if marks < 40:
            reasons.append("Very low marks")
        if lms < 30:
            reasons.append("Very low LMS activity")
        if previous < 40:
            reasons.append("Low previous performance")
        if reasons:
            anomalies.append({
                "student_id": record.get("student_id"),
                "student_name": record.get("student_name"),
                "reasons": reasons,
                "severity": "High"
            })
    return anomalies


def correlation(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0
    try:
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
        denominator = (sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)) ** 0.5
        if denominator == 0:
            return 0
        return round(numerator / denominator, 3)
    except Exception:
        return 0


def get_correlations():
    records = get_records()
    if not records:
        return {"attendance_marks": 0, "attendance_lms": 0, "marks_lms": 0}
    attendance = [float(r["attendance"]) for r in records]
    marks = [float(r["marks"]) for r in records]
    lms = [float(r["lms_activity"]) for r in records]
    return {
        "attendance_marks": correlation(attendance, marks),
        "attendance_lms": correlation(attendance, lms),
        "marks_lms": correlation(marks, lms)
    }


def generate_alerts():
    records = get_records()
    generated = []
    for record in records:
        if record["risk"] == "High":
            message = f"High-risk student detected: {record['student_name'] or record['student_id']}"
            existing = alerts_col.find_one({
                "student_id": record["student_id"],
                "alert_type": "Academic Risk",
                "status": "New"
            })
            if not existing:
                alerts_col.insert_one({
                    "student_id": record["student_id"],
                    "alert_type": "Academic Risk",
                    "message": message,
                    "severity": "High",
                    "created_at": datetime.now().isoformat(),
                    "status": "New"
                })
                generated.append(message)
    return generated


def get_alerts():
    alerts = list(alerts_col.find().sort("_id", -1))
    return _to_dicts(alerts)


def log_event(event, details=""):
    system_logs_col.insert_one({
        "event": event,
        "details": details,
        "created_at": datetime.now().isoformat()
    })


def create_backup():
    records = get_records()
    filename = f"edupredict_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(BACKUP_DIR, filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4)
    return path


def system_status():
    try:
        db.command("ping")
        status = "active"
    except Exception:
        status = "inactive"
    return {
        "database": status,
        "storage": "active",
        "anomaly_detection": "active",
        "correlation_analysis": "active",
        "automated_alerts": "active",
        "backup": "active",
        "timestamp": datetime.now().isoformat()
    }


init_database()
