import csv
import io
import json
import os
import secrets
import hashlib
import pickle
import smtplib
from datetime import datetime
from functools import wraps
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from release_safety import install as install_release_safety, dataset_frame, safe_csv, scoped_query, owns, MAX_FILE_BYTES, teacher_student_ids
import bcrypt
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_file, send_from_directory, has_app_context
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

# Import project configurations and modules
import config
from config import *
from model import (predict_risk, train_and_save_model, load_model,
                   GRADE_MODEL_FILE, GRADE_SCALER_FILE,
                   DEMAND_MODEL_FILE, DEMAND_SCALER_FILE)
from data_processor import process_csv, validate_dataset, generate_sample_data
from analytics_data import (
    init_database, save_record, get_records,
    detect_anomalies, get_correlations, generate_alerts,
    get_alerts, log_event, create_backup, system_status,
    save_prediction, get_predictions
)
from db import (
    db, init_app as init_db_app, users_col, educational_records_col, alerts_col,
    system_logs_col, batches_col, student_batches_col,
    assignments_col, student_marks_col, predictions_col,
    datasets_col, dataset_mappings_col, reports_col
)

app = Flask(__name__)
install_release_safety(app)
app.secret_key = SECRET_KEY
app.config["MONGO_URI"] = MONGO_URI
app.config["DB_NAME"] = DB_NAME
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Setup student assignment submissions upload directory
SUBMISSIONS_UPLOAD_DIR = config.SUBMISSIONS_FOLDER
REPORTS_DIR = config.REPORTS_FOLDER

def _writable_dir(folder):
    """Return a writable storage folder, falling back to /tmp on read-only runtimes."""
    import tempfile
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, ".edupredict_write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return folder
    except Exception:
        base = os.path.join(tempfile.gettempdir(), "edupredict")
        os.makedirs(base, exist_ok=True)
        return base

SUBMISSIONS_UPLOAD_DIR = _writable_dir(SUBMISSIONS_UPLOAD_DIR)
REPORTS_DIR = _writable_dir(REPORTS_DIR)

ALLOWED_SUBMISSION_EXTENSIONS = {'pdf', 'docx', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'txt'}

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "true").lower() == "true",
)
app.config["CORS_ORIGINS"] = [value.strip() for value in os.environ.get("ALLOWED_ORIGINS", "").split(",") if value.strip()]
if app.config["CORS_ORIGINS"]:
    CORS(app, origins=app.config["CORS_ORIGINS"])

init_db_app(app)


# ============================================================
# DATABASE HELPERS & ID NORMALIZERS
# ============================================================

def _db_ok():
    return has_app_context()


def _next_id(collection):
    if not _db_ok():
        return int(datetime.now().timestamp() * 1000) % 1000000
    
    counter = db["counters"]
    
    seq_doc = counter.find_one({"_id": collection.name})
    if not seq_doc:
        max_doc = collection.find_one({}, sort=[("id", -1)])
        try:
            start_seq = int(max_doc["id"]) if max_doc and "id" in max_doc else 1000
        except (ValueError, TypeError):
            start_seq = 1000
            
        try:
            counter.update_one({"_id": collection.name}, {"$setOnInsert": {"seq": start_seq}}, upsert=True)
        except DuplicateKeyError:
            pass

    result = counter.find_one_and_update(
        {"_id": collection.name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    return result["seq"]


def _doc_id(doc):
    if doc is None:
        return None
    val = doc.get("id")
    if val is not None:
        return val
    oid = doc.get("_id")
    if oid is not None:
        return str(oid)
    return None


def _to_dict(doc):
    if doc is None:
        return None
    d = dict(doc)
    if "_id" in d:
        if "id" not in d:
            d["id"] = str(d.pop("_id"))
        else:
            d.pop("_id")
    d.pop("password", None)
    d.pop("file_data", None)
    d.pop("csv_data", None)
    d.pop("file_path_internal", None)
    
    for k, v in list(d.items()):
        if isinstance(v, ObjectId):
            d[k] = str(v)
            
    return d


def _to_dicts(docs):
    return [_to_dict(d) for d in docs]


def _normalize_id(val):
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return val


def _id_query(user_id):
    nid = _normalize_id(user_id)
    if nid is None:
        return {"$in": []}
    alternatives = [nid]
    if isinstance(nid, int):
        alternatives.append(str(nid))
    elif isinstance(nid, str):
        try:
            alternatives.append(int(nid))
        except (ValueError, TypeError):
            pass
    return {"$in": list(set(alternatives))}


def _safe_object_id(oid_str):
    try:
        return ObjectId(str(oid_str))
    except Exception:
        return None


def _find_user_by_id(uid):
    if uid is None or not _db_ok():
        return None
    nid = _normalize_id(uid)
    user = users_col.find_one({"id": nid})
    if not user:
        user = users_col.find_one({"id": str(nid)})
    if not user and _safe_object_id(uid):
        user = users_col.find_one({"_id": _safe_object_id(uid)})
    
    if not user and isinstance(uid, str):
        user = users_col.find_one({"name": uid})
    return user


# ============================================================
# INITIALIZATION & MODEL CALIBRATION
# ============================================================

try:
    with app.app_context():
        init_database()
except Exception as e:
    print(f"Database init warning: {e}")

_MODEL_FILES = [MODEL_FILE, SCALER_FILE, GRADE_MODEL_FILE,
                GRADE_SCALER_FILE, DEMAND_MODEL_FILE, DEMAND_SCALER_FILE]
try:
    if not all(os.path.exists(f) for f in _MODEL_FILES):
        train_and_save_model()
    print("Core Scikit-Learn predictive models initialized successfully!")
except Exception as e:
    print(f"Model initialization warning: {e}")


# ============================================================
# RBAC SECURITY DECORATORS
# ============================================================

def _valid_session():
    if 'user_id' not in session:
        return False
    user = _find_user_by_id(session['user_id'])
    if not user or not user.get('is_active', 1) or session.get('auth_version', 0) != user.get('auth_version', 0):
        session.clear()
        return False
    session['role'] = user.get('role', 'student')
    return True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not _valid_session():
            return (jsonify(success=False, message="Sign in to continue."), 401) if request.path.startswith("/api/") else redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not _valid_session():
                return (jsonify(success=False, message="Sign in to continue."), 401) if request.path.startswith("/api/") else redirect(url_for("login"))
            if session.get("role") not in allowed_roles:
                return jsonify({"success": False, "message": "Unauthorized role clearance"}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================
# AUTOMATED SMTP ALERT & NOTIFICATION SERVICE
# ============================================================

def get_target_email(user_id=None, role=None):
    if user_id and _db_ok():
        user = _find_user_by_id(user_id)
        if user and user.get("email"):
            return user.get("email")
    
    if role and hasattr(config, 'ROLE_EMAILS') and role in config.ROLE_EMAILS:
        return config.ROLE_EMAILS[role]
        
    return getattr(config, 'ADMIN_EMAIL', "")


def send_email(to_email, subject, body, html_body=None, attachment=None, attachment_name=None):
    if not config.EMAIL_ENABLED:
        return False

    if not to_email:
        print("[SMTP ERROR] No recipient email provided")
        return False

    dummy_domains = ["@student.com", "@edupredict.com", "@example.com", "test", "demo"]
    if any(d in to_email.lower() for d in dummy_domains):
        print(f"[SMTP BYPASS] Dummy email detected ({to_email}). Skipping actual dispatch.")
        return True

    try:
        msg = MIMEMultipart('alternative')
        sender = getattr(config, 'EMAIL_USER', EMAIL_USER)
        msg['From'] = f"EduPredict Alerts <{sender}>"
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        if html_body:
            msg.attach(MIMEText(html_body, 'html'))

        if attachment:
            part = MIMEBase('application', 'octet-stream')
            if isinstance(attachment, str):
                part.set_payload(attachment.encode('utf-8'))
            else:
                part.set_payload(attachment)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{attachment_name or "EduPredict_Report.csv"}"')
            msg.attach(part)

        host = getattr(config, 'EMAIL_HOST', EMAIL_HOST)
        port = int(getattr(config, 'EMAIL_PORT', EMAIL_PORT))
        user = getattr(config, 'EMAIL_USER', EMAIL_USER)
        password = getattr(config, 'EMAIL_PASSWORD', EMAIL_PASSWORD)

        if user and password and password != "your-app-password":
            server = smtplib.SMTP(host, port, timeout=15)
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
            server.quit()
            print(f"[SMTP SUCCESS] Email delivered to {to_email}")
            return True
        else:
            print(f"[SMTP SKIPPED] Credentials missing. Email NOT sent to {to_email}")
            return False
    except Exception as e:
        print(f"[SMTP FATAL ERROR] {type(e).__name__}: {e}")
        return False


def build_html_email_template(title, headline, message, metrics=None, action_url=None):
    metrics_html = ""
    if metrics and isinstance(metrics, dict):
        metrics_html = "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:18px 0;'>"
        for k, v in metrics.items():
            metrics_html += f"<div style='margin-bottom:6px;'><strong>{k}:</strong> <span style='color:#0f766e;'>{v}</span></div>"
        metrics_html += "</div>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family:'Inter',sans-serif;background:#f1f5f9;margin:0;padding:24px;color:#1e293b;">
        <div style="max-width:600px;margin:0 auto;background:#ffffff;border-radius:14px;border:1px solid #e2e8f0;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.05);">
            <div style="background:#123c36;padding:24px;color:#ffffff;">
                <span style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#5eead4;">EduPredict Alert Center</span>
                <h2 style="margin:8px 0 0;font-size:22px;">{title}</h2>
            </div>
            <div style="padding:28px;">
                <h3 style="color:#0f172a;margin-top:0;">{headline}</h3>
                <p style="font-size:14px;line-height:1.6;color:#475569;">{message}</p>
                {metrics_html}
                <div style="margin-top:24px;padding-top:18px;border-top:1px solid #f1f5f9;font-size:12px;color:#94a3b8;">
                    This automated alert was dispatched by the EduPredict Big Data Analytics Pipeline.
                </div>
            </div>
        </div>
    </body>
    </html>
    """


# ============================================================
# PRIMARY AUTHENTICATION & SESSION ROUTES
# ============================================================

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        credentials = request.get_json() if request.is_json else request.form
        email = credentials.get("email", "").strip().lower()
        password = credentials.get("password", "")

        if _db_ok():
            try:
                import re
                user_doc = users_col.find_one({"email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
                if user_doc:
                    pw_hash = user_doc.get("password")
                    
                    if hasattr(pw_hash, "decode"):
                        try:
                            pw_bytes = pw_hash.decode('utf-8').encode('utf-8')
                        except Exception:
                            pw_bytes = bytes(pw_hash)
                    elif isinstance(pw_hash, bytes):
                        pw_bytes = pw_hash
                    else:
                        pw_bytes = str(pw_hash).encode('utf-8')

                    if type(pw_bytes).__name__ == 'Binary' or hasattr(pw_bytes, '__bytes__'):
                        pw_bytes = bytes(pw_bytes)

                    is_valid = False
                    try:
                        is_valid = bcrypt.checkpw(password.encode('utf-8'), pw_bytes)
                    except Exception:
                        is_valid = False

                    if is_valid:
                        uid = user_doc.get("id")
                        if uid is None:
                            uid = str(user_doc.get("_id"))

                        session["user_id"] = uid
                        session["user_email"] = user_doc['email']
                        session["role"] = user_doc['role']
                        session["name"] = user_doc['name']
                        session['auth_version'] = user_doc.get('auth_version', 0)

                        log_event("User Login", f"{email} authenticated with role clearance: {user_doc['role']}")
                        return redirect(url_for("dashboard"))
            except Exception as e:
                print(f"Database login error: {e}")

        return render_template("login.html", error="Invalid institutional credentials.")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if request.is_json:
            data = request.get_json() or {}
            name = data.get("name", "").strip()
            email = data.get("email", "").strip().lower()
            password = data.get("password", "")
        else:
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

        role = "student"

        if not name or not email or not password:
            if request.is_json:
                return jsonify({"success": False, "message": "All fields are required"}), 400
            return render_template("login.html", error="All fields are required.")

        if _db_ok():
            try:
                if users_col.find_one({"email": email}):
                    msg = "Email address already registered in identity catalog."
                    if request.is_json:
                        return jsonify({"success": False, "message": msg}), 400
                    return render_template("login.html", error=msg)

                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                uid = _next_id(users_col)
                users_col.insert_one({
                    "id": uid,
                    "name": name,
                    "email": email,
                    "password": hashed,
                    "role": role,
                    "created_at": datetime.now().isoformat(),
                    "is_active": 1
                })
                log_event("User Registered", f"New student account provisioned: {email}")

                welcome_subject = "Welcome to EduPredict"
                welcome_body = f"Hello {name},\n\nYour account has been successfully created. You can now log in using your email: {email}."
                send_email(email, welcome_subject, welcome_body)

                if request.is_json:
                    return jsonify({"success": True, "message": "Student account registered successfully!"})
                return render_template("login.html", success="Registration complete! Please sign in with your student credentials.")
            except Exception as e:
                print(f"Registration error: {e}")

        msg = "Registration pipeline error."
        if request.is_json:
            return jsonify({"success": False, "message": msg}), 500
        return render_template("login.html", error=msg)

    return render_template("register.html")


@app.route("/logout")
def logout():
    if "user_email" in session:
        log_event("User Logout", f"{session['user_email']} logged out")
    session.clear()
    return redirect("/")


# ============================================================
# FIXED DASHBOARD ROUTE - Injects teacher data into template
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    role = session.get("role")
    user_id = session.get("user_id")

    # ============================================================
    # TEACHER DASHBOARD - Full data injection for teacher.html
    # ============================================================
    if role == "teacher":
        if not _db_ok():
            return render_template("teacher.html",
                                   user=session, batches=[], assignments=[],
                                   students=[], submissions=[], grades=[],
                                   high_risk_count=0)

        try:
            t_query = _id_query(user_id)

            # 1. Fetch teacher's assigned batches
            teacher_batches = list(batches_col.find({"teacher_id": t_query}))
            batch_ids = [b.get("id") for b in teacher_batches if b.get("id") is not None]
            bid_alts = list(set(batch_ids + [str(x) for x in batch_ids]))
            batch_name_list = [b.get("name") for b in teacher_batches if b.get("name")]

            # 2. Fetch assignments for these batches
            if bid_alts:
                teacher_assignments = list(assignments_col.find({
                    "$or": [
                        {"teacher_id": t_query},
                        {"batch_id": {"$in": bid_alts}}
                    ]
                }).sort("_id", -1))
            else:
                teacher_assignments = list(assignments_col.find({"teacher_id": t_query}).sort("_id", -1))

            # Enrich assignments with batch_name
            enriched_assignments = []
            for a in teacher_assignments:
                ad = _to_dict(a)
                asg_batch = batches_col.find_one({"$or": [
                    {"id": ad.get("batch_id")},
                    {"id": str(ad.get("batch_id"))},
                    {"_id": _safe_object_id(ad.get("batch_id"))}
                ]})
                ad["batch"] = asg_batch.get("name") if asg_batch else ""
                ad["batch_name"] = ad["batch"]
                ad["max_marks"] = ad.get("total_marks", 100)
                enriched_assignments.append(ad)

            # 3. Fetch students in these batches
            sb_entries = list(student_batches_col.find({"batch_id": {"$in": bid_alts}}))
            student_ids = [s.get("student_id") for s in sb_entries if s.get("student_id") is not None]
            sids_alts = list(set([str(x) for x in student_ids] + [_normalize_id(x) for x in student_ids]))

            # Map student_id -> batch_name
            student_batch_lookup = {}
            for sb in sb_entries:
                sid_str = str(sb.get("student_id"))
                b = batches_col.find_one({"$or": [
                    {"id": sb.get("batch_id")},
                    {"id": str(sb.get("batch_id"))},
                    {"_id": _safe_object_id(sb.get("batch_id"))}
                ]})
                if b:
                    student_batch_lookup[sid_str] = b.get("name", "")

            teacher_students = []
            if sids_alts:
                raw_students = list(users_col.find({"id": {"$in": sids_alts}, "role": "student"}, {"password": 0}))
                for s in raw_students:
                    sd = _to_dict(s)
                    sd["batch"] = student_batch_lookup.get(str(sd.get("id")), "General")
                    pred = predictions_col.find_one(
                        {"student_id": {"$in": [str(sd.get("id")), sd.get("id")]}},
                        sort=[("_id", -1)]
                    )
                    if pred:
                        sd["risk_level"] = pred.get("risk", "Low")
                        sd["risk_score"] = pred.get("confidence", 0)
                        sd["attendance"] = pred.get("attendance")
                        sd["marks"] = pred.get("marks")
                    else:
                        sd["risk_level"] = "Low"
                        sd["risk_score"] = 0
                    teacher_students.append(sd)

            # 4. Fetch submissions
            assignment_ids = [a.get("id") for a in teacher_assignments if a.get("id") is not None]
            aid_alts = list(set(assignment_ids + [str(x) for x in assignment_ids]))

            raw_submissions = list(student_marks_col.find({
                "assignment_id": {"$in": aid_alts}
            }).sort("submitted_at", -1))

            enriched_submissions = []
            for sub in raw_submissions:
                sd = _to_dict(sub)
                asg = assignments_col.find_one({"$or": [
                    {"id": sd.get("assignment_id")},
                    {"id": str(sd.get("assignment_id"))},
                    {"_id": _safe_object_id(sd.get("assignment_id"))}
                ]})
                if asg:
                    sd["assignment_title"] = asg.get("title", "")
                    sd["total_marks"] = asg.get("total_marks", 100)
                user = _find_user_by_id(sd.get("student_id"))
                if user:
                    sd["student_name"] = user.get("name", "")
                    sd["student_email"] = user.get("email", "")
                sd["student_batch"] = student_batch_lookup.get(str(sd.get("student_id")), "")
                sd["marks"] = sd.get("obtained_marks")
                sd["content"] = sd.get("submission_text", "")
                enriched_submissions.append(sd)

            # 5. Grades ledger = graded submissions only
            grades_ledger = [s for s in enriched_submissions if s.get("status") == "graded"]

            # 6. High risk count
            high_risk_count = sum(1 for s in teacher_students
                                  if str(s.get("risk_level", "")).lower() == "high")

            return render_template(
                "teacher.html",
                user=session,
                batches=batch_name_list,
                assignments=enriched_assignments,
                students=teacher_students,
                submissions=enriched_submissions,
                grades=grades_ledger,
                high_risk_count=high_risk_count
            )

        except Exception as e:
            print(f"[TEACHER DASHBOARD] Error: {e}")
            import traceback
            traceback.print_exc()
            return render_template("teacher.html",
                                   user=session, batches=[], assignments=[],
                                   students=[], submissions=[], grades=[],
                                   high_risk_count=0)

    # ============================================================
    # OTHER ROLES
    # ============================================================
    templates = {
        "administrator": "admin_dashboard.html",
        "analyst": "analyst_dashboard.html",
        "student": "student_dashboard.html"
    }
    return render_template(templates.get(role, "dashboard.html"), user=session)


# ============================================================
# DATASET UPLOAD & INGESTION
# ============================================================

@app.route("/api/upload", methods=["POST"])
@app.route("/api/upload-dataset", methods=["POST"])
@role_required(["administrator", "analyst", "teacher"])
def upload_dataset():
    try:
        if "file" in request.files:
            file = request.files["file"]
        elif len(request.files) > 0:
            file = next(iter(request.files.values()))
        else:
            return jsonify({"success": False, "message": "No file uploaded", "error": "No file uploaded"}), 400

        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected", "error": "No file selected"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"success": False, "message": "Only CSV files are allowed", "error": "Only CSV files are allowed"}), 400

        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8', errors='ignore')))

        if df.empty:
            return jsonify({"success": False, "message": "CSV file is empty", "error": "CSV file is empty"}), 400

        csv_data = df.to_csv(index=False)
        if len(csv_data.encode('utf-8')) > MAX_FILE_BYTES or len(df) > 10000 or len(df.columns) > 100:
            return jsonify(success=False, message='Use a CSV under 2 MB, 10,000 rows and 100 columns.'), 400
        if datasets_col.count_documents({'uploaded_by': session.get('user_id')}) >= 20:
            return jsonify(success=False, message='Demo upload limit reached. Contact the project owner to clear old datasets.'), 429
        dataset_id = _next_id(datasets_col)
        datasets_col.insert_one({
            'id': dataset_id, 'name': secure_filename(file.filename), 'filename': secure_filename(file.filename),
            'csv_data': csv_data, 'uploaded_by': session.get('user_id'),
            'row_count': len(df), 'total_records': len(df), 'column_count': len(df.columns),
            'columns': list(df.columns), 'uploaded_at': datetime.now().isoformat()
        })
        log_event('Dataset Uploaded', f'Dataset #{dataset_id} stored durably ({len(df)} rows)')

        send_alert = request.form.get("send_alert") == "true" or request.form.get("send_alert") is True
        alert_email = request.form.get("alert_email") or get_target_email(role="administrator")

        if send_alert and alert_email:
            subj = f"[EduPredict HDFS Ingestion] Partition Mounted: {file.filename}"
            body = f"Hello,\n\nA new dataset partition ({file.filename}) containing {len(df)} records has been successfully mounted into the HDFS ingestion tier.\n\nUploaded By: {session.get('name', 'Analyst')}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nEduPredict Data Center Gateway"
            send_email(alert_email, subj, body)

        return jsonify({
            "success": True,
            "message": "Dataset uploaded and partition mounted successfully!",
            "dataset_id": dataset_id,
            "id": dataset_id,
            "filename": file.filename,
            "total_records": len(df),
            "row_count": len(df),
            "columns": list(df.columns)
        })
    except Exception as e:
        print(f"Upload exception: {e}")
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500


@app.route("/api/process-dataset", methods=["POST"])
@role_required(["administrator", "analyst"])
def process_dataset():
    try:
        data = request.get_json() or {}
        dataset_id = data.get("dataset_id")
        x_columns = data.get("x_columns", [])
        y_column = data.get("y_column")

        if not x_columns or not y_column:
            return jsonify({"success": False, "message": "Please select X features and Y target"}), 400

        dataset = None
        if _db_ok():
            dataset = datasets_col.find_one(scoped_query({
                "$or": [{"id": _normalize_id(dataset_id)}, {"id": str(dataset_id)}, {"_id": _safe_object_id(dataset_id)}]
            }, "uploaded_by"))

        if not dataset:
            return jsonify({"success": False, "message": "Dataset not found or access denied"}), 404

        df = dataset_frame(dataset)
        for col in x_columns + [y_column]:
            if col not in df.columns:
                return jsonify({"success": False, "message": f"Column '{col}' not found"}), 400

        X = df[x_columns].apply(pd.to_numeric, errors='coerce')
        y = df[y_column].apply(pd.to_numeric, errors='coerce')
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < 10 or len(x_columns) > 20:
            return jsonify({"success": False, "message": "Use at least 10 valid numeric rows and at most 20 features"}), 400

        is_classification = 2 <= len(set(y_clean)) <= 10
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score

        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            model_type = "Classification"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            model_type = "Regression"

        if _db_ok():
            dataset_mappings_col.insert_one({
                "dataset_id": dataset_id,
                "x_columns": json.dumps(x_columns),
                "y_column": y_column,
                "model_type": model_type,
                "score": round(float(score), 4),
                "created_at": datetime.now().isoformat()
            })
            log_event("Custom Model Trained", f"Trained {model_type} on dataset #{dataset_id} (Score: {score:.4f})")

        return jsonify({
            "success": True,
            "message": f"Model trained! {model_type} - Score: {score:.4f}",
            "model_type": model_type,
            "score": round(float(score), 4),
            "total_samples": len(X_clean),
            "x_columns": x_columns,
            "y_column": y_column
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/chart-data", methods=["POST"])
@role_required(["analyst", "administrator"])
def get_chart_data():
    try:
        data = request.get_json() or {}
        dataset_id = data.get("dataset_id")
        x_column = data.get("x_column")
        y_column = data.get("y_column")

        dataset = None
        if _db_ok():
            dataset = datasets_col.find_one(scoped_query({
                "$or": [{"id": _normalize_id(dataset_id)}, {"id": str(dataset_id)}, {"_id": _safe_object_id(dataset_id)}]
            }, "uploaded_by"))

        if not dataset:
            return jsonify({"success": False, "message": "Dataset not found"}), 404

        df = dataset_frame(dataset)
        if x_column not in df.columns or y_column not in df.columns:
            return jsonify({"success": False, "message": "Columns not found"}), 400

        df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
        df_clean = df[[x_column, y_column]].dropna()
        df_sample = df_clean.head(min(100, len(df_clean)))

        return jsonify({
            "success": True,
            "x_column": x_column,
            "y_column": y_column,
            "data": {
                "labels": df_sample[x_column].tolist(),
                "values": df_sample[y_column].tolist()
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# PREDICTION ROUTES
# ============================================================

@app.route("/api/predict", methods=["POST"])
@role_required(["administrator", "teacher", "analyst"])
def predict():
    try:
        data = request.get_json() or {}
        att = float(data.get("attendance") or data.get("ATTENDANCE") or 0)
        marks = float(data.get("marks") or data.get("assignment_score") or data.get("score") or 0)
        lms = float(data.get("lms_activity") or data.get("lms") or 0)
        prev = float(data.get("previous_performance") or data.get("study_hours") or data.get("previous_score") or marks)
        s_name = data.get("student_name") or data.get("name") or "Manual Inspection"
        sid = str(data.get("student_id") or data.get("id") or session.get("user_id", "unknown"))

        result = predict_risk(att, marks, lms, prev)

        if _db_ok():
            try:
                predictions_col.insert_one({
                    "student_id": sid,
                    "student_name": s_name,
                    "attendance": round(att, 1),
                    "marks": round(marks, 1),
                    "lms_activity": round(lms, 1),
                    "previous_performance": round(prev, 1),
                    "risk": result["risk"],
                    "confidence": result.get("confidence", 0.85),
                    "predicted_grade": result.get("predicted_grade", "B"),
                    "predicted_at": datetime.now().isoformat()
                })
                save_record(sid, s_name, att, marks, lms, prev)
            except Exception:
                pass

        log_event("Single Prediction", f"Predicted {result['risk']} for {s_name}")
        return jsonify({"success": True, **result})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "message": str(e)}), 400


@app.route("/api/predict-batch", methods=["POST"])
@role_required(["administrator", "teacher", "analyst"])
def predict_batch():
    try:
        data = request.get_json() or {}
        if isinstance(data, list):
            students = data
        elif isinstance(data, dict):
            students = data.get("students") or data.get("records") or data.get("data") or []
        else:
            students = []

        results = []
        for student in students:
            att = float(student.get("attendance") or student.get("ATTENDANCE") or 75.0)
            marks = float(student.get("marks") or student.get("assignment_score") or student.get("score") or 70.0)
            lms = float(student.get("lms_activity") or student.get("lms") or 70.0)
            prev = float(student.get("previous_performance") or student.get("study_hours") or marks)
            s_name = student.get("student_name") or student.get("name") or f"Student #{student.get('student_id', student.get('id', ''))}"
            sid = str(student.get("student_id") or student.get("id") or "unknown")

            pred = predict_risk(att, marks, lms, prev)
            conf_val = round(float(pred.get("confidence", 0.85)), 4)

            res_item = {
                "student_id": sid,
                "id": sid,
                "student_name": s_name,
                "name": s_name,
                "attendance": round(att, 1),
                "marks": round(marks, 1),
                "assessment_score": round(marks, 1),
                "lms_activity": round(lms, 1),
                "previous_performance": round(prev, 1),
                "prior_performance": round(prev, 1),
                "risk": pred["risk"],
                "risk_level": pred.get("risk_level", pred["risk"]),
                "confidence": conf_val,
                "certainty": conf_val,
                "predicted_grade": pred.get("predicted_grade", "B")
            }
            results.append(res_item)

            if _db_ok():
                try:
                    predictions_col.insert_one({
                        "student_id": sid,
                        "student_name": s_name,
                        "attendance": round(att, 1),
                        "marks": round(marks, 1),
                        "lms_activity": round(lms, 1),
                        "previous_performance": round(prev, 1),
                        "risk": pred["risk"],
                        "confidence": conf_val,
                        "predicted_grade": pred.get("predicted_grade", "B"),
                        "predicted_at": datetime.now().isoformat()
                    })
                except Exception:
                    pass

        try:
            generate_alerts()
        except Exception:
            pass

        return jsonify({
            "success": True,
            "predictions": results,
            "results": results,
            "total": len(results)
        })
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return jsonify({"success": False, "message": str(e)}), 400


# ============================================================
# USER MANAGEMENT (ADMIN CONTROL PANEL)
# ============================================================

@app.route("/api/admin/users", methods=["GET", "POST", "PUT", "DELETE"])
@role_required(["administrator"])
def manage_users():
    if request.method == "GET":
        if _db_ok():
            try:
                users = list(users_col.find().sort("_id", -1))
                return jsonify({"success": True, "users": _to_dicts(users)})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)}), 500
        return jsonify({"success": True, "users": []})

    elif request.method == "POST":
        data = request.get_json() or {}
        email = data.get("email", "").strip().lower()
        name = data.get("name", "").strip()
        role = data.get("role", "student")
        password = data.get("password") or "password123"

        if not email or not name:
            return jsonify({"success": False, "message": "Name and email required"}), 400

        if _db_ok():
            try:
                if users_col.find_one({"email": email}):
                    return jsonify({"success": False, "message": "Email already exists"}), 400

                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                uid = _next_id(users_col)
                users_col.insert_one({
                    "id": uid, "email": email, "password": hashed, "name": name,
                    "role": role, "created_at": datetime.now().isoformat(), "is_active": 1
                })
                
                subj = "EduPredict Account Provisioned"
                body = f"Hello {name},\n\nYour account has been created with role: {role.upper()}.\nLogin Email: {email}\n\nPlease contact administration for your temporary password."
                send_email(email, subj, body)

                log_event("User Provisioned", f"Administrator created user: {email} with role {role}")
                return jsonify({"success": True, "message": f"User {email} created successfully with role {role}."})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)}), 400
        return jsonify({"success": True, "message": "User created."})

    elif request.method == "PUT":
        data = request.get_json() or {}
        user_id = data.get("id")
        if _normalize_id(user_id) == 1:
            return jsonify({"success": False, "message": "Master cluster administrator cannot be modified"}), 403

        if _db_ok():
            try:
                update_fields = {}
                if data.get("name"):
                    update_fields["name"] = data.get("name")
                if data.get("role"):
                    update_fields["role"] = data.get("role")
                if "is_active" in data:
                    update_fields["is_active"] = data.get("is_active")

                users_col.update_one(
                    {"$or": [{"id": _normalize_id(user_id)}, {"id": str(user_id)}, {"_id": _safe_object_id(user_id)}]},
                    {"$set": update_fields}
                )
                log_event("User Role Updated", f"Administrator updated user #{user_id} - New Role: {data.get('role')}")
                return jsonify({"success": True, "message": "User credentials and role updated successfully!"})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)}), 400
        return jsonify({"success": True, "message": "User updated."})

    elif request.method == "DELETE":
        user_id = request.args.get("id")
        if str(user_id) == "1":
            return jsonify({"success": False, "message": "Master administrator cannot be deleted"}), 403
        if str(user_id) == str(session.get("user_id")):
            return jsonify({"success": False, "message": "Cannot revoke active administrative session"}), 400

        if _db_ok():
            try:
                nid = _normalize_id(user_id)
                users_col.delete_one({"$or": [{"id": nid}, {"id": str(user_id)}, {"_id": _safe_object_id(user_id)}]})
                return jsonify({"success": True, "message": "User purged from directory."})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)}), 400
        return jsonify({"success": True, "message": "User deleted."})


@app.route("/api/admin/teachers", methods=["GET"])
@role_required(["administrator", "teacher"])
def get_teachers():
    if _db_ok():
        teachers = list(users_col.find({"role": "teacher", "is_active": 1}))
        return jsonify({"success": True, "teachers": _to_dicts(teachers)})
    return jsonify({"success": True, "teachers": []})


@app.route("/api/admin/students", methods=["GET"])
@role_required(["administrator", "teacher", "analyst"])
def get_students():
    if _db_ok():
        role = session.get("role")
        students = list(users_col.find({"role": "student", "is_active": 1}))

        # Batch lookup (student_id -> batch info)
        student_batch_map = {}
        for sb in student_batches_col.find():
            sid = str(sb.get("student_id"))
            b = batches_col.find_one({
                "$or": [
                    {"id": sb.get("batch_id")},
                    {"id": str(sb.get("batch_id"))},
                    {"_id": _safe_object_id(sb.get("batch_id"))}
                ]
            })
            if b:
                student_batch_map[sid] = {"batch_id": b.get("id"), "batch_name": b.get("name", "")}
                nid = _normalize_id(sb.get("student_id"))
                if nid is not None:
                    student_batch_map[str(nid)] = student_batch_map[sid]

        # Teacher: only students in their assigned cohorts (matches dashboard scope)
        allowed_sids = None
        if role == "teacher":
            t_query = _id_query(session.get("user_id"))
            teacher_batches = list(batches_col.find({"teacher_id": t_query}))
            tb_ids = [b.get("id") for b in teacher_batches if b.get("id") is not None]
            tb_alts = list(set(tb_ids + [str(x) for x in tb_ids]))
            allowed_sids = set()
            for sb in student_batches_col.find({"batch_id": {"$in": tb_alts}}):
                sid = sb.get("student_id")
                allowed_sids.add(str(sid))
                nid = _normalize_id(sid)
                if nid is not None:
                    allowed_sids.add(str(nid))

        enriched_students = []
        for s in students:
            sd = _to_dict(s)
            sid_str = str(sd.get("id"))

            if allowed_sids is not None and sid_str not in allowed_sids:
                continue

            binfo = student_batch_map.get(sid_str)
            if binfo:
                sd["batch_id"] = binfo["batch_id"]
                sd["batch_name"] = binfo["batch_name"]
                sd["enrollment_status"] = "Assigned"
            else:
                sd["enrollment_status"] = "Unassigned"
                sd["batch_name"] = sd.get("batch_name") or ""

            # Latest ML prediction — same source as teacher dashboard KPI
            pred = predictions_col.find_one(
                {"student_id": {"$in": [sd.get("id"), str(sd.get("id"))]}},
                sort=[("_id", -1)]
            )
            if pred:
                sd["risk"] = pred.get("risk", "Low")
                sd["risk_level"] = pred.get("risk", "Low")
                if pred.get("attendance") is not None:
                    sd["attendance"] = pred.get("attendance")
                if pred.get("marks") is not None:
                    sd["marks"] = pred.get("marks")
                if pred.get("lms_activity") is not None:
                    sd["lms_activity"] = pred.get("lms_activity")
                sd["confidence"] = pred.get("confidence")
                sd["predicted_at"] = pred.get("predicted_at")
            else:
                # Fallback: educational observation record
                ed = educational_records_col.find_one(
                    {"student_id": {"$in": [str(sd.get("id")), sd.get("id")]}},
                    sort=[("_id", -1)]
                )
                if ed:
                    sd["risk"] = ed.get("risk", "Low")
                    sd["risk_level"] = ed.get("risk", "Low")
                    if ed.get("attendance") is not None:
                        sd["attendance"] = ed.get("attendance")
                    if ed.get("marks") is not None:
                        sd["marks"] = ed.get("marks")
                    if ed.get("lms_activity") is not None:
                        sd["lms_activity"] = ed.get("lms_activity")
                    sd["predicted_at"] = ed.get("created_at")
                else:
                    sd["risk"] = "Low"
                    sd["risk_level"] = "Low"

            enriched_students.append(sd)

        return jsonify({"success": True, "students": enriched_students})
    return jsonify({"success": True, "students": []})


@app.route("/api/admin/overview", methods=["GET"])
@role_required(["administrator"])
def admin_overview():
    data = {
        "users": 0, "students": 0, "teachers": 0,
        "batches": 0, "assignments": 0, "marks": 0,
        "predictions": 0, "records": 0, "alerts": 0, "reports": 0
    }
    if _db_ok():
        try:
            data["users"] = users_col.count_documents({})
            data["students"] = users_col.count_documents({"role": "student"})
            data["teachers"] = users_col.count_documents({"role": "teacher"})
            data["batches"] = batches_col.count_documents({})
            data["assignments"] = assignments_col.count_documents({})
            data["marks"] = student_marks_col.count_documents({})
            data["predictions"] = predictions_col.count_documents({})
            data["records"] = educational_records_col.count_documents({})
            data["alerts"] = alerts_col.count_documents({})
            data["reports"] = reports_col.count_documents({})
        except Exception:
            pass
    return jsonify({"success": True, **data})


# ============================================================
# BATCH PARTITION MANAGEMENT (ADMIN)
# ============================================================
@app.route("/api/admin/batches", methods=["GET", "POST", "PUT", "DELETE"])
@role_required(["administrator"])
def manage_batches():
    if request.method == "GET":
        if _db_ok():
            batches = list(batches_col.find().sort("_id", -1))
            result = []
            for b in batches:
                bd = _to_dict(b)
                teacher = _find_user_by_id(bd.get("teacher_id"))
                bd["teacher_name"] = teacher["name"] if teacher else "Unassigned"
                bd["student_count"] = student_batches_col.count_documents({
                    "$or": [{"batch_id": bd["id"]}, {"batch_id": str(bd["id"])}, {"batch_id": _safe_object_id(bd["id"])}]
                })
                result.append(bd)
            return jsonify({"success": True, "batches": result})
        return jsonify({"success": True, "batches": []})

    elif request.method == "POST":
        data = request.get_json() or {}
        if _db_ok():
            batch_id = _next_id(batches_col)
            raw_tid = data.get("teacher_id") or data.get("instructor_id") or data.get("assigned_instructor") or data.get("teacher")
            
            teacher = _find_user_by_id(raw_tid)
            tid = teacher.get("id") if teacher else _normalize_id(raw_tid)

            batches_col.insert_one({
                "id": batch_id,
                "name": data.get("name") or data.get("batch_name") or data.get("cohort_name"),
                "teacher_id": tid,
                "created_at": datetime.now().isoformat()
            })
            log_event("Batch Partition Created", f"Created cohort: {data.get('name')}")
            return jsonify({"success": True, "message": "Batch cohort created!", "batch_id": batch_id})
        return jsonify({"success": True, "message": "Batch created!"})

    elif request.method == "PUT":
        data = request.get_json() or {}
        if _db_ok():
            raw_tid = data.get("teacher_id") or data.get("instructor_id") or data.get("assigned_instructor") or data.get("teacher")
            
            teacher = _find_user_by_id(raw_tid)
            tid = teacher.get("id") if teacher else _normalize_id(raw_tid)
            
            batch_id = _normalize_id(data.get("id") or data.get("_id") or data.get("batch_id"))
            name = data.get("name") or data.get("batch_name") or data.get("cohort_name")
            
            batches_col.update_one(
                {"$or": [{"id": batch_id}, {"id": str(batch_id)}, {"_id": _safe_object_id(batch_id)}]},
                {"$set": {"name": name, "teacher_id": tid}}
            )
            return jsonify({"success": True, "message": "Batch configuration updated!"})
        return jsonify({"success": True, "message": "Batch updated!"})

    elif request.method == "DELETE":
        batch_id = _normalize_id(request.args.get("id"))
        if _db_ok():
            batches_col.delete_one({"$or": [{"id": batch_id}, {"id": str(batch_id)}, {"_id": _safe_object_id(batch_id)}]})
            student_batches_col.delete_many({"$or": [{"batch_id": batch_id}, {"batch_id": str(batch_id)}, {"batch_id": _safe_object_id(batch_id)}]})
            return jsonify({"success": True, "message": "Batch partition dismantled."})
        return jsonify({"success": True, "message": "Batch deleted."})
    

@app.route("/api/admin/batch-students", methods=["POST", "DELETE"])
@role_required(["administrator"])
def manage_batch_students():
    if request.method == "POST":
        data = request.get_json() or {}
        batch_id = _normalize_id(data.get("batch_id") or data.get("id"))
        student_ids = data.get("student_ids", [])
        if not isinstance(student_ids, list):
            student_ids = [student_ids]

        if _db_ok():
            for sid in student_ids:
                sid_norm = _normalize_id(sid)
                existing = student_batches_col.find_one({
                    "$and": [
                        {"$or": [{"student_id": sid_norm}, {"student_id": str(sid_norm)}, {"student_id": _safe_object_id(sid_norm)}]},
                        {"$or": [{"batch_id": batch_id}, {"batch_id": str(batch_id)}, {"batch_id": _safe_object_id(batch_id)}]}
                    ]
                })
                if not existing:
                    student_batches_col.insert_one({"student_id": sid_norm, "batch_id": batch_id})
            return jsonify({"success": True, "message": f"{len(student_ids)} learners assigned to cohort."})
        return jsonify({"success": True, "message": "Learners assigned."})

    elif request.method == "DELETE":
        batch_id = _normalize_id(request.args.get("batch_id"))
        student_id = _normalize_id(request.args.get("student_id"))
        if _db_ok():
            student_batches_col.delete_many({
                "$and": [
                    {"$or": [{"batch_id": batch_id}, {"batch_id": str(batch_id)}, {"batch_id": _safe_object_id(batch_id)}]},
                    {"$or": [{"student_id": student_id}, {"student_id": str(student_id)}, {"student_id": _safe_object_id(student_id)}]}
                ]
            })
            return jsonify({"success": True, "message": "Learner removed from cohort."})
        return jsonify({"success": True, "message": "Learner removed."})


@app.route("/api/admin/batch-students/<int:batch_id>", methods=["GET"])
@role_required(["administrator"])
def get_batch_students(batch_id):
    if _db_ok():
        assignments = list(student_batches_col.find({
            "$or": [{"batch_id": batch_id}, {"batch_id": str(batch_id)}, {"batch_id": _safe_object_id(batch_id)}]
        }))
        student_ids = [a["student_id"] for a in assignments]
        all_ids = list(set([_normalize_id(s) for s in student_ids] + [str(s) for s in student_ids]))
        students = list(users_col.find({"$or": [{"id": {"$in": all_ids}}, {"_id": {"$in": [_safe_object_id(x) for x in all_ids if _safe_object_id(x)]}}]}))
        return jsonify({"success": True, "students": _to_dicts(students)})
    return jsonify({"success": True, "students": []})


# ============================================================
# TEACHER WORKSPACE & EVALUATIONS
# ============================================================

@app.route("/api/teacher/data", methods=["GET"])
@role_required(["teacher", "administrator"])
def get_teacher_data():
    user_id = session.get("user_id")
    role = session.get("role")

    if _db_ok():
        if role == "administrator":
            batches = list(batches_col.find())
            assignments = list(assignments_col.find().sort("_id", -1))
        else:
            teacher_query = _id_query(user_id)
            batches = list(batches_col.find({"teacher_id": teacher_query}))
            batch_ids = [b.get("id") for b in batches]
            assignments = list(assignments_col.find({"$or": [
                {"teacher_id": teacher_query},
                {"batch_id": {"$in": batch_ids + [str(x) for x in batch_ids]}}
            ]}).sort("_id", -1))

        assignment_ids = [a.get("id") for a in assignments]
        marks = list(student_marks_col.find({"assignment_id": {"$in": assignment_ids + [str(x) for x in assignment_ids]}}).sort("_id", -1))
        
        enriched_marks = []
        for m in marks:
            md = _to_dict(m)
            assignment = assignments_col.find_one({"$or": [{"id": md.get("assignment_id")}, {"id": str(md.get("assignment_id"))}, {"_id": _safe_object_id(md.get("assignment_id"))}]})
            if assignment:
                user = _find_user_by_id(md.get("student_id"))
                md["student_name"] = user["name"] if user else None
                md["student_email"] = user["email"] if user else None
                md["assignment_title"] = assignment.get("title")
                md["total_marks"] = assignment.get("total_marks")
                asg_batch = batches_col.find_one({"$or": [{"id": assignment.get("batch_id")}, {"id": str(assignment.get("batch_id"))}, {"_id": _safe_object_id(assignment.get("batch_id"))}]})
                md["batch_id"] = assignment.get("batch_id")
                md["batch_name"] = asg_batch.get("name") if asg_batch else "Unassigned"
                enriched_marks.append(md)

        enriched_batches = []
        all_student_ids = set()
        student_batch_map = {}
        for b in batches:
            bd = _to_dict(b)
            sb = list(student_batches_col.find({"$or": [{"batch_id": bd["id"]}, {"batch_id": str(bd["id"])}, {"batch_id": _safe_object_id(bd["id"])}]}))
            
            student_ids = [s["student_id"] for s in sb]
            normalized_sids = [x for x in list(set([_normalize_id(s) for s in student_ids] + [str(s) for s in student_ids])) if x is not None]
            
            for nsid in normalized_sids:
                student_batch_map[str(nsid)] = {"batch_id": bd.get("id"), "batch_name": bd.get("name")}
            
            all_student_ids.update(normalized_sids)
            students_list = list(users_col.find({"id": {"$in": normalized_sids}}))
            
            bd["student_names"] = ", ".join([s["name"] for s in students_list])
            bd["student_count"] = len(students_list)
            enriched_batches.append(bd)

        students = []
        if role == "administrator":
            raw_students = list(users_col.find({"role": "student"}, {"password": 0}))
        else:
            if all_student_ids:
                all_sid = [x for x in list(all_student_ids) if x is not None]
                raw_students = list(users_col.find({"id": {"$in": all_sid}, "role": "student"}, {"password": 0}))
            else:
                raw_students = []

        for s in raw_students:
            sd = _to_dict(s)
            sid = sd.get("id")
            pred = predictions_col.find_one({"student_id": {"$in": [str(sid), sid]}}, sort=[("_id", -1)])
            sd["risk"] = pred.get("risk") if pred else "No prediction"
            sd["confidence"] = round(pred.get("confidence", 0) * 100, 1) if pred else 0
            sd["attendance"] = pred.get("attendance") if pred else "N/A"
            sd["marks"] = pred.get("marks") if pred else "N/A"
            sd["predicted_at"] = pred.get("predicted_at") if pred else None
            binfo = student_batch_map.get(str(_normalize_id(sid))) or student_batch_map.get(str(sid))
            sd["batch_id"] = binfo.get("batch_id") if binfo else None
            sd["batch_name"] = binfo.get("batch_name") if binfo else "Unassigned"
            students.append(sd)

        enriched_assignments = []
        for a in assignments:
            ad = _to_dict(a)
            batch = batches_col.find_one({"$or": [{"id": ad.get("batch_id")}, {"id": str(ad.get("batch_id"))}, {"_id": _safe_object_id(ad.get("batch_id"))}]})
            ad["batch_name"] = batch["name"] if batch else None
            enriched_assignments.append(ad)

        return jsonify({
            "success": True,
            "assignments": enriched_assignments,
            "marks": enriched_marks,
            "batches": enriched_batches,
            "students": students
        })

    return jsonify({"success": True, "assignments": [], "marks": [], "batches": [], "students": []})


@app.route("/api/teacher/assignments", methods=["GET", "POST"])
@role_required(["teacher", "administrator"])
def teacher_assignments():
    user_id = session.get("user_id")
    role = session.get("role")

    if request.method == "GET":
        if _db_ok():
            if role == "administrator":
                assignments = list(assignments_col.find().sort("_id", -1))
            else:
                teacher_query = _id_query(user_id)
                assignments = list(assignments_col.find({"teacher_id": teacher_query}).sort("_id", -1))
                
            result = []
            for a in assignments:
                ad = _to_dict(a)
                batch = batches_col.find_one({"$or": [{"id": ad.get("batch_id")}, {"id": str(ad.get("batch_id"))}, {"_id": _safe_object_id(ad.get("batch_id"))}]})
                ad["batch_name"] = batch["name"] if batch else "Unassigned"
                result.append(ad)
            return jsonify({"success": True, "assignments": result})
        return jsonify({"success": True, "assignments": []})

    elif request.method == "POST":
        data = request.get_json() or {}
        title = data.get("title")
        subject = data.get("subject")
        batch_id = _normalize_id(data.get("batch_id"))
        total_marks = data.get("total_marks")
        due_date = data.get("due_date")
        send_email_broadcast = data.get("send_email", False)

        if _db_ok():
            assignment_id = _next_id(assignments_col)
            batch=batches_col.find_one({'id':{'$in':[batch_id,str(batch_id)]}})
            if not batch or (role != 'administrator' and str(batch.get('teacher_id')) != str(user_id)):
                return jsonify(success=False,message='Choose one of your assigned cohorts.'),403
            if not isinstance(title,str) or not title.strip() or len(title)>160:
                return jsonify(success=False,message='Enter a title up to 160 characters.'),400
            try:
                total_marks=float(total_marks)
                if not 0 < total_marks <= 1000:raise ValueError()
                datetime.fromisoformat(due_date)
            except (TypeError,ValueError):
                return jsonify(success=False,message='Enter a valid due date and total marks from 1 to 1000.'),400
            tid = batch.get('teacher_id')
            assignments_col.insert_one({
                "id": assignment_id, "batch_id": batch_id,
                "teacher_id": tid, "title": title,
                "subject": subject, "total_marks": total_marks,
                "due_date": due_date, "created_at": datetime.now().isoformat()
            })
            log_event("Assignment Created", f"Published assignment unit: {title}")

            if send_email_broadcast and batch_id:
                sb = list(student_batches_col.find({"$or": [{"batch_id": batch_id}, {"batch_id": str(batch_id)}]}))
                sids = [s["student_id"] for s in sb]
                normalized_sids = [x for x in list(set([_normalize_id(s) for s in sids] + [str(s) for s in sids])) if x is not None]
                students = list(users_col.find({"id": {"$in": normalized_sids}}))

                for s in students:
                    s_email = s.get("email")
                    if s_email:
                        subj = f"New Assignment Published: {title}"
                        body = f"Hello {s.get('name')},\n\nA new assignment '{title}' ({subject}) has been published.\nTotal Marks: {total_marks}\nDue Date: {due_date}\n\nPlease submit your deliverable through the student workspace."
                        send_email(s_email, subj, body)

            return jsonify({"success": True, "message": "Assignment created successfully!", "assignment_id": assignment_id})
        return jsonify({"success": True, "message": "Assignment created!"})


@app.route("/api/teacher/grade", methods=["POST"])
@role_required(["teacher", "administrator"])
def grade_student():
    data = request.get_json() or {}
    student_id = _normalize_id(data.get("student_id"))
    assignment_id = _normalize_id(data.get("assignment_id"))
    obtained_marks = data.get("obtained_marks")
    feedback = data.get("feedback", "")
    send_email_notif = data.get("send_email", False)

    if _db_ok():
        try:
            assignment_for_limit = assignments_col.find_one({"$or": [{"id": assignment_id}, {"id": str(assignment_id)}, {"_id": _safe_object_id(assignment_id)}]})
            if not assignment_for_limit:
                return jsonify(success=False,message='Assignment not found.'),404
            if session.get('role')=='teacher' and not owns(assignment_for_limit,'teacher_id'):
                return jsonify(success=False,message='Assignment is not assigned to you.'),403
            cohort=assignment_for_limit.get('batch_id')
            if not student_batches_col.find_one({'student_id':{'$in':[student_id,str(student_id)]},'batch_id':{'$in':[cohort,str(cohort)]}}):
                return jsonify(success=False,message='Student is not enrolled in this assignment cohort.'),403
            total_marks = int(assignment_for_limit.get("total_marks", 100)) if assignment_for_limit else 100
            try:
                marks_val = float(obtained_marks)
            except (TypeError, ValueError):
                return jsonify({"success": False, "message": "Invalid marks value."}), 400
            if not np.isfinite(marks_val) or marks_val < 0 or marks_val > total_marks:
                return jsonify({"success": False, "message": f"Marks must be between 0 and {total_marks}."}), 400
            obtained_marks = marks_val

            if session.get("role") == "teacher":
                t_query = _id_query(session.get("user_id"))
                t_batches = list(batches_col.find({"teacher_id": t_query}))
                bids = [b.get("id") for b in t_batches if b.get("id") is not None]
                bid_alts = list(set(bids + [str(x) for x in bids]))
                in_cohort = student_batches_col.find_one({
                    "$and": [
                        {"$or": [{"student_id": student_id}, {"student_id": str(student_id)}, {"student_id": _safe_object_id(student_id)}]},
                        {"$or": [{"batch_id": {"$in": bid_alts}}]}
                    ]
                })
                if not in_cohort:
                    return jsonify({"success": False, "message": "This learner is not part of your assigned cohorts."}), 403

            existing = student_marks_col.find_one({
                "$and": [
                    {"$or": [{"student_id": student_id}, {"student_id": str(student_id)}, {"student_id": _safe_object_id(student_id)}]},
                    {"$or": [{"assignment_id": assignment_id}, {"assignment_id": str(assignment_id)}, {"assignment_id": _safe_object_id(assignment_id)}]}
                ]
            })

            now_iso = datetime.now().isoformat()
            if existing:
                student_marks_col.update_one({"_id": existing["_id"]}, {"$set": {
                    "obtained_marks": int(obtained_marks),
                    "status": "graded",
                    "feedback": feedback,
                    "graded_at": now_iso
                }})
            else:
                student_marks_col.insert_one({
                    "student_id": student_id,
                    "assignment_id": assignment_id,
                    "obtained_marks": int(obtained_marks),
                    "status": "graded",
                    "feedback": feedback,
                    "graded_at": now_iso
                })

            log_event("Student Graded", f"Graded student #{student_id} for assignment #{assignment_id}")

            if send_email_notif:
                student = _find_user_by_id(student_id)
                assignment = assignments_col.find_one({"$or": [{"id": assignment_id}, {"id": str(assignment_id)}, {"_id": _safe_object_id(assignment_id)}]})
                if student and student.get("email"):
                    asg_title = assignment.get("title", "Assignment") if assignment else "Assignment"
                    total = assignment.get("total_marks", 100) if assignment else 100
                    subj = f"Grade Evaluation Published: {asg_title}"
                    body = f"Hello {student.get('name')},\n\nYour score for {asg_title} is {obtained_marks}/{total}.\nFeedback: {feedback or 'No remarks'}\n\nCheck your student portal for updated analytics."
                    send_email(student.get("email"), subj, body)

            return jsonify({"success": True, "message": "Grade recorded successfully!"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 400
    return jsonify({"success": True, "message": "Grade recorded!"})


@app.route("/api/teacher/submissions", methods=["GET"])
@role_required(["teacher", "administrator"])
def get_submissions():
    teacher_id = session.get("user_id")
    role = session.get("role")

    if _db_ok():
        try:
            if role == "administrator":
                submissions = list(student_marks_col.find({}).sort("submitted_at", -1))
            else:
                t_query = _id_query(teacher_id)
                t_batches = list(batches_col.find({"teacher_id": t_query}))
                bids = [b.get("id") for b in t_batches if b.get("id") is not None]
                bid_alts = list(set(bids + [str(x) for x in bids]))
                t_assignments = list(assignments_col.find({"$or": [{"teacher_id": t_query}, {"batch_id": {"$in": bid_alts}}]}))
                aids = [a.get("id") for a in t_assignments if a.get("id") is not None]
                aid_alts = list(set(aids + [str(x) for x in aids]))
                submissions = list(student_marks_col.find({"assignment_id": {"$in": aid_alts}}).sort("submitted_at", -1))

            result = []
            for s in submissions:
                sd = _to_dict(s)
                assignment = assignments_col.find_one({"$or": [{"id": sd.get("assignment_id")}, {"id": str(sd.get("assignment_id"))}, {"_id": _safe_object_id(sd.get("assignment_id"))}]})
                if assignment:
                    sd["assignment_title"] = assignment.get("title")
                    sd["total_marks"] = assignment.get("total_marks")
                    asg_batch = batches_col.find_one({"$or": [{"id": assignment.get("batch_id")}, {"id": str(assignment.get("batch_id"))}, {"_id": _safe_object_id(assignment.get("batch_id"))}]})
                    sd["batch_id"] = assignment.get("batch_id")
                    sd["batch_name"] = asg_batch.get("name") if asg_batch else "Unassigned"
                user = _find_user_by_id(sd.get("student_id"))
                sd["student_name"] = user["name"] if user else None
                sd["student_email"] = user["email"] if user else None
                result.append(sd)

            return jsonify({"success": True, "submissions": result})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 400
    return jsonify({"success": True, "submissions": []})


@app.route("/api/teacher/review-submission", methods=["POST"])
@role_required(["teacher", "administrator"])
def review_submission():
    data = request.get_json() or {}
    submission_id = data.get("submission_id")
    obtained_marks = data.get("obtained_marks")
    feedback = data.get("feedback", "")
    send_email_alert = data.get("send_email", False)

    if not submission_id or obtained_marks is None:
        return jsonify({"success": False, "message": "Submission ID and marks required"}), 400

    if _db_ok():
        try:
            oid = _safe_object_id(submission_id)
            query = {"$or": [{"_id": oid}, {"id": _normalize_id(submission_id)}]} if oid else {"id": _normalize_id(submission_id)}
            submission = student_marks_col.find_one(query)

            asg = None
            if submission:
                asg = assignments_col.find_one({"$or": [
                    {"id": submission.get("assignment_id")},
                    {"id": str(submission.get("assignment_id"))},
                    {"_id": _safe_object_id(submission.get("assignment_id"))}
                ]})
            total_marks = int(asg.get("total_marks", 100)) if asg else 100
            try:
                marks_val = float(obtained_marks)
            except (TypeError, ValueError):
                return jsonify({"success": False, "message": "Invalid marks value."}), 400
            if marks_val < 0 or marks_val > total_marks:
                return jsonify({"success": False, "message": f"Marks must be between 0 and {total_marks}."}), 400
            obtained_marks = marks_val

            if session.get("role") == "teacher" and submission:
                t_query = _id_query(session.get("user_id"))
                t_batches = list(batches_col.find({"teacher_id": t_query}))
                my_bids = set()
                for b in t_batches:
                    my_bids.add(str(b.get("id")))
                if asg and str(asg.get("batch_id")) not in my_bids:
                    return jsonify({"success": False, "message": "This submission is outside your assigned cohorts."}), 403

            student_marks_col.update_one(query, {"$set": {
                "obtained_marks": int(obtained_marks),
                "status": "graded",
                "feedback": feedback,
                "graded_at": datetime.now().isoformat()
            }})

            if send_email_alert and submission:
                target_email = get_target_email(user_id=submission.get("student_id"))
                if target_email:
                    subj = "EduPredict Feedback Alert: Coursework Graded"
                    body = f"Hello,\n\nYour coursework submission #{submission_id} has been evaluated.\nMarks Awarded: {obtained_marks}\nInstructor Feedback: {feedback or 'Satisfactory work'}\n\nPlease visit your portal to review updated risk indicators."
                    send_email(target_email, subj, body)

            return jsonify({"success": True, "message": "Submission graded and recorded!"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 400
    return jsonify({"success": True, "message": "Submission graded!"})


# ============================================================
# STUDENT WORKSPACE & COUNSELING ALERTS
# ============================================================

@app.route("/api/student/data", methods=["GET"])
@role_required(["student"])
def get_student_data():
    student_id = session.get("user_id")

    if _db_ok():
        try:
            s_query = _id_query(student_id)
            sb = list(student_batches_col.find({"student_id": s_query}))
            bids = list(set([s.get("batch_id") for s in sb if s.get("batch_id") is not None]))
            all_bids = list(set(bids + [int(b) for b in bids if str(b).isdigit()] + [str(b) for b in bids]))

            batches = list(batches_col.find({"id": {"$in": all_bids}}))
            enriched_batches = []
            for b in batches:
                bd = _to_dict(b)
                teacher = _find_user_by_id(bd.get("teacher_id"))
                bd["teacher_name"] = teacher["name"] if teacher else "Unassigned"
                enriched_batches.append(bd)

            assignments = list(assignments_col.find({"batch_id": {"$in": all_bids}}))
            enriched_assignments = []
            for a in assignments:
                ad = _to_dict(a)
                batch = batches_col.find_one({"$or": [{"id": ad.get("batch_id")}, {"id": str(ad.get("batch_id"))}, {"_id": _safe_object_id(ad.get("batch_id"))}]})
                ad["batch_name"] = batch["name"] if batch else None

                mark = student_marks_col.find_one({
                    "assignment_id": {"$in": [ad.get("id"), str(ad.get("id"))]},
                    "student_id": s_query
                })
                ad["obtained_marks"] = mark.get("obtained_marks") if mark else None
                ad["graded_at"] = mark.get("graded_at") if mark else None
                ad["status"] = mark.get("status") if mark else None
                ad["submission_text"] = mark.get("submission_text") if mark else None
                ad["file_path"] = mark.get("file_path") if mark else None
                ad["feedback"] = mark.get("feedback", "") if mark else ""
                ad["submitted_at"] = mark.get("submitted_at") if mark else None
                enriched_assignments.append(ad)

            predictions = list(predictions_col.find({"student_id": {"$in": [str(s) for s in s_query["$in"]]}}).sort("_id", -1).limit(10))

            return jsonify({
                "success": True,
                "batches": enriched_batches,
                "assignments": enriched_assignments,
                "predictions": _to_dicts(predictions)
            })
        except Exception as e:
            print(f"Error in student data endpoint: {e}")

    return jsonify({"success": True, "batches": [], "assignments": [], "predictions": []})


@app.route("/api/student/submit-assignment", methods=["POST"])
@role_required(["student"])
def submit_assignment():
    student_id = session.get("user_id")

    if request.is_json:
        data = request.get_json() or {}
        assignment_id = data.get("assignment_id")
        submission_text = data.get("submission_text", "")
        file_obj = None
    else:
        assignment_id = request.form.get("assignment_id")
        submission_text = request.form.get("submission_text", "")
        file_obj = request.files.get("submission_file")

    if not assignment_id:
        return jsonify({"success": False, "message": "Assignment ID required"}), 400

    if _db_ok():
        try:
            aid = _normalize_id(assignment_id)
            sid = _normalize_id(student_id)

            assignment = assignments_col.find_one({'id': {'$in': [aid, str(aid)]}})
            membership = student_batches_col.find_one({'student_id': {'$in': [sid, str(sid)]}, 'batch_id': {'$in': [assignment.get('batch_id'), str(assignment.get('batch_id'))]}}) if assignment else None
            if not assignment or not membership:
                return jsonify(success=False, message='Assignment not assigned to your cohort.'), 403
            if not submission_text.strip() and not file_obj:
                return jsonify(success=False, message='Enter a submission or attach a file.'), 400
            if len(submission_text) > 20000:
                return jsonify(success=False, message='Submission text is too long.'), 400
            file_bytes = None
            saved_file_name = None
            if file_obj and file_obj.filename != "":
                filename = secure_filename(file_obj.filename)
                saved_file_name = f"{sid}_{aid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_SUBMISSION_EXTENSIONS:
                    return jsonify(success=False, message='Unsupported attachment type.'), 400
                file_bytes = file_obj.read(MAX_FILE_BYTES + 1)
                if not file_bytes or len(file_bytes) > MAX_FILE_BYTES:
                    return jsonify(success=False, message='Use a non-empty attachment under 2 MB.'), 400

            now_iso = datetime.now().isoformat()
            existing = student_marks_col.find_one({
                "$and": [
                    {"$or": [{"student_id": sid}, {"student_id": str(sid)}, {"student_id": _safe_object_id(sid)}]},
                    {"$or": [{"assignment_id": aid}, {"assignment_id": str(aid)}, {"assignment_id": _safe_object_id(aid)}]}
                ]
            })

            payload = {
                "student_id": sid,
                "assignment_id": aid,
                "submission_text": submission_text,
                "status": "submitted",
                "submitted_at": now_iso
            }
            if saved_file_name:
                payload["file_path"] = saved_file_name
                payload["file_data"] = file_bytes

            if existing:
                if existing.get("status") == "graded":
                    return jsonify({"success": False, "message": "Coursework already evaluated and closed."}), 400
                student_marks_col.update_one({"_id": existing["_id"]}, {"$set": payload})
            else:
                payload["obtained_marks"] = None
                student_marks_col.insert_one(payload)

            log_event("Coursework Submitted", f"Student #{sid} submitted solution for assignment #{aid}")
            return jsonify({"success": True, "message": "Assignment submitted successfully!"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 400

    return jsonify({"success": True, "message": "Assignment submitted."})


@app.route("/api/student/refresh-risk", methods=["POST"])
@role_required(["student"])
def refresh_student_risk():
    student_id = session.get("user_id")
    s_query = _id_query(student_id)

    marks_records = list(student_marks_col.find({"student_id": s_query, "status": "graded"}))
    ed_rec = educational_records_col.find_one({"student_id": s_query}, sort=[("_id", -1)])

    if not marks_records and not ed_rec:
        return jsonify({
            "success": True,
            "risk": "Not Evaluated",
            "confidence": 0.0,
            "message": "Abhi tak koi academic records ya graded coursework mojood nahi hai."
        })

    if marks_records:
        total_obtained = sum(float(m.get("obtained_marks", 0)) for m in marks_records)
        total_possible = 0
        for m in marks_records:
            assign = assignments_col.find_one({"id": _normalize_id(m.get("assignment_id"))})
            total_possible += float(assign.get("total_marks", 100)) if assign else 100
        marks = (total_obtained / total_possible * 100) if total_possible > 0 else 75.0
    else:
        marks = float(ed_rec.get("marks", 70.0)) if ed_rec and ed_rec.get("marks") is not None else 70.0

    attendance = float(ed_rec.get("attendance", 80.0)) if ed_rec else 80.0
    lms = float(ed_rec.get("lms_activity", 75.0)) if ed_rec else 75.0
    prev = float(ed_rec.get("previous_performance", marks)) if ed_rec else marks

    result = predict_risk(attendance, marks, lms, prev)
    predicted_grade = "A" if marks >= 80 else ("B" if marks >= 60 else "C")

    predictions_col.insert_one({
        "student_id": str(student_id),
        "student_name": session.get("name", "Student"),
        "attendance": round(attendance, 1),
        "marks": round(marks, 1),
        "lms_activity": round(lms, 1),
        "previous_performance": round(prev, 1),
        "risk": result["risk"],
        "confidence": result.get("confidence", 0.85),
        "predicted_grade": predicted_grade,
        "predicted_at": datetime.now().isoformat()
    })

    return jsonify({"success": True, "risk": result["risk"], "confidence": result.get("confidence", 0.85)})


@app.route("/api/student/clear-predictions", methods=["POST", "DELETE"])
@role_required(["student", "administrator"])
def clear_student_predictions():
    student_id = session.get("user_id")
    s_query = _id_query(student_id)
    if _db_ok():
        try:
            all_sids = [str(s) for s in s_query.get("$in", [])] + s_query.get("$in", [])
            result = predictions_col.delete_many({"student_id": {"$in": all_sids}})
            return jsonify({
                "success": True,
                "message": f"Prediction history cleared ({result.deleted_count} records removed)."
            })
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
    return jsonify({"success": False, "message": "Database offline."}), 500


@app.route("/api/student/request-counseling", methods=["POST"])
@role_required(["student"])
def request_counseling():
    try:
        data = request.get_json() or {}
        subject = data.get("subject", "Student Counseling Request")
        message = data.get("message", "")
        student_name = session.get("name", "Student")
        student_email = session.get("user_email", "")
        student_id = session.get("user_id")

        teacher_email = None
        if _db_ok():
            sb = student_batches_col.find_one({"student_id": _id_query(student_id)})
            if sb:
                batch_id = sb.get("batch_id")
                batch = batches_col.find_one({"$or": [{"id": _normalize_id(batch_id)}, {"id": str(batch_id)}, {"_id": _safe_object_id(batch_id)}]})
                if batch and batch.get("teacher_id"):
                    teacher_email = get_target_email(user_id=batch.get("teacher_id"))

        if not teacher_email:
            teacher_email = get_target_email(role="teacher") or "Teacher.ep9@gmail.com"

        email_subject = f"[EduPredict Alert] Counseling Request from {student_name}"
        email_body = f"Student Name: {student_name}\nStudent Email: {student_email}\nSubject: {subject}\n\nMessage:\n{message}\n\nPlease check the teacher portal to review academic metrics and schedule intervention."
        
        html_body = build_html_email_template(
            title="Student Counseling Request",
            headline=f"Advisory Intervention Requested by {student_name}",
            message=message or "Student has requested counseling support.",
            metrics={"Student Name": student_name, "Student Email": student_email, "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )

        sent = send_email(teacher_email, email_subject, email_body, html_body=html_body)

        if sent:
            return jsonify({"success": True, "message": f"Advisor counseling alert successfully dispatched to instructor ({teacher_email})!"})
        else:
            return jsonify({"success": False, "message": "Failed to transmit SMTP packet to instructor."}), 500
    except Exception as e:
        print(f"Counseling route exception: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# MAPREDUCE & BIG DATA ENGINE (ANALYST STUDIO)
# ============================================================

@app.route("/api/analyst/analyze-db-students", methods=["POST"])
@role_required(["administrator", "analyst"])
def analyze_db_students():
    if not _db_ok():
        return jsonify({"success": False, "message": "Database not connected"}), 500

    try:
        students = list(users_col.find({"role": "student", "is_active": 1}))
        if not students:
            return jsonify({"success": False, "message": "No active students cataloged in database."}), 404

        analyzed_count = 0
        results = []

        for s in students:
            sid = s.get("id")
            s_name = s.get("name", "Student")
            s_query = _id_query(sid)

            marks_records = list(student_marks_col.find({"student_id": s_query, "status": "graded"}))
            if marks_records:
                total_obtained = sum(float(m.get("obtained_marks", 0)) for m in marks_records)
                total_possible = 0
                for m in marks_records:
                    assign = assignments_col.find_one({"$or": [{"id": _normalize_id(m.get("assignment_id"))}, {"_id": _safe_object_id(m.get("assignment_id"))}]})
                    total_possible += float(assign.get("total_marks", 100)) if assign else 100
                calc_marks = (total_obtained / total_possible * 100) if total_possible > 0 else 70.0
            else:
                calc_marks = 65.0

            ed_rec = educational_records_col.find_one({"student_id": s_query}, sort=[("_id", -1)])
            attendance = float(ed_rec.get("attendance", 78.0)) if ed_rec else 78.0
            lms_activity = float(ed_rec.get("lms_activity", 72.0)) if ed_rec else 72.0
            previous_perf = float(ed_rec.get("previous_performance", calc_marks)) if ed_rec else calc_marks

            pred_result = predict_risk(attendance, calc_marks, lms_activity, previous_perf)
            risk = pred_result.get("risk", "Low")
            confidence = float(pred_result.get("confidence", 0.82))
            grade = "A" if calc_marks >= 80 else ("B" if calc_marks >= 60 else "C")

            predictions_col.insert_one({
                "student_id": str(sid),
                "student_name": s_name,
                "attendance": round(attendance, 1),
                "marks": round(calc_marks, 1),
                "lms_activity": round(lms_activity, 1),
                "previous_performance": round(previous_perf, 1),
                "risk": risk,
                "predicted_grade": grade,
                "confidence": confidence,
                "predicted_at": datetime.now().isoformat(),
                "analyzed_by": session.get("user_email")
            })

            analyzed_count += 1
            results.append({
                "student_name": s_name,
                "risk": risk,
                "marks": round(calc_marks, 1),
                "attendance": round(attendance, 1),
                "grade": grade
            })

        log_event("MapReduce Batch Reduction", f"Analyzed and partitioned {analyzed_count} student risk profiles.")
        return jsonify({
            "success": True,
            "message": f"Successfully processed {analyzed_count} students using live database scores.",
            "total_analyzed": analyzed_count,
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# MAPREDUCE ANOMALY DETECTION ENGINE
# ============================================================

@app.route("/api/analytics/scan-anomalies", methods=["POST"])
@role_required(["administrator", "analyst", "teacher"])
def scan_anomalies():
    if not _db_ok():
        return jsonify({"success": False, "message": "Database offline"}), 500

    try:
        students = list(users_col.find({"role": "student"}))
        anomalies_detected = 0

        for s in students:
            sid = s.get("id")
            s_query = _id_query(sid)
            s_name = s.get("name", "Student")

            ed_rec = educational_records_col.find_one({"student_id": s_query}, sort=[("_id", -1)])
            marks_records = list(student_marks_col.find({"student_id": s_query, "status": "graded"}))

            avg_marks = 70.0
            if marks_records:
                avg_marks = sum(float(m.get("obtained_marks", 0)) for m in marks_records) / len(marks_records)
            elif ed_rec and ed_rec.get("marks") is not None:
                avg_marks = float(ed_rec.get("marks"))

            attendance = float(ed_rec.get("attendance", 85.0)) if ed_rec else 85.0

            if attendance < 75.0:
                existing_alert = alerts_col.find_one({
                    "student_id": {"$in": [sid, str(sid)]},
                    "alert_type": "Attendance Anomaly"
                })

                if not existing_alert:
                    alerts_col.insert_one({
                        "alert_type": "Attendance Anomaly",
                        "severity": "critical" if attendance < 60.0 else "high",
                        "status": "pending",
                        "message": f"Attendance dropped to {attendance}%, below institutional policy threshold (75%).",
                        "student_id": str(sid),
                        "student_name": s_name,
                        "metric_value": f"{attendance}%",
                        "created_at": datetime.now().isoformat()
                    })
                    anomalies_detected += 1
                else:
                    alerts_col.update_one(
                        {"_id": existing_alert["_id"]},
                        {"$set": {
                            "metric_value": f"{attendance}%",
                            "severity": "critical" if attendance < 60.0 else "high",
                            "message": f"Attendance dropped to {attendance}%, below institutional policy threshold (75%).",
                            "updated_at": datetime.now().isoformat()
                        }}
                    )

            if avg_marks < 45.0:
                existing_alert = alerts_col.find_one({
                    "student_id": {"$in": [sid, str(sid)]},
                    "alert_type": "Grade Degradation"
                })

                if not existing_alert:
                    alerts_col.insert_one({
                        "alert_type": "Grade Degradation",
                        "severity": "high",
                        "status": "pending",
                        "message": f"Mean coursework grade plummeted to {round(avg_marks, 1)}%. Remedial support advised.",
                        "student_id": str(sid),
                        "student_name": s_name,
                        "metric_value": f"{round(avg_marks, 1)}%",
                        "created_at": datetime.now().isoformat()
                    })
                    anomalies_detected += 1
                else:
                    alerts_col.update_one(
                        {"_id": existing_alert["_id"]},
                        {"$set": {
                            "metric_value": f"{round(avg_marks, 1)}%",
                            "message": f"Mean coursework grade plummeted to {round(avg_marks, 1)}%. Remedial support advised.",
                            "updated_at": datetime.now().isoformat()
                        }}
                    )

        return jsonify({
            "success": True,
            "message": f"Scan finished! {anomalies_detected} new anomalies detected.",
            "detected_count": anomalies_detected
        })
    except Exception as e:
        print(f"Anomaly scan error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/generate-report", methods=["POST"])
@app.route("/api/reports/generate", methods=["POST"])
@role_required(["administrator", "analyst", "teacher"])
def generate_report():
    try:
        data = request.get_json() or {}
        report_type = data.get("type", "student_risk")
        if report_type not in ("student_risk", "anomaly_summary", "course_demand"):
            return jsonify(success=False, message="Choose a supported report type."), 400
        custom_title = data.get("title", "").strip() or f"EduPredict {report_type.replace('_', ' ').title()} Report"
        notes = data.get("notes", "")
        send_email_alert = data.get("send_email", False)
        
        recipient = data.get("recipient") or get_target_email(role="analyst")
        urgency = data.get("urgency", "medium")

        output = io.StringIO()
        writer = csv.writer(output)

        if report_type == "course_demand":
            writer.writerow(['Course Domain', 'Current Enrollment', 'Projected Demand', 'Capacity Saturation', 'Status'])
            writer.writerow(['Cloud & Distributed Systems', '180', '260', '88%', 'High Demand'])
            writer.writerow(['Data Science & Machine Learning', '220', '310', '94%', 'Critical Capacity'])
            writer.writerow(['Full-Stack Web Architecture', '140', '165', '65%', 'Nominal'])
            writer.writerow(['Cybersecurity Operations', '90', '150', '82%', 'Growing Demand'])
        elif report_type == "anomaly_summary":
            writer.writerow(['Alert ID', 'Anomaly Type', 'Severity', 'Student Name', 'Diagnostic Message', 'Timestamp'])
            query={'student_id':{'$in':teacher_student_ids()}} if session.get('role')=='teacher' else {}
            alerts = list(alerts_col.find(query).sort("_id", -1).limit(50))
            for a in alerts:
                writer.writerow([
                    a.get("id") or str(a.get("_id")),
                    a.get("alert_type", "Anomaly"),
                    a.get("severity", "Medium"),
                    a.get("student_name", "Global"),
                    a.get("message", ""),
                    a.get("created_at", "")
                ])
        else:
            writer.writerow(['Student ID', 'Legal Name', 'Email', 'Assigned Cohort', 'Attendance %', 'Marks %', 'Risk Level', 'Projected Grade', 'Confidence'])
            query={"role":"student"}
            if session.get('role')=='teacher':query['id']={'$in':teacher_student_ids()}
            students = list(users_col.find(query))
            for s in students:
                sid = s.get("id")
                pred = predictions_col.find_one({"student_id": {"$in": [str(sid), sid]}}, sort=[("_id", -1)])
                writer.writerow([
                    sid,
                    s.get("name", ""),
                    s.get("email", ""),
                    s.get("batch_name", "Cohort-A"),
                    pred.get("attendance", "N/A") if pred else "N/A",
                    pred.get("marks", "N/A") if pred else "N/A",
                    pred.get("risk", "N/A") if pred else "N/A",
                    pred.get("predicted_grade", "N/A") if pred else "N/A",
                    f"{round(pred.get('confidence', 0)*100, 1)}%" if pred else "N/A"
                ])

        csv_content = safe_csv(output.getvalue())
        output.close()

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"report_{report_type}_{timestamp_str}_{secrets.token_hex(4)}.csv"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        report_id = _next_id(reports_col)
        reports_col.insert_one({
            "id": report_id,
            "name": report_filename,
            "title": custom_title,
            "type": report_type,
            "notes": notes,
            "file_path": report_path,
            "csv_data": csv_content,
            "created_by": session.get("user_id"),
            "created_by_name": session.get("name", "Analyst"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        log_event("Report Compiled", f"Compiled {report_type} report: {report_filename}")

        if send_email_alert and recipient:
            email_subject = f"[EduPredict Alert &bull; {urgency.upper()}] {custom_title}"
            plain_body = f"Hello,\n\nAn analytical assessment '{custom_title}' has been compiled by {session.get('name', 'Lead Analyst')}.\n\nExecutive Directives:\n{notes or 'Please inspect the attached CSV artifact.'}\n\nGenerated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nEduPredict Analytics Gateway"
            html_body = build_html_email_template(
                title=custom_title,
                headline=f"Urgency Level: {urgency.upper()} &bull; New Assessment Compiled",
                message=notes or "Analytical findings have been processed across cluster models. Review the attached CSV summary.",
                metrics={"Report Type": report_type.replace('_', ' ').title(), "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Analyst Origin": session.get('name', 'Lead Analyst')}
            )
            send_email(
                to_email=recipient,
                subject=email_subject,
                body=plain_body,
                html_body=html_body,
                attachment=csv_content.encode('utf-8'),
                attachment_name=report_filename
            )

        return jsonify({
            "success": True,
            "message": f"Report '{custom_title}' synthesized successfully!",
            "report_id": report_id,
            "report_filename": report_filename
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/reports", methods=["GET"])
@role_required(["administrator", "analyst", "teacher"])
def get_reports():
    if _db_ok():
        reports = list(reports_col.find(scoped_query({}, "created_by"), {"csv_data": 0}).sort("_id", -1))
        return jsonify({"success": True, "reports": _to_dicts(reports)})
    return jsonify({"success": True, "reports": []})


@app.route("/api/reports/<int:report_id>", methods=["DELETE"])
@role_required(["administrator", "analyst"])
def delete_report(report_id):
    if _db_ok():
        report = reports_col.find_one(scoped_query({"id": report_id}, "created_by"))
        if report:
            if os.path.exists(report.get("file_path", "")):
                try: os.remove(report["file_path"])
                except Exception: pass
            reports_col.delete_one({"id": report_id})
            return jsonify({"success": True, "message": "Report removed."})
    return jsonify({"success": True, "message": "Report deleted."})


@app.route("/api/reports/<report_id>/send-mail", methods=["POST"])
@role_required(["administrator", "analyst", "teacher"])
def resend_report_mail(report_id):
    data = request.get_json() or {}
    recipient = data.get("recipient") or get_target_email(role="analyst")
    note = data.get("note", "")

    report = None
    if _db_ok():
        oid = _safe_object_id(report_id)
        report = reports_col.find_one(scoped_query({"$or": [{"_id": oid}, {"id": _normalize_id(report_id)}]}, "created_by"))

    if not report:
        return jsonify({"success": False, "message": "Report artifact not found."}), 404

    try:
        csv_bytes = None
        if os.path.exists(report.get("file_path", "")):
            with open(report["file_path"], "rb") as f:
                csv_bytes = f.read()
        if csv_bytes is None and report.get("csv_data"):
            csv_bytes = report["csv_data"].encode("utf-8")

        subject = f"[EduPredict Report Alert] {report.get('title') or report.get('name')}"
        plain_body = f"Hello,\n\nPlease find attached the analytical report: {report.get('name')}.\n\nNote:\n{note or 'Forwarded via EduPredict Report Gateway.'}\n\nCompiled on: {report.get('created_at')}"
        html_body = build_html_email_template(
            title=report.get('title') or "Analytical Assessment",
            headline="Forwarded Institutional Report",
            message=note or "Please inspect the attached report artifact.",
            metrics={"Report Name": report.get('name'), "Generated At": report.get('created_at')}
        )

        sent = send_email(recipient, subject, plain_body, html_body=html_body, attachment=csv_bytes, attachment_name=report.get("name"))
        if not sent:
            return jsonify({"success": False, "message": f"Failed to transmit report to {recipient}. Please verify the recipient address and SMTP credentials."}), 500
        return jsonify({"success": True, "message": f"Report alert transmitted to {recipient}!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ALERT TRANSMISSION GATEWAYS
# ============================================================

@app.route("/api/alerts/send-email", methods=["POST"])
@role_required(["administrator", "teacher", "analyst"])
def send_alert_email_endpoint():
    data = request.get_json() or {}
    student_id = data.get("student_id")
    recipient = data.get("recipient") or get_target_email(user_id=student_id)
    note = data.get("note", "")

    student_name = "Learner"
    if student_id:
        student = _find_user_by_id(student_id)
        if student:
            student_name = student.get("name", "Learner")

    subject = f"[EduPredict Alert] Student Risk Escalation: {student_name}"
    plain_body = f"High priority academic risk alert for {student_name} (#{student_id}).\n\nDirectives:\n{note}\n\nPlease inspect the faculty/analyst workspace."
    html_body = build_html_email_template(
        title="Student Risk Escalation",
        headline=f"Intervention Flag for {student_name}",
        message=note or "Anomalous performance patterns require advisory counseling.",
        metrics={"Student": student_name, "Incident ID": student_id or "General", "Escalated By": session.get("name", "Faculty")}
    )

    sent = send_email(recipient, subject, plain_body, html_body=html_body)

    if not sent:
        return jsonify({"success": False, "message": f"Failed to transmit alert to {recipient}. Please verify the recipient address and SMTP credentials."}), 500
    return jsonify({"success": True, "message": f"Alert successfully transmitted to {recipient}!"})


@app.route("/api/alerts/broadcast-email", methods=["POST"])
@role_required(["administrator", "analyst", "teacher"])
def broadcast_alert_digest():
    data = request.get_json() or {}
    recipient = data.get("recipient") or get_target_email(role="administrator")
    notes = data.get("notes", "Critical retention intervention needed for the flagged students listed below.")
    subject = data.get("subject", "EduPredict Executive Anomaly & Risk Digest")

    high_risk_list = []
    if _db_ok():
        recent_alerts = list(alerts_col.find({
            "status": {"$ne": "acknowledged"},
            "severity": {"$in": ["critical", "high", "CRITICAL", "HIGH"]}
        }).sort("_id", -1).limit(10))

        for a in recent_alerts:
            sid = a.get("student_id")
            pred = predictions_col.find_one({"student_id": {"$in": [sid, str(sid)]}}, sort=[("_id", -1)])
            action = "Schedule parent-mentor meeting & remedial tutoring" if str(a.get("severity")).lower() == "critical" else "Review attendance & LMS engagement"
            
            high_risk_list.append({
                "name": a.get("student_name", "Student"),
                "attendance": pred.get("attendance", "N/A") if pred else "N/A",
                "marks": pred.get("marks", "N/A") if pred else "N/A",
                "lms": pred.get("lms_activity", "N/A") if pred else "N/A",
                "risk": str(a.get("severity", "HIGH")).upper() + " RISK",
                "action": action
            })

    if not high_risk_list:
        high_risk_list = [
            {"name": "Hamza Tariq", "attendance": "45", "marks": "40", "lms": "30", "risk": "HIGH RISK", "action": "Urgent academic counselor intervention required"}
        ]

    table_rows = ""
    plain_text_items = ""
    for s in high_risk_list:
        table_rows += f"""
        <tr style="border-bottom: 1px solid #E2E8F0;">
            <td style="padding: 12px 14px; font-weight: 700; color: #0F172A;">{s['name']}</td>
            <td style="padding: 12px 14px; text-align: center; color: #DC2626; font-weight: 700;">{s['attendance']}%</td>
            <td style="padding: 12px 14px; text-align: center; color: #DC2626; font-weight: 700;">{s['marks']}%</td>
            <td style="padding: 12px 14px; text-align: center; color: #475569;">{s['lms']}%</td>
            <td style="padding: 12px 14px; text-align: center;"><span style="background: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 800;">{s['risk']}</span></td>
            <td style="padding: 12px 14px; font-size: 12px; color: #334155;">{s['action']}</td>
        </tr>
        """
        plain_text_items += f"- {s['name']}: Attendance: {s['attendance']}%, Score: {s['marks']}% | Action: {s['action']}\n"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: 'Inter', -apple-system, sans-serif; background: #F1F5F9; margin: 0; padding: 24px; color: #1E293B;">
        <div style="max-width: 720px; margin: 0 auto; background: #FFFFFF; border-radius: 14px; border: 1px solid #E2E8F0; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
            <div style="background: linear-gradient(135deg, #123C36 0%, #0F766E 100%); padding: 24px 28px; color: #FFFFFF;">
                <span style="font-size: 11px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: #5EEAD4;">EduPredict Alert System &bull; MapReduce Anomaly Engine</span>
                <h2 style="margin: 8px 0 0; font-size: 22px; font-weight: 800;">Executive Student Risk &amp; Retention Digest</h2>
            </div>
            <div style="padding: 24px 28px;">
                <p style="font-size: 14px; color: #334155; line-height: 1.6; margin-top: 0;">
                    <strong>Executive Directives:</strong> {notes}
                </p>
                <div style="background: #FEF2F2; border-left: 4px solid #DC2626; padding: 12px 16px; border-radius: 6px; margin: 18px 0; font-size: 13px; color: #991B1B; font-weight: 600;">
                    Warning: The following learners have breached critical retention and performance thresholds (&lt;75% attendance or failing scores).
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px;">
                    <thead>
                        <tr style="background: #123C36; color: #FFFFFF; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
                            <th style="padding: 10px 14px; text-align: left;">Student Name</th>
                            <th style="padding: 10px 14px; text-align: center;">Attendance</th>
                            <th style="padding: 10px 14px; text-align: center;">Score</th>
                            <th style="padding: 10px 14px; text-align: center;">LMS</th>
                            <th style="padding: 10px 14px; text-align: center;">Risk Level</th>
                            <th style="padding: 10px 14px; text-align: left;">Remedial Directive</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>

                <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #F1F5F9; font-size: 12px; color: #94A3B8; text-align: center;">
                    Generated by EduPredict Big Data Analytics Pipeline &bull; Automated Institutional Escalation
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    plain_body = f"""EduPredict Executive Digest:
{notes}

Critical Risk Flags:
{plain_text_items}

Visit your EduPredict faculty portal to view full trajectory logs."""

    sent = send_email(
        to_email=recipient,
        subject=subject,
        body=plain_body,
        html_body=html_content
    )

    if not sent:
        return jsonify({"success": False, "message": f"Failed to dispatch batch digest to {recipient}. Please verify the recipient address and SMTP credentials."}), 500
    return jsonify({"success": True, "message": f"Batch risk alert email successfully dispatched to: {recipient}"})


@app.route("/api/alerts/test-email", methods=["POST"])
@role_required(["administrator"])
def test_email_endpoint():
    data = request.get_json() or {}
    target = data.get("email") or get_target_email(role="administrator")
    
    sent = send_email(target, "EduPredict SMTP Alert Test", "The automated notification daemon is operational.")
    if not sent:
        return jsonify({"success": False, "message": f"Test email could not be delivered to {target}. Please verify SMTP credentials in config.py (EMAIL_USER / EMAIL_PASSWORD)."}), 500
    return jsonify({"success": True, "message": f"Test packet routed to {target}."})


# ============================================================
# UNIVERSAL EMAIL DISPATCH GATEWAY (ALL ROLES)
# ============================================================

@app.route("/api/email/send", methods=["POST"])
@login_required
def universal_email_send():
    data = request.get_json() or {}
    recipient = (data.get("recipient") or "").strip()
    subject = (data.get("subject") or "").strip()
    message = (data.get("message") or "").strip()
    context = data.get("context") or "Email Dispatch"

    if not recipient:
        return jsonify({"success": False, "message": "Recipient email address is required."}), 400
    if not subject:
        return jsonify({"success": False, "message": "Email subject is required."}), 400
    if not message:
        return jsonify({"success": False, "message": "Email message body is required."}), 400

    sender_name = session.get("name", "Platform User")
    sender_role = session.get("role", "user")
    sender_email = session.get("user_email", "")

    body_prefix = f"Sender: {sender_name} ({sender_role})\nSender Email: {sender_email}\n\n{message}"
    html_body = build_html_email_template(
        title=subject,
        headline=f"Message from {sender_name} ({sender_role})",
        message=message,
        metrics={"From": sender_email, "Context": context, "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    )

    sent = send_email(recipient, subject, body_prefix, html_body=html_body)
    if not sent:
        return jsonify({"success": False, "message": f"Email could not be delivered to {recipient}. Please verify the recipient address and SMTP settings."}), 500

    log_event("Email Dispatch", f"{sender_email} dispatched email to {recipient}")
    return jsonify({"success": True, "message": f"Email successfully delivered to {recipient}!"})


# ============================================================
# CONTACT, PROFILE & NOTIFICATION SETTINGS
# ============================================================

@app.route("/api/contact", methods=["POST"])
def contact_endpoint():
    data = request.get_json() or {}
    name = data.get("name")
    email = data.get("email")
    category = data.get("category", "General")
    subject = data.get("subject", "Inquiry")
    message = data.get("message", "")

    log_event("Support Ticket", f"Ticket from {email} - {subject}")

    admin_mail = get_target_email(role="administrator")
    ticket_body = f"New Support Ticket Received:\nFrom: {name} <{email}>\nCategory: {category}\nSubject: {subject}\n\nMessage:\n{message}"
    send_email(admin_mail, f"[Support Desk] {subject}", ticket_body)

    return jsonify({"success": True, "message": "Inquiry submitted! Our engineering team will respond within 24 hours."})


@app.route("/api/profile", methods=["GET", "PUT"])
@login_required
def profile():
    user_id = session.get("user_id")

    if request.method == "GET":
        if _db_ok():
            try:
                user = _find_user_by_id(user_id)
                if user:
                    ud = _to_dict(user)
                    ud.pop("password", None)
                    return jsonify({"success": True, "user": ud})
            except Exception:
                pass
        return jsonify({"success": False, "message": "User not found"}), 404

    elif request.method == "PUT":
        data = request.get_json() or {}
        name = data.get("name")
        curr_pw = data.get("current_password")
        new_pw = data.get("new_password")

        if _db_ok():
            try:
                user = _find_user_by_id(user_id)
                if curr_pw and new_pw:
                    pw_hash = user.get("password")
                    if isinstance(pw_hash, str):
                        pw_hash = pw_hash.encode('utf-8')
                    if not user or not bcrypt.checkpw(curr_pw.encode('utf-8'), pw_hash):
                        return jsonify({"success": False, "message": "Current password is incorrect"}), 400
                    hashed = bcrypt.hashpw(new_pw.encode('utf-8'), bcrypt.gensalt())
                    users_col.update_one({"$or": [{"id": _normalize_id(user_id)}, {"id": str(user_id)}, {"_id": _safe_object_id(user_id)}]}, {"$set": {"password": hashed}})
                    users_col.update_one({'_id': user['_id']}, {'$inc': {'auth_version': 1}})
                    session['auth_version'] = user.get('auth_version', 0) + 1

                if name:
                    users_col.update_one({"$or": [{"id": _normalize_id(user_id)}, {"id": str(user_id)}, {"_id": _safe_object_id(user_id)}]}, {"$set": {"name": name}})
                    session["name"] = name
                return jsonify({"success": True, "message": "Profile updated!"})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)}), 400

        if name:
            session["name"] = name
        return jsonify({"success": True, "message": "Profile updated!"})


@app.route("/api/profile/notification-settings", methods=["POST"])
@login_required
def save_notification_settings():
    data = request.get_json() or {}
    user_id = session.get("user_id")
    if _db_ok():
        users_col.update_one(
            {"$or": [{"id": _normalize_id(user_id)}, {"id": str(user_id)}, {"_id": _safe_object_id(user_id)}]},
            {"$set": {"notification_preferences": data}}
        )
    return jsonify({"success": True, "message": "Alert preferences saved successfully!"})


# ============================================================
# ANALYTICS ENDPOINTS
# ============================================================

@app.route("/api/analytics/records", methods=["GET"])
@role_required(["administrator", "analyst", "teacher"])
def get_all_records():
    records = get_records()
    return jsonify({"success": True, "records": records})


@app.route("/api/analytics/anomalies", methods=["GET"])
@role_required(["administrator", "analyst", "teacher"])
def get_anomalies():
    records = get_records()
    anomalies = detect_anomalies(records)
    return jsonify({"success": True, "anomalies": anomalies})


@app.route("/api/analytics/correlations", methods=["GET"])
@role_required(["administrator", "analyst", "teacher"])
def get_correlations_data():
    correlations = get_correlations()
    return jsonify({"success": True, "correlations": correlations})


@app.route("/api/analytics/alerts", methods=["GET", "DELETE"])
@role_required(["administrator", "analyst", "teacher"])
def get_all_alerts():
    if request.method == "DELETE":
        aid = request.args.get("id")
        if aid and _db_ok():
            oid = _safe_object_id(aid)
            query = {"$or": [{"_id": oid}, {"id": _normalize_id(aid)}]} if oid else {"id": _normalize_id(aid)}
            alerts_col.update_one(query, {"$set": {"status": "acknowledged"}})
            return jsonify({"success": True, "message": "Alert acknowledged!"})
    
    if not _db_ok():
        return jsonify({"success": True, "alerts": []})

    role = session.get("role")
    query = {"status": {"$ne": "acknowledged"}}

    if role == "teacher":
        teacher_id = session.get("user_id")
        t_query = _id_query(teacher_id)
        
        batches = list(batches_col.find({"teacher_id": t_query}))
        bids = [b.get("id") for b in batches]
        
        sb = list(student_batches_col.find({"batch_id": {"$in": bids + [str(x) for x in bids]}}))
        sids = [s["student_id"] for s in sb]
        
        sids_str_int = list(set([str(x) for x in sids] + [_normalize_id(x) for x in sids]))
        
        if not sids_str_int:
            query["student_id"] = {"$in": ["no_students_found"]} 
        else:
            query["student_id"] = {"$in": sids_str_int}
            
    alerts = list(alerts_col.find(query).sort("_id", -1))
    return jsonify({"success": True, "alerts": _to_dicts(alerts)})


@app.route("/api/analytics/alerts/acknowledge-all", methods=["POST"])
@role_required(["administrator", "analyst", "teacher"])
def acknowledge_all_alerts():
    if not _db_ok():
        return jsonify({"success": False, "message": "Database offline"}), 500

    try:
        role = session.get("role")
        query = {"status": {"$ne": "acknowledged"}}

        if role == "teacher":
            teacher_id = session.get("user_id")
            t_query = _id_query(teacher_id)
            batches = list(batches_col.find({"teacher_id": t_query}))
            bids = [b.get("id") for b in batches]
            sb = list(student_batches_col.find({"batch_id": {"$in": bids + [str(x) for x in bids]}}))
            sids = [s["student_id"] for s in sb]
            sids_str_int = list(set([str(x) for x in sids] + [_normalize_id(x) for x in sids]))
            
            if sids_str_int:
                query["student_id"] = {"$in": sids_str_int}
            else:
                query["student_id"] = {"$in": ["no_students_found"]}

        total_before = alerts_col.count_documents(query)
        result = alerts_col.update_many(query, {"$set": {"status": "acknowledged"}})
        
        log_event("Alerts Acknowledged", f"User {session.get('user_email')} acknowledged {result.modified_count} alerts")
        
        return jsonify({
            "success": True,
            "message": f"Successfully acknowledged and cleared {result.modified_count} alerts.",
            "deleted_count": result.modified_count,
            "total_before": total_before
        })
    except Exception as e:
        print(f"Acknowledge all error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/analytics/system-status", methods=["GET"])
@role_required(["administrator", "analyst", "teacher"])
def get_system_status():
    status = system_status()
    return jsonify({"success": True, "status": status})


# ============================================================
# ADMIN DATA MATRIX ROUTES
# ============================================================

@app.route("/api/admin/assignments", methods=["GET", "POST", "PUT", "DELETE"])
@role_required(["administrator"])
def admin_manage_assignments():
    if request.method == "GET":
        if _db_ok():
            assignments = list(assignments_col.find().sort("_id", -1))
            result = []
            for a in assignments:
                ad = _to_dict(a)
                batch = batches_col.find_one({"$or": [{"id": ad.get("batch_id")}, {"id": str(ad.get("batch_id"))}, {"_id": _safe_object_id(ad.get("batch_id"))}]})
                ad["batch_name"] = batch["name"] if batch else "Unassigned"
                teacher = _find_user_by_id(ad.get("teacher_id"))
                ad["teacher_name"] = teacher.get("name") if teacher else "Unassigned"
                result.append(ad)
            return jsonify({"success": True, "assignments": result})
        return jsonify({"success": True, "assignments": []})
        
    elif request.method == "POST":
        data = request.get_json() or {}
        if _db_ok():
            assignment_id = _next_id(assignments_col)
            batch_id = _normalize_id(data.get("batch_id"))
            
            batch = batches_col.find_one({"$or": [{"id": batch_id}, {"id": str(batch_id)}, {"_id": _safe_object_id(batch_id)}]})
            tid = batch.get("teacher_id") if batch else session.get("user_id")
            
            assignments_col.insert_one({
                "id": assignment_id, "batch_id": batch_id,
                "teacher_id": tid, "title": data.get("title"),
                "subject": data.get("subject"), "total_marks": data.get("total_marks"),
                "due_date": data.get("due_date"), "created_at": datetime.now().isoformat()
            })
            log_event("Assignment Created", f"Admin published assignment: {data.get('title')}")
            return jsonify({"success": True, "message": "Assignment created successfully!", "assignment_id": assignment_id})
        return jsonify({"success": True, "message": "Assignment created!"})
        
    elif request.method == "PUT":
        data = request.get_json() or {}
        if _db_ok():
            assignment_id = _normalize_id(data.get("id") or data.get("_id"))
            batch_id = _normalize_id(data.get("batch_id"))
            
            update_data = {
                "title": data.get("title"),
                "subject": data.get("subject"),
                "batch_id": batch_id,
                "total_marks": data.get("total_marks"),
                "due_date": data.get("due_date")
            }
            
            batch = batches_col.find_one({"$or": [{"id": batch_id}, {"id": str(batch_id)}, {"_id": _safe_object_id(batch_id)}]})
            if batch and batch.get("teacher_id"):
                update_data["teacher_id"] = batch.get("teacher_id")

            assignments_col.update_one(
                {"$or": [{"id": assignment_id}, {"id": str(assignment_id)}, {"_id": _safe_object_id(assignment_id)}]},
                {"$set": update_data}
            )
            return jsonify({"success": True, "message": "Assignment updated successfully!"})
        return jsonify({"success": True, "message": "Assignment updated!"})
        
    elif request.method == "DELETE":
        assignment_id = request.args.get("id")
        if _db_ok():
            aid = _normalize_id(assignment_id)
            assignments_col.delete_one({"$or": [{"id": aid}, {"id": str(aid)}, {"_id": _safe_object_id(aid)}]})
            student_marks_col.delete_many({"$or": [{"assignment_id": aid}, {"assignment_id": str(aid)}]})
            return jsonify({"success": True, "message": "Assignment deleted."})
        return jsonify({"success": True, "message": "Assignment deleted."})


@app.route("/api/admin/marks", methods=["GET", "DELETE"])
@role_required(["administrator"])
def admin_manage_marks():
    if request.method == "GET":
        if _db_ok():
            submissions = list(student_marks_col.find().sort("submitted_at", -1))
            result = []
            for s in submissions:
                sd = _to_dict(s)
                assignment = assignments_col.find_one({"$or": [{"id": sd.get("assignment_id")}, {"id": str(sd.get("assignment_id"))}, {"_id": _safe_object_id(sd.get("assignment_id"))}]})
                if assignment:
                    sd["assignment_title"] = assignment.get("title")
                    sd["total_marks"] = assignment.get("total_marks")
                user = _find_user_by_id(sd.get("student_id"))
                sd["student_name"] = user.get("name") if user else "Unknown"
                result.append(sd)
            return jsonify({"success": True, "marks": result})
        return jsonify({"success": True, "marks": []})
        
    elif request.method == "DELETE":
        mark_id = request.args.get("id")
        if _db_ok():
            mid = _normalize_id(mark_id)
            student_marks_col.delete_one({"$or": [{"id": mid}, {"id": str(mid)}, {"_id": _safe_object_id(mid)}]})
            return jsonify({"success": True, "message": "Mark record purged."})
        return jsonify({"success": True, "message": "Mark deleted."})


@app.route("/api/admin/predictions", methods=["GET"])
@role_required(["administrator"])
def admin_get_predictions():
    if _db_ok():
        preds = list(predictions_col.find().sort("_id", -1))
        result = []
        for p in preds:
            pd_dict = _to_dict(p)
            conf = pd_dict.get("confidence", 0)
            pd_dict["probability"] = f"{round(float(conf) * 100, 1)}%" if conf else "N/A"
            if not pd_dict.get("predicted_grade"):
                pd_dict["predicted_grade"] = "N/A"
            result.append(pd_dict)
        return jsonify({"success": True, "predictions": result})
    return jsonify({"success": True, "predictions": []})


@app.route("/api/admin/predictions/<pred_id>", methods=["DELETE"])
@role_required(["administrator"])
def admin_delete_prediction(pred_id):
    if _db_ok():
        oid = _safe_object_id(pred_id)
        query = {"$or": [{"_id": oid}, {"id": _normalize_id(pred_id)}]} if oid else {"id": _normalize_id(pred_id)}
        predictions_col.delete_one(query)
        return jsonify({"success": True, "message": "Prediction record purged successfully."})
    return jsonify({"success": False, "message": "Database offline."}), 500


@app.route("/api/admin/records", methods=["GET"])
@role_required(["administrator"])
def admin_get_records():
    if _db_ok():
        records = list(educational_records_col.find().sort("_id", -1))
        result = []
        for r in records:
            rd = _to_dict(r)
            sid = rd.get("student_id")
            
            student = _find_user_by_id(sid)
            rd["student_name"] = student.get("name") if student else "Unknown Student"
            
            batch_assignment = student_batches_col.find_one({"student_id": {"$in": [sid, str(sid)]}})
            if batch_assignment:
                batch = batches_col.find_one({"$or": [{"id": batch_assignment.get("batch_id")}, {"id": str(batch_assignment.get("batch_id"))}, {"_id": _safe_object_id(batch_assignment.get("batch_id"))}]})
                rd["batch_name"] = batch.get("name") if batch else "Unassigned"
            else:
                rd["batch_name"] = "Unassigned"
                
            rd["gpa_index"] = f"{rd.get('marks', 'N/A')}%"
            rd["lms_score"] = f"{rd.get('lms_activity', 'N/A')}%"
            result.append(rd)
        return jsonify({"success": True, "records": result})
    return jsonify({"success": True, "records": []})


@app.route("/api/admin/records/<record_id>", methods=["DELETE"])
@role_required(["administrator"])
def admin_delete_record(record_id):
    if _db_ok():
        oid = _safe_object_id(record_id)
        query = {"$or": [{"_id": oid}, {"id": _normalize_id(record_id)}]} if oid else {"id": _normalize_id(record_id)}
        educational_records_col.delete_one(query)
        return jsonify({"success": True, "message": "Educational record purged successfully."})
    return jsonify({"success": False, "message": "Database offline."}), 500


# ============================================================
# FRONTEND TEMPLATE PAGE ROUTES
# ============================================================

@app.route("/students")
@role_required(["administrator", "analyst", "teacher"])
def students_page():
    return render_template("students.html", user=session)


@app.route("/predict")
@role_required(["administrator", "analyst", "teacher"])
def predict_page():
    return render_template("predict.html", user=session)


@app.route("/analytics")
@role_required(["administrator", "analyst", "teacher"])
def analytics_page():
    return render_template("analytics.html", user=session)


@app.route("/reports")
@role_required(["administrator", "analyst", "teacher"])
def reports_page():
    return render_template("reports.html", user=session)


@app.route("/alerts")
@role_required(["administrator", "analyst", "teacher"])
def alerts_page():
    return render_template("alerts.html", user=session)


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/features")
def features_page():
    return render_template("features.html")


@app.route("/architecture")
def architecture_page():
    return render_template("architecture.html")


@app.route("/contact")
def contact_page():
    return render_template("contact.html")


@app.route("/teacher")
@role_required(["administrator", "analyst", "teacher"])
def teacher_page():
    return render_template("teacher.html", user=session)


@app.route("/data")
@role_required(["administrator", "analyst", "teacher"])
def data_page():
    return render_template("data.html", user=session)


@app.route("/upload")
@role_required(["administrator", "analyst", "teacher"])
def upload_page():
    return render_template("upload.html", user=session)


@app.route("/predictions")
@role_required(["administrator", "analyst", "teacher"])
def predictions_page():
    return render_template("predictions.html", user=session)


@app.route("/profile")
@login_required
def profile_page():
    return render_template("profile.html", user=session)


@app.route("/support-users")
@role_required(["administrator"])
def support_users_page():
    users = []
    if _db_ok():
        try:
            users = _to_dicts(list(users_col.find()))
        except Exception:
            pass
    return render_template("support_users.html", user=session, users=users)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "database_configured": bool(os.environ.get("MONGO_URI"))})


@app.route("/static/reports/<path:filename>")
@role_required(["administrator", "analyst", "teacher"])
def protected_report_download(filename):
    report = reports_col.find_one(scoped_query({'name': filename}, 'created_by'))
    if not report:
        return jsonify(success=False, message='Report not found.'), 404
    if report.get('csv_data') is not None:
        return send_file(io.BytesIO(report['csv_data'].encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name=report['name'])
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


@app.route("/static/uploads/submissions/<path:filename>")
@role_required(["administrator", "teacher"])
def protected_submission_download(filename):
    mark = student_marks_col.find_one({'file_path': filename})
    if not mark:
        return jsonify(success=False, message='Submission not found.'), 404
    if session.get('role') != 'administrator':
        assignment = assignments_col.find_one({'id': {'$in': [mark.get('assignment_id'), str(mark.get('assignment_id'))]}})
        if not assignment or not owns(assignment, 'teacher_id'):
            return jsonify(success=False, message='Submission not found.'), 404
    if mark.get('file_data') is not None:
        return send_file(io.BytesIO(bytes(mark['file_data'])), mimetype='application/octet-stream', as_attachment=True, download_name=secure_filename(filename))
    return send_from_directory(SUBMISSIONS_UPLOAD_DIR, filename, as_attachment=True)


@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html", user_role=session.get("role", "Unknown")), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=config.DEBUG_MODE, use_reloader=False, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
