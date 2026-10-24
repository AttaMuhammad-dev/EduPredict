"""Request validation and durable, owner-scoped demo artefacts."""
import csv
import io
import math
import re
from urllib.parse import urlsplit
from flask import request, jsonify, session

MAX_FILE_BYTES = 2 * 1024 * 1024

def dataset_frame(record):
    import pandas as pd
    if record.get('csv_data') is not None:
        return pd.read_csv(io.StringIO(record['csv_data']))
    # Compatibility with a local pre-migration upload; new uploads are durable.
    return pd.read_csv(record['file_path'])

def safe_csv(value):
    rows = csv.reader(io.StringIO(value))
    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow(["'" + cell if cell.lstrip().startswith(('=', '+', '-', '@')) else cell for cell in row])
    return output.getvalue()

def owns(record, field):
    return session.get('role') == 'administrator' or str(record.get(field)) == str(session.get('user_id'))

def scoped_query(query, field):
    if session.get('role') == 'administrator':
        return query
    uid = session.get('user_id')
    return {'$and': [query, {field: {'$in': [uid, str(uid)]}}]}

def install(app):
    app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
    @app.before_request
    def validate_request():
        if request.method not in ('POST', 'PUT', 'DELETE', 'PATCH'):
            return
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify(success=False, message='Cross-origin changes are not allowed.'), 403
        if request.is_json:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify(success=False, message='Send a JSON object.'), 400
            if request.path in ('/login', '/register', '/api/profile', '/api/admin/users'):
                for key in ('email', 'name', 'password', 'current_password', 'new_password'):
                    if key in data and (not isinstance(data[key], str) or len(data[key]) > 320):
                        return jsonify(success=False, message='Invalid account fields.'), 400
        else:
            data = request.form
        if request.path in ('/register', '/login'):
            email, secret = data.get('email', ''), data.get('password', '')
            if not isinstance(email, str) or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email.strip()) or not isinstance(secret, str) or not 1 <= len(secret.encode()) <= 72:
                return jsonify(success=False, message='Enter a valid email and password (maximum 72 bytes).'), 400
            if request.path == '/register' and len(secret) < 8:
                return jsonify(success=False, message='Use a password of at least 8 characters.'), 400
        if request.path in ('/api/predict', '/api/student/refresh-risk') and request.is_json:
            for key in ('attendance', 'marks', 'lms_activity', 'previous_performance'):
                if key in data:
                    try:
                        value = float(data[key])
                        if isinstance(data[key], bool) or not math.isfinite(value) or not 0 <= value <= 100:
                            raise ValueError()
                    except (TypeError, ValueError):
                        return jsonify(success=False, message=f'{key} must be a percentage from 0 to 100.'), 400
        if request.path == '/api/profile' and data.get('new_password'):
            if not data.get('current_password') or not 8 <= len(data['new_password']) or len(data['new_password'].encode()) > 72:
                return jsonify(success=False, message='Enter your current password and a new password of 8–72 bytes.'), 400
    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        if request.path.startswith('/api/') or 'user_id' in session:
            response.headers['Cache-Control'] = 'no-store'
        if response.status_code >= 500 and request.path.startswith('/api/'):
            response.set_data(app.json.dumps({'success': False, 'message': 'The service could not finish this request. Please try again.'}))
            response.content_type = 'application/json'
        return response

def teacher_student_ids():
    from db import batches_col, student_batches_col
    uid=session.get('user_id')
    batches=list(batches_col.find({'teacher_id': {'$in': [uid, str(uid)]}}))
    ids=[value for batch in batches for value in [batch.get('id'),str(batch.get('id'))]]
    return list({value for row in student_batches_col.find({'batch_id': {'$in':ids}}) for value in [row.get('student_id'),str(row.get('student_id'))]})
