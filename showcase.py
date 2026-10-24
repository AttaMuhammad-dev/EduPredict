"""Public, read-only fixtures. Never connects to institutional/demo account data."""
import csv
import io
from flask import Blueprint, Response, render_template

showcase = Blueprint('showcase', __name__)
SAMPLE_ROWS = (
    ('demo_001', 92, 86, 81, 84),
    ('demo_002', 78, 71, 64, 74),
    ('demo_003', 62, 48, 41, 56),
    ('demo_004', 96, 93, 90, 91),
    ('demo_005', 70, 59, 52, 63),
    ('demo_006', 85, 79, 75, 78),
)

@showcase.get('/showcase')
def page():
    return render_template('showcase.html', rows=SAMPLE_ROWS)

@showcase.get('/showcase/sample.csv')
def sample():
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['sample_id', 'attendance', 'marks', 'lms_activity', 'previous_performance'])
    writer.writerows(SAMPLE_ROWS)
    return Response(output.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': 'attachment; filename=edupredict-synthetic-sample.csv',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
    })
