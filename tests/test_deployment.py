import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['SECRET_KEY'] = 'workspace-test-session-key-only-' + 'x' * 32
os.environ['COOKIE_SECURE'] = 'false'
os.environ['EMAIL_ENABLED'] = 'false'
os.environ['MONGO_URI'] = 'mongodb://127.0.0.1/edupredict_test'
os.environ['DATA_DIR'] = tempfile.mkdtemp(prefix='runtime-', dir=Path(__file__).resolve().parent)

import mongomock
import db
db.client = mongomock.MongoClient()
import app as application


class DeploymentChecks(unittest.TestCase):
    def setUp(self):
        db.client = mongomock.MongoClient()
        self.client = application.app.test_client()

    def test_no_predictable_accounts_created_at_startup(self):
        with application.app.app_context():
            self.assertEqual(db.users_col.count_documents({}), 0)

    def test_public_pages_and_health_render(self):
        for path in ['/', '/login', '/about', '/features', '/api/health']:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_student_cannot_read_staff_analytics_or_report_files(self):
        with application.app.app_context():
            db.users_col.insert_one({'id':1001,'role':'student','name':'Synthetic student'})
        with self.client.session_transaction() as session:
            session.update(user_id=1001, role='student', name='Synthetic student')
        for path in ['/api/analytics/records', '/api/analytics/alerts', '/api/reports',
                     '/static/reports/sample.csv', '/static/uploads/submissions/sample.txt']:
            self.assertEqual(self.client.get(path).status_code, 403, path)

    def test_email_disabled_never_connects_to_smtp(self):
        with patch('smtplib.SMTP', side_effect=AssertionError('SMTP must stay disabled')):
            self.assertFalse(application.send_email('recipient@example.org', 'Demo', 'Demo'))

    def test_analyst_can_run_model_on_synthetic_input(self):
        with application.app.app_context():
            db.users_col.insert_one({'id':1002,'role':'analyst','name':'Synthetic analyst'})
        with self.client.session_transaction() as session:
            session.update(user_id=1002, role='analyst', name='Synthetic analyst')
        response = self.client.post('/api/predict', json={
            'attendance': 80, 'marks': 75, 'lms_activity': 70, 'previous_performance': 72,
            'student_name': 'Synthetic Student', 'student_id': 'demo-100',
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertIn(response.json['risk'], ['Low', 'Medium', 'High'])


if __name__ == '__main__':
    unittest.main()
