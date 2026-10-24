import io
import unittest
from unittest.mock import patch
import bcrypt
import mongomock
from test_deployment import application, db

class ReleaseWorkflows(unittest.TestCase):
    def setUp(self):
        db.client=mongomock.MongoClient()
        self.client=application.app.test_client()
        self.password='synthetic-pass-123'
        with application.app.app_context():
            for uid,role in [(1,'student'),(2,'analyst'),(3,'student'),(4,'teacher'),(5,'analyst'),(6,'administrator'),(7,'teacher')]:
                db.users_col.insert_one({'id':uid,'role':role,'name':f'Synthetic {uid}','email':f'user-{uid}@example.invalid','is_active':1,'password':bcrypt.hashpw(self.password.encode(),bcrypt.gensalt(4))})
            db.batches_col.insert_one({'id':10,'teacher_id':4,'name':'Synthetic cohort'})
            db.student_batches_col.insert_one({'student_id':1,'batch_id':10})
            db.assignments_col.insert_one({'id':20,'batch_id':10,'teacher_id':4,'title':'Synthetic task','total_marks':100})
    def role(self,uid):
        with application.app.app_context():user=db.users_col.find_one({'id':uid})
        with self.client.session_transaction() as session:session.update(user_id=uid,role=user['role'],name=user['name'],auth_version=user.get('auth_version',0))
    def upload(self):
        csv='attendance,marks,lms_activity,previous_performance\n'+''.join(f'{50+i},{40+i},{45+i},{42+i}\n' for i in range(30))
        response=self.client.post('/api/upload',data={'file':(io.BytesIO(csv.encode()),'synthetic.csv')})
        self.assertEqual(response.status_code,200,response.json)
        return response.json['dataset_id']
    def test_upload_survives_missing_local_files_and_processes(self):
        self.role(2);identifier=self.upload()
        with application.app.app_context():
            dataset=db.datasets_col.find_one({'id':identifier})
            self.assertIn('csv_data',dataset);self.assertNotIn('file_path',dataset)
        response=self.client.post('/api/chart-data',json={'dataset_id':identifier,'x_column':'attendance','y_column':'marks'})
        self.assertEqual(response.status_code,200);self.assertEqual(len(response.json['data']['labels']),30)
        response=self.client.post('/api/process-dataset',json={'dataset_id':identifier,'x_columns':['attendance','lms_activity'],'y_column':'marks'})
        self.assertEqual(response.status_code,200,response.json);self.assertTrue(response.json['success'])
    def test_another_analyst_cannot_read_uploaded_data(self):
        self.role(2);identifier=self.upload();self.role(5)
        self.assertEqual(self.client.post('/api/chart-data',json={'dataset_id':identifier,'x_column':'attendance','y_column':'marks'}).status_code,404)
    def test_empty_or_non_csv_upload_rejected(self):
        self.role(2)
        for content,name in [(b'a,b\n','empty.csv'),(b'abc','wrong.html')]:
            self.assertEqual(self.client.post('/api/upload',data={'file':(io.BytesIO(content),name)}).status_code,400)
    def test_small_training_set_rejected(self):
        self.role(2)
        identifier=self.client.post('/api/upload',data={'file':(io.BytesIO(b'x,y\n1,2\n2,3\n'),'small.csv')}).json['id']
        self.assertEqual(self.client.post('/api/process-dataset',json={'dataset_id':identifier,'x_columns':['x'],'y_column':'y'}).status_code,400)
    def test_durable_report_download_and_owner_isolation(self):
        self.role(2)
        response=self.client.post('/api/reports/generate',json={'type':'student_risk','title':'Synthetic report'})
        self.assertEqual(response.status_code,200,response.json)
        name=response.json['report_filename']
        with patch('os.path.exists',return_value=False):
            downloaded=self.client.get('/static/reports/'+name)
        self.assertEqual(downloaded.status_code,200);self.assertIn(b'Student ID',downloaded.data)
        self.role(5)
        self.assertEqual(self.client.get('/static/reports/'+name).status_code,404)
        self.assertEqual(self.client.get('/api/reports').json['reports'],[])
    def test_unknown_report_type_rejected(self):
        self.role(2);self.assertEqual(self.client.post('/api/reports/generate',json={'type':'../../private'}).status_code,400)
    def test_assigned_student_submission_is_durable_and_scoped(self):
        self.role(1)
        response=self.client.post('/api/student/submit-assignment',data={'assignment_id':'20','submission_text':'Synthetic answer','submission_file':(io.BytesIO(b'Synthetic attachment'),'answer.txt')})
        self.assertEqual(response.status_code,200,response.json)
        with application.app.app_context():
            mark=db.student_marks_col.find_one({'student_id':1});name=mark['file_path'];self.assertEqual(bytes(mark['file_data']),b'Synthetic attachment')
        self.role(4)
        response=self.client.get('/static/uploads/submissions/'+name)
        self.assertEqual(response.status_code,200);self.assertEqual(response.data,b'Synthetic attachment')
        self.role(7);self.assertEqual(self.client.get('/static/uploads/submissions/'+name).status_code,404)
    def test_unassigned_student_cannot_submit(self):
        self.role(3);self.assertEqual(self.client.post('/api/student/submit-assignment',json={'assignment_id':20,'submission_text':'Not assigned'}).status_code,403)
    def test_bad_attachment_type_rejected(self):
        self.role(1);self.assertEqual(self.client.post('/api/student/submit-assignment',data={'assignment_id':'20','submission_file':(io.BytesIO(b'<script>'),'attack.html')}).status_code,400)
    def test_staff_role_revocation_applies_to_existing_session(self):
        self.role(6)
        with application.app.app_context():db.users_col.update_one({'id':6},{'$set':{'role':'student'}})
        self.assertEqual(self.client.get('/api/admin/users').status_code,403)
    def test_disabled_account_session_rejected(self):
        self.role(2)
        with application.app.app_context():db.users_col.update_one({'id':2},{'$set':{'is_active':0}})
        self.assertEqual(self.client.get('/api/reports').status_code,401)
    def test_password_change_invalidates_other_sessions(self):
        self.role(1);old=self.client.get_cookie('session')
        response=self.client.put('/api/profile',json={'current_password':self.password,'new_password':'synthetic-new-pass-123'})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(self.client.get('/api/profile').status_code,200)
        other=application.app.test_client();other.set_cookie('session',old.value)
        self.assertEqual(other.get('/api/profile').status_code,401)
    def test_malformed_account_and_prediction_inputs(self):
        for payload in [[],{'email':{'$ne':None},'password':'synthetic-test'},{'name':'Test','email':'bad','password':'abc'}]:
            self.assertEqual(self.client.post('/register',json=payload).status_code,400)
        self.role(2)
        for value in [-1,101,True,'bad']:
            self.assertEqual(self.client.post('/api/predict',json={'attendance':value,'marks':70,'lms_activity':60,'previous_performance':50}).status_code,400)
    def test_cross_origin_changes_denied(self):
        self.role(1)
        self.assertEqual(self.client.put('/api/profile',json={'name':'Changed'},headers={'Origin':'https://unrelated.example'}).status_code,403)
    def test_teacher_report_contains_only_assigned_students(self):
        self.role(4)
        response=self.client.post('/api/reports/generate',json={'type':'student_risk'})
        self.assertEqual(response.status_code,200,response.json)
        content=self.client.get('/static/reports/'+response.json['report_filename']).data
        self.assertIn(b'Synthetic 1',content);self.assertNotIn(b'Synthetic 3',content)
        self.assertNotIn(b'85.0%',content)
    def test_teacher_cannot_publish_in_another_teachers_cohort(self):
        self.role(7)
        response=self.client.post('/api/teacher/assignments',json={'batch_id':10,'title':'Synthetic task','total_marks':100,'due_date':'2026-11-01','teacher_id':4})
        self.assertEqual(response.status_code,403)
    def test_teacher_can_grade_only_assigned_work_and_valid_marks(self):
        self.role(7)
        payload={'student_id':1,'assignment_id':20,'obtained_marks':75}
        self.assertEqual(self.client.post('/api/teacher/grade',json=payload).status_code,403)
        self.role(4)
        self.assertEqual(self.client.post('/api/teacher/grade',json={**payload,'obtained_marks':101}).status_code,400)
        self.assertEqual(self.client.post('/api/teacher/grade',json=payload).status_code,200)

if __name__=='__main__':unittest.main()
