import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from test_deployment import application, db
import mongomock
import email_service

class EmailChecks(unittest.TestCase):
    def setUp(self):
        db.client = mongomock.MongoClient()
        self.config = SimpleNamespace(EMAIL_ENABLED=True, EMAIL_USER='owner@allowed.test', EMAIL_PASSWORD='synthetic-private-password', EMAIL_HOST='smtp.gmail.com', EMAIL_PORT=587, ROLE_EMAILS={'administrator':'owner@allowed.test'}, EMAIL_ALLOWED_RECIPIENTS='')
        self.server = MagicMock()
        self.server.send_message.return_value = {}
        self.factory = MagicMock()
        self.factory.return_value.__enter__.return_value = self.server
    def send(self, recipient='owner@allowed.test', subject='Synthetic sample', body='Sample data only', **options):
        with application.app.test_request_context('/api/contact', method='POST'):
            return email_service.deliver(self.config, recipient, subject, body, smtp_factory=self.factory, **options)
    def test_tls_and_auth_before_send_and_real_multipart_attachment(self):
        self.assertTrue(self.send(html_body='<p>Sample only</p>',attachment=b'a,b\n1,2\n',attachment_name='sample.csv'))
        self.factory.assert_called_once_with('smtp.gmail.com',587,timeout=10)
        self.server.starttls.assert_called_once()
        self.server.login.assert_called_once_with(self.config.EMAIL_USER,self.config.EMAIL_PASSWORD)
        message=self.server.send_message.call_args.args[0]
        self.assertEqual(message.get_content_type(),'multipart/mixed')
        self.assertEqual(next(message.iter_attachments()).get_filename(),'sample.csv')
    def test_unapproved_recipient_never_connects(self):
        self.assertFalse(self.send(recipient='unapproved@elsewhere.test'));self.factory.assert_not_called()
    def test_no_header_injection_or_oversize_attachments(self):
        for kwargs in [{'recipient':'owner@allowed.test\r\nBcc:bad@elsewhere.test'},{'subject':'Hello\r\nBcc:bad@elsewhere.test'},{'attachment':b'x'*(2*1024*1024+1)}]:
            self.assertFalse(self.send(**kwargs))
        self.factory.assert_not_called()
    def test_disabled_and_missing_credentials_do_not_connect(self):
        self.config.EMAIL_ENABLED=False;self.assertFalse(self.send())
        self.config.EMAIL_ENABLED=True;self.config.EMAIL_PASSWORD='';self.assertFalse(self.send());self.factory.assert_not_called()
    def test_rejected_provider_recipient_is_not_success(self):
        self.server.send_message.return_value={'owner@allowed.test':(550,'Synthetic rejection')};self.assertFalse(self.send())
    def test_provider_outage_is_not_success_and_does_not_log_credentials(self):
        self.server.login.side_effect=RuntimeError('synthetic-private-password')
        with patch('builtins.print') as log:self.assertFalse(self.send());log.assert_not_called()
    def test_shared_database_outage_fails_closed(self):
        with patch('db.get_db',side_effect=RuntimeError('database unavailable')):self.assertFalse(self.send())
        self.factory.assert_not_called()
    def test_shared_per_actor_limit_survives_service_calls(self):
        for _ in range(5):self.assertTrue(self.send())
        self.assertFalse(self.send());self.assertEqual(self.server.send_message.call_count,5)
    def test_student_cannot_use_email_relay(self):
        client=application.app.test_client()
        with application.app.app_context():db.users_col.insert_one({'id':4242,'role':'student','name':'Synthetic student','is_active':1})
        with client.session_transaction() as session:session.update(user_id=4242,role='student',name='Synthetic student')
        response=client.post('/api/email/send',json={'recipient':'owner@allowed.test','subject':'sample','message':'sample'})
        self.assertEqual(response.status_code,403)
    def test_contact_failure_does_not_report_delivery(self):
        with patch.object(application,'send_email',return_value=False):
            response=application.app.test_client().post('/api/contact',json={'name':'Synthetic tester','email':'tester@example.test','subject':'Sample','message':'Sample data only; this is a test.'})
        self.assertEqual(response.status_code,503);self.assertFalse(response.json['success'])
    def test_contact_validation_rejects_bad_fields(self):
        with patch.object(application,'send_email') as sender:
            response=application.app.test_client().post('/api/contact',json={'name':'Synthetic tester','email':'invalid','message':'x'})
        self.assertEqual(response.status_code,400);sender.assert_not_called()
    def test_html_template_escapes_untrusted_content(self):
        rendered=application.build_html_email_template('<img src=x>','Headline','<script>bad()</script>',{'Name':'<b>unsafe</b>'})
        self.assertNotIn('<script>',rendered);self.assertNotIn('<img src=x>',rendered);self.assertIn('&lt;script&gt;',rendered)
    def test_empty_digest_does_not_fabricate_student_alerts(self):
        client=application.app.test_client()
        with application.app.app_context():db.users_col.insert_one({'id':4243,'role':'administrator','name':'Synthetic admin','is_active':1})
        with client.session_transaction() as session:session.update(user_id=4243,role='administrator',name='Synthetic admin')
        with patch.object(application,'send_email') as sender:response=client.post('/api/alerts/broadcast-email',json={})
        self.assertEqual(response.status_code,400);sender.assert_not_called()
    def test_teacher_cannot_email_outside_cohort(self):
        client=application.app.test_client()
        with application.app.app_context():db.users_col.insert_one({'id':4244,'role':'teacher','name':'Synthetic teacher','is_active':1})
        with client.session_transaction() as session:session.update(user_id=4244,role='teacher',name='Synthetic teacher')
        with patch.object(application,'send_email') as sender:response=client.post('/api/alerts/send-email',json={'student_id':9999})
        self.assertEqual(response.status_code,403);sender.assert_not_called()

if __name__=='__main__':unittest.main()
