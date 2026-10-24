"""Bounded SMTP delivery for the public portfolio demo.

Only owner-configured recipients receive mail. Never turn a public demo into an
arbitrary-recipient relay. Limits are stored atomically in the existing database.
"""
import hashlib
import re
import ssl
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from flask import has_app_context, has_request_context, request, session
from pymongo.errors import DuplicateKeyError

def valid_address(value):
    return isinstance(value, str) and len(value) <= 254 and re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}", value) is not None

def allowed_addresses(config):
    values = [config.EMAIL_USER, *config.ROLE_EMAILS.values(), *getattr(config, 'EMAIL_ALLOWED_RECIPIENTS', '').split(',')]
    return {value.strip().lower() for value in values if valid_address(value.strip())}

def reserve_delivery():
    if not has_app_context():
        return False
    from db import get_db
    now = datetime.now(timezone.utc)
    actor = str(session.get('user_id') or request.remote_addr or 'anonymous') if has_request_context() else 'owner-verification'
    actor_hash = hashlib.sha256(actor.encode()).hexdigest()[:24]
    limits = [(f"day:{now:%Y-%m-%d}", 30), (f"actor:{actor_hash}:{now:%Y-%m-%d-%H}", 5)]
    try:
        collection = get_db()['email_delivery_limits']
        collection.create_index('expiresAt', expireAfterSeconds=0)
        for key, limit in limits:
            collection.find_one_and_update(
                {'_id': key, 'count': {'$lt': limit}},
                {'$inc': {'count': 1}, '$setOnInsert': {'expiresAt': now + timedelta(days=2)}}, upsert=True,
            )
        return True
    except DuplicateKeyError:
        return False
    except Exception:
        # Shared-limit storage failure must not silently open unlimited SMTP sending.
        return False

def deliver(config, recipient, subject, body, html_body=None, attachment=None, attachment_name=None, smtp_factory=None):
    if not config.EMAIL_ENABLED:
        return False
    if not valid_address(recipient) or recipient.lower() not in allowed_addresses(config):
        return False
    if not valid_address(config.EMAIL_USER) or not config.EMAIL_PASSWORD:
        return False
    if not isinstance(subject, str) or not 1 <= len(subject) <= 180 or '\r' in subject or '\n' in subject:
        return False
    if not isinstance(body, str) or not 1 <= len(body) <= 50_000:
        return False
    if html_body is not None and (not isinstance(html_body, str) or len(html_body) > 100_000):
        return False
    if attachment is not None:
        if isinstance(attachment, str): attachment = attachment.encode('utf-8')
        if not isinstance(attachment, bytes) or len(attachment) > 2 * 1024 * 1024:
            return False
    if not reserve_delivery():
        return False
    try:
        message = EmailMessage()
        message['From'] = f'EduPredict Alerts <{config.EMAIL_USER}>'
        message['To'] = recipient
        message['Subject'] = subject
        message.set_content(body)
        if html_body: message.add_alternative(html_body, subtype='html')
        if attachment is not None:
            name = re.sub(r'[^A-Za-z0-9._-]', '_', attachment_name or 'EduPredict_Report.csv')[:100]
            message.add_attachment(attachment, maintype='application', subtype='octet-stream', filename=name)
        factory = smtp_factory or smtplib.SMTP
        with factory(config.EMAIL_HOST, int(config.EMAIL_PORT), timeout=10) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
            refused = server.send_message(message)
        return not refused
    except Exception:
        # Never print credentials, recipient addresses or provider response bodies.
        return False
