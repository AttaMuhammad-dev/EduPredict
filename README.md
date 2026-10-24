# EduPredict

A Python/Flask educational analytics portfolio application with role-based dashboards, academic data workflows and synthetic prediction examples.

**Live demo:** https://edupredict-demo.vercel.app  
**Project notes:** https://atta-muhammad-portfolio.vercel.app/projects/edupredict/

## Current demonstration

Public registration creates a student account. Student profile and dashboard flows are available; staff analytics and reports require the corresponding role. No default administrator accounts are created. Email delivery is disabled. Use invented sample information only.

The prediction models use generated data, and the risk score is a heuristic. These examples do not establish real-world predictive accuracy and must not be used to determine student outcomes.

## Run locally

Use Python 3.12, create a virtual environment, then:

```sh
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set a random `SECRET_KEY` of at least 32 characters, your own dedicated MongoDB URI, and `DB_NAME=edupredict_demo`. Keep `EMAIL_ENABLED=false`. Local HTTP uses `COOKIE_SECURE=false`; hosted HTTPS uses `true`.

```sh
python app.py
```

The six included model/scaler artifacts were regenerated using the synthetic generator in `model.py` and the pinned scikit-learn version. Keep those versions aligned. New normalized CSV uploads, generated CSV reports and student attachment bytes are stored in MongoDB, so those records survive serverless restarts. Legacy local-file records still require migration or re-upload. See [RELEASE-20261024.md](RELEASE-20261024.md) for data limits and access rules.

## Deploy and verify

The Vercel configuration runs the Flask application through `api/index.py`. Store environment credentials privately in your hosting account. Free Vercel hosting is the current deployment target. The optional Render configuration is retained, but it was not deployed because the account requested card verification.

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Twenty-two isolated checks cover public/student access, staff ownership, durable uploads/reports/attachments, assigned-cohort submission and grading, revoked roles, password-session invalidation, malformed requests, cross-origin changes and synthetic predictions. Each test uses a new in-memory database. Live checks on 24 October also passed student registration, sign-in/out, profile save/read and denial of staff prediction access. Elevated staff workflows were exercised with synthetic fixtures in isolation; this does not establish real-world prediction accuracy or email delivery.

## Source and attribution

This is a fresh prepared-source repository for Atta Muhammad’s portfolio deployment. It retains the application’s source attribution without copying the old Git history or local databases, uploaded files and generated reports.

Original project: https://github.com/Jokerisstupid/EduPredict
