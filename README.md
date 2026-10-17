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

The six included model/scaler artifacts were regenerated using the synthetic generator in `model.py` and the pinned scikit-learn version. Keep those versions aligned. Runtime uploads and reports use temporary private folders and are not durable across instances or restarts.

## Deploy and verify

The Vercel configuration runs the Flask application through `api/index.py`. Store environment credentials privately in your hosting account. Free Vercel hosting is the current deployment target. The optional Render configuration is retained, but it was not deployed because the account requested card verification.

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Five isolated checks cover public pages, absence of predictable default accounts, student restrictions on staff analytics/reports, disabled email, and synthetic predictions. Separate live checks covered student registration, sign-in/out and profile save/read. A complete review of staff, upload, reporting and storage workflows remains pending.

## Source and attribution

This is a fresh prepared-source repository for Atta Muhammad’s portfolio deployment. It retains the application’s source attribution without copying the old Git history or local databases, uploaded files and generated reports.

Original project: https://github.com/Jokerisstupid/EduPredict
