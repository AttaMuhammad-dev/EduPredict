from urllib.parse import urlparse

from flask import current_app, g
from pymongo import MongoClient


# Global client to utilize MongoDB's built-in connection pooling.
client = None


class _DatabaseProxy:
    """Resolve the active database lazily for existing module imports."""

    def __getitem__(self, name):
        return get_db()[name]


db = _DatabaseProxy()


def get_db():
    """Establish and return the MongoDB database for the current app context."""
    global client

    if client is None:
        client = MongoClient(current_app.config["MONGO_URI"], serverSelectionTimeoutMS=5000, maxPoolSize=5)

    if "db" not in g:
        mongo_uri = current_app.config["MONGO_URI"]
        configured_name = current_app.config.get("DB_NAME")
        db_name = configured_name or urlparse(mongo_uri).path.lstrip("/")
        if not db_name:
            db_name = "edupredict"
        g.db = client[db_name]

    return g.db


def close_db(e=None):
    """Remove the database reference from the request context."""
    g.pop("db", None)


def init_app(app):
    """Register database teardown and application configuration."""
    app.teardown_appcontext(close_db)


class _CollectionProxy:
    """Resolve collections lazily so existing module imports remain compatible."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, attribute):
        return getattr(get_db()[self.name], attribute)


def _collection(name):
    return _CollectionProxy(name)


users_col = _collection("users")
educational_records_col = _collection("educational_records")
alerts_col = _collection("alerts")
system_logs_col = _collection("system_logs")
batches_col = _collection("batches")
student_batches_col = _collection("student_batches")
assignments_col = _collection("assignments")
student_marks_col = _collection("student_marks")
predictions_col = _collection("predictions")
datasets_col = _collection("datasets")
dataset_mappings_col = _collection("dataset_mappings")
reports_col = _collection("reports")