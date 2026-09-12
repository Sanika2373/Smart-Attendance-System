import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

    # Locally: falls back to a SQLite file at instance/attendance.db
    # In production: set DATABASE_URL (e.g. Render/Railway Postgres connection string)
    _db_url = os.environ.get("DATABASE_URL")
    if _db_url and _db_url.startswith("postgres://"):
        # SQLAlchemy needs "postgresql://", not the older "postgres://" some hosts still give you
        _db_url = _db_url.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_DATABASE_URI = _db_url or (
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "attendance.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    FACES_DIR = os.path.join(BASE_DIR, "static", "faces")
    SAMPLES_PER_USER = 15          # how many face images captured during registration
    RECOGNITION_TOLERANCE = 0.5    # lower = stricter match (0.4-0.6 is typical)
    ATTENDANCE_COOLDOWN_HOURS = 12  # prevents marking the same person twice in one session

    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
