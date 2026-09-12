from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120))
    encoding = db.Column(db.PickleType, nullable=False)   # stores the 128-d face encoding array
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)

    attendances = db.relationship("Attendance", backref="user", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "roll_no": self.roll_no,
            "email": self.email,
            "registered_on": self.registered_on.strftime("%Y-%m-%d %H:%M"),
        }


class Attendance(db.Model):
    __tablename__ = "attendance"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date = db.Column(db.Date, default=lambda: datetime.utcnow().date())
    time_in = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="Present")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.user.name,
            "roll_no": self.user.roll_no,
            "date": self.date.strftime("%Y-%m-%d"),
            "time_in": self.time_in.strftime("%H:%M:%S"),
            "status": self.status,
        }
