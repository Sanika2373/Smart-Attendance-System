import os
import io
from datetime import datetime, timedelta
from functools import wraps

import cv2
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, Response, send_file
)

from config import Config
from models import db, User, Attendance
from modules.face_utils import capture_samples, generate_encoding, load_known_encodings, recognize_face

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.create_all()

# in-memory cooldown tracker: {user_id: last_marked_datetime}
# prevents marking the same person again within ATTENDANCE_COOLDOWN_HOURS
_last_marked = {}


# ---------- Auth helpers ----------

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == app.config["ADMIN_USERNAME"] and password == app.config["ADMIN_PASSWORD"]:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Dashboard ----------

@app.route("/")
@login_required
def dashboard():
    total_users = User.query.count()
    today = datetime.utcnow().date()
    today_count = Attendance.query.filter_by(date=today).count()
    recent = Attendance.query.order_by(Attendance.time_in.desc()).limit(10).all()
    return render_template(
        "dashboard.html",
        total_users=total_users,
        today_count=today_count,
        recent=[a.to_dict() for a in recent],
    )


# ---------- Registration ----------

@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "POST":
        name = request.form.get("name")
        roll_no = request.form.get("roll_no")
        email = request.form.get("email")

        if User.query.filter_by(roll_no=roll_no).first():
            flash("A user with this roll number already exists.", "error")
            return redirect(url_for("register"))

        save_dir = os.path.join(app.config["FACES_DIR"], roll_no)
        # opens a local webcam window on the machine running this server
        image_paths = capture_samples(roll_no, save_dir, app.config["SAMPLES_PER_USER"])

        if not image_paths:
            flash("No face samples captured. Try again with better lighting.", "error")
            return redirect(url_for("register"))

        encoding = generate_encoding(image_paths)
        if encoding is None:
            flash("Could not generate a face encoding from the captured images.", "error")
            return redirect(url_for("register"))

        user = User(name=name, roll_no=roll_no, email=email, encoding=encoding)
        db.session.add(user)
        db.session.commit()

        flash(f"{name} registered successfully with {len(image_paths)} samples.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


# ---------- Live recognition / attendance marking ----------

def gen_attendance_frames():
    """Video stream generator: reads webcam, recognizes faces, marks attendance,
    and yields annotated JPEG frames for the browser <img> tag."""
    with app.app_context():
        users = User.query.all()
        known_encodings, known_ids = load_known_encodings(users)
        id_to_name = {u.id: u.name for u in users}

    cam = cv2.VideoCapture(0)
    tolerance = app.config["RECOGNITION_TOLERANCE"]
    cooldown = timedelta(hours=app.config["ATTENDANCE_COOLDOWN_HOURS"])

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        matches = recognize_face(frame, known_encodings, known_ids, tolerance)

        for user_id, (top, right, bottom, left), confidence in matches:
            now = datetime.utcnow()
            last_time = _last_marked.get(user_id)

            if last_time is None or (now - last_time) > cooldown:
                with app.app_context():
                    db.session.add(Attendance(user_id=user_id))
                    db.session.commit()
                _last_marked[user_id] = now
                label = f"{id_to_name.get(user_id, 'Unknown')} - Marked ({confidence}%)"
                color = (0, 200, 0)
            else:
                label = f"{id_to_name.get(user_id, 'Unknown')} - Already marked"
                color = (0, 165, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cam.release()


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_attendance_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/take_attendance")
@login_required
def take_attendance():
    if User.query.count() == 0:
        flash("Register at least one user before taking attendance.", "error")
        return redirect(url_for("dashboard"))
    return render_template("attendance.html")


# ---------- Reports ----------

@app.route("/report")
@login_required
def report():
    date_filter = request.args.get("date")
    query = Attendance.query
    if date_filter:
        query = query.filter_by(date=datetime.strptime(date_filter, "%Y-%m-%d").date())
    records = query.order_by(Attendance.time_in.desc()).all()
    return render_template("report.html", records=[r.to_dict() for r in records], date_filter=date_filter)


@app.route("/report/export")
@login_required
def export_report():
    records = Attendance.query.order_by(Attendance.time_in.desc()).all()
    df = pd.DataFrame([r.to_dict() for r in records])

    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"attendance_report_{datetime.utcnow().strftime('%Y%m%d')}.csv",
    )


# ---------- Users list ----------

@app.route("/users")
@login_required
def users_list():
    users = User.query.order_by(User.registered_on.desc()).all()
    return render_template("users.html", users=[u.to_dict() for u in users])


@app.route("/users/delete/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash("User deleted.", "success")
    return redirect(url_for("users_list"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
