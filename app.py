import os
import io
import base64
from datetime import datetime, timedelta
from functools import wraps

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, Response, send_file, jsonify
)

from config import Config
from models import db, User, Attendance
from modules.face_utils import (
    capture_samples, generate_encoding, generate_encoding_from_frames,
    load_known_encodings, recognize_face, find_duplicate_face
)

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
    present_user_ids = {
        row[0] for row in db.session.query(Attendance.user_id)
        .filter(Attendance.date == today).distinct().all()
    }
    today_count = len(present_user_ids)
    absent_count = total_users - today_count
    recent = Attendance.query.order_by(Attendance.time_in.desc()).limit(10).all()
    return render_template(
        "dashboard.html",
        total_users=total_users,
        today_count=today_count,
        absent_count=absent_count,
        recent=[a.to_dict() for a in recent],
    )


# ---------- Registration ----------

@app.route("/register")
@login_required
def register():
    # Registration now happens entirely via the browser's camera (see
    # register.html + /api/register_user below), so this route just
    # renders the page. Works identically on localhost and on the
    # deployed site, from a laptop or a phone.
    return render_template("register.html")


def _decode_base64_image(image_data_url):
    """Shared helper: turns a 'data:image/jpeg;base64,...' string into an
    OpenCV BGR frame. Returns None if it can't be decoded."""
    if "," not in image_data_url:
        return None
    try:
        _, encoded = image_data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.route("/api/register_user", methods=["POST"])
@login_required
def register_user_api():
    """
    Browser-camera-based registration. The browser captures several photos
    (via getUserMedia + canvas, same technique as attendance) and POSTs
    them here as base64 JPEGs along with the form fields. Works from any
    device's camera, local or deployed.
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    roll_no = (data.get("roll_no") or "").strip()
    email = (data.get("email") or "").strip()
    images = data.get("images") or []

    if not name or not roll_no:
        return jsonify({"error": "Name and Roll No are required."}), 400

    if User.query.filter_by(roll_no=roll_no).first():
        return jsonify({"error": "A user with this roll number already exists."}), 400

    if not images:
        return jsonify({"error": "No face samples were captured. Try again."}), 400

    frames = []
    for image_data_url in images:
        frame = _decode_base64_image(image_data_url)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return jsonify({"error": "Could not decode any captured images."}), 400

    # Optionally save the captured samples to disk, same folder structure
    # as the old local-webcam flow, purely for record-keeping.
    save_dir = os.path.join(app.config["FACES_DIR"], roll_no)
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(save_dir, f"{roll_no}_{i}.jpg"), frame)

    encoding = generate_encoding_from_frames(frames)
    if encoding is None:
        return jsonify({
            "error": "No face was clearly detected in the captured photos. "
                     "Try again with better lighting, facing the camera directly."
        }), 400

    existing_users = User.query.all()
    duplicate = find_duplicate_face(encoding, existing_users, app.config["RECOGNITION_TOLERANCE"])
    if duplicate is not None:
        return jsonify({
            "error": f"This face is already registered as '{duplicate.name}' "
                     f"(Roll No: {duplicate.roll_no}). Each person can only be registered once."
        }), 400

    user = User(name=name, roll_no=roll_no, email=email or None, encoding=encoding)
    db.session.add(user)
    db.session.commit()

    return jsonify({"success": True, "message": f"{name} registered successfully with {len(frames)} samples."})


# ---------- Live recognition / attendance marking ----------

def gen_attendance_frames():
    """Video stream generator: reads webcam, recognizes faces, marks attendance,
    and yields annotated JPEG frames for the browser <img> tag.
    NOTE: kept for local/legacy use only — not used by the deployed site anymore."""
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
    # NOTE: this only works when the Flask server itself has a webcam attached
    # (i.e. running locally with `python app.py`). It will NOT work on Render
    # or any cloud host, since cloud servers have no physical camera.
    return Response(gen_attendance_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/recognize_frame", methods=["POST"])
@login_required
def recognize_frame():
    """
    Browser-camera-based recognition (works both locally AND on a deployed
    server like Render, since the camera runs in the USER'S browser, not
    on the server). The browser sends one JPEG frame at a time as base64;
    this route decodes it, runs recognition, marks attendance, and returns
    JSON describing any recognized faces + their box coordinates so the
    browser can draw them on screen.
    """
    data = request.get_json(silent=True) or {}
    image_data_url = data.get("image", "")

    if "," not in image_data_url:
        return jsonify({"error": "No image received"}), 400

    # Strip the "data:image/jpeg;base64," prefix and decode
    header, encoded = image_data_url.split(",", 1)
    try:
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Could not decode image"}), 400

    if frame is None:
        return jsonify({"error": "Empty frame"}), 400

    users = User.query.all()
    known_encodings, known_ids = load_known_encodings(users)
    id_to_name = {u.id: u.name for u in users}

    tolerance = app.config["RECOGNITION_TOLERANCE"]
    cooldown = timedelta(hours=app.config["ATTENDANCE_COOLDOWN_HOURS"])
    matches = recognize_face(frame, known_encodings, known_ids, tolerance)

    results = []
    for user_id, (top, right, bottom, left), confidence in matches:
        now = datetime.utcnow()
        last_time = _last_marked.get(user_id)

        if last_time is None or (now - last_time) > cooldown:
            db.session.add(Attendance(user_id=user_id))
            db.session.commit()
            _last_marked[user_id] = now
            status = "marked"
        else:
            status = "already_marked"

        results.append({
            "name": id_to_name.get(user_id, "Unknown"),
            "confidence": confidence,
            "status": status,
            "box": {"top": top, "right": right, "bottom": bottom, "left": left},
        })

    return jsonify({"faces": results})


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