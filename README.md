# Smart Attendance System (Face Recognition)

A Flask web app that registers users' faces via webcam, then recognizes them
live and automatically marks attendance with a timestamp.

## Folder Structure
```
smart_attendance_system/
├── app.py                 # Main Flask app (routes, video streaming)
├── config.py               # App configuration
├── models.py                # Database models (User, Attendance)
├── requirements.txt         # Python dependencies
├── .env.example              # Copy to .env and edit
├── modules/
│   └── face_utils.py         # Face capture / encoding / recognition logic
├── static/
│   ├── css/style.css          # App styling
│   └── faces/                  # Saved registration face images (auto-created)
├── templates/
│   ├── base.html               # Shared layout + nav
│   ├── login.html
│   ├── dashboard.html
│   ├── register.html
│   ├── attendance.html          # Live camera feed page
│   ├── report.html
│   └── users.html
└── instance/
    └── attendance.db            # SQLite database (auto-created on first run)
```

## Setup Steps

### 1. Install system dependencies (needed for `dlib` / `face_recognition`)

**Windows:** Install "CMake" and "Visual Studio Build Tools" (C++ workload) first,
then pip installing `dlib` should work. If it still fails, install dlib via:
`pip install dlib-bin`

**Mac:** `brew install cmake`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Python packages
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac/Linux
```
Open `.env` and set your own `SECRET_KEY`, `ADMIN_USERNAME`, and `ADMIN_PASSWORD`.

### 5. Run the app
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

Default login (unless you changed `.env`): `admin` / `admin123`

## How to Use
1. **Login** with the admin credentials.
2. **Register User** → fill the form → a webcam window opens on the server
   machine → look at the camera and move your head slightly until enough
   samples are captured (default 15) → window closes automatically → user saved.
3. **Take Attendance** → opens a live camera feed in the browser. Every
   recognized face is automatically marked present (green box). A person
   already marked within the cooldown window (default 12 hours) shows an
   orange box and is not marked again.
4. **Reports** → view/filter attendance by date, export as CSV.
5. **Users** → view all registered users, delete if needed.

## Notes & Limitations (good to mention in your project report)
- The webcam must be connected to the machine running this Flask server
  (OpenCV opens the local camera directly) — this is a classic limitation
  of browser-based camera access for server-side processing, which is why
  many student/institutional deployments run this as a local kiosk app
  (e.g., a PC + camera at the entrance) rather than a public cloud website.
- No liveness/anti-spoofing check is included by default — a printed photo
  could fool it. This is listed as a natural "future scope" extension.
- Recognition accuracy depends on lighting and sample quality during
  registration — capture samples in good, even lighting.
- `RECOGNITION_TOLERANCE` in `config.py` controls strictness: lower value =
  fewer false positives but may reject valid matches in poor lighting.

## Database

Locally, this project uses **SQLite** — a single file auto-created at
`instance/attendance.db` the first time you run `python app.py`. Nothing to
install or configure; it just appears. You can open it with a free tool like
"DB Browser for SQLite" to inspect the `users` and `attendance` tables directly.

`config.py` is already written to auto-switch: if a `DATABASE_URL` environment
variable is present (as it will be on Render/Railway once you attach a Postgres
database), it uses that instead of SQLite automatically — no code change needed.

## Deployment (optional, for a hosted dashboard)

Remember: the live camera recognition needs a webcam physically connected to
whatever machine runs this server, so most deployments keep that part local
(e.g., a PC at the entrance) and only host the dashboard/reports/database in
the cloud. Steps below assume **Render** (free tier, simplest for students) —
Railway is nearly identical.

1. **Push the project to GitHub** (a plain `git init`, `git add .`, `git commit`, then push to a new repo).
2. **Create a Postgres database** on Render (or Railway/ElephantSQL/Supabase) — it will give you a `DATABASE_URL` connection string.
3. **Create a new Web Service on Render**, connect your GitHub repo. It auto-detects the `Procfile` (`web: gunicorn app:app`) — no need for the Dockerfile unless you prefer containers.
4. **Set environment variables** in Render's dashboard (not in a committed `.env` file):
   - `DATABASE_URL` → paste the Postgres connection string from step 2
   - `SECRET_KEY` → any long random string
   - `ADMIN_USERNAME`, `ADMIN_PASSWORD` → your chosen admin login
5. **Deploy.** Render builds from `requirements.txt` and runs the `Procfile` command. On first load, `db.create_all()` in `app.py` creates the tables in your new Postgres database automatically.
6. **Access the hosted dashboard** at the Render-provided URL — Reports, Users, and CSV export all work remotely. For actually taking attendance with a camera, run `python app.py` locally on the machine with the webcam (pointing its data at the same `DATABASE_URL` via your local `.env`), so both the local kiosk and the cloud dashboard share one database.

If you'd rather containerize: a `Dockerfile` is included — `docker build -t attendance . && docker run -p 5000:5000 --env-file .env attendance` works the same way, and most platforms (Render, Railway, AWS ECS) can deploy directly from it.

Keep `debug=False` in production — the code already uses `app.run(debug=True)` only under `if __name__ == "__main__"`, and gunicorn (used by the Procfile/Dockerfile) doesn't use that flag at all, so this is already handled correctly for deployment.
