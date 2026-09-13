"""
Core face-recognition logic:
- capturing registration samples from the webcam
- generating a face encoding from saved images
- comparing a live frame against all known encodings in the database
"""

import os
import cv2
import face_recognition
import numpy as np


def capture_samples(roll_no, save_dir, num_samples=15):
    """
    Opens the webcam, detects a face each frame, and saves `num_samples`
    cropped face images to save_dir. Returns the list of saved file paths.
    Press 'q' to stop early.
    """
    os.makedirs(save_dir, exist_ok=True)
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    saved_paths = []
    count = 0

    while count < num_samples:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            file_path = os.path.join(save_dir, f"{roll_no}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            saved_paths.append(file_path)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{num_samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # only take one face per frame

        cv2.imshow("Registering Face - press q to stop early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= num_samples:
            break

    cam.release()
    cv2.destroyAllWindows()
    return saved_paths


def generate_encoding(image_paths):
    """
    Takes a list of face image paths belonging to ONE person and returns
    a single averaged 128-d encoding representing that person.
    """
    encodings = []
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue
        enc = face_recognition.face_encodings(image, face_locations)
        if enc:
            encodings.append(enc[0])

    if not encodings:
        return None

    return np.mean(encodings, axis=0)


def generate_encoding_from_frames(frames):
    """
    Same idea as generate_encoding(), but takes a list of already-decoded
    OpenCV BGR frames (in memory) instead of file paths. Used by the
    browser-camera registration flow, where images arrive from the
    client rather than from a local webcam capture.
    """
    encodings = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        if not face_locations:
            continue
        enc = face_recognition.face_encodings(rgb, face_locations)
        if enc:
            encodings.append(enc[0])

    if not encodings:
        return None

    return np.mean(encodings, axis=0)


def load_known_encodings(users):
    """
    users: list of User model objects (each has .encoding and .id)
    Returns (list_of_encodings, list_of_user_ids) for fast comparison.
    """
    known_encodings = [u.encoding for u in users]
    known_ids = [u.id for u in users]
    return known_encodings, known_ids


def recognize_face(frame, known_encodings, known_ids, tolerance=0.5):
    """
    Given a single BGR frame (from OpenCV) and the known encodings/ids,
    returns a list of (user_id, face_location, confidence) for every
    recognized face in the frame. Unrecognized faces are skipped.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for encoding, location in zip(face_encodings, face_locations):
        if not known_encodings:
            continue
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = int(np.argmin(distances))
        best_distance = distances[best_match_index]

        if best_distance <= tolerance:
            confidence = round((1 - best_distance) * 100, 1)
            results.append((known_ids[best_match_index], location, confidence))

    return results


def find_duplicate_face(new_encoding, existing_users, tolerance=0.5):
    """
    Checks a newly-captured face encoding against every already-registered
    user's encoding. Returns the matching User object if this face (or a
    very similar one) is already registered under a different name/roll_no,
    otherwise returns None. Used during registration to block accidental
    double-registration of the same person.
    """
    if not existing_users:
        return None

    known_encodings = [u.encoding for u in existing_users]
    distances = face_recognition.face_distance(known_encodings, new_encoding)
    best_match_index = int(np.argmin(distances))

    if distances[best_match_index] <= tolerance:
        return existing_users[best_match_index]

    return None