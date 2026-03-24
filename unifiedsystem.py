"""
modules/unified_system.py
─────────────────────────
Core AttendMood engine.

Pipeline (per frame):
  1. Detect all faces with Haar Cascade
  2. For each face:
       a. Run face_recognition → identify person → mark attendance
       b. Run emotion CNN       → classify emotion  → log sentiment
  3. Overlay results on the live frame
  4. Optionally trigger MoodMate AI assistant on new emotions
"""

import cv2
import numpy as np
import os
import csv
import time
import datetime
import pickle
import threading

# ── optional imports (graceful degradation) ─────────────────────────────────
try:
    import face_recognition as fr
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face_recognition not installed – attendance will use name=Unknown")

try:
    from tensorflow.keras.models import load_model as keras_load
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed – emotion detection disabled")

from modules.moodmate_assistant import MoodMateAssistant
from modules.attendance_manager import AttendanceManager

# ── constants ────────────────────────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Happy':    (0,   215, 255),   # gold
    'Sad':      (255, 100,  50),   # blue-ish
    'Angry':    (0,    0,  255),   # red
    'Neutral':  (180, 180, 180),   # grey
    'Fear':     (128,   0, 128),   # purple
    'Disgust':  (0,   128,   0),   # dark green
    'Surprise': (0,   165, 255),   # orange
}

KNOWN_FACES_DIR  = "data/known_faces"
ENCODINGS_FILE   = "data/face_encodings.pkl"
EMOTION_MODEL    = "models/face_sentiment_model.h5"
ATTENDANCE_DIR   = "attendance_logs"

SNAPSHOT_DIR     = "snapshots"
RECOGNITION_CONF = 0.50          # face_recognition distance threshold
EMOTION_INTERVAL = 2.0           # seconds between MoodMate AI responses


class AttendMoodSystem:
    """Unified system: attendance (face recognition) + sentiment (emotion CNN)."""

    def __init__(self, camera_index: int = 0, voice_enabled: bool = True):
        self.camera_index  = camera_index
        self.voice_enabled = voice_enabled

        # sub-systems
        self.attendance_mgr = AttendanceManager(ATTENDANCE_DIR)
        self.moodmate       = MoodMateAssistant(voice_enabled=voice_enabled)

        # face detector (always available via OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # face recognition encodings
        self.known_encodings = []
        self.known_names     = []
        self._load_encodings()

        # emotion model
        self.emotion_model = None
        if TF_AVAILABLE:
            self._load_emotion_model()

        # state
        self._last_emotion_time: dict = {}   # name → timestamp of last AI response
        self._emotion_thread: threading.Thread | None = None

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # ── setup helpers ────────────────────────────────────────────────────────

    def _load_encodings(self):
        """Load pre-computed face encodings from disk, or build from image folder."""
        if not FACE_RECOGNITION_AVAILABLE:
            return
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names     = data["names"]
            print(f"✅ Loaded {len(self.known_names)} face encoding(s).")
        else:
            self._build_encodings()

    def _build_encodings(self):
        """Scan known_faces/ folder and encode every face image."""
        if not FACE_RECOGNITION_AVAILABLE:
            return
        print("🔄 Building face encodings from known_faces/ …")
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = fr.load_image_file(img_path)
                encs = fr.face_encodings(img)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(person_name)

        if self.known_encodings:
            os.makedirs("data", exist_ok=True)
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(
                    {"encodings": self.known_encodings, "names": self.known_names}, f
                )
            print(f"✅ Encoded {len(self.known_names)} face(s) and saved.")
        else:
            print("⚠️  No faces found in data/known_faces/. Run --mode register first.")

    def _load_emotion_model(self):
        if not os.path.exists(EMOTION_MODEL):
            print(f"⚠️  Emotion model not found at {EMOTION_MODEL}.")
            print("     Run  python main.py --mode train  to train it first.")
            return
        self.emotion_model = keras_load(EMOTION_MODEL)
        print("✅ Emotion model loaded.")

    # ── registration ─────────────────────────────────────────────────────────

    def register_person(self, name: str, num_samples: int = 30):
        """Capture face images from webcam and save to known_faces/<name>/."""
        save_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.camera_index)
        count = 0
        print(f"📸 Capturing {num_samples} images for '{name}'. Look at the camera!")

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = frame[y:y+h, x:x+w]
                path = os.path.join(save_dir, f"{name}_{count:03d}.jpg")
                cv2.imwrite(path, face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Captured {count}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

            cv2.imshow(f"Registering: {name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Saved {count} images for '{name}'. Run --mode live to start.")

        # rebuild encodings with new person
        if os.path.exists(ENCODINGS_FILE):
            os.remove(ENCODINGS_FILE)
        self._build_encodings()

    # ── core live loop ───────────────────────────────────────────────────────

    def run_live(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("❌ Cannot open camera.")
            return

        print("🎥 Camera opened. Press 'q' to quit, 's' for snapshot.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces with Haar
            face_rects = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in face_rects:
                # ── 1. Identify person ──────────────────────────────────────
                name = self._identify_person(frame, x, y, w, h)

                # ── 2. Mark attendance ──────────────────────────────────────
                self.attendance_mgr.mark(name)

                # ── 3. Detect emotion ───────────────────────────────────────
                emotion, confidence = self._detect_emotion(gray, x, y, w, h)

                # ── 4. Log sentiment ────────────────────────────────────────
                self.attendance_mgr.log_emotion(name, emotion)

                # ── 5. Trigger MoodMate (throttled, in background) ─────────
                self._maybe_trigger_moodmate(name, emotion)

                # ── 6. Overlay visuals ──────────────────────────────────────
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

                # name tag
                cv2.rectangle(display, (x, y-30), (x+w, y), color, -1)
                cv2.putText(
                    display, name,
                    (x+4, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
                )

                # emotion tag
                emo_label = f"{emotion} ({confidence*100:.0f}%)" if emotion else "No Model"
                cv2.putText(
                    display, emo_label,
                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            # HUD
            self._draw_hud(display)

            cv2.imshow("AttendMood – Attendance + Sentiment", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_path = os.path.join(
                    SNAPSHOT_DIR,
                    f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(snap_path, display)
                print(f"📸 Snapshot saved: {snap_path}")

        cap.release()
        cv2.destroyAllWindows()
        self.attendance_mgr.save()
        print("✅ Session ended. Attendance saved.")
        self.print_report()

    # ── recognition helpers ──────────────────────────────────────────────────

    def _identify_person(self, frame, x, y, w, h) -> str:
        """Return person name from face region using face_recognition library."""
        if not FACE_RECOGNITION_AVAILABLE or not self.known_encodings:
            return "Unknown"

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        loc   = [(y, x+w, y+h, x)]   # top, right, bottom, left
        encs  = fr.face_encodings(rgb, loc)

        if not encs:
            return "Unknown"

        distances = fr.face_distance(self.known_encodings, encs[0])
        best_idx  = int(np.argmin(distances))
        if distances[best_idx] < RECOGNITION_CONF:
            return self.known_names[best_idx]
        return "Unknown"

    def _detect_emotion(self, gray, x, y, w, h):
        """Return (emotion_label, confidence) for the face ROI."""
        if self.emotion_model is None:
            return None, 0.0

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds      = self.emotion_model.predict(roi, verbose=0)[0]
        label_idx  = int(np.argmax(preds))
        return EMOTION_LABELS[label_idx], float(preds[label_idx])

    # ── MoodMate integration ─────────────────────────────────────────────────

    def _maybe_trigger_moodmate(self, name: str, emotion: str):
        """Fire MoodMate AI response at most once per EMOTION_INTERVAL seconds per person."""
        if emotion is None:
            return
        now = time.time()
        last = self._last_emotion_time.get(name, 0)
        if now - last < EMOTION_INTERVAL:
            return
        self._last_emotion_time[name] = now

        # run in background so it doesn't block the video loop
        t = threading.Thread(
            target=self.moodmate.respond, args=(name, emotion), daemon=True
        )
        t.start()

    # ── HUD ─────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        present = self.attendance_mgr.present_count()

        cv2.rectangle(frame, (0, h-40), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"📅 {ts}  |  👥 Present: {present}",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame, "AttendMood",
                    (w-130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # ── report ───────────────────────────────────────────────────────────────

    def print_report(self):
        self.attendance_mgr.print_report()
