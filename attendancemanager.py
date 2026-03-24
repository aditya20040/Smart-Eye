"""
modules/attendance_manager.py
──────────────────────────────
Handles attendance marking, emotion logging, CSV persistence, and reporting.

CSV schema  (attendance_logs/YYYY-MM-DD.csv):
  Name, First_Seen, Last_Seen, Status, Emotions
"""

import os
import csv
import datetime
from collections import defaultdict


class AttendanceManager:
    def __init__(self, log_dir: str = "attendance_logs"):
        self.log_dir   = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._today      = datetime.date.today().isoformat()
        self._log_file   = os.path.join(log_dir, f"{self._today}.csv")

        # in-memory state
        self._attendance: dict[str, dict] = {}   # name → {first_seen, last_seen, emotions:[]}
        self._load_existing()

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_existing(self):
        """Reload today's CSV so sessions survive restarts."""
        if not os.path.exists(self._log_file):
            return
        with open(self._log_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._attendance[row["Name"]] = {
                    "first_seen": row.get("First_Seen", ""),
                    "last_seen":  row.get("Last_Seen",  ""),
                    "status":     row.get("Status", "Present"),
                    "emotions":   [e for e in row.get("Emotions", "").split("|") if e],
                }

    # ── marking ──────────────────────────────────────────────────────────────

    def mark(self, name: str):
        """Mark a person as present (called every frame they are detected)."""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        if name not in self._attendance:
            self._attendance[name] = {
                "first_seen": now,
                "last_seen":  now,
                "status":     "Present",
                "emotions":   [],
            }
            print(f"✅ ATTENDANCE: {name} marked PRESENT at {now}")
        else:
            self._attendance[name]["last_seen"] = now

    def log_emotion(self, name: str, emotion: str):
        """Append an emotion reading for a person."""
        if emotion is None:
            return
        if name not in self._attendance:
            self.mark(name)
        self._attendance[name]["emotions"].append(emotion)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self):
        """Write current state to today's CSV."""
        fields = ["Name", "First_Seen", "Last_Seen", "Status",
                  "Dominant_Emotion", "Emotions"]
        with open(self._log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for name, data in self._attendance.items():
                dominant = self._dominant_emotion(data["emotions"])
                writer.writerow({
                    "Name":             name,
                    "First_Seen":       data["first_seen"],
                    "Last_Seen":        data["last_seen"],
                    "Status":           data["status"],
                    "Dominant_Emotion": dominant,
                    "Emotions":         "|".join(data["emotions"][-200:]),  # cap at 200
                })
        print(f"💾 Attendance saved → {self._log_file}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def present_count(self) -> int:
        return sum(1 for v in self._attendance.values() if v["status"] == "Present")

    def _dominant_emotion(self, emotions: list) -> str:
        if not emotions:
            return "N/A"
        counts = defaultdict(int)
        for e in emotions:
            counts[e] += 1
        return max(counts, key=counts.get)

    # ── report ────────────────────────────────────────────────────────────────

    def print_report(self):
        if not self._attendance:
            print("\n📋 No attendance recorded yet.")
            return

        print(f"\n{'='*60}")
        print(f"  📋  AttendMood Daily Report — {self._today}")
        print(f"{'='*60}")
        print(f"  {'Name':<20} {'In':<10} {'Out':<10} {'Mood'}")
        print(f"  {'-'*55}")

        for name, data in sorted(self._attendance.items()):
            dom = self._dominant_emotion(data["emotions"])
            mood_emoji = {
                "Happy": "😊", "Sad": "😞", "Angry": "😠",
                "Neutral": "😐", "Surprise": "😮", "Fear": "😨",
                "Disgust": "🤢", "N/A": "—"
            }.get(dom, "")
            print(f"  {name:<20} {data['first_seen']:<10} {data['last_seen']:<10} "
                  f"{mood_emoji} {dom}")

        print(f"{'='*60}")
        print(f"  Total present: {self.present_count()}")
        print(f"{'='*60}\n")

        # emotion breakdown
        all_emotions = defaultdict(int)
        for data in self._attendance.values():
            for e in data["emotions"]:
                all_emotions[e] += 1

        if all_emotions:
            total = sum(all_emotions.values())
            print("  📊 Classroom Emotion Distribution:")
            for emo, cnt in sorted(all_emotions.items(), key=lambda x: -x[1]):
                bar = "█" * int((cnt / total) * 30)
                print(f"     {emo:<10} {bar} {cnt} ({cnt*100//total}%)")
            print()
