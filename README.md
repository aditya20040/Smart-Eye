# 🎓 AttendMood
### Attendance Detection + Sentiment Analysis — One Unified System

AttendMood merges two independent projects into a single real-time pipeline:

| Source Repo | Capability | What it does here |
|---|---|---|
| **Face_Recognition** (Aishwarya-846) | Face Recognition | Identifies each person → marks attendance |
| **Moodmate** (Devika9705) | Emotion Detection | Reads facial emotion → logs sentiment |

Both run **simultaneously on the same webcam frame** — one camera feed, one window, zero overlap.

---

## 🗂️ Project Structure

```
AttendMood/
├── main.py                        ← Single entry point (all modes)
├── requirements.txt
│
├── modules/
│   ├── unified_system.py          ← Core pipeline (recognition + emotion + overlay)
│   ├── attendance_manager.py      ← CSV attendance logging & reports
│   ├── moodmate_assistant.py      ← MoodMate AI voice/text responses
│   └── emotion_trainer.py         ← FER2013 CNN trainer
│
├── data/
│   ├── known_faces/               ← One subfolder per person
│   │   ├── Alice/  (*.jpg)
│   │   └── Bob/    (*.jpg)
│   └── face_encodings.pkl         ← Auto-generated encoding cache
│
├── models/
│   └── face_sentiment_model.h5    ← Trained emotion CNN (generate with --mode train)
│
├── attendance_logs/
│   └── YYYY-MM-DD.csv             ← Daily attendance + emotion logs
│
└── snapshots/                     ← Press 's' during live mode to save frames
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
# Recommended: create a virtual environment first
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **dlib / face_recognition note:**
> On macOS/Linux: `brew install cmake` or `sudo apt install cmake build-essential`
> On Windows: install Visual Studio Build Tools before pip install face-recognition

### 2. Train the emotion model (once)

```bash
python main.py --mode train
```
Downloads FER2013 via kagglehub and trains a CNN (~30 epochs). Saved to `models/face_sentiment_model.h5`.
Needs a Kaggle account. Run `kaggle` CLI setup once: https://www.kaggle.com/docs/api

### 3. Register known people

```bash
python main.py --mode register --name Alice
python main.py --mode register --name Bob
```
Captures ~30 webcam photos per person and builds face encodings automatically.

### 4. Run the live system

```bash
python main.py --mode live
```

**Controls during live mode:**
| Key | Action |
|-----|--------|
| `q` | Quit and save attendance |
| `s` | Save snapshot |

### 5. View today's report

```bash
python main.py --mode report
```

---

## 🔧 Command Reference

```
python main.py --mode live       # Real-time attendance + emotion (default)
python main.py --mode register   # Register a new person
python main.py --mode train      # Train emotion model
python main.py --mode report     # Print today's summary

Options:
  --name NAME      Person name (required for register)
  --camera N       Camera index (default: 0)
  --no-voice       Disable TTS voice responses
```

---

## 📊 Attendance Log Format

Each session writes `attendance_logs/YYYY-MM-DD.csv`:

```
Name,First_Seen,Last_Seen,Status,Dominant_Emotion,Emotions
Alice,09:05:12,09:55:43,Present,Happy,Happy|Happy|Neutral|Happy|...
Bob,09:06:01,09:54:12,Present,Neutral,Neutral|Sad|Neutral|...
Unknown,09:10:22,09:10:22,Present,Surprise,Surprise
```

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
Haar Cascade Face Detector  ←── detects face bounding boxes
     │
     ├──► face_recognition  →  identify person  →  AttendanceManager.mark()
     │                                              AttendanceManager.log_emotion()
     │
     └──► Emotion CNN (48×48 grayscale ROI)
              │
              └──► Emotion label + confidence
                        │
                        └──► MoodMateAssistant.respond()  (throttled, background thread)
                                    │
                                    ├── print to console
                                    └── pyttsx3 TTS (optional)
```

---

## 🔑 Optional: OpenAI-Powered MoodMate

Set your OpenAI API key for richer AI-generated responses:

```bash
export OPENAI_API_KEY="sk-..."     # Linux/macOS
set OPENAI_API_KEY=sk-...          # Windows
```

Without the key, built-in rule-based responses are used (works 100% offline).

---

## 🧩 Tech Stack

| Component | Library |
|---|---|
| Face detection | OpenCV Haar Cascade |
| Face recognition / attendance | `face_recognition` (dlib) |
| Emotion classification | TensorFlow / Keras CNN (FER2013) |
| AI responses | OpenAI GPT-3.5 or built-in fallback |
| Voice output | pyttsx3 (offline TTS) |
| Attendance storage | CSV via Python `csv` module |

---

## 🔮 Future Enhancements

- [ ] Streamlit web dashboard for attendance + mood analytics
- [ ] Email/Slack alerts when attendance falls below threshold
- [ ] Mood-based music recommendations via Spotify API
- [ ] Multi-camera support
- [ ] Model fine-tuning on custom emotion datasets

---

## 📝 Credits

- **Moodmate** — emotion detection & MoodMate assistant by [Devika9705](https://github.com/Devika9705/Moodmate)
- **Face_Recognition** — attendance face recognition by [Aishwarya-846](https://github.com/Aishwarya-846/Face_Recognition)
- Merged and extended into AttendMood
