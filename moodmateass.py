"""
modules/moodmate_assistant.py
──────────────────────────────
MoodMate AI assistant.

Generates emotion-aware responses for identified persons.
Uses OpenAI API if a key is configured; falls back to built-in
rule-based responses so the system works without an API key.
"""

import os
import threading

# optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ── Built-in fallback responses (no API key needed) ──────────────────────────
FALLBACK_RESPONSES = {
    "Happy": [
        "Great energy today! Keep that smile going.",
        "You're radiating positivity — wonderful!",
        "That happiness is contagious. Have a great session!",
    ],
    "Sad": [
        "You seem a little down. Take a deep breath — you've got this.",
        "It's okay to have tough moments. I'm here with you.",
        "Small steps forward still count. You're doing well.",
    ],
    "Angry": [
        "Take a short pause and breathe slowly. You deserve calm.",
        "Channel that energy constructively. You're capable.",
        "A few deep breaths can reset everything. Try it.",
    ],
    "Neutral": [
        "Calm and focused — a great state to learn.",
        "Steady and balanced. Ready for the session!",
        "Neutral is powerful. Let's make the most of it.",
    ],
    "Fear": [
        "You're safe here. Take it one step at a time.",
        "Courage isn't the absence of fear — it's acting anyway.",
        "I see you. Breathe in slowly, breathe out slowly.",
    ],
    "Disgust": [
        "Something bothering you? Take a moment to reset.",
        "It's okay to step back and refocus.",
        "Clear your mind — fresh starts are always possible.",
    ],
    "Surprise": [
        "Wow, something caught you off guard! Life is full of surprises.",
        "That look of surprise means you're engaged — excellent!",
        "Surprised? Stay curious — that's the best mindset.",
    ],
}

import random


class MoodMateAssistant:
    """Generates and optionally speaks mood-based responses."""

    def __init__(self, voice_enabled: bool = True):
        self.voice_enabled = voice_enabled and TTS_AVAILABLE
        self._tts_lock     = threading.Lock()

        self._openai_key = os.getenv("OPENAI_API_KEY", "")
        self._tts_engine = None

        if self.voice_enabled:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 160)
            except Exception:
                self.voice_enabled = False

    # ── public API ────────────────────────────────────────────────────────────

    def respond(self, name: str, emotion: str):
        """Generate and (optionally) speak a response for the detected emotion."""
        message = self._generate_message(name, emotion)
        print(f"\n💬 MoodMate [{name}|{emotion}]: {message}\n")

        if self.voice_enabled and self._tts_engine:
            with self._tts_lock:
                try:
                    self._tts_engine.say(message)
                    self._tts_engine.runAndWait()
                except Exception:
                    pass   # TTS is best-effort

    # ── message generation ────────────────────────────────────────────────────

    def _generate_message(self, name: str, emotion: str) -> str:
        if OPENAI_AVAILABLE and self._openai_key:
            return self._openai_response(name, emotion)
        return self._fallback_response(name, emotion)

    def _fallback_response(self, name: str, emotion: str) -> str:
        pool = FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["Neutral"])
        base = random.choice(pool)
        if name and name != "Unknown":
            return f"{name}, {base[0].lower()}{base[1:]}"
        return base

    def _openai_response(self, name: str, emotion: str) -> str:
        try:
            client = openai.OpenAI(api_key=self._openai_key)
            prompt = (
                f"You are MoodMate, an empathetic AI assistant in a classroom attendance system. "
                f"A student named '{name}' has just been detected showing the emotion '{emotion}'. "
                f"Give a short (1–2 sentence), warm, and supportive message appropriate for a "
                f"classroom setting. Be concise and encouraging."
            )
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  OpenAI error: {e}. Using fallback.")
            return self._fallback_response(name, emotion)
