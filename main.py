"""
AttendMood - Attendance Detection + Sentiment Analysis System
============================================================
Single entry point that runs both face recognition (attendance)
and real-time emotion detection simultaneously from one webcam feed.
"""

import argparse
import sys
from modules.unified_system import AttendMoodSystem


def main():
    parser = argparse.ArgumentParser(
        description="AttendMood: Attendance + Sentiment Analysis"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "register", "train", "report"],
        default="live",
        help=(
            "live     → Run real-time attendance + emotion detection\n"
            "register → Capture and save face images for a new person\n"
            "train    → Train/retrain the emotion model (FER2013)\n"
            "report   → Print today's attendance + mood summary"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Person name (required for --mode register)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable TTS voice responses from MoodMate assistant",
    )

    args = parser.parse_args()

    system = AttendMoodSystem(
        camera_index=args.camera,
        voice_enabled=not args.no_voice,
    )

    if args.mode == "live":
        print("\n🚀 Starting AttendMood Live System...")
        print("   ✅ Face Recognition  → Attendance Tracking")
        print("   ✅ Emotion Detection → Sentiment Analysis")
        print("   Press 'q' to quit | 's' to take snapshot\n")
        system.run_live()

    elif args.mode == "register":
        if not args.name:
            print("❌ Please provide --name <person_name> for registration.")
            sys.exit(1)
        system.register_person(args.name)

    elif args.mode == "train":
        from modules.emotion_trainer import train_emotion_model
        train_emotion_model()

    elif args.mode == "report":
        system.print_report()


if __name__ == "__main__":
    main()
