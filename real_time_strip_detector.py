import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import pyttsx3
from ultralytics import YOLO


DEFAULT_MODEL_CANDIDATES = [
    Path(r"C:\Users\HANUMAN NARAYANA\OneDrive\Apps\Desktop\runs\detect\train-2\weights\best.pt"),
    Path(r"C:\Users\HANUMAN NARAYANA\OneDrive\Apps\Desktop\runs\detect\train\weights\best.pt"),
    Path("runs/detect/train-2/weights/best.pt"),
    Path("runs/detect/train/weights/best.pt"),
    Path("yolo11n.pt"),
]

LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")


def local_now():
    return datetime.now(LOCAL_TIMEZONE)


def clean_for_speech(text):
    return (
        text.replace(" g ", " gram ")
        .replace("mg", " milligram")
        .replace("ml", " milliliter")
        .replace("  ", " ")
    )


def resolve_model_path(model_path=None):
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n".join(f"  - {path}" for path in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(f"No model found. Tried:\n{candidates}")


CSV_COLUMNS = [
    "Date",
    "Time",
    "TimeZone",
    "SessionID",
    "Medicine",
    "Dose",
    "Event",
    "ReminderNumber",
    "PillTakenStatus",
    "Message",
    "Confidence",
    "Box",
    "EvidenceFile",
]


def build_dose_message(args):
    dose = clean_for_speech(args.dose)
    medicine = clean_for_speech(args.medicine)
    return f"Please take only {dose} of {medicine} now. Do not take more than {dose}."


def init_voice_engine(args):
    if args.no_voice:
        return None

    engine = pyttsx3.init()
    engine.setProperty("rate", args.voice_rate)
    engine.setProperty("volume", args.voice_volume)

    voices = engine.getProperty("voices")
    if args.voice_id:
        engine.setProperty("voice", args.voice_id)
    elif args.voice_index is not None:
        if args.voice_index < 0 or args.voice_index >= len(voices):
            raise ValueError(f"voice-index must be between 0 and {len(voices) - 1}")
        engine.setProperty("voice", voices[args.voice_index].id)

    return engine


def speak(engine, message):
    print("VOICE:", message)
    if engine is None:
        return
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as exc:
        print(f"Voice failed: {exc}")


def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    for index, voice in enumerate(voices):
        print(f"{index}: {voice.name} | {voice.id}")


def ensure_csv_schema(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_COLUMNS:
            return
        rows = list(reader)

    backup_path = path.with_suffix(".backup.csv")
    if not backup_path.exists():
        path.replace(backup_path)
    else:
        timestamp = local_now().strftime("%Y%m%d_%H%M%S")
        path.replace(path.with_suffix(f".backup_{timestamp}.csv"))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            migrated = {column: row.get(column, "") for column in CSV_COLUMNS}
            if not migrated["TimeZone"]:
                migrated["TimeZone"] = "IST"
            old_status = row.get("Status", "")
            if old_status and not migrated["PillTakenStatus"]:
                if old_status.startswith("Waiting"):
                    migrated["PillTakenStatus"] = "Pending"
                    migrated["Message"] = old_status
                else:
                    migrated["PillTakenStatus"] = old_status
            writer.writerow(migrated)


def append_event(
    csv_path,
    session_id,
    medicine,
    dose,
    event,
    reminder_number,
    pill_taken_status,
    message,
    confidence=0.0,
    box=None,
    evidence_file="",
):
    ensure_csv_schema(csv_path)
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        now = local_now()
        writer.writerow(
            [
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                now.strftime("%Z"),
                session_id,
                medicine,
                dose,
                event,
                reminder_number,
                pill_taken_status,
                message,
                f"{confidence:.3f}",
                " ".join(str(int(v)) for v in box) if box is not None else "",
                evidence_file,
            ]
        )


def start_evidence_writer(args, session_id, frame):
    if not args.save_evidence:
        return None, ""

    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / f"dose_{session_id}.mp4"

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(evidence_path), fourcc, args.evidence_fps, (width, height))
    if not writer.isOpened():
        print(f"Could not start evidence recording: {evidence_path}")
        return None, ""

    print(f"Recording dose evidence video: {evidence_path}")
    return writer, str(evidence_path)


def stop_evidence_writer(writer):
    if writer is not None:
        writer.release()


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    return cap


def draw_status(frame, detected, confidence, visible_seconds, state, reminders_sent, args):
    if detected:
        label = f"MedicineStrip: {confidence:.2f} | visible {visible_seconds:.1f}s"
        color = (0, 180, 0)
    else:
        label = "No MedicineStrip detected"
        color = (0, 0, 220)

    state_label = f"State: {state} | reminders: {reminders_sent}/{args.max_reminders}"

    cv2.rectangle(frame, (10, 10), (700, 92), (0, 0, 0), -1)
    cv2.putText(frame, label, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, state_label, (22, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

    if state == "taken":
        cv2.putText(
            frame,
            "TAKEN LOGGED",
            (22, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 0),
            2,
        )
    elif state == "waiting_for_taken":
        cv2.putText(
            frame,
            "Press t when tablet is taken",
            (22, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def run_detector(args):
    model_path = resolve_model_path(args.model)
    model = YOLO(str(model_path))
    engine = init_voice_engine(args)
    dose_message = build_dose_message(args)

    print(f"Loaded model: {model_path}")
    print("Show the tablet strip to the webcam. Press q or Esc to quit.")
    print(f"Dose message: {dose_message}")
    session_id = local_now().strftime("%Y%m%d_%H%M%S")

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 1 if you have another camera."
        )

    state = "idle"
    visible_since = None
    missing_since = None
    dose_spoken_at = 0
    last_reminder_at = 0
    reminders_sent = 0
    best_session_conf = 0.0
    best_session_box = None
    evidence_writer = None
    evidence_file = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]

        best_conf = 0.0
        best_box = None
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        detected = best_box is not None
        now = time.time()
        if detected:
            if visible_since is None:
                visible_since = now
            missing_since = None

            x1, y1, x2, y2 = [int(v) for v in best_box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"MedicineStrip {best_conf:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 0),
                2,
            )

            visible_seconds = now - visible_since
            if best_conf > best_session_conf:
                best_session_conf = best_conf
                best_session_box = best_box

            if state == "idle" and visible_seconds >= args.confirm_seconds:
                message = dose_message
                speak(engine, message)
                evidence_writer, evidence_file = start_evidence_writer(args, session_id, frame)
                append_event(
                    args.csv,
                    session_id,
                    args.medicine,
                    args.dose,
                    "dose_instruction",
                    0,
                    "Pending",
                    message,
                    best_session_conf,
                    best_session_box,
                    evidence_file,
                )
                state = "waiting_for_taken"
                dose_spoken_at = now
                last_reminder_at = now
        else:
            visible_since = None
            visible_seconds = 0.0
            if state == "waiting_for_taken":
                if missing_since is None:
                    missing_since = now
                missing_seconds = now - missing_since
                if missing_seconds >= args.taken_absence_seconds:
                    state = "taken"
                    stop_evidence_writer(evidence_writer)
                    evidence_writer = None
                    speak(engine, f"Thank you. {args.medicine} dose has been recorded.")
                    append_event(
                        args.csv,
                        session_id,
                        args.medicine,
                        args.dose,
                        "taken_confirmed",
                        reminders_sent,
                        "Taken",
                        "Strip left camera view after dose instruction.",
                        best_session_conf,
                        best_session_box,
                        evidence_file,
                    )
                    print("Taken confirmed because the detected strip left the camera view.")

        if state == "waiting_for_taken" and now - last_reminder_at >= args.reminder_interval:
            reminders_sent += 1
            if reminders_sent <= args.max_reminders:
                message = (
                    f"Reminder {reminders_sent}. {dose_message}"
                )
                speak(engine, message)
                append_event(
                    args.csv,
                    session_id,
                    args.medicine,
                    args.dose,
                    "reminder",
                    reminders_sent,
                    "Not taken yet",
                    message,
                    best_session_conf,
                    best_session_box,
                    evidence_file,
                )
                last_reminder_at = now

            if reminders_sent >= args.max_reminders:
                state = "missed"
                stop_evidence_writer(evidence_writer)
                evidence_writer = None
                speak(engine, f"{args.medicine} was not confirmed as taken.")
                append_event(
                    args.csv,
                    session_id,
                    args.medicine,
                    args.dose,
                    "missed_after_reminders",
                    reminders_sent,
                    "Not taken",
                    f"{args.medicine} was not confirmed as taken after {args.max_reminders} reminders.",
                    best_session_conf,
                    best_session_box,
                    evidence_file,
                )

        if state in {"taken", "missed"} and detected:
            cv2.putText(
                frame,
                "Press r to reset for next dose",
                (22, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        draw_status(frame, detected, best_conf, visible_seconds, state, reminders_sent, args)
        if evidence_writer is not None:
            evidence_writer.write(frame)
        cv2.imshow("Real-time Tablet Strip Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            if state == "waiting_for_taken":
                stop_evidence_writer(evidence_writer)
                evidence_writer = None
                append_event(
                    args.csv,
                    session_id,
                    args.medicine,
                    args.dose,
                    "stopped_before_confirmation",
                    reminders_sent,
                    "Not confirmed",
                    "Program stopped before taken/not-taken status was finalized.",
                    best_session_conf,
                    best_session_box,
                    evidence_file,
                )
            break
        if key == ord("t") and state == "waiting_for_taken":
            state = "taken"
            stop_evidence_writer(evidence_writer)
            evidence_writer = None
            speak(engine, f"Thank you. {args.medicine} dose has been recorded.")
            append_event(
                args.csv,
                session_id,
                args.medicine,
                args.dose,
                "taken_confirmed_manual",
                reminders_sent,
                "Taken",
                "Operator confirmed tablet was taken while video evidence was recording.",
                best_session_conf,
                best_session_box,
                evidence_file,
            )
            print("Taken confirmed manually with key 't'.")
        if key == ord("r"):
            state = "idle"
            visible_since = None
            missing_since = None
            dose_spoken_at = 0
            last_reminder_at = 0
            reminders_sent = 0
            best_session_conf = 0.0
            best_session_box = None
            stop_evidence_writer(evidence_writer)
            evidence_writer = None
            evidence_file = ""
            session_id = local_now().strftime("%Y%m%d_%H%M%S")
            print("Reset for next dose.")

    stop_evidence_writer(evidence_writer)
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time MedicineStrip dose reminder")
    parser.add_argument("--model", help="Path to best.pt. Defaults to the newest trained model found.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Use 1 for a second camera.")
    parser.add_argument("--conf", type=float, default=0.08, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--confirm-seconds", type=float, default=2.0, help="Seconds strip must be visible before speaking the dose.")
    parser.add_argument("--taken-absence-seconds", type=float, default=4.0, help="Seconds strip must disappear before marking taken.")
    parser.add_argument("--reminder-interval", type=float, default=120.0, help="Seconds between reminders.")
    parser.add_argument("--max-reminders", type=int, default=3, help="Maximum reminders before marking not taken.")
    parser.add_argument("--medicine", default="15 g tablet", help="Medicine name to speak and log.")
    parser.add_argument("--dose", default="1 tablet", help="Dose to speak and log.")
    parser.add_argument("--voice-rate", type=int, default=160, help="Text-to-speech speed.")
    parser.add_argument("--voice-volume", type=float, default=1.0, help="Text-to-speech volume from 0.0 to 1.0.")
    parser.add_argument("--voice-index", type=int, help="Windows voice index. Use --list-voices to see options.")
    parser.add_argument("--voice-id", help="Exact Windows voice id.")
    parser.add_argument("--no-voice", action="store_true", help="Print voice messages without speaking.")
    parser.add_argument("--list-voices", action="store_true", help="List available system voices and exit.")
    parser.add_argument("--test-voice", action="store_true", help="Speak the dose message once and exit.")
    parser.add_argument("--csv", default="medicine_dose_history.csv", help="CSV file for dose events.")
    parser.add_argument("--save-evidence", action=argparse.BooleanOptionalAction, default=True, help="Save a video clip for each dose session.")
    parser.add_argument("--evidence-dir", default="dose_evidence", help="Folder where dose evidence videos are saved.")
    parser.add_argument("--evidence-fps", type=float, default=10.0, help="FPS for saved evidence videos.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.list_voices:
        list_voices()
    elif parsed_args.test_voice:
        test_engine = init_voice_engine(parsed_args)
        speak(test_engine, build_dose_message(parsed_args))
    else:
        run_detector(parsed_args)
