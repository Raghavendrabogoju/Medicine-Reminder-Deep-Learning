# Medicine Reminder System — Deep Learning with YOLO11

This project uses a trained YOLO11 neural network to detect a single
`MedicineStrip` class. OpenCV is used only to capture and display webcam
frames; it does not perform the medicine detection. The trained detector
provides the visual input for the reminder and dose-history workflow.

## Dataset

```
data.yaml
train/images/    train/labels/
valid/images/    valid/labels/
test/images/     test/labels/
```

YOLO-format bounding-box labels are provided alongside the images. The
current dataset is small, so collect and label more varied images before
relying on model predictions outside the training conditions.

## Setup

```bash
pip install ultralytics opencv-python pyttsx3 pandas pyyaml
```

GPU is optional. The notebook automatically selects CUDA when available
and otherwise trains on CPU.

## Quick dose reminder

Run this from the project folder:

```bash
venv\Scripts\python.exe real_time_strip_detector.py --conf 0.08 --confirm-seconds 2 --medicine "15 g tablet" --dose "1 tablet" --reminder-interval 120 --max-reminders 3
```

The script opens the webcam, uses the trained YOLO11 model to detect
`MedicineStrip`, speaks
"Please take 1 tablet of 15 g tablet now", records a short video clip
for that dose session, and logs dose events to `medicine_dose_history.csv`.
Train the model using the notebook before starting the reminder system.
By default, the script selects the newest trained checkpoint under
`runs/detect/train*/weights/best.pt` (or a root-level `best.pt`). Pass
`--model path\to\best.pt` to select a different checkpoint.

After the dose message, the script watches whether the detected strip
leaves the camera view for a few seconds. If it does, it logs `Taken`.
If not, it gives a voice reminder every 2 minutes, up to 3 reminders,
and logs each reminder plus the final `Not taken` status.

If the webcam opens but detection is weak, try:

```bash
venv\Scripts\python.exe real_time_strip_detector.py --conf 0.05 --confirm-seconds 2 --medicine "15 g tablet" --dose "1 tablet" --reminder-interval 120 --max-reminders 3
```

If the wrong camera opens, try `--camera 1`.

Dose videos are saved in `dose_evidence/`, and the CSV stores the video
path in the `EvidenceFile` column.

## Voice check

List available Windows voices:

```bash
venv\Scripts\python.exe real_time_strip_detector.py --list-voices
```

Test the dose message without opening the camera:

```bash
venv\Scripts\python.exe real_time_strip_detector.py --test-voice --voice-index 2 --medicine "15 g tablet" --dose "1 tablet"
```

If the voice is too fast or unclear, lower the rate:

```bash
venv\Scripts\python.exe real_time_strip_detector.py --test-voice --voice-index 2 --voice-rate 130 --medicine "15 g tablet" --dose "1 tablet"
```

## Running it

1. Open `Medicine_Reminder_System.ipynb` in Jupyter.
2. Run Parts 1–4 to fine-tune YOLO11 on the labeled strip dataset and
   evaluate predictions.
3. Run Part 5 to use the trained detector with the real-time reminder
   system. Press `q` in the webcam window to quit.

## What the reminder logic does

- When the tablet strip is detected for 2 continuous seconds, it speaks
  the exact dose: "Please take 1 tablet of 15 g tablet now."
- If the strip disappears from the camera view for a few seconds after
  the instruction, it logs `Taken`.
- If the strip does not disappear, it speaks a reminder every 2 minutes,
  up to 3 times.
- It logs every dose instruction, reminder, taken confirmation, or final
  not-taken result to `medicine_dose_history.csv`, including a saved
  video path in `EvidenceFile`.
