# Medicine Reminder System — Single Class (MedicineStrip)

## Folder structure

```
MedicineReminderProject/
├── Medicine_Reminder_System.ipynb   <- run this
├── data.yaml                        <- dataset config (1 class: MedicineStrip)
├── train/
│   ├── images/                      <- 10 images already here
│   └── labels/                      <- EMPTY — you still need to label these
├── valid/
│   ├── images/                      <- 3 images already here
│   └── labels/                      <- EMPTY — you still need to label these
├── test/
│   ├── images/                      <- 2 images already here
│   └── labels/                      <- EMPTY — you still need to label these
└── detection_history.csv            <- created automatically the first time you run Part 5
```

`best.pt` is NOT included — it gets created when you run the training
cells in the notebook (Part 4), saved at
`runs/detect/train/weights/best.pt`. Copy or rename that file to
`best.pt` in this project's root folder before running Part 5
(real-time detection), or just edit `MODEL_PATH` in the notebook to
point at the `runs/detect/...` path directly.

## Before you can train: labeling

The `images/` folders already have your 15 photos split 10/3/2, but
**none of them have bounding-box labels yet** — that step can't be
automated. Use Roboflow (recommended, free, browser-based) or
LabelImg (offline) to draw a box around the strip in each image and
export/save in YOLO format so a `.txt` file lands next to each image
in the matching `labels/` folder. See the earlier `LABELING_GUIDE.md`
I sent you if you need the walkthrough again.

Heads up: 15 images is small for object detection. It's enough to get
the whole pipeline working end-to-end, but don't be surprised if the
first trained model is shaky outside the backgrounds/angles it's seen.
Add more photos with `capture_images.py` any time and retrain — no
need to start over, just relabel the new ones and merge them in.

## Setup

```bash
pip install ultralytics opencv-python pyttsx3 pandas pyyaml
```

GPU is optional but speeds up training a lot. If you don't have one,
change `device=0` to `device="cpu"` in the training cell (Part 4) —
training will just take longer.

## Quick dose reminder

Run this from the project folder:

```bash
venv\Scripts\python.exe real_time_strip_detector.py --conf 0.08 --confirm-seconds 2 --medicine "15 g tablet" --dose "1 tablet" --reminder-interval 120 --max-reminders 3
```

The script opens the webcam, detects `MedicineStrip`, speaks
"Please take 1 tablet of 15 g tablet now", records a short video clip
for that dose session, and logs dose events to `medicine_dose_history.csv`.
It automatically uses the newer trained model at
`C:\Users\HANUMAN NARAYANA\OneDrive\Apps\Desktop\runs\detect\train-2\weights\best.pt`
when available.

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
2. Run Part 1–4 top to bottom to train the model (only needs to be
   done once, or again later if you add more photos).
3. Confirm `best.pt` exists / `MODEL_PATH` points to your trained
   weights.
4. Run Part 5 to start the real-time webcam reminder system.
   Press `q` in the webcam window to quit.

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
