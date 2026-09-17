# SafeZone AI — Project Starter (70%)

Real-time workplace safety monitoring: detects missing helmets (PPE
non-compliance) and restricted-zone intrusions from a video feed, with
a live Streamlit dashboard.

## What's already built (this package)

| File | What it does |
|---|---|
| `train.py` | Fine-tunes YOLOv8 on the Hard Hat Workers dataset |
| `zone_utils.py` | Restricted-zone polygon logic + a click-to-calibrate tool |
| `detector.py` | Core pipeline: PPE check + zone check, draws boxes on frames |
| `app.py` | Streamlit dashboard: live video, violation counters, log table |
| `requirements.txt` | All dependencies (all free/open source) |

This is the full architecture and working logic — what's left is
plugging in your trained model and your specific demo footage.

## Setup

```bash
pip install -r requirements.txt
```

## Step-by-step: what your team does next (the remaining 30%)

### 1. Train the model (Person 1 — ML Engineer)
- Move your downloaded dataset folder into this project, rename it `dataset/`
  (so the structure is `dataset/data.yaml`, `dataset/train/`, etc.)
- **Important**: open `dataset/data.yaml` and confirm the class order —
  it should list `head`, `helmet`, `person`. If the order is different,
  no code changes are needed (the code reads names automatically), just
  double check it matches what you expect.
- Run training (ideally in Google Colab with a free GPU):
  ```bash
  python train.py
  ```
- When it finishes, copy `runs/safezone_ai/weights/best.pt` into this
  project's root folder (same place as `app.py`).

### 2. Calibrate the restricted zone (Person 2 — CV Pipeline)
- Record or choose your demo video (see tip below).
- Run:
  ```bash
  python zone_utils.py your_demo_video.mp4
  ```
- Click 4+ points on the frame to outline your "restricted zone"
  (e.g. an area near machinery). Press `q` when done — it prints
  coordinates in the terminal.
- Paste those coordinates into `RESTRICTED_ZONE` in `zone_utils.py`.

### 3. Record your demo video (Person 4 — Integration/Pitch)
Real construction footage is hard to source — instead, film a 30-60
second clip of teammates:
- Walking around with and without a helmet on
- Walking in and out of a taped-off area on the floor (this becomes
  your "restricted zone" after calibration)

This gives you full control over triggering both violation types live.

### 4. Run the dashboard (whole team, for testing)
```bash
streamlit run app.py
```
- Choose "Upload a video" in the sidebar, upload your demo clip
- Click "Start monitoring"
- You should see bounding boxes: green = compliant, red = no helmet,
  orange = zone intrusion — with a live violation log on the right

### 5. Deploy for judges (Person 4)
- Push this project to a GitHub repo
- Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud)
  (free) so judges can open a live link without installing anything
- **Note**: `best.pt` can be large — if it's over GitHub's 100MB limit,
  use Git LFS, or host the weights file on Google Drive and download
  it at app startup (ask me for this code if needed)

## Tuning tips

- If you get too many false "NO HELMET" flags, raise the confidence
  slider in the sidebar, or increase the IOU overlap threshold in
  `detector.py` (`> 0.2` → try `> 0.3`)
- If training accuracy seems low after 30 epochs, try `yolov8s.pt`
  instead of `yolov8n.pt` in `train.py` — slower but more accurate

## What's NOT included (the last ~30%, up to your team)

- The actual trained weights file (`best.pt`) — depends on your dataset run
- Your specific demo video and calibrated zone coordinates
- Visual polish/branding on the Streamlit dashboard (colors, logo, layout tweaks)
- Deployment to Streamlit Cloud (a few clicks, described above)
- Optional stretch: SMS/email alerts via Twilio for real violations
