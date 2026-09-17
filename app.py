"""
SafeZone AI - Live Dashboard
------------------------------
Streamlit app: upload a video (or use a webcam), see PPE and zone
violations flagged in real time, and view a running violation log.

HOW TO RUN:
    streamlit run app.py

BEFORE RUNNING:
    Make sure best.pt (your trained YOLOv8 weights) is in this same
    folder, or update WEIGHTS_PATH below.
"""

import time
import cv2
import pandas as pd
import streamlit as st

from detector import SafeZoneDetector

WEIGHTS_PATH = "best.pt"

st.set_page_config(page_title="SafeZone AI", layout="wide")

st.title("🦺 SafeZone AI")
st.caption("Real-time PPE compliance and restricted-zone monitoring")

# --- Sidebar controls ---
st.sidebar.header("Controls")
source_type = st.sidebar.radio("Video source", ["Upload a video", "Webcam"])
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.4, 0.05)
start_button = st.sidebar.button("▶ Start monitoring")
stop_button = st.sidebar.button("⏹ Stop")

# --- Session state for violation log ---
if "violations" not in st.session_state:
    st.session_state.violations = []
if "running" not in st.session_state:
    st.session_state.running = False

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# --- Layout: video on the left, metrics + log on the right ---
col_video, col_stats = st.columns([2, 1])
video_placeholder = col_video.empty()

with col_stats:
    st.subheader("Live Stats")
    metric_ppe = st.empty()
    metric_zone = st.empty()
    st.subheader("Violation Log")
    log_placeholder = st.empty()

uploaded_file = None
if source_type == "Upload a video":
    uploaded_file = st.sidebar.file_uploader("Upload demo video", type=["mp4", "mov", "avi"])

if st.session_state.running:
    try:
        detector = SafeZoneDetector(weights_path=WEIGHTS_PATH, conf_threshold=conf_threshold)
    except Exception as e:
        st.error(f"Could not load model weights from '{WEIGHTS_PATH}'. "
                 f"Make sure you've trained the model and copied best.pt here.\n\n{e}")
        st.stop()

    if source_type == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if uploaded_file is None:
            st.warning("Please upload a video file in the sidebar first.")
            st.stop()
        # Save uploaded file to disk temporarily so OpenCV can read it
        temp_path = "temp_uploaded_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_path)

    ppe_count = 0
    zone_count = 0

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, violations = detector.process_frame(frame)

        for v in violations:
            st.session_state.violations.append({
                "Time": time.strftime("%H:%M:%S", time.localtime(v.timestamp)),
                "Type": v.type,
                "Confidence": f"{v.confidence:.2f}",
                "Details": v.details,
            })
            if v.type == "PPE":
                ppe_count += 1
            elif v.type == "ZONE":
                zone_count += 1

        # Display frame (convert BGR -> RGB for Streamlit)
        video_placeholder.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        metric_ppe.metric("PPE Violations", ppe_count)
        metric_zone.metric("Zone Intrusions", zone_count)

        if st.session_state.violations:
            df = pd.DataFrame(st.session_state.violations[::-1])  # newest first
            log_placeholder.dataframe(df, use_container_width=True, height=300)

        time.sleep(0.03)  # small delay so the UI stays responsive

    cap.release()
else:
    video_placeholder.info("Click **Start monitoring** in the sidebar to begin.")
