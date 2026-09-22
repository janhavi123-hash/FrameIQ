import streamlit as st
import cv2
import time
import tempfile
import os
from ultralytics import YOLO

def draw_text_with_background(frame, text, position, font_scale=0.6,
                                color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - text_height - 8),
                  (x + text_width + 5, y + baseline), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def process_video(input_path, output_path, progress_bar, status_text):
    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (960, 540))

    seen_ids = {}
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        results = model.track(frame, verbose=False, conf=0.5, persist=True,
                               tracker="bytetrack.yaml")
        annotated_frame = results[0].plot(line_width=2)

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.tolist()
            classes = boxes.cls.tolist()
            for track_id, cls_id in zip(ids, classes):
                class_name = model.names[int(cls_id)]
                seen_ids.setdefault(class_name, set()).add(int(track_id))

        y_offset = 25
        for class_name, id_set in seen_ids.items():
            text = f"{class_name}: {len(id_set)} total"
            draw_text_with_background(annotated_frame, text, (10, y_offset),
                                       font_scale=0.55, color=(0, 255, 0))
            y_offset += 28

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        draw_text_with_background(annotated_frame, f"FPS: {fps:.1f}", (830, 25),
                                   font_scale=0.55, color=(0, 255, 255))

        out.write(annotated_frame)
        frame_count += 1
        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count} of {total_frames}...")

    cap.release()
    out.release()
    return seen_ids


st.set_page_config(page_title="Object Detection and Tracking", layout="centered")
st.title("Object Detection and Tracking")
st.write("Upload a video to detect and track objects in real time.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.subheader("Original video")
    st.video(uploaded_file)

    # Only process if this is a NEW file (different name) than what's already
    # stored in session_state — this is what stops reprocessing on every click.
    if ("processed_filename" not in st.session_state or
            st.session_state.processed_filename != uploaded_file.name):

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_file.read())
        temp_input.close()

        output_path = os.path.join(tempfile.gettempdir(), "result.mp4")

        with st.spinner("Processing your video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            seen_ids = process_video(temp_input.name, output_path, progress_bar, status_text)
            status_text.empty()

        # Save results in session_state so they persist across re-runs
        st.session_state.processed_filename = uploaded_file.name
        st.session_state.output_path = output_path
        st.session_state.seen_ids = seen_ids

    st.success("Processing complete!")

    st.subheader("Processed video")
    st.video(st.session_state.output_path)

    st.subheader("Detection summary")
    col1, col2 = st.columns(2)
    items = list(st.session_state.seen_ids.items())
    half = (len(items) + 1) // 2
    with col1:
        for class_name, id_set in items[:half]:
            st.metric(label=class_name.capitalize(), value=len(id_set))
    with col2:
        for class_name, id_set in items[half:]:
            st.metric(label=class_name.capitalize(), value=len(id_set))

    with open(st.session_state.output_path, "rb") as f:
        st.download_button(
            label="Download processed video",
            data=f,
            file_name="result.mp4",
            mime="video/mp4"
        )