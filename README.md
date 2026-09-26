# FrameIQ — Object Detection and Tracking

A real-time object detection and tracking tool. Detects and tracks multiple objects in video, assigns persistent IDs, and counts unique objects seen — all through a browser-based interface.

## Demo Video
📹 [Watch the demo video](https://drive.google.com/file/d/1BiYy_0TpdD3FvtYBh7Or9oUjE6xzN_hj/view?usp=drivesdk)

## Features
- Object detection using YOLOv8 (Ultralytics)
- Multi-object tracking with persistent IDs using ByteTrack
- Cumulative unique-object counting per class (not per-frame)
- FPS counter
- Streamlit web interface: upload a video, process it, preview and download the result
- Original vs. processed video comparison view

## Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8 (yolov8s)
- ByteTrack (via Ultralytics' built-in tracker)
- Streamlit

## How It Works
1. Each frame of the uploaded video is passed through a YOLOv8 model, which detects objects and draws bounding boxes with labels.
2. ByteTrack assigns each detected object a persistent ID across frames, so the same object keeps the same ID even as it moves.
3. A running count of unique IDs per object class is kept, giving an accurate "total objects seen" count rather than a per-frame count.
4. The processed video is saved and made available to preview and download in the browser.

## Known Limitations
- YOLO occasionally misclassifies objects due to the limitations of its pretrained dataset (COCO) — e.g., unusual clothing shapes can be misread as an everyday object.
- In crowded or heavily occluded scenes, objects can briefly lose tracking and get re-assigned a new ID, which can inflate the unique-count for busy videos — a known challenge in multi-object tracking called re-identification.
