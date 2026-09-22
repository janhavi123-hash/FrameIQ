import cv2
import time
from ultralytics import YOLO

def draw_text_with_background(frame, text, position, font_scale=0.6,
                                color=(255, 255, 255), thickness=2):
    """Draws text with a dark semi-transparent background box behind it,
    so it stays readable no matter what's in the video behind it."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # Draw a filled dark rectangle behind the text
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - text_height - 8),
                  (x + text_width + 5, y + baseline), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw the text on top of that background
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def main():
    VIDEO_SOURCE = "videos/sample.mp4"
    OUTPUT_PATH = "output/result.mp4"

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Video opened successfully. Press 'q' to quit.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (960, 540))

    seen_ids = {}
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
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
                if class_name not in seen_ids:
                    seen_ids[class_name] = set()
                seen_ids[class_name].add(int(track_id))

        # --- Draw the count panel with readable backgrounds ---
        y_offset = 25
        for class_name, id_set in seen_ids.items():
            text = f"{class_name}: {len(id_set)} total"
            draw_text_with_background(annotated_frame, text, (10, y_offset),
                                       font_scale=0.55, color=(0, 255, 0))
            y_offset += 28

        # --- FPS counter, top-right, same readable style ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        draw_text_with_background(annotated_frame, f"FPS: {fps:.1f}", (830, 25),
                                   font_scale=0.55, color=(0, 255, 255))

        out.write(annotated_frame)
        cv2.imshow("Object Detection and Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved output video to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()