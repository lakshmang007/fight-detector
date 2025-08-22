"""
main.py

Captures video frames from the default webcam, runs an action recognition
model to detect fights, and triggers a notification via utils when a fight
is likely. This is an MVP scaffold with clear extension points.

Usage:
    python main.py
"""

import time
from typing import Optional

import cv2

from model import ActionRecognizer
from utils import notify_fight_detected


def draw_status(frame, text: str, color=(0, 255, 0)):
    """Overlay status text on the frame."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def main(camera_index: int = 0, prob_threshold: float = 0.6, window_name: str = "Fight Detector"):
    """
    - Opens a webcam stream
    - Reads frames in a loop
    - Sends batches/clips to the model for prediction
    - If fight is detected above threshold, triggers notification
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    # Initialize action recognizer (uses a pre-trained model if available)
    recognizer = ActionRecognizer()

    # For simple temporal modeling, collect a small buffer of frames
    clip: list = []
    clip_len = recognizer.required_clip_len
    last_notification_time: Optional[float] = None
    notification_cooldown_sec = 10.0  # avoid spamming notifications

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed; exiting.")
                break

            # Resize for speed if needed
            display_frame = frame.copy()

            # Accumulate frames for a temporal clip
            clip.append(frame)
            if len(clip) > clip_len:
                clip.pop(0)

            status_text = "Collecting..." if len(clip) < clip_len else "Predicting..."
            status_color = (0, 255, 255)

            fight_prob = 0.0
            label = "unknown"
            if len(clip) == clip_len:
                # Predict once we have enough frames
                label, probs = recognizer.predict_clip(clip)
                fight_prob = float(probs.get("fight", 0.0))
                status_text = f"{label} ({fight_prob:.2f})"
                status_color = (0, 255, 0) if label != "fight" else (0, 0, 255)

                # Notification on threshold with cooldown
                now = time.time()
                if label == "fight" and fight_prob >= prob_threshold:
                    if not last_notification_time or (now - last_notification_time) > notification_cooldown_sec:
                        notify_fight_detected(prob=fight_prob)
                        last_notification_time = now

            # Draw overlay and show
            draw_status(display_frame, status_text, status_color)
            cv2.imshow(window_name, display_frame)

            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

