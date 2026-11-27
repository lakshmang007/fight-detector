"""Webcam fight detector: flags sustained heavy motion and prints an alert. Run `python main.py`, press q to quit."""
import time
from datetime import datetime

import cv2

from model import MotionDetector


def notify_fight_detected(prob):
    # Swap this print for email/SMS/push when needed.
    print(f"[ALERT {datetime.now():%Y-%m-%d %H:%M:%S}] Fight likely detected! probability={prob:.2f}")


def main(camera_index=0, prob_threshold=0.6, cooldown_sec=10.0, window_name="Fight Detector"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")
    detector, last_alert = MotionDetector(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break
            prob = detector.update(frame)
            if prob is None:
                text, color = "Collecting...", (0, 255, 255)
            else:
                fight = prob >= prob_threshold
                text, color = f"{'fight' if fight else 'normal'} ({prob:.2f})", (0, 0, 255) if fight else (0, 255, 0)
                now = time.monotonic()
                if fight and (last_alert is None or now - last_alert > cooldown_sec):  # cooldown stops alert spam
                    notify_fight_detected(prob)
                    last_alert = now
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
