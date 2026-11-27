import numpy as np

from model import MotionDetector


def test_motion_score():
    black, white = np.zeros((48, 64, 3), np.uint8), np.full((48, 64, 3), 255, np.uint8)
    still, flicker = MotionDetector(), MotionDetector()
    still_out = [still.update(black) for _ in range(16)]
    flicker_out = [flicker.update(white if i % 2 else black) for i in range(16)]
    assert still_out[:15] == [None] * 15  # needs 16 frames before it scores
    assert still_out[15] == 0.0 and flicker_out[15] == 1.0


if __name__ == "__main__":
    test_motion_score()
    print("ok")
