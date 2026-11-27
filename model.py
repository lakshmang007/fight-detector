"""Motion-based fight score: how much the scene changes across the last 16 frames."""
from collections import deque

import cv2


class MotionDetector:
    def __init__(self, clip_len=16, scale=25.0):
        self.diffs = deque(maxlen=clip_len - 1)  # 16 frames -> 15 frame-to-frame diffs
        self.prev = None
        self.scale = scale  # mean pixel change that reads as 100% fight; tune per camera/scene

    def update(self, frame):
        """Add one BGR frame. Returns fight probability in [0, 1], or None until 16 frames are in."""
        # Each frame is processed once and only its diff is kept, instead of redoing all 16 every frame.
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.prev is not None:
            self.diffs.append(cv2.absdiff(gray, self.prev).mean())
        self.prev = gray
        if len(self.diffs) < self.diffs.maxlen:
            return None
        return min(1.0, sum(self.diffs) / len(self.diffs) / self.scale)
