# Fight Detector

A small Python + OpenCV app that watches a webcam and prints an alert when it sees sustained, heavy motion, a rough proxy for a fight.

It is a motion heuristic, not a trained AI model: it reacts to *how much* the picture changes, not to *what* people are doing. Treat it as a starting point or a demo, not a safety system.

## Features

- Live webcam feed with an on-screen label: `Collecting...` (yellow), `normal (0.12)` (green) or `fight (0.83)` (red)
- Scores motion over a sliding window of the last 16 frames (about half a second at 30 fps)
- Prints a timestamped alert when the score reaches 0.6, at most once every 10 seconds
- One dependency (`opencv-python`); no GPU, no model download
- `MotionDetector` can be used on its own, for example on a recorded video file

## Installation

You need Python 3 (tested with Python 3.13 and OpenCV 5.0), a webcam, and a desktop session, because the app opens a preview window.

```bash
git clone https://github.com/lakshmang007/fight-detector.git
cd fight-detector
pip install -r requirements.txt
```

## Quick start

```bash
python main.py
```

A window titled **Fight Detector** opens. Make big, fast movements close to the camera and the label should turn red, with an alert like this in the terminal:

```text
[ALERT 2026-11-27 14:03:52] Fight likely detected! probability=0.83
```

Press **q** in the window to quit.

## Usage

### Change the settings

There are no command-line flags. Pass settings to `main()` instead, for example to use a second camera and a stricter threshold:

```bash
python -c "from main import main; main(camera_index=1, prob_threshold=0.75)"
```

### Score a recorded video

`MotionDetector` takes one BGR frame at a time and returns a score from 0 to 1, or `None` until it has seen 16 frames:

```python
import cv2
from model import MotionDetector

cap = cv2.VideoCapture("clip.mp4")
detector = MotionDetector()
frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    prob = detector.update(frame)
    if prob is not None and prob >= 0.6:
        print(f"frame {frame_no}: fight likely ({prob:.2f})")
    frame_no += 1
```

### Send real alerts

Alerts are a `print` in `notify_fight_detected()` in [main.py](main.py). Replace that line with your email, SMS or push code.

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `camera_index` | `main()` | `0` | Which camera OpenCV opens |
| `prob_threshold` | `main()` | `0.6` | Score at or above which a window counts as a fight |
| `cooldown_sec` | `main()` | `10.0` | Minimum seconds between two alerts |
| `window_name` | `main()` | `"Fight Detector"` | Title of the preview window |
| `clip_len` | `MotionDetector()` | `16` | Frames in the sliding window |
| `scale` | `MotionDetector()` | `25.0` | Average pixel change (out of 255) that counts as a score of 1.0. Tune it per camera and scene |

There are no environment variables or config files.

## How it works

1. Each frame is turned grayscale and blurred with a 5×5 Gaussian filter to remove sensor noise.
2. The app takes the average absolute difference from the previous frame and keeps the last 15 of these differences.
3. Score = average of those 15 differences ÷ `scale`, capped at 1.0. With the defaults, a fight means the picture changes by an average of 15 out of 255 per frame.

Each frame is processed once, so scoring costs about 0.5 ms per 640×480 frame on a desktop CPU.

## Limitations

- Anything that moves a lot looks like a fight: dancing, sports, a person close to the lens, camera shake, or lights switching on and off.
- Two people fighting far from the camera may not move enough pixels to trigger it.
- `scale` and `prob_threshold` need tuning for each camera position and scene.
- `cv2.imshow` needs a display, so it won't run on a headless server or with `opencv-python-headless` as it is.

## Project layout

| File | Purpose |
|---|---|
| [main.py](main.py) | Webcam loop, on-screen label, alert with cooldown |
| [model.py](model.py) | `MotionDetector`, the motion score |
| [test_model.py](test_model.py) | Test with synthetic frames, no camera needed |
| [CHANGELOG.md](CHANGELOG.md) | What changed between versions |

## Contributing

1. Create a branch from `main`.
2. Run the test, which needs no camera:

   ```bash
   python test_model.py
   ```

3. Open a pull request describing what you changed and how you tested it.

## License

This repository has no license yet. Until one is added, default copyright applies, so others may view the code but have no legal permission to reuse it.
