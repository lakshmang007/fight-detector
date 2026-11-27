# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project has no version tags yet, so past entries are dated.

## [Unreleased]

### Removed
- The PyTorch path (torchvision `r3d_18` video model). It crashed on its first prediction with a `RuntimeError` (the clip reached the model's preprocessing in the wrong tensor shape), and its output never fed into the fight score anyway.
- `torch` and `torchvision` from `requirements.txt`: about 1.2 GiB less to install, and no 127 MiB weights download.
- `utils.py`. `notify_fight_detected()` now lives in `main.py`.

### Changed
- `ActionRecognizer.predict_clip(frames)` is replaced by `MotionDetector.update(frame)`, which processes each frame once instead of re-processing all 16 frames on every new frame. Scoring went from 6.3 ms to 0.45 ms per 640×480 frame (14× faster, measured on one desktop CPU), with identical scores.
- Startup time went from 6.7 s to 0.5 s, because PyTorch is no longer imported.
- `prob_threshold` now sets both the on-screen label and the alert. Before, the label used a hard-coded 0.6.
- The alert cooldown uses `time.monotonic()`, so changing the system clock no longer affects it.

### Added
- `test_model.py`, a camera-free test of the motion score.
- A full README.

## 2026-07-23
### Added
- Project title in the README.

## 2025-08-22
### Added
- First version: webcam loop, motion-based fight score, optional PyTorch `r3d_18` path, and a printed alert with a 10-second cooldown.
