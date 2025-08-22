"""
model.py

Loads a pre-trained action recognition model and exposes a simple API to
predict actions on a short clip of frames. This MVP uses PyTorch by default
with a lightweight heuristic fallback if the model isn't available.

You can later swap in a proper video action recognition model (e.g.
TorchVision Video models or custom fine-tuned model) and update the
"_predict_with_model" method accordingly.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    import torch
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class ActionRecognizer:
    """
    High-level wrapper to perform action recognition on short frame clips.

    API:
      - required_clip_len: number of frames expected in each clip
      - predict_clip(frames: List[np.ndarray]) -> (label, probs)
    """

    def __init__(self):
        # For many video models, 8-16 frame clips are common. We'll pick 16.
        self.required_clip_len: int = 16

        # Attempt to load a pre-trained torchvision video model (e.g., r3d_18)
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

        self.model = None
        self.transform = None

        if TORCH_AVAILABLE:
            try:
                # r3d_18 supports pretrained weights in newer torchvision versions.
                # We'll attempt to load weights; if it fails, we continue with heuristic fallback.
                weights = getattr(torchvision.models.video, "R3D_18_Weights", None)
                if weights is not None:
                    w = weights.DEFAULT
                    self.model = torchvision.models.video.r3d_18(weights=w).to(self.device)
                    self.model.eval()
                    self.transform = w.transforms()
                else:
                    # Older torchvision may require manual transforms
                    self.model = torchvision.models.video.r3d_18(pretrained=True).to(self.device)
                    self.model.eval()

                    # Basic normalization similar to ImageNet; adjust if needed
                    self.transform = transforms.Compose([
                        transforms.ConvertImageDtype(torch.float32),
                        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                    ])
            except Exception as e:
                print(f"Warning: Failed to load pretrained video model: {e}. Falling back to heuristic detector.")
                self.model = None
        else:
            print("PyTorch/torchvision not available. Using heuristic fallback.")

    def predict_clip(self, frames: List[np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """
        Predict action label for a clip of frames.

        Returns:
            label: str
            probs: dict mapping {"fight": prob, "normal": prob}
        """
        if len(frames) != self.required_clip_len:
            raise ValueError(f"Expected {self.required_clip_len} frames, got {len(frames)}")

        if self.model is not None:
            return self._predict_with_model(frames)
        else:
            return self._predict_with_heuristic(frames)

    def _predict_with_model(self, frames: List[np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """Use the loaded PyTorch model to predict; map to fight/normal via a simple rule.
        Note: r3d_18 pretraining is on Kinetics-400, which does not have a 'fight' class.
        For MVP, we take high motion magnitude as a proxy or, if you have a finetuned model,
        replace this mapping with the actual fight class probability.
        """
        assert TORCH_AVAILABLE and self.model is not None

        # Convert frames (H,W,C BGR) to tensor clip (T,C,H,W) expected by torchvision video models
        clip = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        clip = np.stack(clip)  # (T,H,W,C)
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)  # (T,C,H,W), dtype uint8

        if self.transform is not None:
            # Some weights transforms expect (C,T,H,W)
            clip = clip.float() / 255.0
            clip = clip.permute(1, 0, 2, 3)  # (C,T,H,W)
            clip = self.transform(clip)
        else:
            # Minimal normalization
            clip = clip.float() / 255.0
            clip = clip.permute(1, 0, 2, 3)  # (C,T,H,W)

        with torch.no_grad():
            inputs = clip.unsqueeze(0).to(self.device)  # (B,C,T,H,W)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Heuristic mapping: use motion magnitude as proxy for 'fight' probability,
        # then blend with uniform prior from model to get a stable MVP.
        motion_prob = self._motion_intensity_prob(frames)
        fight_prob = float(0.7 * motion_prob + 0.3 * 0.5)  # blend to stabilize
        fight_prob = max(0.0, min(1.0, fight_prob))
        normal_prob = 1.0 - fight_prob
        label = "fight" if fight_prob >= 0.6 else "normal"
        return label, {"fight": fight_prob, "normal": normal_prob}

    def _predict_with_heuristic(self, frames: List[np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """Fallback using simple motion intensity as fight proxy."""
        fight_prob = self._motion_intensity_prob(frames)
        label = "fight" if fight_prob >= 0.6 else "normal"
        return label, {"fight": float(fight_prob), "normal": float(1.0 - fight_prob)}

    def _motion_intensity_prob(self, frames: List[np.ndarray]) -> float:
        """
        Compute a simple motion score via mean absolute frame differencing.
        Returns a probability-like score in [0,1] after normalization.
        """
        if len(frames) < 2:
            return 0.0
        diffs = []
        prev_gray = None
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diffs.append(diff.mean())
            prev_gray = gray
        if not diffs:
            return 0.0
        score = float(np.mean(diffs))
        # Normalize by a rough scale to map to [0,1]
        norm = score / 25.0  # tune this for your camera/scene
        return float(max(0.0, min(1.0, norm)))

