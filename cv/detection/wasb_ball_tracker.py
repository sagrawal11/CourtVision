"""cv/detection/wasb_ball_tracker.py — WASB tennis ball detector (drop-in for BallTracker).

Uses the vendored WASB HRNet (cv/detection/wasb/) with pretrained tennis weights
(models/ball/wasb_tennis_best.pth.tar). Same interface as BallTracker.detect_ball,
so it slots into the pipeline and extract_features unchanged.

Measured on real footage (Indoor Match 2, annotated hit/bounce frames): WASB
detects ~81% of clearly-visible balls vs ~63% for the previous TrackNet post-proc.
Cost is ~0.6s/frame on CPU (fine on the GPU worker; slow for long CPU runs).

WASB preprocessing/decoding (must match the pretrained model exactly):
  - 3 consecutive RGB frames [t-2, t-1, t] warped to 512x288 via affine, stacked
    to 9 channels; ImageNet normalized.
  - Output = 3 heatmaps at 512x288; index 2 (current frame) is read.
  - Decode: sigmoid -> pixel threshold 0.5 -> connected components -> heatmap-
    weighted centroid -> inverse affine back to original pixel coords.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .wasb.hrnet import HRNet
from .wasb.image_utils import get_affine_transform, affine_transform

logger = logging.getLogger("cv.detection.wasb_ball_tracker")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INP_W, _INP_H = 512, 288
_FRAMES_IN = 3
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SCORE_THRESH = 0.5  # heatmap pixel threshold (WASB default)
_CONF_SCALE = 15.0   # rough normaliser: blob score (sum of hm weights, ~0.5-17) -> 0-1


class WASBBallTracker:
    """WASB HRNet ball detector with the BallTracker.detect_ball interface."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.frame_history: deque = deque(maxlen=_FRAMES_IN)
        self.video_width: Optional[int] = None
        self.video_height: Optional[int] = None
        self.model = None
        self.model_loaded = False
        self._load(Path(model_path) if model_path else PROJECT_ROOT / "models" / "ball" / "wasb_tennis_best.pth.tar")

    def _load(self, model_path: Path) -> None:
        if not model_path.exists():
            logger.error(f"WASB weights not found at {model_path}")
            return
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(Path(__file__).resolve().parent / "wasb" / "wasb_tennis.yaml")
            self.model = HRNet(cfg)
            ckpt = torch.load(model_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device).eval()
            self.model_loaded = True
            logger.info(f"Loaded WASB tennis ball detector from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load WASB model: {e}")
            self.model = None
            self.model_loaded = False

    def _preprocess(self, frames):
        h, w = frames[0].shape[:2]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = max(h, w) * 1.0
        trans_in = get_affine_transform(center, scale, 0, [_INP_W, _INP_H], inv=0)
        trans_out_inv = get_affine_transform(center, scale, 0, [_INP_W, _INP_H], inv=1)
        chans = []
        for bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            warped = cv2.warpAffine(rgb, trans_in, (_INP_W, _INP_H), flags=cv2.INTER_LINEAR)
            t = (warped.astype(np.float32) / 255.0 - _MEAN) / _STD
            chans.append(np.transpose(t, (2, 0, 1)))  # C,H,W
        inp = torch.from_numpy(np.concatenate(chans, axis=0)).unsqueeze(0)  # 1,9,H,W
        return inp, trans_out_inv

    def _decode(self, hm: np.ndarray):
        """Highest-scoring connected-component, heatmap-weighted centroid."""
        if float(np.max(hm)) <= _SCORE_THRESH:
            return None
        _, th = cv2.threshold(hm, _SCORE_THRESH, 1, cv2.THRESH_BINARY)
        n_lbl, labels = cv2.connectedComponents(th.astype(np.uint8))
        best, best_score = None, -1.0
        for m in range(1, n_lbl):
            ys, xs = np.where(labels == m)
            ws = hm[ys, xs]
            s = float(ws.sum())
            if s > best_score:
                best_score = s
                best = (float((xs * ws).sum() / ws.sum()), float((ys * ws).sum() / ws.sum()))
        return (best, best_score) if best is not None else None

    @torch.no_grad()
    def detect_ball(
        self,
        frame: np.ndarray,
        text_prompt: str = "tennis ball",
        threshold: float = 0.3,
    ) -> Optional[Tuple[Tuple[int, int], float, None]]:
        """Detect the ball in the current frame (needs 3 consecutive frames of history).
        Returns ((x, y), confidence, None) in original pixel coords, or None.
        `text_prompt`/`threshold` kept for interface compatibility (ignored)."""
        if not self.model_loaded:
            return None
        if self.video_width is None:
            self.video_height, self.video_width = frame.shape[:2]
        self.frame_history.append(frame)
        if len(self.frame_history) < _FRAMES_IN:
            return None

        inp, trans_out_inv = self._preprocess(list(self.frame_history))
        preds = self.model(inp.to(self.device))
        hm_tensor = preds[0] if isinstance(preds, dict) else preds
        hm = torch.sigmoid(hm_tensor).cpu().numpy()[0, _FRAMES_IN - 1]  # current-frame heatmap

        res = self._decode(hm)
        if res is None:
            return None
        (xh, yh), score = res
        xy = affine_transform(np.array([xh, yh], dtype=np.float32), trans_out_inv)
        conf = float(min(score / _CONF_SCALE, 1.0))
        return (int(round(xy[0])), int(round(xy[1]))), conf, None


def create_ball_tracker(device: Optional[str] = None, prefer: str = "wasb"):
    """Return the best available ball tracker: WASB (higher detection rate) when
    its weights are present, else the legacy TrackNet BallTracker. Keeps the same
    detect_ball(frame) interface either way."""
    wasb_weights = PROJECT_ROOT / "models" / "ball" / "wasb_tennis_best.pth.tar"
    if prefer == "wasb" and wasb_weights.exists():
        tracker = WASBBallTracker(device=device)
        if tracker.model_loaded:
            logger.info("Using WASB ball detector")
            return tracker
        logger.warning("WASB weights present but model failed to load — falling back to TrackNet")
    from cv.detection.ball_tracker import BallTracker
    logger.info("Using TrackNet ball detector")
    return BallTracker(device=device)
