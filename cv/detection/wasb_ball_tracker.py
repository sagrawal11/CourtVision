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
_MAX_DISP = 300      # px (original coords): reject candidate blobs that jump farther than a ball can

# Default weights: prefer the Stage-2 our-footage fine-tune (wasb_stage2_ep12) when present —
# validated to recover outdoor recall + kill car/windscreen clutter AND improve indoor recall
# +10-15pp while retaining RacketVision (F1 0.868). Falls back to the base tennis weights.
_MODELS_DIR = PROJECT_ROOT / "models" / "ball"
_DEFAULT_WEIGHTS = _MODELS_DIR / "wasb_stage2_ep12.pth.tar"
if not _DEFAULT_WEIGHTS.exists():
    _DEFAULT_WEIGHTS = _MODELS_DIR / "wasb_tennis_best.pth.tar"


class _Track:
    """Per-frame (x, y, visible) history for the online tracker."""

    def __init__(self):
        self._xy: dict = {}
        self._visi: dict = {}

    def add(self, fid, x, y, visi):
        self._xy[fid] = np.array([x, y], dtype=np.float32)
        self._visi[fid] = visi

    def is_visible(self, fid):
        return self._visi.get(fid, False)

    def xy(self, fid):
        return self._xy[fid]


class OnlineBallTracker:
    """WASB's online motion gate (nttcom/WASB-SBDT src/trackers/online.py, faithful port).

    Given ALL candidate blobs for a frame, keep only those within `max_disp` of the previous
    frame's ball position, then take the highest-score one. This suppresses the confident
    clutter false positives (other court / line marker / parked cars) that appear far from
    the ball's trajectory — the precision half of WASB (0.877 -> ~0.937 on RacketVision).
    Stateful across a video; call refresh() at the start of each new clip.
    """

    def __init__(self, max_disp: int = _MAX_DISP):
        self._max_disp = max_disp
        self._fid = 0
        self._track = _Track()

    def refresh(self):
        self._fid = 0
        self._track = _Track()

    def update(self, dets):
        """dets: list of {'xy': np.array([x,y]), 'score': float} in original coords."""
        if self._fid > 0 and self._track.is_visible(self._fid - 1):
            prev = self._track.xy(self._fid - 1)
            dets = [d for d in dets if np.linalg.norm(d['xy'] - prev) < self._max_disp]
        best_score, x, y, visi = -np.inf, -np.inf, -np.inf, False
        for d in dets:
            if d['score'] > best_score:
                best_score, x, y, visi = d['score'], float(d['xy'][0]), float(d['xy'][1]), True
        self._track.add(self._fid, x, y, visi)
        self._fid += 1
        return {'x': x, 'y': y, 'visi': visi, 'score': best_score}


class WASBBallTracker:
    """WASB HRNet ball detector with the BallTracker.detect_ball interface."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None,
                 use_tracker: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tracker = OnlineBallTracker() if use_tracker else None  # motion gate (precision fix)
        self.frame_history: deque = deque(maxlen=_FRAMES_IN)
        self.video_width: Optional[int] = None
        self.video_height: Optional[int] = None
        self.model = None
        self.model_loaded = False
        self._load(Path(model_path) if model_path else _DEFAULT_WEIGHTS)

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

    def _decode_all(self, hm: np.ndarray):
        """All connected-component blobs above threshold: list of ((x, y), score) in
        heatmap coords. Feeds the online motion gate, which needs every candidate."""
        if float(np.max(hm)) <= _SCORE_THRESH:
            return []
        _, th = cv2.threshold(hm, _SCORE_THRESH, 1, cv2.THRESH_BINARY)
        n_lbl, labels = cv2.connectedComponents(th.astype(np.uint8))
        out = []
        for m in range(1, n_lbl):
            ys, xs = np.where(labels == m)
            ws = hm[ys, xs]
            out.append(((float((xs * ws).sum() / ws.sum()), float((ys * ws).sum() / ws.sum())),
                        float(ws.sum())))
        return out

    def reset(self):
        """Clear frame history + tracker state — call at the start of each new video."""
        self.frame_history.clear()
        if self.tracker is not None:
            self.tracker.refresh()

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

        if self.tracker is None:                      # legacy: single best blob, no motion gate
            res = self._decode(hm)
            if res is None:
                return None
            (xh, yh), score = res
            xy = affine_transform(np.array([xh, yh], dtype=np.float32), trans_out_inv)
            conf = float(min(score / _CONF_SCALE, 1.0))
            return (int(round(xy[0])), int(round(xy[1]))), conf, None

        # motion-gated: decode ALL candidate blobs -> original coords -> tracker gates + picks
        dets = []
        for (xh, yh), score in self._decode_all(hm):
            xy = affine_transform(np.array([xh, yh], dtype=np.float32), trans_out_inv)
            dets.append({'xy': np.array([xy[0], xy[1]], dtype=np.float32), 'score': score})
        out = self.tracker.update(dets)
        if not out['visi']:
            return None
        conf = float(min(out['score'] / _CONF_SCALE, 1.0))
        return (int(round(out['x'])), int(round(out['y']))), conf, None


def create_ball_tracker(device: Optional[str] = None, prefer: str = "wasb", use_tracker: bool = True):
    """Return the best available ball tracker: WASB (higher detection rate) when
    its weights are present, else the legacy TrackNet BallTracker. Keeps the same
    detect_ball(frame) interface either way. `use_tracker` enables WASB's online motion
    gate (clutter-FP suppression) — reset it per video via tracker.reset()."""
    wasb_weights = _DEFAULT_WEIGHTS
    if prefer == "wasb" and wasb_weights.exists():
        tracker = WASBBallTracker(device=device, use_tracker=use_tracker)
        if tracker.model_loaded:
            logger.info("Using WASB ball detector")
            return tracker
        logger.warning("WASB weights present but model failed to load — falling back to TrackNet")
    from cv.detection.ball_tracker import BallTracker
    logger.info("Using TrackNet ball detector")
    return BallTracker(device=device)
