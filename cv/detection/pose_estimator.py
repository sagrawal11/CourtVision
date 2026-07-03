"""SAM-3D-Body pose estimation for stroke classification — SCAFFOLD (incomplete).

Per docs/cv-pipeline.md, the plan is to replace the centroid-based stroke
features with body-pose features (which arm leads, shoulder rotation, arm
extension at contact), which generalise across handedness and stance far better.

COST CONTROL — the whole point of this scaffold: SAM-3D-Body is heavy, so it must
run ONLY in small windows around detected hit frames (~1-1.5k contacts/match),
never on every frame. `gated_hit_windows()` implements that gating and is real,
tested logic; the model itself is stubbed.

Status: inert. `PoseEstimator.is_available()` returns False until a model is
wired in, and the pipeline falls back to centroid stroke features unchanged. To
finish: load SAM-3D-Body in `_load`, implement `estimate_pose_at` +
`pose_stroke_features`, retrain the stroke model on pose features, and pass a
PoseEstimator into HitDetector._classify_types_ml (see the hook there).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def gated_hit_windows(hit_frames: List[int], window: int = 3) -> List[Tuple[int, int]]:
    """Merge per-hit ±window frame ranges into deduped (start, end) intervals.

    This is what keeps the pose pass cheap — pose runs only inside these
    intervals (around contacts), not across the whole video. Overlapping or
    adjacent windows are merged so nothing is processed twice.
    """
    if not hit_frames:
        return []
    ranges = sorted((max(0, f - window), f + window) for f in hit_frames)
    merged: List[List[int]] = [list(ranges[0])]
    for start, end in ranges[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


class PoseEstimator:
    """Wrapper around SAM-3D-Body. Inert until a model is loaded."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self._model = None  # TODO: load SAM-3D-Body here (self._load()).

    def is_available(self) -> bool:
        """True once a pose model is loaded. False keeps the pipeline on the
        existing centroid stroke features."""
        return self._model is not None

    def _load(self) -> None:
        # TODO: load SAM-3D-Body weights onto self.device and set self._model.
        raise NotImplementedError("SAM-3D-Body loading not implemented yet")

    def estimate_pose_at(self, frame, bbox: Optional[Tuple[float, float, float, float]] = None):
        """Estimate body pose for the player in `bbox` at a single frame.

        TODO: crop to bbox, run SAM-3D-Body, return joint positions / angles.
        """
        raise NotImplementedError("pose estimation not implemented yet")

    def pose_stroke_features(self, pose) -> List[float]:
        """Turn a pose into stroke-classifier features (leading arm, shoulder
        rotation, arm extension at contact).

        TODO: implement once estimate_pose_at returns real poses.
        """
        raise NotImplementedError("pose feature extraction not implemented yet")
