#!/usr/bin/env python3
"""
cv/detection/court_roi.py — Per-court region-of-interest.

A court ROI is a hand-drawn polygon (full-res pixel coords) outlining the ONE
court a match is played on. It is drawn once per court camera with
cv/tools/court_roi_editor.py and reused across every match on that court.

extract_features.py uses it to ignore adjacent-court players and off-court balls
on multi-court (college dual-match) footage, where the pretrained detectors
otherwise pick up players/balls from neighbouring courts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_polygon(path: str) -> Optional[np.ndarray]:
    """Load a court polygon from a court_roi_editor.py JSON. None if missing/invalid."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        pts = json.loads(p.read_text()).get("polygon", [])
    except (ValueError, OSError):
        return None
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32)


def contains(polygon: np.ndarray, x: float, y: float) -> bool:
    """True if (x, y) is inside (or on the edge of) the polygon."""
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def expand_polygon(polygon: np.ndarray, up: int = 95, down: int = 155, side: int = 35) -> np.ndarray:
    """Add margin outward from a court traced on its lines.

    Players stand *behind* the baselines and the ball flies *above* the far
    baseline, so a line-tight polygon misses them. This pushes the top edge up
    (far player + ball airspace), the bottom edge down (near player behind his
    baseline), and the sides out (wide play). Assumes near = bottom of frame,
    which holds for these fixed end-court PlaySight cameras.
    """
    poly = polygon.astype(float)
    mid_x, mid_y = poly[:, 0].mean(), poly[:, 1].mean()
    out = poly.copy()
    out[:, 0] += np.where(poly[:, 0] > mid_x, side, -side)
    out[:, 1] += np.where(poly[:, 1] > mid_y, down, -up)
    return out.astype(np.int32)
