"""
Player-of-Interest (POI) tracker for tennis match analysis.

Tracks which of the two detected players is the "target" player across the
entire match, including through side switches that happen every game.

Key concepts:
    - "near"  = player in the bottom portion of the frame (closer to camera)
    - "far"   = player in the top portion of the frame (farther from camera)
    - The POI label alternates between near/far at each game changeover.
    - During a single point, each player stays in their half of the court.

Usage:
    tracker = POITracker(
        start_side="near",          # POI starts at the bottom of the frame
        court_mid_y=360,            # pixel y-coordinate of the net
    )
    # Per-frame inference
    poi_box, opp_box = tracker.assign(frame_idx, bounding_boxes)
    # At game changeover
    tracker.switch_sides()
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple
import numpy as np


# -------------------------------------------------------------------------
# Data types
# -------------------------------------------------------------------------

@dataclasses.dataclass
class PlayerBox:
    """A detected player bounding box with identity label."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str     # "poi" | "opp" | "unknown"
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def feet_y(self) -> float:
        """Bottom of bounding box — closest to the ground plane."""
        return self.y2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


# -------------------------------------------------------------------------
# POI Tracker
# -------------------------------------------------------------------------

class POITracker:
    """
    Assigns the "POI" and "OPP" labels to player bounding boxes per frame.

    Algorithm:
        1. Accept the list of detected bounding boxes for a frame.
        2. Sort boxes by feet_y (larger y = lower in frame = "near").
        3. Assign labels based on current serve_side.
        4. Maintain a smoothing history to avoid flickering.
        5. At game changeovers (signalled externally), flip the side assignment.

    Parameters:
        start_side: "near" | "far" — which side the POI starts on.
        frame_height: Video height in pixels (used to compute net_y if not given).
        net_y: Y-coordinate of the net in the frame (auto-estimated if None).
        smoothing_frames: Number of past frames used to smooth identity assignment.
    """

    def __init__(
        self,
        start_side: str = "near",
        frame_height: int = 720,
        net_y: Optional[float] = None,
        smoothing_frames: int = 15,
    ):
        assert start_side in ("near", "far"), "start_side must be 'near' or 'far'"
        self.poi_side = start_side            # current side for the POI
        self.frame_height = frame_height
        self.net_y = net_y or frame_height / 2
        self.smoothing_frames = smoothing_frames

        # For smoothing: track last N frame assignments
        self._poi_y_history: List[float] = []

    def switch_sides(self) -> None:
        """Call at each game changeover to flip POI's side of the court."""
        self.poi_side = "far" if self.poi_side == "near" else "near"
        self._poi_y_history.clear()
        print(f"[POITracker] Side switch → POI is now on '{self.poi_side}' side")

    def assign(
        self,
        frame_idx: int,
        boxes: np.ndarray,   # shape (N, 4), each row = [x1, y1, x2, y2]
    ) -> Tuple[Optional[PlayerBox], Optional[PlayerBox]]:
        """
        Assign POI / OPP labels to the provided bounding boxes.

        Returns:
            (poi_box, opp_box) — either may be None if not detected.
        """
        if boxes is None or len(boxes) == 0:
            return None, None

        player_boxes = [
            PlayerBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label="unknown")
            for b in boxes
        ]

        # Filter out ball boys / line judges (too small or near edges)
        player_boxes = self._filter_non_players(player_boxes)

        if len(player_boxes) == 0:
            return None, None

        if len(player_boxes) == 1:
            # Only one player visible — infer which they are
            box = player_boxes[0]
            side = self._classify_side(box)
            if side == self.poi_side:
                box.label = "poi"
                return box, None
            else:
                box.label = "opp"
                return None, box

        # Two (or more) players — pick the closest two to the net area
        # then sort by y-position
        player_boxes = sorted(player_boxes, key=lambda b: b.feet_y)

        # "far" player = smaller feet_y (higher in frame = farther from camera)
        # "near" player = larger feet_y (lower in frame = closer to camera)
        far_box  = player_boxes[0]   # smallest y
        near_box = player_boxes[-1]  # largest y

        if self.poi_side == "near":
            poi_box = near_box
            opp_box = far_box
        else:
            poi_box = far_box
            opp_box = near_box

        poi_box.label = "poi"
        opp_box.label = "opp"

        # Update smoothing history
        self._poi_y_history.append(poi_box.cy)
        if len(self._poi_y_history) > self.smoothing_frames:
            self._poi_y_history.pop(0)

        return poi_box, opp_box

    # ── Helpers ──────────────────────────────────────────────────────────

    def _classify_side(self, box: PlayerBox) -> str:
        """Classify a single box as 'near' or 'far' based on y-position relative to net."""
        return "near" if box.feet_y > self.net_y else "far"

    def _filter_non_players(self, boxes: List[PlayerBox]) -> List[PlayerBox]:
        """
        Remove detections that are likely ball boys, line judges, or spectators.
        Heuristic: keep only the 2 largest boxes within the court area.
        """
        if not boxes:
            return boxes

        # Sort by area descending, keep top 2
        boxes_sorted = sorted(boxes, key=lambda b: b.area, reverse=True)
        return boxes_sorted[:2]

    @property
    def poi_is_near(self) -> bool:
        return self.poi_side == "near"


# -------------------------------------------------------------------------
# Side-Switch Detector
# -------------------------------------------------------------------------

class SideSwitchDetector:
    """
    Detects game changeovers from ball and player tracking signals.

    A changeover is characterised by:
        1. Ball not visible for ≥ MIN_GAP_FRAMES frames
        2. Both players stationary (< MOVEMENT_THRESHOLD pixels of motion)
        3. Duration: changeovers take 60–90 seconds; we detect the transition

    This is used to drive POITracker.switch_sides() automatically.
    """

    MIN_GAP_FRAMES = 60         # ~2 seconds at 30fps
    MOVEMENT_THRESHOLD_PX = 20  # pixels the player can move and still be "still"
    MIN_STILL_FRAMES = 45       # how many consecutive still frames = changeover

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._still_counter = 0
        self._ball_gap_counter = 0

    def update(
        self,
        ball_visible: bool,
        player_near_vel: float,  # pixels/frame velocity
        player_far_vel: float,
    ) -> bool:
        """
        Update internal counters. Returns True on the frame a changeover is detected.
        """
        if not ball_visible:
            self._ball_gap_counter += 1
        else:
            self._ball_gap_counter = 0

        both_still = (
            player_near_vel < self.MOVEMENT_THRESHOLD_PX and
            player_far_vel < self.MOVEMENT_THRESHOLD_PX
        )

        if both_still and self._ball_gap_counter >= self.MIN_GAP_FRAMES:
            self._still_counter += 1
        else:
            self._still_counter = 0

        if self._still_counter == self.MIN_STILL_FRAMES:
            self._still_counter = 0
            self._ball_gap_counter = 0
            return True  # Changeover detected

        return False

    @staticmethod
    def compute_velocity(
        positions: List[Optional[Tuple[float, float]]],
        frame_idx: int,
        lookback: int = 5,
    ) -> float:
        """Compute average pixel velocity over last N frames."""
        if frame_idx < lookback + 1:
            return 0.0

        pts = [
            positions[j] for j in range(frame_idx - lookback, frame_idx + 1)
            if positions[j] is not None
        ]
        if len(pts) < 2:
            return 0.0

        dists = [
            np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
            for i in range(1, len(pts))
        ]
        return float(np.mean(dists))
