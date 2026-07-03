"""
cv/analysis/player_identity.py — Colour-based player identity tracker.

Why this exists
---------------
The two players physically swap court ends (changeovers) after every odd
game. Per-frame YOLO detections only tell us "a person was here at frame F";
they don't say "this is the same person who was on the far side ten minutes
ago." Without identity, downstream code that depends on per-player attributes
(handedness, stats by player) silently breaks at every changeover.

YOLO also doesn't distinguish players from coaches, ball kids, line judges,
or anybody else who walks onto the court. So we need to lock in references
during a stable two-players-only configuration (a real point) and reject
later detections that don't match either reference.

How it works
------------
1. Initialisation phase
   Wait until we observe `INIT_STABLE_FRAMES` consecutive frames containing
   exactly two strong detections, one in the upper half of the frame and one
   in the lower half. Use the average colour signatures from those frames as
   the references for "player 1" (initial near) and "player 2" (initial far).
   Frames before this happen contribute nothing — pre-match warm-up, coaches
   walking around, etc. are skipped automatically.

2. Per-frame tracking
   Compute a colour signature for every person bbox detected in the frame.
   Greedily assign the best match to player 1 and the best match to player 2.
   If a match's distance exceeds `MAX_REF_DISTANCE`, that side is marked
   unknown for the frame rather than picking a coach by mistake. References
   slowly drift via EMA so they track lighting changes.

3. Changeover detection
   Smooth the per-frame "p1 is on near side" signal aggressively (median over
   ~3 s). Report a transition only when it persists for `PERSIST_FRAMES`
   afterwards, so brief occlusions (one player walking in front of the other,
   replay graphics) don't create spurious side-swaps.

Public API
----------
    extract_color_signature(frame, bbox)          → np.ndarray | None
    color_distance(sig_a, sig_b)                  → float in [0, 1]
    PlayerIdentityTracker
        .update(frame_idx, frame, all_bboxes)
            → (near_bbox|None, far_bbox|None, smoothed_assignment)
        .get_changeover_frames()                  → List[int]
        .is_initialized                           → bool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour signature
# ─────────────────────────────────────────────────────────────────────────────

HUE_BINS = 8
SAT_BINS = 8

TORSO_Y_TOP_FRAC    = 0.20
TORSO_Y_BOTTOM_FRAC = 0.55
TORSO_X_INSET_FRAC  = 0.15


def extract_color_signature(
    frame: np.ndarray,
    bbox:  Tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """Return a normalised hue-saturation histogram over the torso region."""
    if frame is None or bbox is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0.0, min(w_img - 1.0, x1)); x2 = max(0.0, min(w_img, x2))
    y1 = max(0.0, min(h_img - 1.0, y1)); y2 = max(0.0, min(h_img, y2))

    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return None

    tx1 = int(x1 + bw * TORSO_X_INSET_FRAC)
    tx2 = int(x2 - bw * TORSO_X_INSET_FRAC)
    ty1 = int(y1 + bh * TORSO_Y_TOP_FRAC)
    ty2 = int(y1 + bh * TORSO_Y_BOTTOM_FRAC)

    torso = frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], channels=[0, 1], mask=None,
        histSize=[HUE_BINS, SAT_BINS],
        ranges=[0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def color_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1] — 0 = identical, 1 = nothing in common."""
    bc = float(np.sum(np.sqrt(sig_a * sig_b)))
    bc = max(0.0, min(1.0, bc))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pick the "near + far" candidate pair from a frame's detections
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_pair(
    bboxes:  Sequence[Tuple[float, float, float, float]],
    frame_h: int,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]:
    """
    From a list of detections, pick the largest box whose feet are in the
    bottom half ("near") and the largest whose feet are in the top half
    ("far"). Either may be None.
    """
    near_cand, far_cand = None, None
    near_area, far_area = 0.0, 0.0
    for b in bboxes:
        x1, y1, x2, y2 = b
        area   = max(0.0, (x2 - x1) * (y2 - y1))
        feet_y = (y1 + y2) / 2.0
        if feet_y > frame_h * 0.5:
            if area > near_area:
                near_cand, near_area = b, area
        else:
            if area > far_area:
                far_cand, far_area = b, area
    return near_cand, far_cand


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

# How many consecutive frames we need with exactly one detection in each half
# (and good colour signatures) before we lock in references. ~1.5s at 30 fps.
INIT_STABLE_FRAMES = 45

# Maximum colour distance for accepting a detection as a known player. Larger
# means we forgive bigger lighting/pose changes but also risk accepting a
# coach in a similar shirt. 0.55 is a permissive but practical threshold.
MAX_REF_DISTANCE = 0.55

# Smoothing window for the per-frame "p1 on near" signal (frames of samples).
SMOOTH_WINDOW = 90

# After a transition in the smoothed signal, require this many SOURCE frames
# of stable opposite-side assignment before calling it a real changeover.
PERSIST_FRAMES = 300

# Within that persistence window, also require at least this many VALID (non -1)
# samples actually agreeing with the new side. A medical/TV timeout or replay
# leaves the ball gone and detections sparse, so a lone flip could otherwise pass
# the "nothing contradicts it" check on an almost-empty window — this demands
# positive corroboration that the players really did swap ends.
MIN_PERSIST_SAMPLES = 10

# Slow reference drift to track lighting/pose changes during long matches.
REFERENCE_LR = 0.02


@dataclass
class PlayerIdentityTracker:
    """Per-frame player identity tracker with smart init and coach filtering."""

    smooth_window:    int = SMOOTH_WINDOW
    persist_frames:   int = PERSIST_FRAMES
    min_persist_samples: int = MIN_PERSIST_SAMPLES
    max_ref_distance: float = MAX_REF_DISTANCE
    init_stable_frames: int = INIT_STABLE_FRAMES

    # Per-sample state
    _frame_indices: List[int] = field(default_factory=list)
    _raw_history:   List[int] = field(default_factory=list)   # 1/0/-1

    # References (None until initialised)
    _sig_p1: Optional[np.ndarray] = None
    _sig_p2: Optional[np.ndarray] = None

    # Initialisation buffer: (near_sig, far_sig) per recent qualifying frame
    _init_buf: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._sig_p1 is not None and self._sig_p2 is not None

    def update(
        self,
        frame_idx: int,
        frame:     np.ndarray,
        bboxes:    Sequence[Tuple[float, float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]], int]:
        """
        Process one frame's detections.

        Args:
            frame_idx : source-video frame index
            frame     : BGR frame array
            bboxes    : list of (x1, y1, x2, y2) for every person detected

        Returns:
            (near_bbox, far_bbox, smoothed_assignment)
              near_bbox / far_bbox : the bbox of each verified player on each
                  side right now (None if we couldn't confidently identify one)
              smoothed_assignment  : 1 = p1 on near, 0 = p1 on far, -1 = unknown
        """
        self._frame_indices.append(frame_idx)
        frame_h = frame.shape[0] if frame is not None else 720

        if not self.is_initialized:
            return self._init_step(frame_idx, frame, bboxes, frame_h)

        # ── Already initialised: score every detection against both refs ──
        return self._track_step(frame, bboxes, frame_h)

    # ── Initialisation ───────────────────────────────────────────────────

    def _init_step(
        self,
        frame_idx: int,
        frame:     np.ndarray,
        bboxes:    Sequence[Tuple[float, float, float, float]],
        frame_h:   int,
    ) -> Tuple[Optional[tuple], Optional[tuple], int]:
        near_cand, far_cand = _candidate_pair(bboxes, frame_h)

        if near_cand is None or far_cand is None:
            # Not a "two clean players" frame — reset the initialisation buffer
            # so we only lock in references during a stretch of stable rally.
            self._init_buf.clear()
            self._raw_history.append(-1)
            return None, None, -1

        near_sig = extract_color_signature(frame, near_cand)
        far_sig  = extract_color_signature(frame, far_cand)
        if near_sig is None or far_sig is None:
            self._init_buf.clear()
            self._raw_history.append(-1)
            return None, None, -1

        self._init_buf.append((near_sig, far_sig))
        if len(self._init_buf) >= self.init_stable_frames:
            # Average colour signatures across the init window for robustness
            avg_near = np.mean([s[0] for s in self._init_buf], axis=0).astype(np.float32)
            avg_far  = np.mean([s[1] for s in self._init_buf], axis=0).astype(np.float32)
            self._sig_p1 = self._renormalise(avg_near)
            self._sig_p2 = self._renormalise(avg_far)
            self._init_buf.clear()
            print(f"[PlayerIdentity] Locked in references at frame {frame_idx}")
            self._raw_history.append(1)
            return near_cand, far_cand, 1

        self._raw_history.append(-1)
        return None, None, -1

    # ── Per-frame tracking ───────────────────────────────────────────────

    def _track_step(
        self,
        frame:   np.ndarray,
        bboxes:  Sequence[Tuple[float, float, float, float]],
        frame_h: int,
    ) -> Tuple[Optional[tuple], Optional[tuple], int]:
        # Score every detection against both references
        scored: List[Tuple[tuple, np.ndarray, float, float, float]] = []
        for b in bboxes:
            sig = extract_color_signature(frame, b)
            if sig is None:
                continue
            d1 = color_distance(sig, self._sig_p1)
            d2 = color_distance(sig, self._sig_p2)
            feet_y = (b[1] + b[3]) / 2.0
            scored.append((b, sig, d1, d2, feet_y))

        if not scored:
            self._raw_history.append(-1)
            return None, None, self._smoothed_at(len(self._raw_history) - 1)

        # Greedy assignment: best p1-match across all detections, then best
        # p2-match across the remaining ones. Reject matches above threshold.
        scored_p1 = sorted(scored, key=lambda s: s[2])
        p1_pick = scored_p1[0] if scored_p1[0][2] <= self.max_ref_distance else None

        remaining = [s for s in scored if s is not p1_pick]
        scored_p2 = sorted(remaining, key=lambda s: s[3])
        p2_pick = scored_p2[0] if scored_p2 and scored_p2[0][3] <= self.max_ref_distance else None

        # Decide whose side each is on, using feet_y vs midline
        def _side(item) -> str:
            return "near" if item[4] > frame_h * 0.5 else "far"

        near_bbox: Optional[tuple] = None
        far_bbox:  Optional[tuple] = None
        p1_on_near: Optional[bool] = None

        if p1_pick is not None:
            if _side(p1_pick) == "near":
                near_bbox = p1_pick[0]
                p1_on_near = True
            else:
                far_bbox = p1_pick[0]
                p1_on_near = False
        if p2_pick is not None:
            if _side(p2_pick) == "near":
                # If p1 was also on near, prefer p1 — discard p2 here
                if near_bbox is None:
                    near_bbox = p2_pick[0]
                    if p1_on_near is None:
                        p1_on_near = False
            else:
                if far_bbox is None:
                    far_bbox = p2_pick[0]
                    if p1_on_near is None:
                        p1_on_near = True

        # Update references via EMA only when we have a confident match
        if p1_pick is not None and p1_pick[2] < self.max_ref_distance:
            self._sig_p1 = self._blend(self._sig_p1, p1_pick[1])
        if p2_pick is not None and p2_pick[3] < self.max_ref_distance:
            self._sig_p2 = self._blend(self._sig_p2, p2_pick[1])

        if p1_on_near is None:
            self._raw_history.append(-1)
        else:
            self._raw_history.append(1 if p1_on_near else 0)

        return near_bbox, far_bbox, self._smoothed_at(len(self._raw_history) - 1)

    # ── Smoothing & changeover detection ─────────────────────────────────

    def _smoothed_at(self, sample_idx: int) -> int:
        if sample_idx < 0 or not self._raw_history:
            return -1
        lo = max(0, sample_idx - self.smooth_window + 1)
        window = [v for v in self._raw_history[lo : sample_idx + 1] if v != -1]
        if not window:
            return -1
        return 1 if sum(window) * 2 >= len(window) else 0

    def smoothed_history(self) -> List[Tuple[int, int]]:
        return [
            (self._frame_indices[i], self._smoothed_at(i))
            for i in range(len(self._raw_history))
        ]

    def get_changeover_frames(self) -> List[int]:
        if not self._raw_history:
            return []
        smoothed: List[int] = [self._smoothed_at(i) for i in range(len(self._raw_history))]

        changeovers: List[int] = []
        last_committed = next((v for v in smoothed if v != -1), 1)

        for i in range(1, len(smoothed)):
            cur = smoothed[i]
            if cur == -1 or cur == last_committed:
                continue

            trans_frame = self._frame_indices[i]
            persistence_end = trans_frame + self.persist_frames
            persistent = True
            agree = 0  # valid samples in the window confirming the new side
            for j in range(i + 1, len(smoothed)):
                if self._frame_indices[j] > persistence_end:
                    break
                if smoothed[j] == -1:
                    continue
                if smoothed[j] != cur:
                    persistent = False
                    break
                agree += 1

            # Commit only when nothing contradicts the flip AND enough valid
            # samples positively confirm it — so a lone flip during a sparse
            # detection gap (timeout/replay) isn't mistaken for a changeover.
            if persistent and agree >= self.min_persist_samples:
                changeovers.append(trans_frame)
                last_committed = cur

        return changeovers

    # ── EMA / renormalisation helpers ────────────────────────────────────

    @staticmethod
    def _blend(prev: np.ndarray, new: np.ndarray) -> np.ndarray:
        out = (1.0 - REFERENCE_LR) * prev + REFERENCE_LR * new
        return PlayerIdentityTracker._renormalise(out)

    @staticmethod
    def _renormalise(hist: np.ndarray) -> np.ndarray:
        s = float(np.sum(hist))
        return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers callers commonly need
# ─────────────────────────────────────────────────────────────────────────────

def near_is_player1_at(changeover_frames: List[int], frame_idx: int) -> bool:
    """True iff the player who started on the near side is still on the near side."""
    n_swaps = sum(1 for cf in changeover_frames if cf <= frame_idx)
    return (n_swaps % 2) == 0


def effective_handedness_at(
    changeover_frames: List[int],
    frame_idx:         int,
    side_at_frame:     str,    # "near" | "far"
    p1_handedness:     str,    # "R" | "L"  (player who started on near side)
    p2_handedness:     str,    # "R" | "L"  (player who started on far side)
) -> str:
    """Return the handedness of whichever physical player is on `side_at_frame`."""
    p1_on_near = near_is_player1_at(changeover_frames, frame_idx)
    if side_at_frame == "near":
        return p1_handedness if p1_on_near else p2_handedness
    else:
        return p2_handedness if p1_on_near else p1_handedness
