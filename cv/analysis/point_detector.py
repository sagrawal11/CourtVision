"""
Point detection and rally segmentation for tennis match analysis.

This module implements a state machine that segments a continuous match video
into discrete points/rallies using ball tracking data, player positions,
and court geometry.

Pipeline:
    1. BounceDetector   – identifies ball bounce events from xy-trajectory
    2. PointStateMachine – segments video into: SERVING → RALLY → POINT_OVER → CHANGEOVER
    3. PointSegmenter   – orchestrates above + produces list of Point records
"""

from __future__ import annotations

import dataclasses
import enum
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class RallyState(enum.Enum):
    """The current phase within a tennis point."""
    IDLE         = "idle"         # Between points / startup
    SERVING      = "serving"      # Server tossing / motion
    RALLY        = "rally"        # Ball in play
    POINT_OVER   = "point_over"   # Rally just ended (brief)
    CHANGEOVER   = "changeover"   # Long break between games


@dataclasses.dataclass
class BounceEvent:
    """A detected ball bounce on the court surface."""
    frame_idx: int
    x: float                   # court-space x (pixel in original frame)
    y: float                   # court-space y (pixel in original frame)
    court_x: float             # normalised 0-1 court coordinate, x
    court_y: float             # normalised 0-1 court coordinate, y
    is_in_bounds: bool


@dataclasses.dataclass
class PointRecord:
    """A complete characterisation of a single tennis point."""
    point_idx: int
    start_frame: int
    end_frame: int

    serve_player: str          # "near" | "far"  (relative to camera)
    outcome: str               # "winner" | "error_net" | "error_out" | "in_play"
    error_player: Optional[str]     # who made the error or "?"

    bounces: List[BounceEvent] = dataclasses.field(default_factory=list)
    serve_bounce: Optional[BounceEvent] = None

    # Shot-level data (populated later by shot classifier)
    shots: List[dict] = dataclasses.field(default_factory=list)

    @property
    def rally_length(self) -> int:
        return len(self.shots)


# ---------------------------------------------------------------------------
# Bounce Detector
# ---------------------------------------------------------------------------

class BounceDetector:
    """
    Detects ball bounces from a time-series of (x, y) ball positions.

    A bounce is characterised by:
      - Smooth arc downward (y increasing toward ground in image coordinates)
      - Local minimum in y, followed by upward trajectory
      - Trajectory consistent with parabolic motion (not artefact jumps)

    This is a heuristic approach that works well without requiring the
    s-ganguli LSTM model (which would need separate training data).
    When we have a pre-trained LSTM checkpoint, swap it in via
    BounceDetector(use_lstm=True, model_path=...).
    """

    def __init__(
        self,
        window: int = 7,
        min_drop_px: float = 8.0,
        min_rise_px: float = 5.0,
        use_lstm: bool = False,
        model_path: Optional[str] = None,
    ):
        self.window = window
        self.min_drop_px = min_drop_px
        self.min_rise_px = min_rise_px
        self._use_lstm = use_lstm
        self._lstm: Optional[object] = None

        if use_lstm and model_path:
            self._load_lstm(model_path)

    def _load_lstm(self, model_path: str) -> None:
        """Load the s-ganguli LSTM bounce predictor (optional)."""
        try:
            import torch
            import torch.nn as nn

            class BounceLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
                    self.fc = nn.Linear(64, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.sigmoid(self.fc(out[:, -1, :]))

            model = BounceLSTM()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            self._lstm = model
            print(f"✓ Bounce LSTM loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load bounce LSTM: {e}. Falling back to heuristic.")
            self._use_lstm = False

    def detect(
        self,
        positions: List[Optional[Tuple[float, float]]],
        fps: float = 30.0,
    ) -> List[Tuple[int, float, float]]:
        """
        Main entry point. Takes a list of (x,y) or None per frame and returns
        a list of (frame_idx, x, y) bounce events.

        Args:
            positions: Per-frame ball (x, y) or None if ball not detected.
            fps: Video frame rate (used for velocity sanity checks).

        Returns:
            List of (frame_idx, x, y) for each detected bounce.
        """
        if self._use_lstm and self._lstm is not None:
            return self._detect_lstm(positions)
        return self._detect_heuristic(positions, fps)

    def _interpolate(
        self, positions: List[Optional[Tuple[float, float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill gaps in ball trajectory via linear interpolation.
        Returns (xs, ys) as numpy arrays with NaN where ball is not detected.
        """
        xs = np.full(len(positions), np.nan)
        ys = np.full(len(positions), np.nan)

        for i, p in enumerate(positions):
            if p is not None:
                xs[i], ys[i] = p

        # Interpolate short gaps (≤ 10 frames)
        known = np.where(~np.isnan(xs))[0]
        if len(known) < 2:
            return xs, ys

        for start, end in zip(known[:-1], known[1:]):
            if 0 < (end - start) <= 10:
                xs[start:end+1] = np.interp(
                    np.arange(start, end + 1), [start, end], [xs[start], xs[end]]
                )
                ys[start:end+1] = np.interp(
                    np.arange(start, end + 1), [start, end], [ys[start], ys[end]]
                )
        return xs, ys

    def _detect_heuristic(
        self,
        positions: List[Optional[Tuple[float, float]]],
        fps: float,
    ) -> List[Tuple[int, float, float]]:
        """Parabola-valley based bounce detection."""
        xs, ys = self._interpolate(positions)
        bounces = []

        half = self.window // 2
        for i in range(half, len(ys) - half):
            if np.isnan(ys[i]):
                continue

            before = ys[max(0, i - half) : i]
            after  = ys[i + 1 : min(len(ys), i + half + 1)]

            before = before[~np.isnan(before)]
            after  = after[~np.isnan(after)]

            if len(before) < 2 or len(after) < 2:
                continue

            # In image coordinates, y increases downward.
            # A bounce = ball descends (y rises) then ascends (y falls).
            drop = ys[i] - before.min()    # how much y rose before this frame
            rise = ys[i] - after.min()     # how much y will fall after

            # Check for local y-maximum (bounce peak in image coords)
            if drop >= self.min_drop_px and rise >= self.min_rise_px:
                # Deduplicate: skip if we already have a bounce within 10 frames
                if bounces and abs(i - bounces[-1][0]) < 10:
                    # Keep the one with bigger combined drop+rise
                    prev_i = bounces[-1][0]
                    prev_drop = ys[prev_i] - ys[max(0, prev_i - half):prev_i][~np.isnan(ys[max(0, prev_i - half):prev_i])].min() if not np.all(np.isnan(ys[max(0, prev_i - half):prev_i])) else 0
                    prev_rise = ys[prev_i] - ys[prev_i + 1:min(len(ys), prev_i + half + 1)][~np.isnan(ys[prev_i + 1:min(len(ys), prev_i + half + 1)])].min() if not np.all(np.isnan(ys[prev_i + 1:min(len(ys), prev_i + half + 1)])) else 0
                    if drop + rise > prev_drop + prev_rise:
                        bounces[-1] = (i, float(xs[i]), float(ys[i]))
                else:
                    bounces.append((i, float(xs[i]), float(ys[i])))

        return bounces

    def _detect_lstm(
        self, positions: List[Optional[Tuple[float, float]]]
    ) -> List[Tuple[int, float, float]]:
        """Run the LSTM bounce predictor on a sliding window."""
        import torch
        xs, ys = self._interpolate(positions)
        bounces = []
        seq_len = 16  # window size LSTM expects

        coords = np.stack([xs, ys], axis=1)  # (N, 2)
        # Normalise to 0-1 range
        coord_max = np.nanmax(np.abs(coords)) or 1.0
        coords_norm = coords / coord_max

        for i in range(seq_len, len(positions)):
            seq = coords_norm[i - seq_len : i]
            if np.any(np.isnan(seq)):
                continue
            inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = self._lstm(inp).item()
            if prob > 0.5:
                if bounces and abs(i - bounces[-1][0]) < 8:
                    continue
                bounces.append((i, float(xs[i]), float(ys[i])))

        return bounces


# ---------------------------------------------------------------------------
# Point State Machine
# ---------------------------------------------------------------------------

class PointStateMachine:
    """
    Segments a tennis video into discrete points.

    Signals consumed (all optional; machine is robust to missing data):
        - ball_positions: List[Optional[Tuple[float, float]]]
        - bounce_events:  List[Tuple[int, float, float]]  from BounceDetector
        - player_positions: List[Optional[Tuple[float, float]]]  (feet centroids per frame, 2 players)
        - homography: np.ndarray (3x3)  to map pixels → court coords
        - court_height_px: reference distance for "in-bounds" checks

    Parameters:
        fps: Video frame rate.
        min_stillness_frames: Frames both players must be still to declare changeover.
        max_ball_gap_frames: Ball tracker gap beyond which rally is considered over.
        player_start_side: "near" (bottom of frame) or "far" (top) for the target player at start.
    """

    # How many frames of missing ball before we conclude point is over
    MAX_BALL_GAP = 45        # ~1.5s at 30fps
    # How many frames of player stillness to declare a changeover
    CHANGEOVER_STILL_FRAMES = 90   # ~3s
    # Minimum rally frames (filter out detection glitches)
    MIN_RALLY_FRAMES = 15
    # Frames between consecutive points
    MIN_BETWEEN_POINTS = 30

    def __init__(
        self,
        fps: float = 30.0,
        player_start_side: str = "near",
        homography: Optional[np.ndarray] = None,
        video_height: Optional[int] = None,
    ):
        self.fps = fps
        self.player_start_side = player_start_side
        self.homography = homography
        self.video_height = video_height

        self._state = RallyState.IDLE
        self._point_start_frame: Optional[int] = None
        self._last_ball_seen_frame: Optional[int] = None
        self._serve_side = player_start_side
        self._points: List[PointRecord] = []
        self._game_count = 0

    def process(
        self,
        ball_positions: List[Optional[Tuple[float, float]]],
        bounce_events: Optional[List[Tuple[int, float, float]]] = None,
        player_near_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
        player_far_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> List[PointRecord]:
        """
        Main entry: process an entire match worth of tracking data.

        Returns list of PointRecord (one per detected point/rally).
        """
        bounce_set = {b[0]: b for b in (bounce_events or [])}
        n_frames = len(ball_positions)

        # Running state
        state = RallyState.IDLE
        point_start: Optional[int] = None
        last_ball_frame: Optional[int] = None
        consecutive_still = 0
        point_bounces: List[BounceEvent] = []
        point_idx = 0

        serve_side = self.player_start_side
        last_point_end: Optional[int] = None

        for i, ball in enumerate(ball_positions):
            ball_visible = ball is not None

            if ball_visible:
                last_ball_frame = i

            # ── Bounce event at this frame ──────────────────────────────
            if i in bounce_set:
                bx, by = bounce_set[i][1], bounce_set[i][2]
                court_x, court_y, in_bounds = self._classify_bounce(bx, by)
                bounce_evt = BounceEvent(
                    frame_idx=i, x=bx, y=by,
                    court_x=court_x, court_y=court_y,
                    is_in_bounds=in_bounds,
                )
                point_bounces.append(bounce_evt)

            # ── Player stillness (changeover heuristic) ─────────────────
            near_still = self._is_still(player_near_positions, i)
            far_still  = self._is_still(player_far_positions,  i)
            both_still = near_still and far_still

            if both_still:
                consecutive_still += 1
            else:
                consecutive_still = 0

            # ── State transitions ────────────────────────────────────────

            if state == RallyState.IDLE:
                if ball_visible:
                    # Could be a serve toss
                    if last_point_end is None or i - last_point_end >= self.MIN_BETWEEN_POINTS:
                        state = RallyState.SERVING
                        point_start = i
                        point_bounces = []

            elif state == RallyState.SERVING:
                if ball_visible:
                    state = RallyState.RALLY
                elif last_ball_frame and i - last_ball_frame > self.MAX_BALL_GAP:
                    # Ball lost immediately after serve — probably a fault
                    state = RallyState.POINT_OVER

            elif state == RallyState.RALLY:
                ball_gap = i - last_ball_frame if last_ball_frame else 0

                # Changeover (both players stood still for a while)
                if consecutive_still >= self.CHANGEOVER_STILL_FRAMES:
                    state = RallyState.CHANGEOVER

                # Ball has been lost for too long → point over
                elif ball_gap >= self.MAX_BALL_GAP:
                    state = RallyState.POINT_OVER

            if state == RallyState.POINT_OVER and point_start is not None:
                duration = i - point_start
                if duration >= self.MIN_RALLY_FRAMES:
                    outcome = self._classify_outcome(point_bounces)
                    record = PointRecord(
                        point_idx=point_idx,
                        start_frame=point_start,
                        end_frame=i,
                        serve_player=serve_side,
                        outcome=outcome,
                        error_player=None,
                        bounces=list(point_bounces),
                        serve_bounce=point_bounces[0] if point_bounces else None,
                    )
                    self._points.append(record)
                    point_idx += 1
                    last_point_end = i

                state = RallyState.IDLE
                point_start = None

            elif state == RallyState.CHANGEOVER:
                if point_start is not None:
                    duration = i - point_start
                    if duration >= self.MIN_RALLY_FRAMES:
                        record = PointRecord(
                            point_idx=point_idx,
                            start_frame=point_start,
                            end_frame=i,
                            serve_player=serve_side,
                            outcome="in_play",
                            error_player=None,
                            bounces=list(point_bounces),
                        )
                        self._points.append(record)
                        point_idx += 1

                # Alternate serve side at each game
                serve_side = "far" if serve_side == "near" else "near"
                self._game_count += 1
                state = RallyState.IDLE
                point_start = None
                consecutive_still = 0

        return self._points

    # ── Helpers ─────────────────────────────────────────────────────────

    def _is_still(
        self,
        positions: Optional[List[Optional[Tuple[float, float]]]],
        frame_idx: int,
        lookback: int = 10,
        threshold_px: float = 15.0,
    ) -> bool:
        """Return True if the player hasn't moved more than threshold_px in the last N frames."""
        if positions is None or frame_idx < lookback:
            return False
        window = [positions[j] for j in range(frame_idx - lookback, frame_idx + 1)
                  if positions[j] is not None]
        if len(window) < 3:
            return False
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        return (max(xs) - min(xs)) < threshold_px and (max(ys) - min(ys)) < threshold_px

    def _classify_bounce(
        self, px: float, py: float
    ) -> Tuple[float, float, bool]:
        """
        Map pixel (px, py) to normalised court coordinates (0-1) via homography.
        Returns (court_x, court_y, in_bounds).
        """
        if self.homography is None:
            # No homography: treat as in-bounds
            return 0.5, 0.5, True

        pt = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.homography)[0][0]
        court_x, court_y = float(mapped[0]), float(mapped[1])

        # A standard court is 23.77m × 10.97m (singles) with doubles 23.77×13.40
        # We normalise 0-1 over the full court bbox, so:
        #   in-bounds = x in [0,1] and y in [0,1]  (roughly)
        in_bounds = 0.0 <= court_x <= 1.0 and 0.0 <= court_y <= 1.0

        return court_x, court_y, in_bounds

    def _classify_outcome(self, bounces: List[BounceEvent]) -> str:
        """
        Infer the point outcome from the final bounce:
          - out-of-bounds bounce → error_out
          - No bounce at all → error_net (ball didn't reach other side)
          - In-bounds, no return → winner (handled later by shot classifier)
        """
        if not bounces:
            return "error_net"

        final_bounce = bounces[-1]
        if not final_bounce.is_in_bounds:
            return "error_out"

        # Without shot classifier, we can't distinguish winner from in-play
        return "in_play"


# ---------------------------------------------------------------------------
# High-level PointSegmenter
# ---------------------------------------------------------------------------

class PointSegmenter:
    """
    Orchestrates BounceDetector + PointStateMachine over a processed frame sequence.

    Usage:
        segmenter = PointSegmenter(fps=fps, player_start_side="near")
        points = segmenter.run(ball_positions, player_near_positions, player_far_positions)
    """

    def __init__(
        self,
        fps: float = 30.0,
        player_start_side: str = "near",
        homography: Optional[np.ndarray] = None,
        bounce_lstm_path: Optional[str] = None,
    ):
        self.fps = fps
        self.player_start_side = player_start_side

        use_lstm = bounce_lstm_path is not None
        self.bounce_detector = BounceDetector(use_lstm=use_lstm, model_path=bounce_lstm_path)
        self.state_machine = PointStateMachine(
            fps=fps,
            player_start_side=player_start_side,
            homography=homography,
        )

    def run(
        self,
        ball_positions: List[Optional[Tuple[float, float]]],
        player_near_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
        player_far_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> List[PointRecord]:
        """
        Full segmentation pipeline.
        Returns list of PointRecord sorted by start_frame.
        """
        print(f"[PointSegmenter] Detecting bounces from {len(ball_positions)} frames...")
        bounce_events = self.bounce_detector.detect(ball_positions, self.fps)
        print(f"[PointSegmenter] Found {len(bounce_events)} bounce events")

        print("[PointSegmenter] Running point state machine...")
        points = self.state_machine.process(
            ball_positions=ball_positions,
            bounce_events=bounce_events,
            player_near_positions=player_near_positions,
            player_far_positions=player_far_positions,
        )
        print(f"[PointSegmenter] Segmented {len(points)} points/rallies")

        return sorted(points, key=lambda p: p.start_frame)


# ---------------------------------------------------------------------------
# Convenience: import cv2 only when needed
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore
