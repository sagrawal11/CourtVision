"""
cv/analysis/visualizer.py — Debug video renderer for CV pipeline verification.

Generates an annotated video overlaying:
  - Court lines (drawn from the 14 confirmed keypoints)
  - Court zone boundaries (all 24 zones, colour-coded with labels)
  - Ball tracking (TrackNet — 30-frame trail)
  - Player bounding boxes (YOLO)

All ML detection is optional — the script gracefully falls back to court
lines + zones only if models are unavailable. This is intentional: the most
important thing to verify first is that the court geometry is correct.

Usage (CLI):
    python cv/analysis/visualizer.py \\
        --input path/to/video.mp4 \\
        --keypoints "0.1,0.2 0.9,0.2 ..." \\  # 14 x,y pairs space-separated
        --output debug.mp4 \\
        --max-seconds 30

Or pass keypoints as a JSON file:
    python cv/analysis/visualizer.py --input video.mp4 --keypoints-json kps.json

The keypoints must be in video pixel space (same format as stored in Supabase
court_configs: kp0_x, kp0_y, ..., kp13_x, kp13_y).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.analysis.court_zones import ALL_ZONES, CourtZone
from cv.detection.court_detector import REF_X_MIN, REF_X_MAX, REF_Y_MIN, REF_Y_MAX

logger = logging.getLogger(__name__)

# ── Court line connectivity ────────────────────────────────────────────────────
# Pairs of keypoint indices that should be connected with a line.
# Derived from CourtReference key_points ordering:
#  0: far-left baseline (doubles)    1: far-right baseline (doubles)
#  2: near-left baseline (doubles)   3: near-right baseline (doubles)
#  4: far-left singles sideline      5: near-left singles sideline
#  6: far-right singles sideline     7: near-right singles sideline
#  8: far service line left          9: far service line right
# 10: near service line left        11: near service line right
# 12: center service line far       13: center service line near

COURT_LINES = [
    (0, 1),   # Far baseline (doubles width)
    (2, 3),   # Near baseline (doubles width)
    (0, 2),   # Left doubles sideline (full length)
    (1, 3),   # Right doubles sideline (full length)
    (4, 5),   # Left singles sideline (full length)
    (6, 7),   # Right singles sideline (full length)
    (8, 9),   # Far service line
    (10, 11), # Near service line
    (12, 13), # Center service line
    # Net segments (derived from midpoints — kps 4/6 at y=0.5 between their endpoints)
    # We draw the net as a separate overlay from zone corners, handled in _draw_net()
]

# Zone colours: baseline zones in blue tones, service zones in orange/green tones
_BASELINE_COLOUR = (200, 120, 50)    # BGR – blue-ish
_SERVICE_L_COLOUR = (50, 180, 50)    # BGR – green
_SERVICE_R_COLOUR = (50, 100, 220)   # BGR – orange-red
_ALLEY_COLOUR = (80, 80, 180)        # BGR – purple for alleys

ZONE_COLOURS = {
    "AA": _ALLEY_COLOUR, "DD": _ALLEY_COLOUR,
    "A":  _BASELINE_COLOUR, "B": _BASELINE_COLOUR,
    "C":  _BASELINE_COLOUR, "D": _BASELINE_COLOUR,
    "wide": _SERVICE_L_COLOUR, "body": _SERVICE_L_COLOUR, "tee": _SERVICE_R_COLOUR,
}

# Overlay opacity for zone fills
ZONE_ALPHA = 0.18


# ── Homography helpers ────────────────────────────────────────────────────────

def _build_h(keypoints: list[tuple[float, float]]) -> Optional[np.ndarray]:
    """
    Build homography from video pixel space → reference pixel space.
    Uses REFERENCE_KEYPOINTS from CourtDetector as the dst (reference) anchors.
    """
    from cv.detection.court_detector import CourtDetector
    ref = CourtDetector.REFERENCE_KEYPOINTS

    src, dst = [], []
    for i, kp in enumerate(keypoints):
        if kp is not None and kp[0] is not None:
            src.append([float(kp[0]), float(kp[1])])
            dst.append([float(ref[i][0]), float(ref[i][1])])

    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    return H


def _normalize(ref_x: float, ref_y: float) -> tuple[float, float]:
    """Convert reference pixel coords to normalised 0-1."""
    return (ref_x - REF_X_MIN) / (REF_X_MAX - REF_X_MIN), (ref_y - REF_Y_MIN) / (REF_Y_MAX - REF_Y_MIN)


def _ref_to_video(H_inv: np.ndarray, ref_x: float, ref_y: float) -> Optional[tuple[int, int]]:
    """Project a reference pixel coordinate into video pixel space."""
    pt = np.array([[[ref_x, ref_y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_inv)
    x, y = int(out[0, 0, 0]), int(out[0, 0, 1])
    return x, y


def _norm_to_video(H_inv: np.ndarray, nx: float, ny: float) -> tuple[int, int]:
    """Project a normalised (0-1) court coordinate into video pixel space."""
    ref_x = nx * (REF_X_MAX - REF_X_MIN) + REF_X_MIN
    ref_y = ny * (REF_Y_MAX - REF_Y_MIN) + REF_Y_MIN
    return _ref_to_video(H_inv, ref_x, ref_y)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_court_lines(frame: np.ndarray, keypoints: list, colour=(255, 255, 255), thickness=2) -> None:
    """Draw the 9 major court line segments between the 14 keypoints."""
    for i, j in COURT_LINES:
        kp_i = keypoints[i] if i < len(keypoints) else None
        kp_j = keypoints[j] if j < len(keypoints) else None
        if kp_i and kp_i[0] is not None and kp_j and kp_j[0] is not None:
            cv2.line(frame,
                     (int(kp_i[0]), int(kp_i[1])),
                     (int(kp_j[0]), int(kp_j[1])),
                     colour, thickness, cv2.LINE_AA)


def _draw_net(frame: np.ndarray, H_inv: np.ndarray, colour=(200, 200, 200), thickness=3) -> None:
    """Draw the net by projecting two points at y=0.5 from both doubles sidelines."""
    left  = _norm_to_video(H_inv, 0.0, 0.5)
    right = _norm_to_video(H_inv, 1.0, 0.5)
    cv2.line(frame, left, right, colour, thickness, cv2.LINE_AA)


def _draw_zone_overlay(
    frame: np.ndarray,
    H_inv: np.ndarray,
    alpha: float = ZONE_ALPHA,
) -> None:
    """
    Draw semi-transparent filled polygons for all 24 court zones.
    Colour-codes by zone type and adds a small text label in the zone centre.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for zone in ALL_ZONES:
        # 4 corners of the zone rectangle in normalised space
        corners_norm = [
            (zone.x1, zone.y1),
            (zone.x2, zone.y1),
            (zone.x2, zone.y2),
            (zone.x1, zone.y2),
        ]
        pts_video = [_norm_to_video(H_inv, nx, ny) for nx, ny in corners_norm]

        # Clip to frame bounds
        pts_arr = np.array(pts_video, dtype=np.int32)
        colour = ZONE_COLOURS.get(zone.sub_zone, (150, 150, 150))
        cv2.fillPoly(overlay, [pts_arr], colour)
        cv2.polylines(frame, [pts_arr], isClosed=True, color=colour, thickness=1, lineType=cv2.LINE_AA)

    # Blend overlay onto frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw zone labels at the zone centroid in video space
    for zone in ALL_ZONES:
        cx_n = (zone.x1 + zone.x2) / 2
        cy_n = (zone.y1 + zone.y2) / 2
        cx_v, cy_v = _norm_to_video(H_inv, cx_n, cy_n)
        if 0 <= cx_v < w and 0 <= cy_v < h:
            text = zone.sub_zone.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.35
            thickness_t = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness_t)
            cv2.putText(frame, text,
                        (cx_v - tw // 2, cy_v + th // 2),
                        font, scale, (255, 255, 255), thickness_t, cv2.LINE_AA)


def _draw_keypoint_dots(frame: np.ndarray, keypoints: list, radius=5, colour=(0, 255, 255)) -> None:
    """Draw numbered dots for each of the 14 keypoints."""
    for i, kp in enumerate(keypoints):
        if kp is not None and kp[0] is not None:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), radius, colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(i), (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_ball(frame: np.ndarray, center: tuple[int, int], trail: list,
               colour=(50, 200, 50), radius=8) -> None:
    """Draw ball with trajectory trail."""
    for i, pt in enumerate(trail):
        alpha_t = int(255 * (i + 1) / len(trail))
        tr_colour = (0, alpha_t // 2, alpha_t)
        cv2.circle(frame, pt, max(2, radius - 4), tr_colour, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, colour, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_players(frame: np.ndarray, bboxes: list, colour=(50, 200, 50)) -> None:
    """Draw player bounding boxes."""
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"P{i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_debug_video(
    input_path: str | Path,
    output_path: str | Path,
    keypoints: list[tuple[float, float]],   # 14 (x, y) in video pixel space
    max_seconds: Optional[float] = 60.0,
    enable_ball_tracking: bool = True,
    enable_player_tracking: bool = True,
    trail_length: int = 30,
) -> None:
    """
    Render a debug video annotated with court lines, zone overlays,
    ball tracking, and player bounding boxes.

    Args:
        input_path:              Source video file
        output_path:             Output video file (mp4)
        keypoints:               14 (x, y) pairs in video pixel space
        max_seconds:             Stop after this many seconds (None = full video)
        enable_ball_tracking:    Use TrackNet for ball detection (needs model)
        enable_player_tracking:  Use YOLO for player detection (needs model)
        trail_length:            Number of frames in ball trail
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build homography
    H = _build_h(keypoints)
    if H is None:
        logger.warning("Could not build homography — zone overlay will be skipped")
        H_inv = None
    else:
        H_inv = np.linalg.inv(H)

    # Load optional detectors
    ball_tracker = None
    if enable_ball_tracking:
        try:
            from cv.detection.ball_tracker import BallTracker
            ball_model = PROJECT_ROOT / "models" / "ball" / "pretrained_ball_detection.pt"
            ball_tracker = BallTracker(model_path=str(ball_model) if ball_model.exists() else None)
            logger.info("BallTracker loaded")
        except Exception as e:
            logger.warning(f"Ball tracker unavailable: {e}")

    player_detector = None
    if enable_player_tracking:
        try:
            from cv.detection.player_detector import PlayerDetector
            yolo_model = PROJECT_ROOT / "models" / "player" / "playersnball5.pt"
            if yolo_model.exists():
                player_detector = PlayerDetector(model_path=yolo_model)
                logger.info("PlayerDetector loaded")
        except Exception as e:
            logger.warning(f"Player detector unavailable: {e}")

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * fps) if max_seconds else total_frames

    logger.info(f"Input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames total")
    logger.info(f"Rendering up to {max_frames} frames...")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    ball_trail: list[tuple[int, int]] = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Zone overlay (only if homography available)
        if H_inv is not None:
            _draw_zone_overlay(frame, H_inv)

        # 2. Court lines
        _draw_court_lines(frame, keypoints)

        # 3. Net
        if H_inv is not None:
            _draw_net(frame, H_inv)

        # 4. Keypoint dots (on first 100 frames only — so you can verify placement)
        if frame_idx < 100:
            _draw_keypoint_dots(frame, keypoints)

        # 5. Ball tracking
        if ball_tracker is not None:
            try:
                result = ball_tracker.detect_ball(frame)
                if result:
                    center, conf, _ = result
                    ball_trail.append(center)
                    if len(ball_trail) > trail_length:
                        ball_trail.pop(0)
                    _draw_ball(frame, center, ball_trail[:-1])
                elif ball_trail:
                    ball_trail.pop(0)
            except Exception:
                pass

        # 6. Player detection
        if player_detector is not None:
            try:
                bboxes = player_detector.run_human_detection(frame, bbox_thr=0.4)
                if len(bboxes) > 0:
                    _draw_players(frame, bboxes[:2])  # max 2 players
            except Exception:
                pass

        # 7. Frame counter HUD
        elapsed_s = frame_idx / fps
        cv2.putText(frame,
                    f"Frame {frame_idx}  |  {elapsed_s:.1f}s",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    "COURT ZONES DEBUG" + (" + BALL" if ball_tracker else "") + (" + PLAYERS" if player_detector else ""),
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            logger.info(f"  {frame_idx}/{max_frames} frames rendered ({frame_idx/fps:.0f}s elapsed)")

    cap.release()
    out.release()
    logger.info(f"Debug video saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_keypoints(raw: str) -> list[tuple[float, float]]:
    """Parse keypoints from space-separated 'x,y' string."""
    pairs = raw.strip().split()
    return [tuple(float(v) for v in p.split(",")) for p in pairs]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate debug annotation video")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="debug_output.mp4", help="Output video path")
    parser.add_argument("--keypoints", help="14 keypoints as 'x1,y1 x2,y2 ...' in video pixel space")
    parser.add_argument("--keypoints-json", help="JSON file with {kp0_x, kp0_y, ..., kp13_x, kp13_y}")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Max seconds to render")
    parser.add_argument("--no-ball", action="store_true", help="Skip ball tracking")
    parser.add_argument("--no-players", action="store_true", help="Skip player detection")
    args = parser.parse_args()

    # Resolve keypoints
    if args.keypoints_json:
        with open(args.keypoints_json) as f:
            data = json.load(f)
        keypoints = [(data.get(f"kp{i}_x"), data.get(f"kp{i}_y")) for i in range(14)]
    elif args.keypoints:
        keypoints = _parse_keypoints(args.keypoints)
    else:
        parser.error("Either --keypoints or --keypoints-json is required")

    render_debug_video(
        input_path=args.input,
        output_path=args.output,
        keypoints=keypoints,
        max_seconds=args.max_seconds,
        enable_ball_tracking=not args.no_ball,
        enable_player_tracking=not args.no_players,
    )


if __name__ == "__main__":
    main()
