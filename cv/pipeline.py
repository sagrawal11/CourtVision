"""
cv/pipeline.py — Tennis Analytics CV Pipeline

Orchestrates per-frame processing across all three detection modules:
  1. Court detection  → homography matrix (run once, cached for entire video)
  2. Ball detection   → TrackNet 3-frame sliding window
  3. Player detection → YOLO bounding boxes

Outputs structured per-frame data that can be written to Supabase or
returned as a JSON result for the analysis API.

Usage (standalone, for local testing):
    python cv/pipeline.py --input path/to/video.mp4 --output results.json

Usage (as a module):
    from cv.pipeline import AnalyticsPipeline
    pipeline = AnalyticsPipeline()
    results = pipeline.process(video_path, court_keypoints=kps)
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.detection.court_detector import CourtDetector
from cv.detection.ball_tracker import BallTracker
from cv.detection.player_detector import PlayerDetector
from cv.analysis.court_zones import classify as classify_zone, CourtZone

logger = logging.getLogger(__name__)

# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BallState:
    """Ball position in a single frame."""
    frame: int
    x: Optional[float]        # Pixel x in original video frame (or None if not detected)
    y: Optional[float]        # Pixel y in original video frame (or None if not detected)
    court_x: Optional[float]  # Normalised court x (0=left doubles sideline, 1=right)
    court_y: Optional[float]  # Normalised court y (0=far baseline, 1=near baseline)
    confidence: float = 0.0
    zone: Optional[str] = None  # Court zone name from court_zones.classify(), e.g. "near_service_left_tee"


@dataclass
class PlayerState:
    """Player bounding box in a single frame."""
    frame: int
    player_id: int            # 0 = near-court player, 1 = far-court player
    bbox: Optional[Tuple[int, int, int, int]]   # (x1, y1, x2, y2) in video pixels
    center_x: Optional[float]  # Bbox centre x in video pixels
    center_y: Optional[float]  # Bbox centre y in video pixels
    court_x: Optional[float]   # Normalised court x (0–1)
    court_y: Optional[float]   # Normalised court y (0–1)
    confidence: float = 0.0
    zone: Optional[str] = None  # Court zone the player's centre falls in


@dataclass
class FrameResult:
    """All detections for a single frame."""
    frame: int
    timestamp_ms: float
    ball: Optional[BallState] = None
    players: List[PlayerState] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Full video analysis output."""
    match_id: Optional[str]
    video_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    court_keypoints: List[Optional[Tuple[float, float]]]
    frames: List[FrameResult] = field(default_factory=list)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AnalyticsPipeline:
    """
    Lightweight analytics pipeline. Significantly faster than the old hero
    video generator — no 3D body estimation, no video output, just data.

    On a MacBook with MPS, expect ~5-15 fps processing speed for a 1080p video.
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        logger.info(f"AnalyticsPipeline using device: {device}")

        self.ball_tracker = BallTracker(device=device)
        self.player_detector = PlayerDetector(device=device)
        self.court_detector = CourtDetector(device=device)

    def _build_homography(
        self,
        keypoints: List[Optional[Tuple[float, float]]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[np.ndarray]:
        """
        Build a homography matrix mapping video pixel coordinates→ normalised court space.

        src_pts: detected keypoint pixel coordinates in the video frame
        dst_pts: corresponding anchor positions from CourtDetector.REFERENCE_KEYPOINTS

        The raw output of cv2.perspectiveTransform will be in the court reference
        pixel space (x: 286–1379, y: 561–2935). _apply_homography then normalises
        that to 0–1 using the REF_X/Y_MIN/MAX bounds.

        Returns None if fewer than 4 valid keypoints are provided.
        """
        from cv.detection.court_detector import REF_X_MIN, REF_X_MAX, REF_Y_MIN, REF_Y_MAX

        ref = CourtDetector.REFERENCE_KEYPOINTS  # list of 14 (x, y) in reference pixel space
        src_pts, dst_pts = [], []
        for i, kp in enumerate(keypoints):
            if kp is not None and kp[0] is not None:
                src_pts.append([float(kp[0]), float(kp[1])])
                dst_pts.append([float(ref[i][0]), float(ref[i][1])])

        if len(src_pts) < 4:
            logger.warning(f"Only {len(src_pts)} keypoints — need ≥4 for homography")
            return None

        H, mask = cv2.findHomography(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            cv2.RANSAC,
            5.0,
        )
        # Store normalisation bounds on the matrix so _apply_homography can use them
        # without repeating the import
        if H is not None:
            self._ref_bounds = (REF_X_MIN, REF_X_MAX, REF_Y_MIN, REF_Y_MAX)
        return H

    def _apply_homography(
        self,
        H: Optional[np.ndarray],
        x: Optional[float],
        y: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Transform a video-pixel (x, y) to normalised court coordinates (0-1).

        The homography H maps video pixels → reference pixel space.
        We then normalise using the court reference bounds so the output
        is in the [0, 1] range that court_zones.classify() expects.
        """
        if H is None or x is None or y is None:
            return None, None
        pt = np.array([[[x, y]]], dtype=np.float32)
        raw = cv2.perspectiveTransform(pt, H)[0, 0]  # still in reference pixel space
        bounds = getattr(self, "_ref_bounds", (286, 1379, 561, 2935))
        x_min, x_max, y_min, y_max = bounds
        cx = (float(raw[0]) - x_min) / (x_max - x_min)
        cy = (float(raw[1]) - y_min) / (y_max - y_min)
        return cx, cy

    def process(
        self,
        video_path: str | Path,
        court_keypoints: Optional[List[Optional[Tuple[float, float]]]] = None,
        match_id: Optional[str] = None,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        auto_detect_court: bool = False,
    ) -> AnalysisResult:
        """
        Run the full analytics pipeline on a video file.

        Args:
            video_path: Path to the video file (local).
            court_keypoints: Pre-confirmed 14-point list from the court editor.
                             If None, court detection runs on the first frame.
            match_id: Supabase match UUID (stored in output for reference).
            frame_skip: Process every Nth frame (1 = every frame).
            max_frames: Stop after this many frames (for testing).

        Returns:
            AnalysisResult with per-frame ball and player detections.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

        # ── Step 1: Resolve court keypoints ──────────────────────────────────
        # In the final product, court_keypoints ALWAYS come from the manual court editor.
        # AI auto-detection is an explicit opt-in (auto_detect_court=True) for local testing only.
        if court_keypoints is None:
            if not auto_detect_court:
                raise ValueError(
                    "court_keypoints are required. "
                    "Either pass confirmed keypoints from Supabase or set auto_detect_court=True "
                    "to fall back to the AI detector (local testing only)."
                )
            logger.warning("auto_detect_court=True — using AI detection (not for production)")
            ret, first_frame = cap.read()
            if ret:
                court_keypoints = self.court_detector.detect_court_in_frame(
                    first_frame, apply_homography=True
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                court_keypoints = [(None, None)] * 14

        H = self._build_homography(court_keypoints, width, height)
        if H is not None:
            logger.info("Homography matrix built successfully")
        else:
            logger.warning("Could not build homography — court coordinates will be null")

        # ── Step 2: Per-frame processing ──────────────────────────────────────
        result = AnalysisResult(
            match_id=match_id,
            video_path=str(video_path),
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            court_keypoints=court_keypoints,
        )

        frames_to_process = min(
            total_frames,
            max_frames if max_frames else total_frames,
        )

        frame_idx = 0
        with tqdm(total=frames_to_process // frame_skip, desc="Analysing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_idx >= frames_to_process:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = (frame_idx / fps) * 1000.0
                frame_result = FrameResult(frame=frame_idx, timestamp_ms=timestamp_ms)

                # Ball detection
                try:
                    ball_det = self.ball_tracker.detect_ball(frame)
                    if ball_det:
                        center, conf, _ = ball_det
                        cx, cy = self._apply_homography(H, center[0], center[1])
                        zone_name = classify_zone(cx, cy).name if (cx is not None and cy is not None and classify_zone(cx, cy)) else None
                        frame_result.ball = BallState(
                            frame=frame_idx,
                            x=float(center[0]),
                            y=float(center[1]),
                            court_x=cx,
                            court_y=cy,
                            confidence=float(conf),
                            zone=zone_name,
                        )
                except Exception as e:
                    logger.debug(f"Ball detection failed on frame {frame_idx}: {e}")

                # Player detection
                try:
                    players = self.player_detector.detect_players(frame)
                    for pid, det in enumerate(players[:2]):   # Max 2 players
                        if det is None:
                            continue
                        bbox, conf = det
                        x1, y1, x2, y2 = bbox
                        px, py = (x1 + x2) / 2, (y1 + y2) / 2
                        pcx, pcy = self._apply_homography(H, px, py)
                        p_zone = classify_zone(pcx, pcy)
                        frame_result.players.append(PlayerState(
                            frame=frame_idx,
                            player_id=pid,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center_x=float(px),
                            center_y=float(py),
                            court_x=pcx,
                            court_y=pcy,
                            confidence=float(conf),
                            zone=p_zone.name if p_zone else None,
                        ))
                except Exception as e:
                    logger.debug(f"Player detection failed on frame {frame_idx}: {e}")

                result.frames.append(frame_result)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        logger.info(f"Analysis complete. {len(result.frames)} frames processed.")
        return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Tennis Analytics CV Pipeline")
    parser.add_argument("--input", required=True, help="Path to video file")
    parser.add_argument("--output", default="results.json", help="Path to output JSON")
    parser.add_argument("--match-id", default=None, help="Supabase match UUID")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit for testing")
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument(
        "--auto-detect-court",
        action="store_true",
        help="Use AI court detection instead of manual keypoints (for local testing only)",
    )
    args = parser.parse_args()

    pipeline = AnalyticsPipeline(device=args.device)
    result = pipeline.process(
        video_path=args.input,
        match_id=args.match_id,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        auto_detect_court=args.auto_detect_court,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print(f"\n✓ Results written to {out_path}")
    print(f"  {len(result.frames)} frames with ball + player detections")


if __name__ == "__main__":
    main()
