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
from cv.detection.wasb_ball_tracker import create_ball_tracker
from cv.detection.player_detector import PlayerDetector
from cv.analysis.court_zones import classify as classify_zone, CourtZone
from cv.analysis.point_detector import PointSegmenter
from cv.analysis.match_stats import MatchStatsAggregator

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
    match_stats: Optional[Dict[str, Any]] = None   # populated after stats aggregation
    shots: List[Dict[str, Any]] = field(default_factory=list)
    _points: List[Any] = field(default_factory=list, repr=False)  # PointRecord list for DB storage


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AnalyticsPipeline:
    """
    Lightweight analytics pipeline. Significantly faster than the old hero
    video generator — no 3D body estimation, no video output, just data.

    On a MacBook with MPS, expect ~5-15 fps processing speed for a 1080p video.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        poi_start_side: str = "near",
        near_handedness: str = "R",
        far_handedness:  str = "R",
    ):
        import torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.poi_start_side = poi_start_side   # 'near' | 'far'
        self.near_handedness = str(near_handedness).upper()
        self.far_handedness  = str(far_handedness).upper()
        logger.info(
            f"AnalyticsPipeline using device: {device}, poi_start_side: {poi_start_side}, "
            f"handedness: near={self.near_handedness}, far={self.far_handedness}"
        )

        self.ball_tracker = create_ball_tracker(device=device)  # WASB if available, else TrackNet
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
        roi_polygon=None,
    ) -> AnalysisResult:
        """
        Run the full analytics pipeline on a video file.

        Args:
            video_path: Path to the video file (local).
            court_keypoints: Pre-confirmed 14-point list from the court editor
                             (manual override). If None, the court is auto-detected
                             from the best-fitting frame of the video.
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
        # Auto-detect the static court by default (best-fitting frame across the
        # clip). Manual keypoints from the court editor, when provided, override
        # this — auto is the default, manual is the fallback/nudge.
        if court_keypoints is None:
            logger.info("No manual keypoints — auto-detecting court from best frame")
            court_keypoints, court_score = self.court_detector.detect_best_frame(
                video_path, roi_polygon=roi_polygon
            )
            if not court_score.get("trustworthy"):
                logger.warning(
                    f"Auto court detection is NOT trustworthy ({court_score}) — likely a "
                    "multi-court view or occluded court. Court-space stats may be wrong; "
                    "confirm keypoints in the court editor."
                )

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

        # Live player-identity tracker — populated during the per-frame loop
        # and queried after to extract changeover frame indices.
        from cv.analysis.player_identity import PlayerIdentityTracker
        identity_tracker = PlayerIdentityTracker()

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

                # Player detection — get EVERY person detected, then let the
                # identity tracker decide which two are the actual players.
                # This filters out coaches, ball kids, refs, spectators, etc.
                detections: List[Tuple[Tuple[float, float, float, float], float]] = []
                try:
                    detections = self.player_detector.detect_players(frame)
                except Exception as e:
                    logger.debug(f"Player detection failed on frame {frame_idx}: {e}")

                bboxes_only = [d[0] for d in detections]
                near_bb, far_bb, _ = identity_tracker.update(frame_idx, frame, bboxes_only)

                # Build PlayerState entries for only the two tracker-verified players.
                # player_id 0 = near, player_id 1 = far — stable across the whole video.
                def _conf_for(bb):
                    if bb is None:
                        return 0.0
                    for det_bb, det_conf in detections:
                        if det_bb == bb:
                            return det_conf
                    return 0.0

                for pid, bbox in ((0, near_bb), (1, far_bb)):
                    if bbox is None:
                        continue
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
                        confidence=float(_conf_for(bbox)),
                        zone=p_zone.name if p_zone else None,
                    ))

                result.frames.append(frame_result)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        detected_changeovers = identity_tracker.get_changeover_frames()
        if detected_changeovers:
            logger.info(f"Detected {len(detected_changeovers)} court changeover(s): {detected_changeovers}")
        logger.info(f"Analysis complete. {len(result.frames)} frames processed.")

        # ── Step 3: Post-processing — point segmentation + match stats ──────────
        try:
            ball_positions = [
                (f.ball.x, f.ball.y) if f.ball else None
                for f in result.frames
            ]

            # Collect per-player positions by physical side. player_id 0 is
            # always "near" and player_id 1 is always "far" by construction
            # (PlayerIdentityTracker assigns it that way).
            near_positions: List[Optional[Tuple[float, float]]] = []
            far_positions:  List[Optional[Tuple[float, float]]] = []
            for f in result.frames:
                near_pos = None
                far_pos  = None
                for p in f.players:
                    if p.center_x is None:
                        continue
                    if p.player_id == 0:
                        near_pos = (p.center_x, p.center_y)
                    elif p.player_id == 1:
                        far_pos = (p.center_x, p.center_y)
                near_positions.append(near_pos)
                far_positions.append(far_pos)

            # Auto-use trained ML models when available
            _ml_bounce = (PROJECT_ROOT / "cv" / "models" / "bounce_model.cbm").exists()
            _ml_hit    = (PROJECT_ROOT / "cv" / "models" / "hit_model.cbm").exists()
            if _ml_bounce or _ml_hit:
                logger.info("Trained ML models found — using CatBoost detectors")

            segmenter = PointSegmenter(
                fps=fps,
                player_start_side=self.poi_start_side,
                homography=H,
                use_ml_bounce=_ml_bounce,
                use_ml_hit=_ml_hit,
                near_handedness=self.near_handedness,
                far_handedness=self.far_handedness,
                changeover_frames=detected_changeovers,
            )
            points = segmenter.run(ball_positions, near_positions, far_positions)

            aggregator = MatchStatsAggregator(poi_start_side=self.poi_start_side)
            stats = aggregator.aggregate(points)
            result.match_stats = stats.to_dict()

            # Store PointRecord list so analysis_job can write them to the points table
            result._points = points

            # Flatten shots — tag each with point_idx so they link back to a point row.
            # is_winner/is_error are intentionally omitted: coaches classify via review UI.
            for pt in points:
                for s in pt.shots:
                    result.shots.append({
                        "frame": s.frame_idx,
                        "point_idx": pt.point_idx,
                        "x": s.court_x if s.court_x is not None else s.x,
                        "y": s.court_y if s.court_y is not None else s.y,
                        "player": s.player,
                        "speed_kmh": s.speed_kmh,
                        "shot_type": s.shot_type,
                    })

            logger.info(
                f"Stats: {len(points)} points, "
                f"POI {stats.poi_points_won}/{stats.total_points} points won, "
                f"Detected {len(result.shots)} total shots"
            )
        except Exception as e:
            logger.warning(f"Stats aggregation failed (non-fatal): {e}")

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
        "--poi-start-side",
        default="near",
        choices=["near", "far"],
        help="Which side of the court the target player starts on (near=bottom, far=top of frame)",
    )
    parser.add_argument(
        "--near-handedness", choices=["R", "L"], default="R",
        help="Handedness of the player who starts on the near side (R=right, L=left)",
    )
    parser.add_argument(
        "--far-handedness",  choices=["R", "L"], default="R",
        help="Handedness of the player who starts on the far side (R=right, L=left)",
    )
    parser.add_argument(
        "--court-roi", default=None,
        help="Path to a court polygon JSON (cv/tools/court_roi_editor.py). Crops to one "
             "court for keypoint auto-detection on multi-court footage.",
    )
    args = parser.parse_args()

    roi_polygon = None
    if args.court_roi:
        from cv.detection.court_roi import load_polygon
        roi_polygon = load_polygon(args.court_roi)
        if roi_polygon is None:
            logger.warning(f"Could not load court ROI from {args.court_roi}")

    pipeline = AnalyticsPipeline(
        device=args.device,
        poi_start_side=args.poi_start_side,
        near_handedness=args.near_handedness,
        far_handedness=args.far_handedness,
    )
    result = pipeline.process(
        video_path=args.input,
        match_id=args.match_id,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        roi_polygon=roi_polygon,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print(f"\n✓ Results written to {out_path}")
    print(f"  {len(result.frames)} frames with ball + player detections")


if __name__ == "__main__":
    main()
