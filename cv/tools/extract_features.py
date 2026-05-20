#!/usr/bin/env python3
"""
cv/tools/extract_features.py — Extract per-frame CV features from a tennis video.

Runs BallTracker + PlayerDetector on every frame and saves the raw tracking
arrays to a compressed NumPy file.  These features are the inputs used by
train_models.py together with the annotation CSVs produced by annotate.py.

Output (.npz) arrays
--------------------
  ball_x, ball_y         float32  (N,)   pixel coords; NaN where ball not visible
  near_x, near_y         float32  (N,)   near-player centroid; NaN when not detected
  far_x,  far_y          float32  (N,)   far-player centroid;  NaN when not detected
  frame_w, frame_h       float32  scalar video dimensions (needed for normalisation)
  fps                    float32  scalar
  total_frames           int32    scalar

Usage
-----
    python cv/tools/extract_features.py --video tests/match1.mov

    # Speed up with frame-skip (positions are filled via interpolation in train_models.py)
    python cv/tools/extract_features.py --video tests/match1.mov --frame-skip 2

Output is written to cv/training_data/<video_stem>_features.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-frame ball/player positions for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video",      required=True, help="Input video path")
    parser.add_argument("--output",     default=None,  help="Output .npz path")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (2 = half the work, minor accuracy loss)")
    parser.add_argument("--device",     default=None, choices=["cuda", "mps", "cpu"],
                        help="Torch device (default: auto)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (for quick testing)")
    args = parser.parse_args()

    # ── Determine output path ────────────────────────────────────────────────
    if args.output is None:
        stem        = Path(args.video).stem
        args.output = str(PROJECT_ROOT / "cv" / "training_data" / f"{stem}_features.npz")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_process    = min(total_frames, args.max_frames) if args.max_frames else total_frames
    print(f"Video   : {args.video}")
    print(f"Frames  : {total_frames}  (processing {n_process})   FPS: {fps:.1f}   {frame_w}×{frame_h}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models …")
    from cv.detection.ball_tracker import BallTracker
    from cv.detection.player_detector import PlayerDetector

    ball_tracker = BallTracker(device=args.device)
    player_det   = PlayerDetector(device=args.device)
    print("Models loaded.\n")

    # ── Allocate output arrays ────────────────────────────────────────────────
    # All float32; NaN marks "not detected"
    ball_x = np.full(n_process, np.nan, dtype=np.float32)
    ball_y = np.full(n_process, np.nan, dtype=np.float32)
    near_x = np.full(n_process, np.nan, dtype=np.float32)
    near_y = np.full(n_process, np.nan, dtype=np.float32)
    far_x  = np.full(n_process, np.nan, dtype=np.float32)
    far_y  = np.full(n_process, np.nan, dtype=np.float32)

    # ── Main extraction loop ──────────────────────────────────────────────────
    ball_detected = 0

    for fi in tqdm(range(n_process), desc="Extracting features", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        if fi % args.frame_skip != 0:
            continue

        # — Ball —
        result = ball_tracker.detect_ball(frame)
        if result is not None:
            center, _conf, _radius = result
            ball_x[fi] = float(center[0])
            ball_y[fi] = float(center[1])
            ball_detected += 1

        # — Players (near = larger y centroid = closer to camera baseline) —
        bboxes = player_det.run_human_detection(frame, bbox_thr=0.35)
        if len(bboxes) >= 2:
            # Sort descending by centroid-y so index-0 is the "near" player
            sorted_bb = sorted(bboxes[:2], key=lambda b: (b[1] + b[3]) / 2, reverse=True)
            nb, fb = sorted_bb[0], sorted_bb[1]
            near_x[fi] = (nb[0] + nb[2]) / 2.0
            near_y[fi] = (nb[1] + nb[3]) / 2.0
            far_x[fi]  = (fb[0] + fb[2]) / 2.0
            far_y[fi]  = (fb[1] + fb[3]) / 2.0
        elif len(bboxes) == 1:
            bb = bboxes[0]
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0
            if cy > frame_h * 0.5:
                near_x[fi] = cx; near_y[fi] = cy
            else:
                far_x[fi]  = cx; far_y[fi]  = cy

    cap.release()

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez_compressed(
        args.output,
        ball_x=ball_x,  ball_y=ball_y,
        near_x=near_x,  near_y=near_y,
        far_x=far_x,    far_y=far_y,
        frame_w=np.float32(frame_w),
        frame_h=np.float32(frame_h),
        fps=np.float32(fps),
        total_frames=np.int32(n_process),
    )

    detect_pct = 100.0 * ball_detected / max(n_process, 1)
    print(f"\nBall detected in {ball_detected}/{n_process} frames ({detect_pct:.1f}%)")
    print(f"Saved → {args.output}")
    print("\nNext steps:")
    print(f"  1. Annotate:  python cv/tools/annotate.py --video {args.video}")
    print(f"  2. Train:     python cv/tools/train_models.py")


if __name__ == "__main__":
    main()
