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
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _save_checkpoint(ckpt_path: Path, *, ball_x, ball_y, near_x, near_y, far_x, far_y,
                     frame_w, frame_h, fps, total_frames, video_stem, next_frame) -> None:
    """Atomically write a resumable checkpoint: the arrays as they stand plus the next
    unprocessed frame. Same layout as the final .npz, with next_frame/video_stem added so a
    re-run can validate + resume. Written to a temp file then os.replace'd (never half-written).
    """
    tmp = str(ckpt_path) + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            ball_x=ball_x, ball_y=ball_y,
            near_x=near_x, near_y=near_y,
            far_x=far_x,   far_y=far_y,
            frame_w=np.float32(frame_w), frame_h=np.float32(frame_h),
            fps=np.float32(fps),         total_frames=np.int32(total_frames),
            video_stem=np.array(video_stem), next_frame=np.int64(next_frame),
        )
    os.replace(tmp, ckpt_path)


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
    parser.add_argument("--court-roi", default=None,
                        help="Path to a court polygon JSON (from cv/tools/court_roi_editor.py). "
                             "Restricts players + ball to that one court — needed for multi-court "
                             "(college dual-match) footage; omit for clean single-court video.")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="Flush a resumable checkpoint every N processed frames (0 disables). "
                             "If the run dies, re-running the same command resumes from it.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore any existing checkpoint and start fresh.")
    args = parser.parse_args()

    # ── Determine output path ────────────────────────────────────────────────
    if args.output is None:
        stem        = Path(args.video).stem
        args.output = str(PROJECT_ROOT / "cv" / "training_data" / f"{stem}_features.npz")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint lives next to the output: <stem>_features.ckpt.npz
    video_stem = Path(args.video).stem
    ckpt_path  = Path(args.output).with_name(Path(args.output).stem + ".ckpt.npz")

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
    from cv.detection.wasb_ball_tracker import create_ball_tracker
    from cv.detection.player_detector import PlayerDetector

    ball_tracker = create_ball_tracker(device=args.device)  # WASB if available, else TrackNet
    player_det   = PlayerDetector(device=args.device)
    print("Models loaded.\n")

    # ── Allocate output arrays (or resume from a checkpoint) ──────────────────
    # All float32; NaN marks "not detected"
    ball_x = np.full(n_process, np.nan, dtype=np.float32)
    ball_y = np.full(n_process, np.nan, dtype=np.float32)
    near_x = np.full(n_process, np.nan, dtype=np.float32)
    near_y = np.full(n_process, np.nan, dtype=np.float32)
    far_x  = np.full(n_process, np.nan, dtype=np.float32)
    far_y  = np.full(n_process, np.nan, dtype=np.float32)

    resume_from = 0
    if args.restart and ckpt_path.exists():
        print(f"--restart: ignoring existing checkpoint {ckpt_path}")
    elif ckpt_path.exists():
        ck = None
        try:
            ck = np.load(ckpt_path)
        except Exception as e:
            print(f"WARNING: could not read checkpoint {ckpt_path} ({e}); starting fresh")
        if ck is not None:
            try:
                if (str(ck["video_stem"]) == video_stem
                        and int(ck["total_frames"]) == n_process
                        and ck["ball_x"].shape[0] == n_process):
                    ball_x, ball_y = ck["ball_x"].copy(), ck["ball_y"].copy()
                    near_x, near_y = ck["near_x"].copy(), ck["near_y"].copy()
                    far_x,  far_y  = ck["far_x"].copy(),  ck["far_y"].copy()
                    resume_from    = int(ck["next_frame"])
                    print(f"Resuming from checkpoint at frame {resume_from}/{n_process}")
                else:
                    print(f"WARNING: checkpoint {ckpt_path} is for a different video/length; "
                          f"starting fresh")
            finally:
                ck.close()

    # ── Main extraction loop ──────────────────────────────────────────────────
    # Player selection is MOTION-BASED so static objects (net-post chairs, ball
    # baskets, standing officials/ball-kids) can't be mistaken for the far player:
    # keep only detections that MOVE (accumulated frame-difference), then assign
    # near = lowest in frame, far = highest. Positional, not appearance — works even
    # when both players wear similar kit. (Clean single-court footage; multi-court
    # footage needs an added per-court region to drop adjacent-court players.)
    PLAYER_CONF_THR = 0.10    # low so the tiny far player is detected (ROI + motion filter the noise)
    PLAYER_IMGSZ    = 1280    # hi-res detection: at 640 the 1920px frame shrinks 3x and the ~20px far player vanishes
    MOTION_THR      = 2.5     # mean accumulated frame-diff below which a box is "static"
    MOTION_DECAY    = 0.9     # recent-motion memory (bridges a briefly-still player)

    # Optional per-court region to ignore adjacent-court players / off-court balls
    court_poly = None
    if args.court_roi:
        from cv.detection.court_roi import load_polygon, contains, expand_polygon
        court_poly = load_polygon(args.court_roi)
        if court_poly is None:
            print(f"WARNING: could not load court ROI '{args.court_roi}' — using full frame")
        else:
            court_poly = expand_polygon(court_poly)   # margin for players behind baselines + ball airspace
            print(f"Court ROI: {len(court_poly)}-point polygon (auto-margined) from {args.court_roi}")

    prev_gray:  Optional[np.ndarray] = None
    motion_acc: Optional[np.ndarray] = None
    ball_detected = int(np.count_nonzero(~np.isnan(ball_x)))

    # ── Resume: seek the video + re-warm the ball tracker's 3-frame window ─────
    loop_start = 0
    if resume_from > 0:
        warm_start = max(0, resume_from - 2)   # WASB needs the 2 prior frames buffered
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(warm_start))
        pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if pos > warm_start:                   # seek overshot → exact fallback from 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            pos = 0
        while pos < warm_start:                # decode-advance to warm_start (no NN)
            if not cap.read()[0]:
                break
            pos += 1
        loop_start = warm_start

    for fi in tqdm(range(loop_start, n_process), desc="Extracting features", unit="frame",
                   initial=loop_start, total=n_process):
        ret, frame = cap.read()
        if not ret:
            break

        if fi % args.frame_skip != 0:
            continue

        # — Ball (restricted to the court region if one was given) —
        result = ball_tracker.detect_ball(frame)   # also primes the deque during warm-up
        if fi < resume_from:
            continue                               # warm-up frame: already in the checkpoint
        if result is not None:
            center, _conf, _radius = result
            if court_poly is None or contains(court_poly, center[0], center[1]):
                ball_x[fi] = float(center[0])
                ball_y[fi] = float(center[1])
                ball_detected += 1

        # — Players: keep in-court detections, then near = lowest, far = highest —
        bboxes = player_det.run_human_detection(frame, bbox_thr=PLAYER_CONF_THR, imgsz=PLAYER_IMGSZ)
        incourt = [bb for bb in bboxes
                   if court_poly is None or contains(court_poly, (bb[0] + bb[2]) / 2.0, bb[3])]
        if len(incourt) >= 2:
            incourt.sort(key=lambda b: (b[1] + b[3]) / 2.0)   # ascending centroid-y
            fb, nb = incourt[0], incourt[-1]                   # top = far, bottom = near
            near_x[fi] = (nb[0] + nb[2]) / 2.0
            near_y[fi] = (nb[1] + nb[3]) / 2.0
            far_x[fi]  = (fb[0] + fb[2]) / 2.0
            far_y[fi]  = (fb[1] + fb[3]) / 2.0
        elif len(incourt) == 1:
            bb = incourt[0]
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0
            if cy > frame_h * 0.5:
                near_x[fi] = cx; near_y[fi] = cy
            else:
                far_x[fi]  = cx; far_y[fi]  = cy

        # — Periodic resumable checkpoint (never let a save failure kill the run) —
        if args.checkpoint_every and (fi + 1) % args.checkpoint_every == 0:
            try:
                _save_checkpoint(ckpt_path, ball_x=ball_x, ball_y=ball_y,
                                 near_x=near_x, near_y=near_y, far_x=far_x, far_y=far_y,
                                 frame_w=frame_w, frame_h=frame_h, fps=fps,
                                 total_frames=n_process, video_stem=video_stem,
                                 next_frame=fi + 1)
            except Exception as e:
                tqdm.write(f"WARNING: checkpoint save failed at frame {fi + 1}: {e}")

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

    # Extraction finished cleanly → drop the checkpoint so a re-run starts fresh.
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except OSError:
            pass

    detect_pct = 100.0 * ball_detected / max(n_process, 1)
    print(f"\nBall detected in {ball_detected}/{n_process} frames ({detect_pct:.1f}%)")
    print(f"Saved → {args.output}")
    print("\nNext steps:")
    print(f"  1. Annotate:  python cv/tools/annotate.py --video {args.video}")
    print(f"  2. Train:     python cv/tools/train_models.py")


if __name__ == "__main__":
    main()
