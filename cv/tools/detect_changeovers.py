#!/usr/bin/env python3
"""
cv/tools/detect_changeovers.py — Find court-changeover frames in a tennis video.

Uses cv/analysis/player_identity.PlayerIdentityTracker (colour signatures)
to detect when the two physical players swap court ends. Saves a JSON file
listing the changeover frame indices that the training pipeline and
inference pipeline both consume.

Output
------
    cv/training_data/<video_stem>_changeovers.json
    {
        "video_stem":     "Outdoor Match 1 15.53.25",
        "fps":             30.0,
        "frame_skip":      5,
        "changeovers":    [12453, 45678, 79123, ...],
        "n_samples":      8431
    }

Usage
-----
    # Default: sample every 5 frames (6fps), plenty for changeover detection
    python cv/tools/detect_changeovers.py --video tests/match1.mp4

    # Faster: sample every 10 frames
    python cv/tools/detect_changeovers.py --video tests/match1.mp4 --frame-skip 10

    # With MPS GPU
    python cv/tools/detect_changeovers.py --video tests/match1.mp4 --device mps

A changeover is reported only if the side-swap persists for ~10 seconds
afterwards, so brief detection noise (e.g. one player walking in front of
the other) won't be miscounted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.analysis.player_identity import PlayerIdentityTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect court-changeover frames using player colour tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video",       required=True, help="Path to input video file")
    parser.add_argument("--output",      default=None,  help="Output JSON path (default: cv/training_data/<stem>_changeovers.json)")
    parser.add_argument("--frame-skip",  type=int, default=5,
                        help="Process every Nth frame (5 = 6fps at 30fps source; good enough for changeover detection)")
    parser.add_argument("--device",      default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--max-frames",  type=int, default=None, help="Stop after N source frames (for testing)")
    args = parser.parse_args()

    if args.output is None:
        stem        = Path(args.video).stem
        args.output = str(PROJECT_ROOT / "cv" / "training_data" / f"{stem}_changeovers.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    n_process = min(total_frames, args.max_frames) if args.max_frames else total_frames

    print(f"Video        : {args.video}")
    print(f"Frames       : {total_frames}  (processing {n_process} at every Nth = {args.frame_skip})")
    print(f"Output       : {args.output}")
    print()

    print("Loading YOLO player detector …")
    from cv.detection.player_detector import PlayerDetector
    player_det = PlayerDetector(device=args.device)
    print("Loaded.\n")

    # Persistence threshold scales with frame rate: 10 seconds of real video.
    # We adjust the tracker's frame-count thresholds to match our sample rate.
    sample_fps   = fps / args.frame_skip
    smooth_win   = max(10, int(sample_fps * 3))   # ~3 s of samples
    persist_win  = max(30, int(sample_fps * 10))  # ~10 s of real video

    tracker = PlayerIdentityTracker(
        smooth_window  = smooth_win,
        persist_frames = persist_win,   # measured in source-frame units
    )

    # We iterate every Nth frame using cap.set for accurate seeks (cv2 .read in
    # tight loop on large videos can desync; .set per sample is safer).
    pbar = tqdm(range(0, n_process, args.frame_skip), desc="Identifying players", unit="sample")
    for fi in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Get ALL person detections, not just the two largest — the tracker
        # decides which ones are the real players by colour matching.
        detections = player_det.detect_players(frame, bbox_thr=0.4)
        bboxes     = [d[0] for d in detections]
        tracker.update(fi, frame, bboxes)

    cap.release()

    changeovers = tracker.get_changeover_frames()
    print(f"\nDetected {len(changeovers)} changeover(s):")
    for cf in changeovers:
        t = cf / fps
        print(f"   frame {cf:>7}   ({int(t // 60):02d}:{t % 60:05.2f})")

    payload = {
        "video_stem":  Path(args.video).stem,
        "fps":         fps,
        "frame_skip":  args.frame_skip,
        "changeovers": changeovers,
        "n_samples":   len(tracker._raw_history),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
