#!/usr/bin/env python3
"""
cv/tools/prelabel.py — Draft annotations from a video's extracted features.

Runs the trained event models (PointSegmenter: bounce / hit / stroke / point
start+end) on the ball + player tracks in <video>_features.npz and writes the
predicted events to cv/training_data/<video>_annotations.csv — the exact
6-column format annotate.py auto-loads. A human then CORRECTS these in
annotate.py instead of labelling from scratch.

Usage
-----
    python cv/tools/prelabel.py --video "/path/clip.mp4"
    # options: --near-hand R --far-hand L   (handedness for stroke mirroring)
    #          --features <npz>  --output <csv>

Note: homography is None here (feature extraction needs no court keypoints), so
bounce/hit/point FRAMES — the time-savers — are predicted, while point OUTCOMES
are rough guesses for the reviewer to confirm.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# PointRecord.outcome  ->  annotation point_end_type (the reviewer confirms/fixes)
OUTCOME_MAP = {
    "winner":    "winner",
    "error_net": "net_error",
    "error_out": "unforced_error",
    "in_play":   "unforced_error",
}


def build_court_homography(polygon) -> Optional[np.ndarray]:
    """4 court corners (pixels) -> normalised [0,1] court coords PointSegmenter expects
    (x: 0=left→1=right, y: 0=far baseline→1=near baseline, net≈0.5). Corners are assigned
    by position so the draw order in the editor doesn't matter."""
    if polygon is None or len(polygon) != 4:
        return None
    pts = polygon.astype(np.float32)
    order = pts[pts[:, 1].argsort()]          # sort by y (top→bottom)
    far, near = order[:2], order[2:]          # 2 smallest y = far, 2 largest y = near
    fl, fr = far[far[:, 0].argsort()]         # far-left, far-right
    nl, nr = near[near[:, 0].argsort()]       # near-left, near-right
    src = np.array([nl, nr, fr, fl], dtype=np.float32)
    dst = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _positions(d, xk: str, yk: str) -> List[Optional[Tuple[float, float]]]:
    xs, ys = d[xk], d[yk]
    out: List[Optional[Tuple[float, float]]] = []
    for i in range(len(xs)):
        out.append(None if (np.isnan(xs[i]) or np.isnan(ys[i])) else (float(xs[i]), float(ys[i])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draft annotations (pre-labels) from extracted features",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    ap.add_argument("--video", required=True, help="Video (used to locate features + name outputs)")
    ap.add_argument("--features", default=None, help="Features .npz (default: cv/training_data/<stem>_features.npz)")
    ap.add_argument("--output", default=None, help="Output CSV (default: cv/training_data/<stem>_annotations.csv)")
    ap.add_argument("--near-hand", default="R", help="Handedness of the player starting NEAR (R/L)")
    ap.add_argument("--far-hand", default="R", help="Handedness of the player starting FAR (R/L)")
    ap.add_argument("--poi-start-side", default="near", choices=["near", "far"])
    ap.add_argument("--court-roi", default=None, help="Court polygon JSON — builds a homography so the point logic has court coordinates (strongly recommended)")
    ap.add_argument("--max-frames", type=int, default=None, help="Only process the first N feature frames (for quick tests)")
    args = ap.parse_args()

    stem = Path(args.video).stem
    feat = Path(args.features) if args.features else (PROJECT_ROOT / "cv" / "training_data" / f"{stem}_features.npz")
    out  = Path(args.output) if args.output else (PROJECT_ROOT / "cv" / "training_data" / f"{stem}_annotations.csv")
    if not feat.exists():
        sys.exit(f"ERROR: features not found: {feat}\n  Run: python cv/tools/extract_features.py --video '{args.video}' ...")

    d = np.load(feat)
    fps = float(d["fps"])
    ball = _positions(d, "ball_x", "ball_y")
    near = _positions(d, "near_x", "near_y")
    far  = _positions(d, "far_x", "far_y")
    if args.max_frames:
        ball, near, far = ball[:args.max_frames], near[:args.max_frames], far[:args.max_frames]
    print(f"Loaded features: {len(ball)} frames @ {fps:.1f} fps  ({feat.name})")

    H = None
    if args.court_roi:
        from cv.detection.court_roi import load_polygon
        H = build_court_homography(load_polygon(args.court_roi))
        print(f"Court homography from ROI corners: {'built' if H is not None else 'FAILED (need a 4-corner polygon)'}")

    from cv.analysis.point_detector import PointSegmenter
    seg = PointSegmenter(
        fps=fps,
        player_start_side=args.poi_start_side,
        homography=H,
        use_ml_bounce=True,
        use_ml_hit=True,
        near_handedness=args.near_hand,
        far_handedness=args.far_hand,
        changeover_frames=[],
    )
    points = seg.run(ball, near, far)

    rows: List[Tuple[int, str, str, str]] = []   # (frame, event_type, stroke_type, point_end_type)
    for pt in points:
        rows.append((int(pt.start_frame), "point_start", "", ""))
        for s in pt.shots:
            rows.append((int(s.frame_idx), "hit", s.shot_type or "", ""))
        for b in pt.bounces:
            rows.append((int(b.frame_idx), "bounce", "", ""))
        rows.append((int(pt.end_frame), "point_end", "", OUTCOME_MAP.get(pt.outcome, "unforced_error")))
    rows.sort(key=lambda r: r[0])

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "frame", "event_type", "stroke_type", "point_end_type", "player"])
        for frame, et, stroke, endtype in rows:
            w.writerow([stem, frame, et, stroke, endtype, ""])

    n_by_type = {}
    for _, et, _, _ in rows:
        n_by_type[et] = n_by_type.get(et, 0) + 1
    print(f"Wrote {len(rows)} predicted events from {len(points)} points -> {out}")
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(n_by_type.items())))
    print(f"\nNext: review in annotate.py --\n  python cv/tools/annotate.py --video '{args.video}' --output '{out}'")


if __name__ == "__main__":
    main()
