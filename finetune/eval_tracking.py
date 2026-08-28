#!/usr/bin/env python3
"""finetune/eval_tracking.py — does a tracking/trajectory layer help on ep12's clean detections?

Compares, against our GT (indoor + outdoor labeled segs), three trajectories built from the
SAME per-frame ep12 detections:
  raw    — ep12 detections as-is
  track  — link into motion tracks, drop short/static (residual clutter), linear-fill gaps
  arc    — the existing physics engine (cv.analysis.ball_trajectory.clean_trajectory:
           static-clutter drop + per-run bounce-agnostic parabola fit + densify)

Metric (native px, tol=15):
  recall    = GT ball frames with a trajectory position within tol / all GT ball frames
  precision = trajectory positions within tol of a GT ball / all trajectory positions
              (a position on a GT no-ball frame, or far from the GT ball, is a false positive)

    python finetune/eval_tracking.py --device mps
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from cv.tools.autolabel import run_wasb
from cv.analysis.ball_trajectory import clean_trajectory
from cv.detection.court_roi import load_polygon, expand_polygon

EP12 = "models/ball/wasb_stage2_ep12.pth.tar"
SEGS = [
    ("indoor",  "tests/Indoor Match 2 15.53.25.mp4",  None, None),   # filled below from annotations? no — use Match1 train GT
]
# We have ball-position GT on the TRAIN segs (labeled by hand):
GTSEGS = [
    ("indoor",  "tests/Indoor Match 1 15.53.25.mp4",  "cv/court_rois/Indoor_Match.json",
     93400, 700,  "cv/training_data/ball_labels/indoor1_labels.csv"),
    ("outdoor", "tests/Outdoor Match 1 15.53.25.mp4", "cv/court_rois/Outdoor_Match_1.json",
     10845, 1351, "cv/training_data/ball_labels/outdoor1_seg10845_labels.csv"),
]

SPEED, GAP, MIN_LEN, STATIC, MAX_FILL = 150.0, 5, 4, 40.0, 8


def load_gt(path):
    return {int(r["frame"]): (float(r["x"]), float(r["y"]))
            for r in csv.DictReader(open(path)) if (r.get("visibility") or "").strip() == "1"}


def track_and_fill(xs, ys):
    """Link detections into motion tracks, drop short/static (residual clutter), linear-fill
    gaps <= MAX_FILL within a track. Returns {frame_idx: (x, y, 'det'|'fill')}."""
    N = len(xs)
    dets = [(i, float(xs[i]), float(ys[i])) for i in range(N) if not np.isnan(xs[i])]
    tracks, cur = [], []
    for d in dets:
        if cur and (d[0] - cur[-1][0] <= GAP) and (np.hypot(d[1] - cur[-1][1], d[2] - cur[-1][2]) <= SPEED * (d[0] - cur[-1][0])):
            cur.append(d)
        else:
            if cur: tracks.append(cur)
            cur = [d]
    if cur: tracks.append(cur)
    traj = {}
    for t in tracks:
        xl = [p[1] for p in t]; yl = [p[2] for p in t]
        moving = (max(xl) - min(xl)) >= STATIC or (max(yl) - min(yl)) >= STATIC
        if len(t) < MIN_LEN or not moving:
            continue
        for i, x, y in t: traj[i] = (x, y, "det")
        fi = [p[0] for p in t]
        for a, b in zip(fi[:-1], fi[1:]):
            if 1 < b - a <= MAX_FILL:
                xa, ya = traj[a][0], traj[a][1]; xb, yb = traj[b][0], traj[b][1]
                for k in range(a + 1, b):
                    f = (k - a) / (b - a); traj[k] = (xa + f * (xb - xa), ya + f * (yb - ya), "fill")
    return traj


def score(traj_abs, gt, tol=15.0):
    """traj_abs: {abs_frame: (x,y,...)}. gt: {abs_frame:(x,y)}."""
    rec = sum(1 for f, (gx, gy) in gt.items()
              if f in traj_abs and np.hypot(traj_abs[f][0] - gx, traj_abs[f][1] - gy) <= tol)
    pos = list(traj_abs)
    prec = sum(1 for f in pos
               if f in gt and np.hypot(traj_abs[f][0] - gt[f][0], traj_abs[f][1] - gt[f][1]) <= tol)
    recall = rec / max(1, len(gt)); precision = prec / max(1, len(pos))
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return recall, precision, f1, len(pos)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--tol", type=float, default=15.0)
    args = ap.parse_args()
    for tag, video, roi, start, n, gtpath in GTSEGS:
        exp = expand_polygon(load_polygon(roi))
        gt = load_gt(gtpath)
        xs, ys, cf = run_wasb(video, start, n, args.device, exp, EP12)
        # raw
        raw = {start + i: (float(xs[i]), float(ys[i])) for i in range(n) if not np.isnan(xs[i])}
        # track+fill
        tr = {start + k: v for k, v in track_and_fill(xs, ys).items()}
        # arc (physics engine)
        atr_rel, _clut, _arcs = clean_trajectory(xs, ys)
        atr = {start + k: v for k, v in atr_rel.items()}
        print(f"\n=== {tag}  (GT ball-present {len(gt)}, seg {n} frames) ===")
        print(f"{'method':>7} | recall  prec   f1   | positions")
        for name, traj in [("raw", raw), ("track", tr), ("arc", atr)]:
            r, p, f1, npos = score(traj, gt, args.tol)
            print(f"{name:>7} |  {r:.3f}  {p:.3f} {f1:.3f} | {npos}")


if __name__ == "__main__":
    main()
