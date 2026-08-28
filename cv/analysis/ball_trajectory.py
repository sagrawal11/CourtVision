"""cv/analysis/ball_trajectory.py — physics-informed ball trajectory estimation.

Two-stage cleanup of noisy WASB detections, exploiting ball physics:

  1. STATIC-RECURRENCE filter — a fixed object (parked car, windscreen, line marker) makes
     the detector fire at the SAME pixel across many frames; the ball MOVES, so each of its
     positions is visited once. Detections whose (x,y) recurs at >= min_recur frames are
     clutter, dropped. (Empirically removes the parking-lot band cleanly.)

  2. Temporally-LOCAL ballistic arcs — group the surviving (moving) detections into
     contiguous runs (gap <= max_gap frames), and within each run robustly fit a parabola
     (image x,y are ~quadratic in frame over one flight), dropping outliers. A run is kept
     as an arc only if it has enough inliers, spans enough frames, AND actually moves.
     Positions between inliers are then physically determined -> densify to fill misses.

Used by autolabel.py (cleaner + denser candidates) and the tracking layer.
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np


def track_and_fill(xs, ys, speed=150.0, gap=5, min_len=4, static=40.0, max_fill=8):
    """Light tracking layer for CLEAN per-frame detections (e.g. the Stage-2 ep12 WASB).
    Link detections into motion tracks, drop short/static ones (residual clutter), and
    linear-fill gaps <= max_fill within a track. Returns {frame: (x, y, 'det'|'fill')}.

    Validated vs hand GT on ep12 detections (finetune/eval_tracking.py): OUTDOOR
    recall 0.848->0.901 at ~equal precision (f1 0.876->0.899); INDOOR ~neutral
    (recall 0.799->0.820, f1 flat). This SIMPLE linear layer is the right one on clean
    detections — the parabola engine below (clean_trajectory) collapses to f1 0.20-0.31
    (multi-flight runs don't fit one parabola; static-recurrence over-flags a real rally).
    The detector does the heavy lifting; this only polishes."""
    N = len(xs)
    dets = [(i, float(xs[i]), float(ys[i])) for i in range(N) if not np.isnan(xs[i])]
    tracks, cur = [], []
    for d in dets:
        if cur and (d[0] - cur[-1][0] <= gap) and (np.hypot(d[1] - cur[-1][1], d[2] - cur[-1][2]) <= speed * (d[0] - cur[-1][0])):
            cur.append(d)
        else:
            if cur: tracks.append(cur)
            cur = [d]
    if cur: tracks.append(cur)
    traj: Dict[int, Tuple[float, float, str]] = {}
    for t in tracks:
        xl = [p[1] for p in t]; yl = [p[2] for p in t]
        moving = (max(xl) - min(xl)) >= static or (max(yl) - min(yl)) >= static
        if len(t) < min_len or not moving:
            continue
        for i, x, y in t:
            traj[i] = (x, y, "det")
        fi = [p[0] for p in t]
        for a, b in zip(fi[:-1], fi[1:]):
            if 1 < b - a <= max_fill:
                xa, ya = traj[a][0], traj[a][1]; xb, yb = traj[b][0], traj[b][1]
                for k in range(a + 1, b):
                    f = (k - a) / (b - a)
                    traj[k] = (xa + f * (xb - xa), ya + f * (yb - ya), "fill")
    return traj


def _static_clutter(dets, radius=12, min_recur=6) -> Set[int]:
    clutter = set()
    for f, x, y in dets:
        c = sum(1 for f2, x2, y2 in dets if f2 != f and abs(x - x2) < radius and abs(y - y2) < radius)
        if c >= min_recur:
            clutter.add(f)
    return clutter


def _contiguous_runs(dets, max_gap):
    runs, cur = [], []
    for d in sorted(dets):
        if cur and d[0] - cur[-1][0] > max_gap:
            runs.append(cur); cur = []
        cur.append(d)
    if cur:
        runs.append(cur)
    return runs


def _fit_arc(run, gate):
    """Robust quadratic fit (x,y vs frame); iteratively drop the worst outlier until all
    residuals <= gate. Returns (cx, cy, inliers) or None."""
    pts = list(run)
    for _ in range(len(pts)):
        if len(pts) < 3:
            return None
        t = np.array([p[0] for p in pts], float)
        cx = np.polyfit(t, [p[1] for p in pts], 2)
        cy = np.polyfit(t, [p[2] for p in pts], 2)
        res = [max(abs(np.polyval(cx, p[0]) - p[1]), abs(np.polyval(cy, p[0]) - p[2])) for p in pts]
        w = int(np.argmax(res))
        if res[w] <= gate:
            return cx, cy, pts
        pts = pts[:w] + pts[w + 1:]
    return None


def extract_ballistic_arcs(dets, gate=14.0, min_inliers=5, min_span=6, max_gap=15,
                           min_motion=30.0, static_radius=12, static_min_recur=6):
    static = _static_clutter(dets, static_radius, static_min_recur)
    live = [d for d in dets if d[0] not in static]
    arcs, used = [], set()
    for run in _contiguous_runs(live, max_gap):
        if len(run) < min_inliers:
            continue
        fit = _fit_arc(run, gate)
        if fit is None:
            continue
        cx, cy, inl = fit
        if len(inl) < min_inliers:
            continue
        span = inl[-1][0] - inl[0][0]
        motion = (max(p[1] for p in inl) - min(p[1] for p in inl)) + \
                 (max(p[2] for p in inl) - min(p[2] for p in inl))
        if span < min_span or motion < min_motion:
            continue
        arcs.append({"inliers": inl, "cx": cx, "cy": cy})
        used.update(p[0] for p in inl)
    clutter = [d[0] for d in dets if d[0] not in used]
    return arcs, clutter


def densify(arcs, max_fill=25) -> Dict[int, Tuple[float, float, str]]:
    traj: Dict[int, Tuple[float, float, str]] = {}
    for a in arcs:
        inl = a["inliers"]; det_frames = {p[0] for p in inl}
        for (fa, _, _), (fb, _, _) in zip(inl, inl[1:]):
            if fb - fa <= max_fill:
                for f in range(fa, fb + 1):
                    traj[f] = (float(np.polyval(a["cx"], f)), float(np.polyval(a["cy"], f)),
                               "det" if f in det_frames else "fill")
        for f, _, _ in inl:
            traj[f] = (float(np.polyval(a["cx"], f)), float(np.polyval(a["cy"], f)), "det")
    return traj


def clean_trajectory(xs, ys, **kw):
    """xs, ys: per-frame arrays (NaN where no detection). Returns (traj, clutter_frames, arcs).
    traj: frame -> (x, y, 'det'|'fill'). clutter_frames: rejected detection frames.

    ⚠️ EXPERIMENTAL / not recommended. Validated vs GT (finetune/eval_tracking.py) it
    collapses to f1 0.20 (indoor) / 0.31 (outdoor) even on CLEAN ep12 detections — the
    per-run single-parabola can't fit runs with multiple flights+bounces, and
    _static_clutter over-flags a real rally's revisited court regions. Use track_and_fill
    instead. Kept for reference / a future bounce-segmented rewrite (detect y-reversals ->
    split runs -> parabola per segment), which is real R&D (SOTA uses learned inpainting)."""
    max_fill = kw.pop("max_fill", 25)
    dets = [(i, float(xs[i]), float(ys[i])) for i in range(len(xs)) if not np.isnan(xs[i])]
    arcs, clutter = extract_ballistic_arcs(dets, **kw)
    return densify(arcs, max_fill), set(clutter), arcs
