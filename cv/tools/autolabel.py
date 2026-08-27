#!/usr/bin/env python3
"""cv/tools/autolabel.py — semi-automated ball-position labeling.

Runs WASB (with confidence) over a video segment, links detections into ballistic
tracks, then AUTO-ACCEPTS high-confidence on-track detections + interpolates short
gaps within a track, and FLAGS the uncertain frames (low-conf / isolated / static /
long gaps) for human correction. Output is cheap ball-position ground truth — the
fine-tuning data + eval set we lack for outdoor. ~5-10x cheaper than frame-by-frame:
the human only reviews FLAG frames.

    python cv/tools/autolabel.py --video "tests/Outdoor Match 1 15.53.25.mp4" \
        --start-frame 1200 --max-frames 900 --device mps \
        --output /tmp/outdoor_autolabels.csv --viz outputs/autolabel/outdoor.png
"""
from __future__ import annotations
import argparse, csv, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SPEED, GAP = 150.0, 5      # track linking: max px/frame, max frame gap
MIN_LEN, STATIC = 4, 40    # confident track: >= MIN_LEN dets AND moving >= STATIC px


def run_wasb(video, start, n, device, court_poly=None, weights=None):
    from cv.detection.wasb_ball_tracker import create_ball_tracker, WASBBallTracker
    from cv.detection.court_roi import contains
    tracker = WASBBallTracker(model_path=weights, device=device) if weights else create_ball_tracker(device=device)
    cap = cv2.VideoCapture(video)
    xs = np.full(n, np.nan, np.float32); ys = np.full(n, np.nan, np.float32); cf = np.full(n, np.nan, np.float32)
    warm = max(0, start - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(warm))
    for _ in range(start - warm):                       # prime the 3-frame deque
        ok, fr = cap.read()
        if ok: tracker.detect_ball(fr)
    got = n
    for i in range(n):
        ok, fr = cap.read()
        if not ok: got = i; break
        r = tracker.detect_ball(fr)
        if r is not None:
            (x, y), conf, _ = r
            if court_poly is None or contains(court_poly, x, y):   # gate to the court (drops off-court clutter)
                xs[i], ys[i], cf[i] = x, y, conf
    cap.release()
    return xs[:got], ys[:got], cf[:got]


def link_tracks(xs, ys, cf):
    dl = [(i, float(xs[i]), float(ys[i]), float(cf[i])) for i in range(len(xs)) if not np.isnan(xs[i])]
    tracks, cur = [], []
    for d in dl:
        if cur and (d[0]-cur[-1][0] <= GAP) and (np.hypot(d[1]-cur[-1][1], d[2]-cur[-1][2]) <= SPEED*(d[0]-cur[-1][0])):
            cur.append(d)
        else:
            if cur: tracks.append(cur)
            cur = [d]
    if cur: tracks.append(cur)
    return tracks


def classify(xs, ys, cf):
    N = len(xs)
    status = ["noball"] * N
    lx = np.full(N, np.nan, np.float32); ly = np.full(N, np.nan, np.float32)
    for t in link_tracks(xs, ys, cf):
        xl = [p[1] for p in t]; yl = [p[2] for p in t]
        moving = (max(xl)-min(xl)) >= STATIC or (max(yl)-min(yl)) >= STATIC
        confident = len(t) >= MIN_LEN and moving
        for i, x, y, c in t:
            lx[i], ly[i] = x, y
            status[i] = "auto" if confident else "flag"
        if confident:
            fi = [p[0] for p in t]
            for a, b in zip(fi[:-1], fi[1:]):
                if 1 < b - a <= GAP:
                    for k in range(a+1, b):
                        f = (k-a)/(b-a)
                        lx[k] = lx[a] + f*(lx[b]-lx[a]); ly[k] = ly[a] + f*(ly[b]-ly[a])
                        status[k] = "interp"
    return status, lx, ly


def make_viz(video, start, status, lx, ly, path):
    COL = {"auto": (0, 255, 0), "interp": (0, 255, 255), "flag": (0, 0, 255)}
    marks = [i for i in range(len(status)) if status[i] in COL]
    if not marks: return
    idx = [marks[int(k)] for k in np.linspace(0, len(marks)-1, min(9, len(marks)))]
    cap = cv2.VideoCapture(video); tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start + i)); ok, fr = cap.read()
        if not ok: continue
        x, y = int(lx[i]), int(ly[i])
        cv2.circle(fr, (x, y), 13, COL[status[i]], 3)
        cv2.putText(fr, f"f{start+i} {status[i]}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COL[status[i]], 3)
        tiles.append(cv2.resize(fr, (640, 360)))
    cap.release()
    while len(tiles) < 9: tiles.append(np.zeros((360, 640, 3), np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, cv2.vconcat([cv2.hconcat(tiles[r*3:r*3+3]) for r in range(3)]))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--court-roi", default=None, help="Court polygon JSON — gate detections to the court (drops off-court clutter)")
    ap.add_argument("--weights", default=None, help="Override WASB weights (e.g. a fine-tuned checkpoint)")
    ap.add_argument("--output", default="/tmp/autolabels.csv")
    ap.add_argument("--viz", default=None)
    args = ap.parse_args()

    court_poly = None
    if args.court_roi:
        from cv.detection.court_roi import load_polygon, expand_polygon
        court_poly = expand_polygon(load_polygon(args.court_roi))
    xs, ys, cf = run_wasb(args.video, args.start_frame, args.max_frames, args.device, court_poly, args.weights)
    N = len(xs); det = ~np.isnan(xs)
    print(f"WASB fired on {int(det.sum())}/{N} frames ({100*det.mean():.1f}%)")
    if det.any():
        print(f"conf: min {np.nanmin(cf):.2f} / median {np.nanmedian(cf):.2f} / max {np.nanmax(cf):.2f}")
    status, lx, ly = classify(xs, ys, cf)
    cnt = Counter(status)
    auto = cnt["auto"] + cnt["interp"]
    print(f"classification: {dict(cnt)}")
    print(f"  AUTO-labeled (accept+interp): {auto} ({100*auto/N:.0f}%)   FLAG (human reviews): {cnt['flag']} ({100*cnt['flag']/N:.0f}%)   noball: {cnt['noball']}")
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "x", "y", "status"])
        for i in range(N):
            fr = args.start_frame + i
            if status[i] == "noball": w.writerow([fr, "", "", "noball"])
            else: w.writerow([fr, f"{lx[i]:.1f}", f"{ly[i]:.1f}", status[i]])
    print(f"wrote {args.output}")
    if args.viz: make_viz(args.video, args.start_frame, status, lx, ly, args.viz)


if __name__ == "__main__":
    main()
