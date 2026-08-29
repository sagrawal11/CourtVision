#!/usr/bin/env python3
"""cv/tools/find_rally.py — locate real rallies by player-in-court density.

Outdoor WASB fires on background clutter (parked cars, windscreen), so
ball-detection density is a BAD rally signal — it picks dead stretches on
adjacent courts. Instead scan for sustained segments where >=2 players' FEET
sit inside the court ROI AND the two players straddle the court (large feet
y-spread = near baseline vs far baseline). That pattern = a rally in progress;
a changeover has both players on the SAME side (small y-spread), and dead-time
has <2 players on court. Ranks candidate segments so you can autolabel the best.

    python cv/tools/find_rally.py --video "tests/Outdoor Match 1 15.53.25.mp4" \
        --court-roi cv/court_rois/Outdoor_Match_1.json --device mps \
        --stride 45 --viz outputs/find_rally/outdoor1.png
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import cv2, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def scan(video, court_poly, device, start, n, stride, bbox_thr):
    """Sample frames every `stride`; per sample record how many player feet fall
    inside court_poly and the near/far feet y-spread. Returns list of
    (frame, count_in_roi, yspread, feet_in_roi)."""
    from cv.detection.player_detector import PlayerDetector
    from cv.detection.court_roi import contains
    det = PlayerDetector(device=device)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if n <= 0 else min(total, start + n)
    samples = []
    t0 = time.time()
    f = start
    while f < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(f))
        ok, fr = cap.read()
        if not ok:
            break
        feet = []
        for (x1, y1, x2, y2), _c in det.detect_players(fr, bbox_thr=bbox_thr):
            fx, fy = (x1 + x2) / 2.0, y2
            if contains(court_poly, fx, fy):
                feet.append((fx, fy))
        ys = [p[1] for p in feet]
        yspread = (max(ys) - min(ys)) if len(feet) >= 2 else 0.0
        samples.append((f, len(feet), yspread, feet))
        f += stride
        if len(samples) % 200 == 0:
            done = (f - start) / max(1, end - start)
            print(f"  scan {f}/{end} ({100*done:.0f}%)  {len(samples)/(time.time()-t0):.1f} samp/s", flush=True)
    cap.release()
    return samples


def segment(samples, min_yspread, max_gap_samples, min_dur_frames, stride):
    """Group samples where >=2 players straddle the court (yspread>=min_yspread),
    bridging up to max_gap_samples cold samples. Return segments sorted by duration."""
    hot = [i for i, s in enumerate(samples) if s[1] >= 2 and s[2] >= min_yspread]
    groups = []
    for i in hot:
        if groups and i - groups[-1][1] <= max_gap_samples:
            groups[-1][1] = i
        else:
            groups.append([i, i])
    out = []
    for a, b in groups:
        f0, f1 = samples[a][0], samples[b][0]
        if f1 - f0 < min_dur_frames:
            continue
        sub = samples[a:b + 1]
        out.append({
            "f0": f0, "f1": f1, "n_samples": b - a + 1,
            "mean_count": float(np.mean([s[1] for s in sub])),
            "mean_spread": float(np.mean([s[2] for s in sub])),
            "hot_frac": float(np.mean([1.0 if (s[1] >= 2) else 0.0 for s in sub])),
        })
    out.sort(key=lambda d: d["f1"] - d["f0"], reverse=True)
    return out


def make_viz(video, raw_poly, exp_poly, seg, path, device, bbox_thr):
    """Montage of 9 evenly-spaced frames across a segment, drawing court ROI
    (raw yellow, expanded cyan) + player boxes (green in-ROI / gray out) + feet."""
    from cv.detection.player_detector import PlayerDetector
    from cv.detection.court_roi import contains
    det = PlayerDetector(device=device)
    frames = [int(round(x)) for x in np.linspace(seg["f0"], seg["f1"], 9)]
    cap = cv2.VideoCapture(video)
    tiles = []
    for fr_i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fr_i))
        ok, fr = cap.read()
        if not ok:
            continue
        cv2.polylines(fr, [exp_poly.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)
        cv2.polylines(fr, [raw_poly.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        for (x1, y1, x2, y2), _c in det.detect_players(fr, bbox_thr=bbox_thr):
            fx, fy = (x1 + x2) / 2.0, y2
            inside = contains(exp_poly, fx, fy)
            col = (0, 255, 0) if inside else (150, 150, 150)
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), col, 2 if inside else 1)
            cv2.circle(fr, (int(fx), int(fy)), 5, col, -1)
        cv2.putText(fr, f"f{fr_i}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        tiles.append(cv2.resize(fr, (640, 360)))
    cap.release()
    while len(tiles) < 9:
        tiles.append(np.zeros((360, 640, 3), np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, cv2.vconcat([cv2.hconcat(tiles[r * 3:r * 3 + 3]) for r in range(3)]))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--court-roi", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=-1, help="-1 = to end of video")
    ap.add_argument("--stride", type=int, default=45, help="sample every N frames (30fps: 45=1.5s)")
    ap.add_argument("--bbox-thr", type=float, default=0.25, help="YOLO conf (low to catch the small far player)")
    ap.add_argument("--up", type=int, default=40, help="ROI expand up (catch far player behind baseline)")
    ap.add_argument("--down", type=int, default=80, help="ROI expand down (catch near player behind baseline)")
    ap.add_argument("--side", type=int, default=20)
    ap.add_argument("--min-yspread", type=float, default=200.0, help="min near/far feet separation (px) for a rally")
    ap.add_argument("--max-gap-sec", type=float, default=6.0, help="bridge cold gaps up to this many seconds")
    ap.add_argument("--min-dur-sec", type=float, default=4.0, help="min segment duration to report")
    ap.add_argument("--top", type=int, default=12, help="how many segments to print")
    ap.add_argument("--viz", default=None, help="montage of the top segment")
    ap.add_argument("--viz-rank", type=int, default=0, help="which ranked segment to viz (0=top)")
    args = ap.parse_args()

    from cv.detection.court_roi import load_polygon, expand_polygon
    raw = load_polygon(args.court_roi)
    exp = expand_polygon(raw, up=args.up, down=args.down, side=args.side)

    fps = 30.0
    max_gap_samples = max(1, int(round(args.max_gap_sec * fps / args.stride)))
    min_dur_frames = int(round(args.min_dur_sec * fps))

    print(f"scanning {args.video}  stride={args.stride}  bbox_thr={args.bbox_thr}")
    samples = scan(args.video, exp, args.device, args.start_frame, args.max_frames, args.stride, args.bbox_thr)
    n2 = sum(1 for s in samples if s[1] >= 2)
    print(f"scanned {len(samples)} samples; {n2} had >=2 players in ROI ({100*n2/max(1,len(samples)):.0f}%)")

    segs = segment(samples, args.min_yspread, max_gap_samples, min_dur_frames, args.stride)
    print(f"\nfound {len(segs)} rally-candidate segments (>=2 straddling players, yspread>={args.min_yspread:.0f}px):")
    print(f"{'rank':>4} {'frames':>18} {'dur':>7} {'mean_n':>7} {'spread':>7} {'hot%':>5}")
    for r, s in enumerate(segs[:args.top]):
        dur = (s["f1"] - s["f0"]) / fps
        t0 = s["f0"] / fps
        print(f"{r:>4} {s['f0']:>7}-{s['f1']:<7} {dur:>5.1f}s {s['mean_count']:>7.2f} "
              f"{s['mean_spread']:>7.0f} {100*s['hot_frac']:>4.0f}%   (@{int(t0//60)}m{int(t0%60):02d}s)")

    if args.viz and segs:
        s = segs[min(args.viz_rank, len(segs) - 1)]
        print(f"\nvisualizing rank {args.viz_rank}: frames {s['f0']}-{s['f1']}")
        make_viz(args.video, raw, exp, s, args.viz, args.device, args.bbox_thr)


if __name__ == "__main__":
    main()
