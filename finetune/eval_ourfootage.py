#!/usr/bin/env python3
"""finetune/eval_ourfootage.py — compare WASB checkpoints on OUR outdoor footage.

Answers the Stage-2 question: did outdoor recall recover (vs ep4's crash) while the
car/windscreen clutter stayed dead (vs current WASB's false positives)?

For each checkpoint:
  - TRAIN seg (10845-12195, has GT): recall/precision vs the human labels @tol px.
    (optimistic — it's training data — but the RELATIVE trend across weights is the signal.)
  - HELD-OUT seg (a different rally, no GT): court-gated firing% + median conf + a viz
    montage so we can SEE whether detections land on the ball or on clutter.

    python finetune/eval_ourfootage.py --tags current,ep4,ep12 --device mps
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from cv.tools.autolabel import run_wasb, classify, make_viz
from cv.detection.court_roi import load_polygon, expand_polygon, contains

VIDEO = "tests/Outdoor Match 1 15.53.25.mp4"
ROI = "cv/court_rois/Outdoor_Match_1.json"
TRAIN = (10845, 1351, "cv/training_data/ball_labels/outdoor1_seg10845_labels.csv")   # start, n, gt
HELDOUT = (41985, 856)                                                               # start, n (rank-5 @23m)
WEIGHTS = {
    "current": None,   # -> create_ball_tracker default = models/ball/wasb_tennis_best.pth.tar
    "ep4":  "models/ball/wasb_stage1_ep4.pth.tar",
    "ep6":  "/tmp/wasb_finetune_work/outputs/stage2_ours/checkpoint_ep6.pth.tar",
    "ep8":  "/tmp/wasb_finetune_work/outputs/stage2_ours/checkpoint_ep8.pth.tar",
    "ep10": "/tmp/wasb_finetune_work/outputs/stage2_ours/checkpoint_ep10.pth.tar",
    "ep12": "/tmp/wasb_finetune_work/outputs/stage2_ours/checkpoint_ep12.pth.tar",
}


def load_gt(path):
    gt = {}
    for r in csv.DictReader(open(path)):
        if (r.get("visibility") or "").strip() == "1":
            gt[int(r["frame"])] = (float(r["x"]), float(r["y"]))
    return gt


def score_train(xs, ys, start, gt, tol):
    """recall = GT balls detected within tol; precision = detections that hit a GT ball."""
    n_det = int((~np.isnan(xs)).sum())
    tp = 0
    for i in range(len(xs)):
        if np.isnan(xs[i]):
            continue
        g = gt.get(start + i)
        if g is not None and np.hypot(xs[i] - g[0], ys[i] - g[1]) <= tol:
            tp += 1
    recall = tp / max(1, len(gt))
    precision = tp / max(1, n_det)
    return recall, precision, tp, n_det, len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="current,ep4,ep12")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tol", type=float, default=15.0, help="GT match tolerance (native px)")
    ap.add_argument("--viz-dir", default="outputs/stage2_eval")
    args = ap.parse_args()

    raw = load_polygon(ROI); exp = expand_polygon(raw)
    gt = load_gt(TRAIN[2])
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    Path(args.viz_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in tags:
        w = WEIGHTS[tag]
        if w is not None and not Path(w).exists():
            print(f"[skip] {tag}: {w} not found"); continue
        print(f"\n=== {tag} ({w or 'default wasb_tennis_best'}) ===")
        # TRAIN seg (with GT)
        xs, ys, cf = run_wasb(VIDEO, TRAIN[0], TRAIN[1], args.device, exp, w)
        rec, prec, tp, ndet, ngt = score_train(xs, ys, TRAIN[0], gt, args.tol)
        print(f"  TRAIN  recall {rec:.3f} ({tp}/{ngt})  precision {prec:.3f} ({tp}/{ndet} dets)  fire {100*ndet/TRAIN[1]:.0f}%")
        # HELD-OUT seg (no GT)
        hxs, hys, hcf = run_wasb(VIDEO, HELDOUT[0], HELDOUT[1], args.device, exp, w)
        hdet = ~np.isnan(hxs)
        hfire = 100 * hdet.mean()
        hconf = np.nanmedian(hcf) if hdet.any() else 0.0
        # airspace-clutter proxy: detections OUTSIDE the raw court poly (above far baseline / windscreen band)
        out_ct = sum(1 for i in range(len(hxs)) if not np.isnan(hxs[i]) and not contains(raw, hxs[i], hys[i]))
        print(f"  HELD   fire {hfire:.0f}%  medconf {hconf:.2f}  dets-outside-court(airspace/clutter) {out_ct}/{int(hdet.sum())}")
        st, lx, ly = classify(hxs, hys, hcf)
        vp = f"{args.viz_dir}/heldout_{tag}.png"
        make_viz(VIDEO, HELDOUT[0], st, lx, ly, vp)
        rows.append((tag, rec, prec, hfire, hconf, out_ct, int(hdet.sum())))

    print("\n================ SUMMARY ================")
    print(f"{'tag':>8} | TRAIN rec  prec | HELD fire medconf  out/tot")
    for tag, rec, prec, hfire, hconf, oc, ht in rows:
        print(f"{tag:>8} |  {rec:.3f}  {prec:.3f} |  {hfire:>3.0f}%   {hconf:.2f}    {oc}/{ht}")


if __name__ == "__main__":
    main()
