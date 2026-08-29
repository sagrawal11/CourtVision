#!/usr/bin/env python3
"""Continuous RacketVision eval for OUR WASBBallTracker — tracker ON vs OFF.
The online motion gate is stateful, so we run detect_ball over ALL frames of each clip
(reset per clip) and score on the annotated frames with RacketVision's exact metric.
Both trackers run in one video pass (same frames) to save time."""
import json, math, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path("/Users/sarthak/Desktop/App Projects/tennis_analytics")
DATA = Path("/tmp/racketvision_work/data")
INFO = Path("/tmp/racketvision_work/hf_meta/tennis/info")
W, H, TOL = 512, 288, 4
SC = 1920 / W  # 3.75 (== 1080/288)
sys.path.insert(0, str(REPO))
from cv.detection.wasb_ball_tracker import WASBBallTracker

def classify(pred, true):
    vp = 0 if pred is None else 1
    vt = 0 if (true[0] == 0 and true[1] == 0) else 1
    if vp == 0 and vt == 0: return "TN", 0.0
    if vp == 1 and vt == 0: return "FP2", 0.0
    if vp == 0 and vt == 1: return "FN", 0.0
    d = math.hypot(pred[0]-true[0], pred[1]-true[1])
    return ("FP1" if d > TOL else "TP"), d

def score(tallies):
    TP, TN, FP1, FP2, FN = (tallies[k] for k in ["TP","TN","FP1","FP2","FN"])
    prec = TP/(TP+FP1+FP2) if (TP+FP1+FP2) else 0
    rec  = TP/(TP+FN+FP1) if (TP+FN+FP1) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return prec, rec, f1

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()
    test = json.load(open(INFO/"test.json"))[:args.limit]
    on  = WASBBallTracker(device=args.device, use_tracker=True)
    off = WASBBallTracker(device=args.device, use_tracker=False)
    tal = {"on": {k:0 for k in ["TP","TN","FP1","FP2","FN"]}, "off": {k:0 for k in ["TP","TN","FP1","FP2","FN"]}}
    t0 = time.time()
    for ci, (match, rally) in enumerate(test):
        vid = DATA/f"tennis/videos/{match}_{rally}.mp4"; csvp = DATA/f"tennis/all/{match}/csv/{rally}_ball.csv"
        if not vid.exists() or not csvp.exists(): continue
        df = pd.read_csv(csvp).sort_values("Frame").fillna(0)
        gt = {int(r.Frame): (float(r.X)/SC, float(r.Y)/SC) for r in df.itertuples()}
        on.reset(); off.reset()
        cap = cv2.VideoCapture(str(vid)); maxf = max(gt); idx = 0
        preds = {"on": {}, "off": {}}
        while idx <= maxf:
            ok, fr = cap.read()
            if not ok: break
            r_on = on.detect_ball(fr); r_off = off.detect_ball(fr)
            if idx in gt:
                preds["on"][idx]  = (r_on[0][0]/SC,  r_on[0][1]/SC)  if r_on  else None
                preds["off"][idx] = (r_off[0][0]/SC, r_off[0][1]/SC) if r_off else None
            idx += 1
        cap.release()
        for fid, true in gt.items():
            for mode in ("on", "off"):
                lab, _ = classify(preds[mode].get(fid), true); tal[mode][lab] += 1
        if (ci+1) % 5 == 0 or ci == len(test)-1:
            po, ro, _ = score(tal["on"]); pf, rf, _ = score(tal["off"])
            print(f"[{ci+1}/{len(test)}] {int(time.time()-t0)}s  ON P{po:.3f}/R{ro:.3f}  OFF P{pf:.3f}/R{rf:.3f}")
    print("\n==== RacketVision tennis (continuous, our WASBBallTracker) ====")
    for mode in ("off", "on"):
        p, r, f = score(tal[mode]); t = tal[mode]
        print(f"  tracker {mode:3}: P {p:.3f}  R {r:.3f}  F1 {f:.3f}   (TP{t['TP']} FP1{t['FP1']} FP2{t['FP2']} FN{t['FN']})")
    print("  (baseline per-window WASB was P0.877/R0.801; motion gate should raise precision)")

if __name__ == "__main__":
    main()
