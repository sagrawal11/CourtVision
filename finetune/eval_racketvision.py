#!/usr/bin/env python3
"""RacketVision tennis-test eval harness for OUR WASB detector (+ optional MS-TrackNetV3).

Reproduces the RacketVision BallMetrics (source/BallTrack/metrics/ball_metrics.py) exactly:
  - Evaluate ONLY on annotated frames (the CSV rows), one prediction per annotated frame.
  - Everything scored in the 512x288 model space.
  - GT visibility = (X,Y) != (0,0). Pred visibility = decoded center != (0,0).
  - tolerance = 4 px (in 512x288). dist>tol on a both-visible frame => FP1.
  - recall    = TP / (TP + FN + FP1)
    precision = TP / (TP + FP1 + FP2)
  - GT source = 'position' (from the CSV X,Y scaled by orig/512, orig/288).

Two model backends, scored with the IDENTICAL confusion-matrix logic:
  --model wasb   : OUR cv/detection/wasb_ball_tracker.py (weights models/ball/wasb_tennis_best.pth.tar)
                   native 3-frame window [fid-2,fid-1,fid] (clamped at 0), no median bg.
  --model mstnv3 : RacketVision MS-TrackNetV3 (their tracknet_v3.py + balltrack_best.pth),
                   seq_len=4 window [fid-3..fid] (clamped) + median-bg concat, their decode.

Reads frames directly from the mp4 (no JPG extraction needed). Frame index N in the
CSV == the N-th frame read from the video (0-indexed), matching their extract_frames.py.

Usage:
  python eval_racketvision.py --model wasb  --device mps
  python eval_racketvision.py --model mstnv3 --device mps
  (optional: --limit N to run on first N clips; --dump preds.csv to save per-frame rows)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---- paths -------------------------------------------------------------------
REPO = Path("/Users/sarthak/Desktop/App Projects/tennis_analytics")
DATA = Path("/tmp/racketvision_work/data")           # tennis/videos + tennis/all/*/csv
INFO = Path("/tmp/racketvision_work/hf_meta/tennis/info")
RV = Path("/tmp/racketvision_work/RacketVision/source/BallTrack")
MSTNV3_CKPT = Path("/tmp/racketvision_work/models/checkpoints/balltrack_best.pth")

W, H = 512, 288          # model space (matches RacketVision + WASB)
TOL = 4                  # tolerance in 512x288 space (RacketVision default)
ORIG_W, ORIG_H = 1920, 1080   # tennis image_shape from metainfo.json
W_SCALER, H_SCALER = ORIG_W / W, ORIG_H / H   # 3.75, 3.75


# ---- RacketVision decode (contour, largest bbox center) ----------------------
def rv_predict_center(heatmap01: np.ndarray, thr: float = 0.5):
    """Replicates BallMetrics.predict_location + center calc, in 512x288 space.
    heatmap01: float array HxW in [0,1] at 288x512. Returns (cx,cy) ints; (0,0)=no ball."""
    hbin = (heatmap01 > thr).astype(np.uint8) * 255
    if hbin.max() == 0:
        return 0, 0
    cnts, _ = cv2.findContours(hbin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0
    rects = [cv2.boundingRect(c) for c in cnts]
    areas = [r[2] * r[3] for r in rects]
    x, y, w, h = rects[int(np.argmax(areas))]
    return int(x + w / 2), int(y + h / 2)


def classify(cx_pred, cy_pred, cx_true, cy_true, tol=TOL):
    """RacketVision confusion class for one frame. Returns (label, dist)."""
    vis_pred = 0 if (cx_pred == 0 and cy_pred == 0) else 1
    vis_true = 0 if (cx_true == 0 and cy_true == 0) else 1
    if vis_pred == 0 and vis_true == 0:
        return "TN", 0.0
    if vis_pred > 0 and vis_true == 0:
        return "FP2", 0.0
    if vis_pred == 0 and vis_true > 0:
        return "FN", 0.0
    dist = math.sqrt((cx_pred - cx_true) ** 2 + (cy_pred - cy_true) ** 2)
    return ("FP1" if dist > tol else "TP"), dist


# ---- WASB backend ------------------------------------------------------------
class WASBBackend:
    seq_len = 3

    def __init__(self, device, weights=None):
        sys.path.insert(0, str(REPO))
        import torch
        from cv.detection.wasb.hrnet import HRNet
        from cv.detection.wasb.image_utils import get_affine_transform, affine_transform
        from omegaconf import OmegaConf
        self.torch = torch
        self._affine_transform = affine_transform
        self._get_affine = get_affine_transform
        cfg = OmegaConf.load(REPO / "cv/detection/wasb/wasb_tennis.yaml")
        self.model = HRNet(cfg)
        ckpt = torch.load(weights or (REPO / "models/ball/wasb_tennis_best.pth.tar"), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.device = device
        self.model.to(device).eval()
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_center(self, frames_bgr):
        """frames_bgr: list of 3 BGR frames [t-2,t-1,t] at original res.
        Returns (cx,cy) in 512x288 space; (0,0) = no ball. (matches RacketVision space)"""
        torch = self.torch
        h, w = frames_bgr[0].shape[:2]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = max(h, w) * 1.0
        trans_in = self._get_affine(center, scale, 0, [W, H], inv=0)
        chans = []
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            warped = cv2.warpAffine(rgb, trans_in, (W, H), flags=cv2.INTER_LINEAR)
            t = (warped.astype(np.float32) / 255.0 - self._mean) / self._std
            chans.append(np.transpose(t, (2, 0, 1)))
        inp = torch.from_numpy(np.concatenate(chans, axis=0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(inp)
        hm_t = preds[0] if isinstance(preds, dict) else preds
        hm = torch.sigmoid(hm_t).cpu().numpy()[0, self.seq_len - 1]  # current-frame heatmap, 288x512, [0,1]
        # Decode with RacketVision's exact contour method, in the SAME 512x288 space they score in.
        return rv_predict_center(hm, thr=0.5)


# ---- MS-TrackNetV3 backend (RacketVision) ------------------------------------
class MSTrackNetV3Backend:
    seq_len = 4

    def __init__(self, device):
        sys.path.insert(0, str(RV))
        import torch
        # register the model only (avoid the config's custom_imports of dataset/metrics/hooks)
        os.chdir(RV)
        import model as _m  # noqa  (registers TrackNetV3)
        from mmengine.registry import MODELS
        self.torch = torch
        # matches configs/tracknetv3_base.py: seq_len=4 -> in_dim=3*(4+1)=15, out_dim=4
        seq_len = 4
        model_cfg = dict(type="TrackNetV3", in_dim=3 * (seq_len + 1), out_dim=seq_len,
                         mixup=True, alpha=0.5, last_only=True)
        mdl = MODELS.build(model_cfg)
        state = torch.load(str(MSTNV3_CKPT), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        mdl.load_state_dict(state)
        self.device = device
        self.model = mdl.to(device).eval()

    def predict_center(self, frames_bgr, median_bgr):
        """frames_bgr: 4 BGR frames [t-3..t] at orig res. median_bgr: BGR median at orig res.
        Reproduces uball.__getitem__ (bg_mode='concat'): median frame prepended, /255,
        resized to 512x288, channels-first, concatenated -> 15ch. Their model, last heatmap."""
        torch = self.torch
        imgs = np.array([cv2.resize(f.astype("float32"), (W, H)) for f in frames_bgr])  # 4,H,W,3
        imgs = np.moveaxis(imgs, -1, 1)  # 4,3,H,W
        med = cv2.resize(median_bgr.astype("float32"), (W, H))
        med = np.moveaxis(med, -1, 0).reshape(1, 3, H, W)
        frames = np.concatenate((med, imgs), axis=0)  # 5,3,H,W
        frames /= 255.0
        frames = frames.reshape(-1, H, W)  # 15,H,W
        inp = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.model._forward(inp)          # 1, seq_len, H, W (sigmoid already applied)
        hm = x[0, -1].cpu().numpy()               # last frame heatmap, [0,1]
        return rv_predict_center(hm, thr=0.5)


# ---- per-clip frame reader ---------------------------------------------------
def read_needed_frames(video_path, needed_idx):
    """Read exactly the frame indices in `needed_idx` (set of ints) from the mp4.
    Returns dict {idx: BGR}. Sequential decode (fast, avoids per-frame seeks)."""
    needed = set(int(i) for i in needed_idx)
    if not needed:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    max_needed = max(needed)
    idx = 0
    while idx <= max_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            out[idx] = frame
        idx += 1
    cap.release()
    return out


def compute_median(video_path, max_frames=100):
    """Pixel-wise median over up to max_frames evenly-sampled frames (matches create_median.py idea)."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, max(0, n - 1), min(max_frames, n)).astype(int).tolist())
    frames, idx = [], 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if idx in picks:
            frames.append(f)
        idx += 1
    cap.release()
    return np.median(np.array(frames), axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["wasb", "mstnv3"], default="wasb")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (0=all)")
    ap.add_argument("--dump", default="", help="optional path to write per-frame CSV")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    test = json.load(open(INFO / "test.json"))
    if args.limit:
        test = test[: args.limit]

    if args.model == "wasb":
        backend = WASBBackend(args.device, args.weights)
    else:
        backend = MSTrackNetV3Backend(args.device)
    seq_len = backend.seq_len

    tallies = {"TP": 0, "TN": 0, "FP1": 0, "FP2": 0, "FN": 0}
    dist_sum, n_frames_scored, t0 = 0.0, 0, time.time()
    rows = []

    for ci, (match, rally) in enumerate(test):
        vid = DATA / f"tennis/videos/{match}_{rally}.mp4"
        csv = DATA / f"tennis/all/{match}/csv/{rally}_ball.csv"
        if not vid.exists() or not csv.exists():
            print(f"  [skip] {match}_{rally}: missing files")
            continue

        import pandas as pd
        df = pd.read_csv(csv).sort_values("Frame").fillna(0)
        ann = [(int(r.Frame), float(r.X), float(r.Y)) for r in df.itertuples()]

        # which raw frame indices do we need? for each annotated fid: [fid-(seq_len-1) .. fid], clamped>=0
        needed = set()
        for fid, _, _ in ann:
            for k in range(seq_len):
                needed.add(max(0, fid - (seq_len - 1) + k))
        fr = read_needed_frames(vid, needed)

        median = compute_median(vid) if args.model == "mstnv3" else None

        for fid, gx, gy in ann:
            widx = [max(0, fid - (seq_len - 1) + k) for k in range(seq_len)]
            if any(i not in fr for i in widx):
                # frame beyond video end (shouldn't happen for annotated frames); treat as no-pred
                cx_pred = cy_pred = 0
            else:
                seq = [fr[i] for i in widx]
                if args.model == "wasb":
                    cx_pred, cy_pred = backend.predict_center(seq)
                else:
                    cx_pred, cy_pred = backend.predict_center(seq, median)
            # GT to model space (matches uball: int(coor/scaler))
            cx_true = int(gx / W_SCALER)
            cy_true = int(gy / H_SCALER)
            label, dist = classify(cx_pred, cy_pred, cx_true, cy_true)
            tallies[label] += 1
            dist_sum += dist
            n_frames_scored += 1
            if args.dump:
                rows.append((match, rally, fid, cx_pred, cy_pred, cx_true, cy_true, label, round(dist, 2)))
        if (ci + 1) % 5 == 0 or ci == len(test) - 1:
            el = time.time() - t0
            print(f"  [{ci+1}/{len(test)}] {match}_{rally}  scored={n_frames_scored}  "
                  f"TP={tallies['TP']} FN={tallies['FN']} FP1={tallies['FP1']} FP2={tallies['FP2']} TN={tallies['TN']}  "
                  f"({el:.0f}s)")

    TP, TN, FP1, FP2, FN = (tallies[k] for k in ["TP", "TN", "FP1", "FP2", "FN"])
    gt_true = TP + FN + FP1
    pred_true = TP + FP1 + FP2
    accuracy = (TP + TN) / max(1, TP + TN + FP1 + FP2 + FN)
    precision = TP / pred_true if pred_true else 0.0
    recall = TP / gt_true if gt_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_dist = dist_sum / (TP + FP1) if (TP + FP1) else float("nan")
    elapsed = time.time() - t0

    print("\n" + "=" * 68)
    print(f"MODEL: {args.model}   clips: {len(test)}   frames scored: {n_frames_scored}")
    print(f"  TP={TP}  TN={TN}  FP1={FP1}  FP2={FP2}  FN={FN}")
    print(f"  precision = {precision:.4f}   (TP/(TP+FP1+FP2))")
    print(f"  recall    = {recall:.4f}   (TP/(TP+FN+FP1))")
    print(f"  f1        = {f1:.4f}")
    print(f"  accuracy  = {accuracy:.4f}   mean_dist(px@512x288) = {mean_dist:.3f}")
    print(f"  wall: {elapsed:.0f}s  ({1000*elapsed/max(1,n_frames_scored):.0f} ms/frame incl. decode+IO)")
    print("=" * 68)

    if args.dump:
        import csv as _csv
        with open(args.dump, "w", newline="") as fh:
            wtr = _csv.writer(fh)
            wtr.writerow(["match", "rally", "frame", "cx_pred", "cy_pred", "cx_true", "cy_true", "type", "dist"])
            wtr.writerows(rows)
        print(f"per-frame rows -> {args.dump}")


if __name__ == "__main__":
    main()
