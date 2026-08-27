#!/usr/bin/env python3
"""cv/tools/export_wasb_labels.py — export corrected ball labels to WASB training format.

Takes a video + a per-frame labels CSV over a contiguous range (from ball_labeler.py or
autolabel.py: columns frame,x,y[,visibility/status]) and writes the WASB-SBDT / TrackNet
layout, matching convert_racketvision_to_wasb.py so our footage trains in the SAME root_dir
as the RacketVision clips (mix ~1:1 for Stage-2):

    <out-root>/<match>/<clip>/000000.jpg ...   every frame in the range, contiguous
    <out-root>/<match>/<clip>/Label.csv        file name,visibility,x-coordinate,y-coordinate

    python cv/tools/export_wasb_labels.py --video "tests/Outdoor Match 1 15.53.25.mp4" \
        --labels /tmp/outdoor_corrected.csv --out-root /tmp/wasb_finetune_work/datasets/ours \
        --match outdoor1_seg1200
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--match", required=True, help="clip group name, e.g. outdoor1_seg1200")
    ap.add_argument("--clip", default="000")
    ap.add_argument("--jpg-quality", type=int, default=95)
    args = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(args.labels)):
        f = int(r["frame"]); x = (r.get("x") or "").strip(); y = (r.get("y") or "").strip()
        rows.append((f, x, y))
    rows.sort()
    if not rows:
        raise SystemExit("no label rows")
    f0, f1 = rows[0][0], rows[-1][0]
    by_frame = {f: (x, y) for f, x, y in rows}

    out_dir = Path(args.out_root) / args.match / args.clip
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(f0))
    lab = []
    for i, f in enumerate(range(f0, f1 + 1)):
        ok, fr = cap.read()
        if not ok:
            break
        name = f"{i:06d}.jpg"
        cv2.imwrite(str(out_dir / name), fr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
        x, y = by_frame.get(f, ("", ""))
        if x != "" and y != "":                       # visible = has a ball position
            lab.append((name, 1, round(float(x), 1), round(float(y), 1)))
        else:
            lab.append((name, 0, 0, 0))
    cap.release()
    with open(out_dir / "Label.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file name", "visibility", "x-coordinate", "y-coordinate"])
        w.writerows(lab)
    vis = sum(1 for r in lab if r[1] == 1)
    print(f"wrote {len(lab)} frames ({vis} visible, {100*vis/max(len(lab),1):.0f}%) -> {out_dir}")


if __name__ == "__main__":
    main()
