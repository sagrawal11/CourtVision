#!/usr/bin/env python3
"""
cv/tools/court_roi_editor.py — Draw the court-of-interest polygon for a video.

Click around the outline of YOUR court (the one the match is played on) — the
court where the two players you care about are rallying. Include a little margin
past the baselines and sidelines so a player who steps back or wide still counts,
and keep the top edge around the far baseline so people standing behind it are
excluded. extract_features.py then ignores everything outside this polygon
(adjacent-court players, net officials off to the side, balls on other courts).

The camera is fixed per court, so you only draw each court ONCE — it is reused
for every match played on that court.

Controls
    Left click : add a polygon point
    z          : undo last point
    c          : clear all points
    s          : save
    q / ESC    : save (if any points) and quit

Usage
    python cv/tools/court_roi_editor.py --video "/path/Duke vs. Cal Court 1.mp4"
    # options:  --frame 151425   --key Cal_Court_1
Saves to: cv/court_rois/<key>.json   (key defaults to the video filename)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def derive_key(video: str) -> str:
    return Path(video).stem.replace(" ", "_").replace(".", "")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw a court-of-interest polygon for extract_features.py",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    ap.add_argument("--video", required=True, help="Video to draw the court on")
    ap.add_argument("--frame", type=int, default=None, help="Frame index (default: 40%% in)")
    ap.add_argument("--key", default=None, help="Court key / output name (default: from filename)")
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "cv" / "court_rois"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}"); sys.exit(1)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fi = args.frame if args.frame is not None else int(n_frames * 0.4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: cannot read frame {fi}"); sys.exit(1)
    H, W = frame.shape[:2]

    key = args.key or derive_key(args.video)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.json"

    points: list[tuple[int, int]] = []
    if out_path.exists():
        try:
            points = [tuple(p) for p in json.loads(out_path.read_text()).get("polygon", [])]
            print(f"Loaded {len(points)} existing points from {out_path}")
        except (ValueError, OSError):
            pass

    scale = min(1600.0 / W, 900.0 / H, 1.0)
    disp_w, disp_h = int(W * scale), int(H * scale)

    def save() -> None:
        out_path.write_text(json.dumps({
            "video": Path(args.video).name, "frame": fi, "width": W, "height": H,
            "polygon": [list(p) for p in points],
        }, indent=2))
        print(f"Saved {len(points)}-point polygon -> {out_path}")

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))

    win = "Court ROI editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)
    print(f"Drawing court '{key}' on frame {fi} of {Path(args.video).name}")

    while True:
        disp = cv2.resize(frame, (disp_w, disp_h))
        if points:
            pd = np.array([[int(px * scale), int(py * scale)] for px, py in points], np.int32)
            if len(points) >= 2:
                cv2.polylines(disp, [pd], len(points) >= 3, (0, 255, 255), 2)
            for px, py in pd:
                cv2.circle(disp, (int(px), int(py)), 4, (0, 0, 255), -1)
        cv2.putText(disp, f"{key}   points={len(points)}   [click add | z undo | c clear | s save | q quit]",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("z") and points:
            points.pop()
        elif k == ord("c"):
            points.clear()
        elif k == ord("s"):
            save()
        elif k in (ord("q"), 27):
            if points:
                save()
            break
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                if points:
                    save()
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
