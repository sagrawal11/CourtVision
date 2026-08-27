#!/usr/bin/env python3
"""cv/tools/ball_labeler.py — review + correct semi-auto ball labels for fine-tuning.

Scrub a video with the autolabel.py predictions overlaid; fix the flagged frames (and
any missed balls / clutter false-accepts) by clicking the ball, or mark no-ball. Exports
a corrected per-frame CSV (frame,x,y,visibility) that export_wasb_labels.py turns into
WASB training data. A zoom inset makes the small ball clickable at fit-to-screen scale.

    python cv/tools/ball_labeler.py --video "tests/Indoor Match 2 15.53.25.mp4" \
        --labels /tmp/in2_autolabels.csv --output /tmp/in2_corrected.csv --start 0 --end 1500

Keys:  d / → next · a / ← prev · f next-flagged · click = set ball · n = no ball
       [ / ] zoom out/in · s save · q quit (auto-saves on quit)
"""
from __future__ import annotations
import argparse, csv, os, shutil
from pathlib import Path
import cv2, numpy as np

COLORS = {"auto": (0, 200, 0), "interp": (0, 220, 220), "flag": (0, 0, 255),
          "human": (255, 200, 0), "noball": (140, 140, 140)}


def load_labels(path):
    """Reads either autolabel output (frame,x,y,status) or a prior ball_labeler save
    (frame,x,y,visibility). Empty x or visibility==0 -> no ball."""
    lab = {}
    if Path(path).exists():
        for r in csv.DictReader(open(path)):
            f = int(r["frame"]); x = (r.get("x") or "").strip(); y = (r.get("y") or "").strip()
            vis = (r.get("visibility") or "").strip()
            if x == "" or vis == "0":
                lab[f] = (None, "noball")
            else:
                lab[f] = ((float(x), float(y)), r.get("status") or "human")
    return lab


def render(frame, lab_entry, fi, dw, zoom, hud):
    """Composite the display image: scaled frame + ball marker + HUD + zoom inset. Testable."""
    H, W = frame.shape[:2]
    scale = dw / W
    disp = cv2.resize(frame, (dw, int(H * scale)))
    xy, status = lab_entry
    col = COLORS.get(status, (0, 0, 255))
    if xy is not None:
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        cv2.circle(disp, (px, py), 11, col, 2)
        cv2.line(disp, (px - 16, py), (px + 16, py), col, 1)
        cv2.line(disp, (px, py - 16), (px, py + 16), col, 1)
        # zoom inset (top-right): magnified crop around the ball
        R = 60
        cx, cy = int(xy[0]), int(xy[1])
        crop = frame[max(0, cy-R):min(H, cy+R), max(0, cx-R):min(W, cx+R)]
        if crop.size:
            iz = cv2.resize(crop, (R*2*zoom, R*2*zoom), interpolation=cv2.INTER_NEAREST)
            ih, iw = iz.shape[:2]
            cv2.drawMarker(iz, (iw//2, ih//2), col, cv2.MARKER_CROSS, 20, 1)
            disp[8:8+ih, dw-8-iw:dw-8] = iz
            cv2.rectangle(disp, (dw-8-iw, 8), (dw-8, 8+ih), (255, 255, 255), 1)
    label = "NO BALL" if xy is None else status.upper()
    cv2.rectangle(disp, (0, 0), (dw, 30), (0, 0, 0), -1)
    cv2.putText(disp, f"f{fi}  {label}   {hud}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return disp, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--render-test", default=None, help="dump one rendered frame to this PNG and exit (no GUI)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(args.end if args.end is not None else n - 1, n - 1)
    # RESUME from prior corrections if they exist, else start from the autolabel candidates.
    resume = Path(args.output).exists()
    lab = load_labels(args.output if resume else args.labels)
    print(f"{'RESUMING from ' + args.output if resume else 'loaded candidates ' + args.labels}")
    dw, zoom = args.display_width, 3
    fi = args.start
    frame_cache = {}

    def read(f):
        if f not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(f)); ok, fr = cap.read()
            frame_cache[f] = fr if ok else None
            if len(frame_cache) > 40: frame_cache.pop(next(iter(frame_cache)))
        return frame_cache[f]

    def save():
        if Path(args.output).exists():
            shutil.copy2(args.output, args.output + ".bak")   # keep the previous save
        tmp = args.output + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["frame", "x", "y", "visibility"])
            for f in range(args.start, end + 1):
                xy, st = lab.get(f, (None, "noball"))
                if xy is None: w.writerow([f, "", "", 0])
                else: w.writerow([f, f"{xy[0]:.1f}", f"{xy[1]:.1f}", 1])
        os.replace(tmp, args.output)                          # atomic
        print(f"saved {args.output}")

    # non-GUI render smoke test (for validation without a display)
    if args.render_test:
        fr = read(fi)
        disp, _ = render(fr, lab.get(fi, (None, "noball")), fi, dw, zoom, "render-test")
        cv2.imwrite(args.render_test, disp); print(f"wrote {args.render_test}"); return

    state = {"scale": 1.0}
    def on_mouse(ev, mx, my, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            lab[fi] = ((mx / state["scale"], my / state["scale"]), "human")

    cv2.namedWindow("ball labeler"); cv2.setMouseCallback("ball labeler", on_mouse)
    flagged = sum(1 for f in range(args.start, end+1) if lab.get(f, (None, ""))[1] == "flag")
    print(f"{end-args.start+1} frames, {flagged} flagged.  NAV: a/d (or arrow keys) = prev/next; "
          f"click=set ball, n=no-ball, f=next-flag, [ / ]=zoom, s=save, q=quit")
    while True:
        fr = read(fi)
        if fr is None: fi = min(fi+1, end); continue
        hud = f"[{fi-args.start+1}/{end-args.start+1}]  zoom{zoom}x"
        disp, sc = render(fr, lab.get(fi, (None, "noball")), fi, dw, zoom, hud)
        state["scale"] = sc
        cv2.imshow("ball labeler", disp)
        k = cv2.waitKeyEx(20)                 # Ex = full keycode (arrow keys survive)
        if k == -1:
            continue
        kc = k & 0xFF
        if k in (65363, 63235) or kc in (ord(' '), ord('d')):   # right / d / space = next
            fi = min(fi + 1, end)
        elif k in (65361, 63234) or kc == ord('a'):             # left / a = prev
            fi = max(fi - 1, args.start)
        elif kc == ord('n'): lab[fi] = (None, "noball")
        elif kc == ord('f'):
            nxt = [f for f in range(fi + 1, end + 1) if lab.get(f, (None, ""))[1] == "flag"]
            fi = nxt[0] if nxt else fi
        elif kc == ord(']'): zoom = min(zoom + 1, 8)
        elif kc == ord('['): zoom = max(zoom - 1, 1)
        elif kc == ord('s'): save()
        elif kc == ord('q'): break
    save(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
