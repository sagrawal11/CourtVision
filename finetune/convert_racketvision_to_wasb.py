#!/usr/bin/env python3
"""
Convert RacketVision tennis clips into the on-disk format WASB-SBDT's Tennis
dataset loader expects (TrackNet layout).

RacketVision (per clip):
    <rv_root>/tennis/videos/<match>_<rally>.mp4              # 1920x1080 @ 60fps, ~600 frames
    <rv_root>/tennis/all/<match>/csv/<rally>_ball.csv        # columns: Frame,Visibility,X,Y  (SPARSE, ~1/12 frames; (0,0)=invisible)

WASB-SBDT Tennis expects (per clip), driven by src/datasets/tennis.py + utils/file.py:load_csv_tennis:
    <out_root>/<match>/<clip>/000000.jpg 000001.jpg ...      # EVERY frame, contiguous 0..N-1, stem parses to int
    <out_root>/<match>/<clip>/Label.csv                      # columns: file name,visibility,x-coordinate,y-coordinate
                                                             #   ONE ROW PER EXTRACTED FRAME (dense, contiguous)
                                                             #   visibility in {0,1}; the eval/loss treats visibility in
                                                             #   dataset.visible_flags (default [1,2]) as "ball present".
                                                             #   x,y in ORIGINAL pixel coords (loader affine-warps to net input).

Why dense: tennis.py does `ball_xyvs[j] for j in range(len(ball_xyvs)-frames_in+1)`, indexing the CSV by
CONTIGUOUS integer frame ids starting at 0, and pairs them 1:1 with sorted frame files. So the CSV must have a
row for every extracted frame and frame files must be named by those same contiguous ids.

Label densification (optional): RacketVision labels are sparse (~1 per 12 frames). Unlabeled frames are written
visibility=0 by default. With --interp-max-gap N>0 we LINEARLY interpolate ball (x,y) across gaps of <= N frames
between two consecutive VISIBLE labels and mark those frames visible=1. This is only an approximation (the ball
arcs) and is meant to give the trainer positive targets for a smoke test / short warm-start; for a production
fine-tune, densify with the ballistic-fit + human-correct loop (see FINETUNE_WASB.md), not naive interpolation.

Usage:
    python3 convert_racketvision_to_wasb.py \
        --rv-root /tmp/racketvision_work/data \
        --out-root /tmp/wasb_finetune_work/datasets/tennis_rv \
        --clips match1/000 match10/000 match100/000 \
        --interp-max-gap 12 [--frame-range START END] [--jpg-quality 95]
"""
import argparse
import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import cv2


def parse_clip(clip_str):
    # "match1/000" or "match1_000"
    if '/' in clip_str:
        match, rally = clip_str.split('/')
    else:
        match, rally = clip_str.rsplit('_', 1)
    return match, rally


def load_rv_labels(csv_path):
    """Return dict {frame_id:int -> (visible:bool, x:float, y:float)} from a RacketVision *_ball.csv."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        fid = int(r['Frame'])
        x, y = float(r['X']), float(r['Y'])
        vis = int(r['Visibility']) == 1 and not (x == 0 and y == 0)
        out[fid] = (vis, x, y)
    return out


def densify(labels, n_frames, interp_max_gap):
    """
    Build a dense per-frame list of (visibility, x, y) for frames 0..n_frames-1.
    labels: {fid -> (vis,x,y)} sparse.
    interp_max_gap: max gap (in frames) between two visible labels to linearly interpolate across.
                    0 disables interpolation (only true labels are visible).
    """
    dense = [(0, 0.0, 0.0)] * n_frames
    # place true labels
    for fid, (vis, x, y) in labels.items():
        if 0 <= fid < n_frames:
            dense[fid] = (1 if vis else 0, x if vis else 0.0, y if vis else 0.0)

    if interp_max_gap and interp_max_gap > 0:
        vis_fids = sorted([fid for fid, (v, _, _) in labels.items()
                           if v and 0 <= fid < n_frames])
        for a, b in zip(vis_fids[:-1], vis_fids[1:]):
            gap = b - a
            if 1 < gap <= interp_max_gap:
                _, xa, ya = dense[a]
                _, xb, yb = dense[b]
                for k in range(1, gap):
                    t = k / gap
                    xi = xa + (xb - xa) * t
                    yi = ya + (yb - ya) * t
                    dense[a + k] = (1, xi, yi)
    return dense


def convert_clip(rv_root, out_root, match, rally, interp_max_gap, frame_range, jpg_quality):
    video_path = osp.join(rv_root, 'tennis', 'videos', '{}_{}.mp4'.format(match, rally))
    csv_path   = osp.join(rv_root, 'tennis', 'all', match, 'csv', '{}_ball.csv'.format(rally))
    assert osp.exists(video_path), 'missing video: {}'.format(video_path)
    assert osp.exists(csv_path), 'missing csv: {}'.format(csv_path)

    clip_out_dir = osp.join(out_root, match, '{}'.format(rally))
    os.makedirs(clip_out_dir, exist_ok=True)

    labels = load_rv_labels(csv_path)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_range is not None:
        f0, f1 = frame_range
        f1 = min(f1, total)
    else:
        f0, f1 = 0, total

    # shift labels so that extracted range starts at contiguous id 0
    shifted = {}
    for fid, v in labels.items():
        if f0 <= fid < f1:
            shifted[fid - f0] = v
    n_out = f1 - f0
    dense = densify(shifted, n_out, interp_max_gap)

    # extract frames [f0, f1) -> 000000.jpg .. (n_out-1).jpg
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for i in range(n_out):
        ok, frame = cap.read()
        if not ok:
            break
        out_name = '{:06d}.jpg'.format(i)
        cv2.imwrite(osp.join(clip_out_dir, out_name),
                    frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        written += 1
    cap.release()

    # write dense Label.csv (one row per WRITTEN frame)
    rows = []
    for i in range(written):
        vis, x, y = dense[i]
        rows.append({'file name': '{:06d}.jpg'.format(i),
                     'visibility': int(vis),
                     'x-coordinate': (x if vis else 0),
                     'y-coordinate': (y if vis else 0)})
    df = pd.DataFrame(rows, columns=['file name', 'visibility', 'x-coordinate', 'y-coordinate'])
    df.to_csv(osp.join(clip_out_dir, 'Label.csv'), index=False)

    n_vis = int(sum(1 for r in rows if r['visibility'] == 1))
    print('  {}/{}: {} frames written, {} visible ({:.0f}%), Label.csv rows={}'.format(
        match, rally, written, n_vis, 100.0 * n_vis / max(written, 1), len(rows)))
    return written, n_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rv-root', default='/tmp/racketvision_work/data')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--clips', nargs='+', required=True, help='e.g. match1/000 match10/000')
    ap.add_argument('--interp-max-gap', type=int, default=0,
                    help='linearly interpolate ball across visible-label gaps <= this many frames (0=off)')
    ap.add_argument('--frame-range', nargs=2, type=int, default=None, metavar=('START', 'END'),
                    help='extract only frames [START,END) (contiguous) instead of the whole clip')
    ap.add_argument('--jpg-quality', type=int, default=95)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    print('converting {} clip(s) -> {} (interp_max_gap={})'.format(
        len(args.clips), args.out_root, args.interp_max_gap))
    tot_f, tot_v = 0, 0
    for c in args.clips:
        match, rally = parse_clip(c)
        w, v = convert_clip(args.rv_root, args.out_root, match, rally,
                            args.interp_max_gap, args.frame_range, args.jpg_quality)
        tot_f += w
        tot_v += v
    print('DONE. total frames={}, total visible={} ({:.0f}%)'.format(
        tot_f, tot_v, 100.0 * tot_v / max(tot_f, 1)))


if __name__ == '__main__':
    main()
