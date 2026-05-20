#!/usr/bin/env python3
"""
Temporary demo script — overlays ball + player tracking on a video and saves it.
Delete this file after generating the demo video.

Usage:
    python tests/demo_tracking_overlay.py \
        --input tests/tennis_test6.mov \
        --output tests/tennis_test6_tracked.mp4
"""

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.detection.ball_tracker import BallTracker
from cv.detection.player_detector import PlayerDetector
from cv.detection.court_detector import CourtDetector

# ── Visual constants ──────────────────────────────────────────────────────────

NEAR_COLOR = (80, 220, 80)    # green (BGR)
FAR_COLOR  = (80, 160, 255)   # orange (BGR)
BALL_COLOR = (0, 220, 255)    # yellow (BGR)

TRAIL_LEN     = 8
TRAIL_RADIUS  = 4
BALL_RADIUS   = 8
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.65
FONT_THICK    = 2

COURT_SAMPLE_FRAMES = 5     # frames sampled for court detection
COURT_X_PAD_FRAC    = 0.10  # lateral padding on the perspective X filter
MIN_BOX_H_FRAC      = 0.03  # reject boxes shorter than this fraction of frame height

# Tracker parameters
MAX_LOST_FRAMES = 20   # drop a track after this many consecutive misses
IOU_MATCH_THRESH = 0.05
CENTROID_MAX_MULT = 2.5  # max match distance = box diagonal * this


# ── Reference court Y values (for net estimation) ────────────────────────────

_REF_KP_Y = [561, 561, 2935, 2935,
             561, 2935, 561, 2935,
             1110, 1110, 2386, 2386,
             1110, 2386]
_NET_REF_Y = 1748


# ── Utility functions ────────────────────────────────────────────────────────

def draw_label(frame, text, x, y, color):
    (tw, th), bl = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    pad = 4
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), color, -1)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)


def bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def bbox_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def bbox_diag(box):
    return np.hypot(box[2] - box[0], box[3] - box[1])


# ── Court geometry ───────────────────────────────────────────────────────────

def detect_best_court(cap, detector, n_frames=COURT_SAMPLE_FRAMES):
    best_kps, best_n = None, 0
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        kps = detector.detect(frame, apply_homography=True)
        n = sum(1 for (x, y) in kps if x is not None and y is not None)
        if n > best_n:
            best_n = n
            best_kps = kps
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return best_kps


def estimate_net_y(keypoints):
    pairs = [
        (_REF_KP_Y[i], float(keypoints[i][1]))
        for i in range(min(len(keypoints), len(_REF_KP_Y)))
        if keypoints[i][0] is not None and keypoints[i][1] is not None
    ]
    if len(pairs) < 2:
        return None
    ref = np.array([p[0] for p in pairs])
    vid = np.array([p[1] for p in pairs])
    A = np.vstack([ref, np.ones(len(ref))]).T
    (a, b), *_ = np.linalg.lstsq(A, vid, rcond=None)
    return float(a * _NET_REF_Y + b)


class PerspectiveCourtFilter:
    """
    Perspective-aware court boundary.  At any Y position in the frame the
    valid X range is linearly interpolated between the far baseline edge
    and the near baseline edge — so adjacent courts that are visible at the
    far end (where the court is narrow in pixel space) are correctly excluded.
    """

    def __init__(self, keypoints, pad_frac=COURT_X_PAD_FRAC):
        self.ok = False
        corners = [keypoints[i] for i in range(4)]
        if any(c[0] is None or c[1] is None for c in corners):
            # Fall back: try convex hull of whatever we have
            valid = [(float(x), float(y)) for (x, y) in keypoints
                     if x is not None and y is not None]
            if len(valid) >= 4:
                pts = np.array(valid)
                ys = pts[:, 1]
                far_mask = ys < np.median(ys)
                near_mask = ~far_mask
                self.far_left  = float(pts[far_mask, 0].min())
                self.far_right = float(pts[far_mask, 0].max())
                self.far_y     = float(pts[far_mask, 1].mean())
                self.near_left  = float(pts[near_mask, 0].min())
                self.near_right = float(pts[near_mask, 0].max())
                self.near_y     = float(pts[near_mask, 1].mean())
                self.ok = True
            return

        # kp 0=far-left  1=far-right  2=near-left  3=near-right
        self.far_left   = float(corners[0][0])
        self.far_right  = float(corners[1][0])
        self.far_y      = (float(corners[0][1]) + float(corners[1][1])) / 2
        self.near_left  = float(corners[2][0])
        self.near_right = float(corners[3][0])
        self.near_y     = (float(corners[2][1]) + float(corners[3][1])) / 2
        self.pad_frac   = pad_frac
        self.ok = True

    def contains(self, x: float, y: float) -> bool:
        if not self.ok:
            return True
        dy = self.near_y - self.far_y
        if abs(dy) < 1:
            return True
        t = (y - self.far_y) / dy   # 0 at far baseline, 1 at near baseline; extrapolates beyond

        left  = self.far_left  + t * (self.near_left  - self.far_left)
        right = self.far_right + t * (self.near_right - self.far_right)
        w = max(right - left, 1)
        pad = self.pad_frac * w
        return (left - pad) <= x <= (right + pad)

    def debug_x_at(self, y: float):
        dy = self.near_y - self.far_y
        if abs(dy) < 1:
            return None, None
        t = (y - self.far_y) / dy
        left  = self.far_left  + t * (self.near_left  - self.far_left)
        right = self.far_right + t * (self.near_right - self.far_right)
        w = max(right - left, 1)
        pad = self.pad_frac * w
        return int(left - pad), int(right + pad)


# ── Player tracker ───────────────────────────────────────────────────────────

class PlayerTracker:
    """
    Simple IoU + centroid tracker that locks near/far identity.

    On each update():
      1. Match new detections to existing tracks by IoU, then centroid fallback.
      2. Update matched tracks' bbox but KEEP their side (near/far) assignment.
      3. Create new tracks for unmatched detections (capped at 4 total).
      4. Expire tracks not seen for MAX_LOST_FRAMES.
    """

    def __init__(self, net_y: float, max_lost: int = MAX_LOST_FRAMES):
        self.net_y = net_y
        self.max_lost = max_lost
        self.tracks: List[Dict] = []
        self._next_id = 0

    def _assign_side(self, bbox) -> str:
        foot_y = bbox[3]
        return "near" if foot_y > self.net_y else "far"

    def update(self, detections: List) -> List[Dict]:
        """
        detections: list of [x1, y1, x2, y2] arrays/tuples.
        Returns list of active track dicts {id, bbox, side}.
        """
        matched_t = set()
        matched_d = set()

        # ── Pass 1: IoU matching ──────────────────────────────────────────────
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                s = bbox_iou(tr["bbox"], det)
                if s >= IOU_MATCH_THRESH:
                    pairs.append((ti, di, s))
        pairs.sort(key=lambda x: -x[2])
        for ti, di, _ in pairs:
            if ti in matched_t or di in matched_d:
                continue
            self.tracks[ti]["bbox"] = detections[di]
            self.tracks[ti]["lost"] = 0
            matched_t.add(ti)
            matched_d.add(di)

        # ── Pass 2: centroid fallback for remaining ───────────────────────────
        for ti, tr in enumerate(self.tracks):
            if ti in matched_t:
                continue
            tc = bbox_center(tr["bbox"])
            max_dist = bbox_diag(tr["bbox"]) * CENTROID_MAX_MULT
            best_di, best_d = None, float("inf")
            for di, det in enumerate(detections):
                if di in matched_d:
                    continue
                dc = bbox_center(det)
                d = np.hypot(tc[0] - dc[0], tc[1] - dc[1])
                if d < best_d and d < max_dist:
                    best_d, best_di = d, di
            if best_di is not None:
                self.tracks[ti]["bbox"] = detections[best_di]
                self.tracks[ti]["lost"] = 0
                matched_t.add(ti)
                matched_d.add(best_di)

        # ── Housekeeping ──────────────────────────────────────────────────────
        for ti in range(len(self.tracks)):
            if ti not in matched_t:
                self.tracks[ti]["lost"] += 1

        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]

        # ── New tracks for unmatched detections ───────────────────────────────
        near_count = sum(1 for t in self.tracks if t["side"] == "near")
        far_count  = sum(1 for t in self.tracks if t["side"] == "far")
        for di, det in enumerate(detections):
            if di in matched_d:
                continue
            if len(self.tracks) >= 4:
                break
            side = self._assign_side(det)
            if side == "near" and near_count >= 2:
                continue
            if side == "far" and far_count >= 2:
                continue
            self.tracks.append({
                "id": self._next_id, "bbox": det,
                "side": side, "lost": 0,
            })
            self._next_id += 1
            if side == "near":
                near_count += 1
            else:
                far_count += 1

        return [t for t in self.tracks if t["lost"] == 0]


# ── Main ─────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, frame_skip: int = 1) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_box_h = int(MIN_BOX_H_FRAC * height)

    print(f"Input : {input_path}")
    print(f"Video : {width}x{height} @ {fps:.1f} fps  ({total} frames)")

    # ── Court detection ───────────────────────────────────────────────────────
    print(f"Detecting court from first {COURT_SAMPLE_FRAMES} frames…")
    court_detector = CourtDetector()
    keypoints = detect_best_court(cap, court_detector)

    court_filter = PerspectiveCourtFilter(keypoints) if keypoints else None
    net_y = estimate_net_y(keypoints) if keypoints else height * 0.4

    if court_filter and court_filter.ok:
        print(f"  Court filter : perspective-aware (far {court_filter.far_left:.0f}–{court_filter.far_right:.0f}  "
              f"near {court_filter.near_left:.0f}–{court_filter.near_right:.0f})")
        print(f"  Net Y        : {net_y:.0f}px")
    else:
        print("  ⚠ Court detection failed — no spatial filtering")

    # ── Setup ─────────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))

    ball_tracker    = BallTracker()
    player_detector = PlayerDetector()
    player_tracker  = PlayerTracker(net_y=net_y)
    ball_trail: deque = deque(maxlen=TRAIL_LEN)

    frame_idx  = 0
    written    = 0
    det_counts = {"ball": 0, "player": 0}

    print("Processing frames…")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        annotated = frame.copy()

        # ── Debug: draw perspective bounds at a few Y levels ──────────────────
        if court_filter and court_filter.ok:
            ov = annotated.copy()
            for probe_y in range(0, height, height // 8):
                lx, rx = court_filter.debug_x_at(probe_y)
                if lx is not None:
                    cv2.circle(ov, (lx, probe_y), 3, (255, 255, 255), -1)
                    cv2.circle(ov, (rx, probe_y), 3, (255, 255, 255), -1)
            if net_y:
                cv2.line(ov, (0, int(net_y)), (width, int(net_y)), (200, 200, 0), 1)
            cv2.addWeighted(ov, 0.2, annotated, 0.8, 0, annotated)

        # ── Ball ──────────────────────────────────────────────────────────────
        ball_det = ball_tracker.detect_ball(frame)
        if ball_det is not None:
            (bx, by), _, _ = ball_det
            ball_trail.append((bx, by))
            det_counts["ball"] += 1
        else:
            ball_trail.append(None)

        for i, pt in enumerate(ball_trail):
            if pt is None:
                continue
            alpha  = (i + 1) / len(ball_trail)
            r = max(1, int(TRAIL_RADIUS * alpha))
            c = tuple(int(v * alpha) for v in BALL_COLOR)
            cv2.circle(annotated, pt, r, c, -1, cv2.LINE_AA)
        if ball_det is not None:
            cv2.circle(annotated, (bx, by), BALL_RADIUS, BALL_COLOR, -1, cv2.LINE_AA)
            cv2.circle(annotated, (bx, by), BALL_RADIUS + 2, (0, 0, 0), 2, cv2.LINE_AA)

        # ── Players ───────────────────────────────────────────────────────────
        raw_boxes = player_detector.run_human_detection(frame, bbox_thr=0.15, nms_thr=0.3)

        on_court = []
        for box in raw_boxes:
            x1, y1, x2, y2 = box
            if (y2 - y1) < min_box_h:
                continue
            foot_x, foot_y = (x1 + x2) / 2, y2
            if court_filter and not court_filter.contains(foot_x, foot_y):
                continue
            on_court.append([float(x1), float(y1), float(x2), float(y2)])

        # Feed filtered detections into the tracker
        active = player_tracker.update(on_court)
        det_counts["player"] += len(active)

        near_n, far_n = 0, 0
        for tr in active:
            x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
            if tr["side"] == "near":
                near_n += 1
                color, label = NEAR_COLOR, f"Near {near_n}"
            else:
                far_n += 1
                color, label = FAR_COLOR, f"Far {far_n}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS, cv2.LINE_AA)
            draw_label(annotated, label, x1, max(y1 - 6, 20), color)

        # ── HUD ──────────────────────────────────────────────────────────────
        ts  = frame_idx / fps
        hud = f"Frame {frame_idx}/{total}  |  {ts:.1f}s"
        cv2.putText(annotated, hud, (12, 32), FONT, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(annotated, hud, (12, 32), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(annotated)
        written += 1
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100
            print(f"  {frame_idx}/{total}  ({pct:.0f}%)", flush=True)

    cap.release()
    out.release()

    print(f"\nDone.")
    print(f"  Frames written : {written}")
    print(f"  Ball detections: {det_counts['ball']}")
    print(f"  Player detects : {det_counts['player']}")
    print(f"  Output saved   : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay ball + player tracking on a tennis video")
    parser.add_argument("--input",      required=True,       help="Input video path")
    parser.add_argument("--output",     required=True,       help="Output video path (.mp4)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()
    run(args.input, args.output, frame_skip=args.frame_skip)


if __name__ == "__main__":
    main()
