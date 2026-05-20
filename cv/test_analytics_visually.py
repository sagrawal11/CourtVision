"""
cv/test_analytics_visually.py — Run the full analytics pipeline locally with visual debug output.

Interactive two-step flow:
  1. (Optional) Click 14 court keypoints on the first frame  OR  pass --corners for 4-point quick setup
  2. (Optional) Click to select the target player (POI), OR pass --poi near|far

Then runs:
  - BallTracker  (TrackNet)
  - PlayerDetector (YOLO)
  - BounceDetector
  - HitDetector   ← new, runs directly on raw frame data (no point segmentation needed)

Output video overlays:
  - Court lines from keypoints
  - Ball trajectory trail
  - Player bounding boxes (near=green, far=orange)
  - 💥 HIT annotations (type + speed)
  - 🎾 BOUNCE markers
  - HUD stats panel

Usage:
  tennis_env/bin/python cv/test_analytics_visually.py \\
      --input tests/tennis_test6.mov \\
      --poi near \\
      --corners "531,348 1076,346 64,888 1747,888"  \\
      --max-frames 400 \\
      --output test_run.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # cv/ → project root
sys.path.insert(0, str(PROJECT_ROOT))

from cv.detection.ball_tracker import BallTracker
from cv.detection.player_detector import PlayerDetector
from cv.detection.court_detector import CourtDetector, REF_X_MIN, REF_X_MAX, REF_Y_MIN, REF_Y_MAX
from cv.analysis.point_detector import HitDetector, HitEvent, BounceDetector


# ── Court line connectivity ────────────────────────────────────────────────────

COURT_LINES = [
    (0, 1), (2, 3),   # Baselines
    (0, 2), (1, 3),   # Doubles sidelines
    (4, 5), (6, 7),   # Singles sidelines
    (8, 9), (10, 11), # Service lines
    (12, 13),         # Centre service line
]


def _build_homography(keypoints: list) -> Optional[np.ndarray]:
    ref = CourtDetector.REFERENCE_KEYPOINTS
    src, dst = [], []
    for i, kp in enumerate(keypoints):
        if kp is not None and kp[0] is not None:
            src.append([float(kp[0]), float(kp[1])])
            dst.append([float(ref[i][0]), float(ref[i][1])])
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32), cv2.RANSAC, 5.0)
    return H


def _norm_to_video(H_inv: np.ndarray, nx: float, ny: float) -> Tuple[int, int]:
    ref_x = nx * (REF_X_MAX - REF_X_MIN) + REF_X_MIN
    ref_y = ny * (REF_Y_MAX - REF_Y_MIN) + REF_Y_MIN
    pt = np.array([[[ref_x, ref_y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_inv)
    return int(out[0, 0, 0]), int(out[0, 0, 1])


# ── Interactive setup ──────────────────────────────────────────────────────────

def pick_keypoints_interactive(frame: np.ndarray) -> list:
    """Click 14 court keypoints interactively. Press Q to skip and use rough 4-point mode."""
    frame_copy = frame.copy()
    points: list = []
    names = [
        "0: Far-L baseline (dbl)", "1: Far-R baseline (dbl)",
        "2: Near-L baseline (dbl)", "3: Near-R baseline (dbl)",
        "4: Far-L singles", "5: Near-L singles",
        "6: Far-R singles", "7: Near-R singles",
        "8: Far svc-line L", "9: Far svc-line R",
        "10: Near svc-line L", "11: Near svc-line R",
        "12: Centre svc far", "13: Centre svc near",
    ]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 14:
            points.append((float(x), float(y)))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(len(points) - 1), (x + 6, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            label = names[len(points) - 1] if len(points) <= len(names) else ""
            cv2.putText(frame_copy, f"Next: {label}", (10, frame_copy.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(WIN, frame_copy)

    WIN = "Click 14 keypoints (order: see console). Press ENTER when done, Q to skip."
    print("\nKeypoint order:")
    for n in names:
        print(f"  {n}")
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)
    cv2.imshow(WIN, frame_copy)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 or len(points) == 14:  # Enter
            break
        if key == ord('q'):
            points.clear()
            break
    cv2.destroyAllWindows()

    # Pad to 14
    kps: list = [None] * 14
    for i, pt in enumerate(points[:14]):
        kps[i] = pt
    return kps


def pick_poi_interactive(frame: np.ndarray, player_bboxes: list) -> str:
    """Show the first frame with detected players and let the user click one."""
    frame_copy = frame.copy()
    for i, bbox in enumerate(player_bboxes[:2]):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        colour = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), colour, 3)
        label = "P0 (near)" if i == 0 else "P1 (far)"
        cv2.putText(frame_copy, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    selected = ["near"]
    WIN = "Click the TARGET player. Press ENTER to confirm (default: near/green)"
    cv2.namedWindow(WIN)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(player_bboxes) >= 2:
            cx0 = (player_bboxes[0][0] + player_bboxes[0][2]) / 2
            cx1 = (player_bboxes[1][0] + player_bboxes[1][2]) / 2
            if abs(x - cx1) < abs(x - cx0):
                selected[0] = "far"
            else:
                selected[0] = "near"
            msg_frame = frame_copy.copy()
            cv2.putText(msg_frame, f"Selected: {selected[0].upper()}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow(WIN, msg_frame)

    cv2.setMouseCallback(WIN, on_click)
    cv2.imshow(WIN, frame_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return selected[0]


# ── Rendering ─────────────────────────────────────────────────────────────────

def draw_court(frame: np.ndarray, keypoints: list, H_inv: Optional[np.ndarray]) -> None:
    for i, j in COURT_LINES:
        ki, kj = keypoints[i] if i < len(keypoints) else None, keypoints[j] if j < len(keypoints) else None
        if ki and ki[0] is not None and kj and kj[0] is not None:
            cv2.line(frame, (int(ki[0]), int(ki[1])), (int(kj[0]), int(kj[1])),
                     (255, 255, 255), 2, cv2.LINE_AA)
    if H_inv is not None:
        nl = _norm_to_video(H_inv, 0.0, 0.5)
        nr = _norm_to_video(H_inv, 1.0, 0.5)
        cv2.line(frame, nl, nr, (200, 200, 200), 3, cv2.LINE_AA)


def draw_keypoint_dots(frame: np.ndarray, keypoints: list) -> None:
    for i, kp in enumerate(keypoints):
        if kp and kp[0] is not None:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, str(i), (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1)


def draw_players(frame: np.ndarray, near_bbox, far_bbox) -> None:
    if near_bbox is not None:
        x1, y1, x2, y2 = int(near_bbox[0]), int(near_bbox[1]), int(near_bbox[2]), int(near_bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "NEAR", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if far_bbox is not None:
        x1, y1, x2, y2 = int(far_bbox[0]), int(far_bbox[1]), int(far_bbox[2]), int(far_bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, "FAR", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)


def draw_hud(frame: np.ndarray, hits: list, bounces: list, frame_idx: int, fps: float,
             poi_side: str) -> None:
    h, w = frame.shape[:2]
    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    ts = frame_idx / max(fps, 1)
    poi_hits = [x for x in hits if x.player == poi_side]
    fhs   = [x for x in poi_hits if x.shot_type == "forehand" and x.speed_kmh]
    bhs   = [x for x in poi_hits if x.shot_type == "backhand" and x.speed_kmh]
    serves= [x for x in poi_hits if x.shot_type == "serve" and x.speed_kmh]

    lines = [
        (f"Frame {frame_idx}  |  {ts:.1f}s", (200, 200, 200)),
        (f"POI: {poi_side.upper()}  Hits: {len(poi_hits)}", (0, 255, 255)),
        (f"Bounces: {len(bounces)}", (50, 130, 255)),
        (f"Avg FH:  {(sum(x.speed_kmh for x in fhs)/len(fhs)):.0f} km/h" if fhs
         else "Avg FH:   --", (100, 255, 100)),
        (f"Avg BH:  {(sum(x.speed_kmh for x in bhs)/len(bhs)):.0f} km/h" if bhs
         else "Avg BH:   --", (100, 255, 100)),
        (f"Avg Srv: {(sum(x.speed_kmh for x in serves)/len(serves)):.0f} km/h" if serves
         else "Avg Srv:  --", (100, 255, 100)),
    ]
    for row, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (8, 18 + row * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)


def draw_bounce(frame: np.ndarray, bx: float, by: float, age: int) -> None:
    radius = max(6, 18 - age * 2)
    alpha_c = max(80, 255 - age * 25)
    colour = (0, alpha_c, 255)
    cv2.circle(frame, (int(bx), int(by)), radius, colour, 2, cv2.LINE_AA)
    if age < 5:
        cv2.putText(frame, "BOUNCE", (int(bx) + 8, int(by) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def draw_hit(frame: np.ndarray, hit: HitEvent, age: int, poi_side: str) -> None:
    bx, by = int(hit.x), int(hit.y)

    if hit.is_winner:
        colour = (0, 255, 0)
        tag = "WINNER!"
    elif hit.is_error:
        colour = (0, 0, 255)
        tag = "ERROR!"
    elif hit.player == poi_side:
        colour = (0, 255, 255)
        tag = ""
    else:
        colour = (128, 128, 128)
        tag = ""

    radius = max(8, 20 - age * 2)
    cv2.circle(frame, (bx, by), radius, colour, 2, cv2.LINE_AA)

    label_parts = []
    if hit.shot_type:
        label_parts.append(hit.shot_type.upper())
    if hit.speed_kmh:
        label_parts.append(f"{hit.speed_kmh:.0f}km/h")
    if tag:
        label_parts.append(tag)

    label = "  ".join(label_parts)
    player_label = "POI" if hit.player == poi_side else "OPP"
    full = f"[{player_label}] {label}" if label else f"[{player_label}] HIT"

    # Offset label upward as it ages so it floats away
    ty = max(20, by - 18 - age * 3)
    cv2.putText(frame, full, (bx - 40, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visual analytics debug for tennis CV pipeline")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="analytics_debug.mp4")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--poi", choices=["near", "far"], help="Target player side (skip interactive)")
    parser.add_argument("--corners", help="4 space-separated x,y pairs for quick court setup (fl,fr,nl,nr)")
    args = parser.parse_args()

    # ── Step 1: Open video & get first frame ─────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.input}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        print("ERROR: Cannot read first frame")
        sys.exit(1)

    # ── Step 2: Court keypoints ───────────────────────────────────────────────
    if args.corners:
        pts = [tuple(map(int, p.split(","))) for p in args.corners.split()]
        keypoints: list = [None] * 14
        # Map: far-L, far-R, near-L, near-R → kp4, kp6, kp5, kp7 (singles corners)
        for idx, kp_i in zip([4, 6, 5, 7], pts[:4]):
            keypoints[idx] = (float(pts[0 if idx == 4 else 1 if idx == 6 else 2 if idx == 5 else 3][0]),
                              float(pts[0 if idx == 4 else 1 if idx == 6 else 2 if idx == 5 else 3][1]))
    else:
        print("Opening court keypoint picker... (Press Q to skip to 4-corner mode)")
        keypoints = pick_keypoints_interactive(first_frame.copy())

    # Build homography from keypoints
    H = _build_homography(keypoints)
    H_inv = np.linalg.inv(H) if H is not None else None
    print(f"Homography: {'OK' if H is not None else 'FAILED (too few keypoints)'}")

    # ── Step 3: Load detectors ────────────────────────────────────────────────
    print("Loading models...")
    ball_tracker  = BallTracker()
    player_det    = PlayerDetector()
    print("Models loaded.")

    # ── Step 4: Run full forward pass ─────────────────────────────────────────
    max_frames = min(args.max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ball_positions: list  = []    # Optional[(x, y)]
    near_positions: list  = []    # Optional[(cx, cy)]
    far_positions:  list  = []    # Optional[(cx, cy)]
    all_bboxes:     list  = []    # list of bboxes per frame for drawing

    frame_buf: list = []  # store raw frames for rendering

    print(f"\nPass 1/2 — Running detectors on {max_frames} frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_player_bboxes = []
    for fi in tqdm(range(max_frames), desc="Detecting"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_buf.append(frame)

        # Ball
        result = ball_tracker.detect_ball(frame)
        if result:
            center, _, _ = result
            ball_positions.append((float(center[0]), float(center[1])))
        else:
            ball_positions.append(None)

        # Players
        bboxes = player_det.run_human_detection(frame, bbox_thr=0.35)
        # Sort: near player = larger y centroid (bottom of frame), far = smaller y
        if len(bboxes) >= 2:
            bboxes_sorted = sorted(bboxes[:2], key=lambda b: (b[1] + b[3]) / 2, reverse=True)
            near_box, far_box = bboxes_sorted[0], bboxes_sorted[1]
        elif len(bboxes) == 1:
            bboxes_sorted = bboxes[:1]
            cy = (bboxes[0][1] + bboxes[0][3]) / 2
            near_box = bboxes[0] if cy > height * 0.5 else None
            far_box  = None if cy > height * 0.5 else bboxes[0]
        else:
            near_box = far_box = None
            bboxes_sorted = []

        all_bboxes.append((near_box, far_box))

        near_positions.append(
            ((near_box[0] + near_box[2]) / 2, (near_box[1] + near_box[3]) / 2) if near_box is not None else None
        )
        far_positions.append(
            ((far_box[0] + far_box[2]) / 2, (far_box[1] + far_box[3]) / 2) if far_box is not None else None
        )

        if fi == 0:
            first_player_bboxes = list(bboxes_sorted)

    cap.release()
    print(f"Detection done. Ball visible in {sum(1 for b in ball_positions if b is not None)}/{len(ball_positions)} frames.")

    # ── Step 5: Select POI ────────────────────────────────────────────────────
    if args.poi:
        poi_side = args.poi
    else:
        poi_side = pick_poi_interactive(frame_buf[0].copy(), first_player_bboxes)
    print(f"POI: {poi_side.upper()}")

    # ── Step 6: Detect bounces & hits ─────────────────────────────────────────
    print("\nPass 2/2 — Running BounceDetector + HitDetector...")
    bounce_det = BounceDetector()
    raw_bounces = bounce_det.detect(ball_positions)  # list of (frame_idx, x, y)

    # Only pass homography for speed calculation when we have enough reliable keypoints
    valid_kps = sum(1 for k in keypoints if k is not None and k[0] is not None)
    H_for_speed = H if valid_kps >= 8 else None
    hit_det = HitDetector(fps=fps, homography=H_for_speed)
    hits: List[HitEvent] = hit_det.detect(
        ball_positions=ball_positions,
        player_near_pos=near_positions,
        player_far_pos=far_positions,
        bounces=raw_bounces,
    )
    # Clamp unrealistic speeds (bad homography can produce absurd values)
    for h in hits:
        if h.speed_kmh is not None and (h.speed_kmh > 300 or h.speed_kmh < 0):
            h.speed_kmh = None

    print(f"Bounces detected: {len(raw_bounces)}")
    print(f"Hits detected:    {len(hits)}")
    for h in hits:
        speed_str = f"{h.speed_kmh:.0f} km/h" if h.speed_kmh else "n/a"
        print(f"  Frame {h.frame_idx:4d}  player={h.player:4s}  type={str(h.shot_type):10s}  speed={speed_str:10s}"
              f"  winner={h.is_winner}  error={h.is_error}")

    # ── Step 7: Render annotated video ────────────────────────────────────────
    print(f"\nRendering annotated video → {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    TRAIL_LEN = 25
    ball_trail: list = []

    for fi, frame in enumerate(tqdm(frame_buf, desc="Rendering")):
        # 1. Court
        draw_court(frame, keypoints, H_inv)
        if fi < 90:
            draw_keypoint_dots(frame, keypoints)

        # 2. Players
        near_box, far_box = all_bboxes[fi]
        draw_players(frame, near_box, far_box)

        # 3. Ball + trail
        bp = ball_positions[fi]
        if bp:
            center_px = (int(bp[0]), int(bp[1]))
            ball_trail.append(center_px)
        if len(ball_trail) > TRAIL_LEN:
            ball_trail.pop(0)
        if ball_trail:
            # Draw trail
            for t, pt in enumerate(ball_trail[:-1]):
                alpha = int(220 * (t + 1) / len(ball_trail))
                cv2.circle(frame, pt, max(2, 5 - (len(ball_trail) - t) // 5),
                           (0, alpha // 2, alpha), -1, cv2.LINE_AA)
            if bp:
                cv2.circle(frame, center_px, 8, (30, 220, 30), -1, cv2.LINE_AA)
                cv2.circle(frame, center_px, 9, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Bounces (recent window)
        for bframe, bx, by in raw_bounces:
            age = fi - bframe
            if 0 <= age < 12:
                draw_bounce(frame, bx, by, age)

        # 5. Hits (recent window)
        for hit in hits:
            age = fi - hit.frame_idx
            if 0 <= age < 20:
                draw_hit(frame, hit, age, poi_side)

        # 6. HUD
        draw_hud(frame, hits, raw_bounces, fi, fps, poi_side)

        out_vid.write(frame)

    out_vid.release()
    print(f"\n✅  Done! Output saved to: {Path(args.output).resolve()}")
    print(f"   Total hits: {len(hits)} | Total bounces: {len(raw_bounces)}")


if __name__ == "__main__":
    main()
