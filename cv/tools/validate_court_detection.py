"""cv/tools/validate_court_detection.py — measure the auto court detector's accuracy.

Runs CourtDetector (yastrebksv TennisCourtDetector, models/court/
model_tennis_court_det.pt) on sampled frames of real footage, scores each frame's
14 keypoints with cv/eval/homography.check(), and reports two things per video:

  1. the per-frame distribution (how noisy any single frame is), and
  2. the BEST-frame pick — what production actually uses. Because the camera is
     static, we score many frames and take the single best-fitting one; that lands
     ~1-15px even when most frames are noisy from player occlusion.

check() calls a fit good at inliers>=12 and mean_err<=50px.

    python cv/tools/validate_court_detection.py --all
    python cv/tools/validate_court_detection.py "tests/Indoor Match 1 15.53.25.mp4"
    python cv/tools/validate_court_detection.py --all --every 15 --max-frames 450
    python cv/tools/validate_court_detection.py --all --first-only   # single-frame baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_ALL_CLIPS = [
    "tests/Indoor Match 1 15.53.25.mp4",
    "tests/Indoor Match 2 15.53.25.mp4",
    "tests/Outdoor Match 1 15.53.25.mp4",
    "tests/Outdoor Practice 15.53.25.mp4",
    "tests/tennis_test6.mov",
]


def _median(vals):
    return round(float(np.median(vals)), 1) if vals else None


def validate_video(video_path: Path, detector, *, every: int, max_frames: int, first_only: bool) -> dict:
    """Score sampled frames by NATIVE keypoint inliers; report the best + trust."""
    if first_only:
        from cv.detection.court_detector import fit_native
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = (cap.isOpened(), None)
        if ok:
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return {"video": video_path.name, "error": "could not read frame 0"}
        H, inl, err = fit_native(detector.detect(frame, apply_homography=False))
        best = {"_frame": 0, "native_inliers": inl,
                "mean_err_px": round(err, 1) if H is not None else None,
                "geom_ok": None, "trustworthy": inl >= 6}
        all_scores = [best]
    else:
        _kps, best, all_scores = detector.detect_best_frame(
            video_path, sample_every=every, max_frames=max_frames, return_diagnostics=True
        )

    inls = [s["native_inliers"] for s in all_scores if s.get("native_inliers") is not None]
    return {
        "video": video_path.name,
        "sampled": len(all_scores),
        "native_max": max(inls) if inls else 0,
        "native_median": _median(inls),
        "geom_ok_frames": sum(1 for s in all_scores if s.get("geom_ok")),
        "best_frame": best.get("_frame"),
        "best_native_inliers": best.get("native_inliers"),
        "best_mean_err": best.get("mean_err_px"),
        "best_geom_ok": best.get("geom_ok"),
        "trustworthy": best.get("trustworthy"),
        "reason": best.get("reason"),
    }


def _report(agg: dict) -> None:
    if "error" in agg:
        print(f"\n{agg['video']}: {agg['error']}")
        return
    print(f"\n{agg['video']}   (sampled {agg['sampled']} frames)")
    print(f"  native inliers : max {agg['native_max']}  median {agg['native_median']}  "
          f"(geometry-valid in {agg['geom_ok_frames']} frames)")
    if agg.get("best_native_inliers") is not None:
        verdict = "TRUSTWORTHY ✓" if agg["trustworthy"] else "NOT TRUSTED ✗ — use manual"
        geom = "" if agg["best_geom_ok"] is None else f"  geom {'ok' if agg['best_geom_ok'] else 'BAD'}"
        print(f"  BEST FRAME     : {verdict}  frame {agg['best_frame']}  "
              f"native inliers {agg['best_native_inliers']}  reproj {agg['best_mean_err']}px{geom}")
    else:
        print(f"  BEST FRAME     : none fittable — {agg.get('reason')}")


def main():
    ap = argparse.ArgumentParser(description="Validate the auto court detector on real footage")
    ap.add_argument("videos", nargs="*", help="video paths (relative to project root or absolute)")
    ap.add_argument("--all", action="store_true", help="sweep the bundled tests/ clips")
    ap.add_argument("--every", type=int, default=15, help="sample every Nth frame (default 15)")
    ap.add_argument("--max-frames", type=int, default=450, help="only sample within the first N frames")
    ap.add_argument("--first-only", action="store_true", help="single-frame baseline (frame 0)")
    ap.add_argument("--device", default=None, help="force detector device (cpu/mps/cuda)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    paths = list(args.videos)
    if args.all:
        paths = _ALL_CLIPS + paths
    if not paths:
        ap.error("provide at least one video path or --all")

    from cv.detection.court_detector import CourtDetector
    detector = CourtDetector(device=args.device)
    if detector.model is None:
        raise SystemExit("Court model failed to load — check models/court/model_tennis_court_det.pt")

    results = []
    for p in paths:
        vp = Path(p)
        if not vp.is_absolute():
            vp = PROJECT_ROOT / vp
        if not vp.exists():
            results.append({"video": vp.name, "error": "not found"})
            continue
        results.append(validate_video(
            vp, detector, every=args.every, max_frames=args.max_frames, first_only=args.first_only,
        ))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for agg in results:
            _report(agg)


if __name__ == "__main__":
    main()
