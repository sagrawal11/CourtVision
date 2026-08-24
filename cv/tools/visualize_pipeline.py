"""cv/tools/visualize_pipeline.py — SEE the full pipeline on a video.

Auto-detects the court (best-frame), then renders the existing debug video overlay
(court lines + 24 zones + TrackNet ball trail + YOLO player boxes) so you can watch
the whole detection stack run end-to-end.

    python cv/tools/visualize_pipeline.py --input "tests/Outdoor Match 1 15.53.25.mp4" \
        --max-seconds 6 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Render the full pipeline debug video")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/court_detection/pipeline.mp4")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from cv.detection.court_detector import CourtDetector
    from cv.analysis.visualizer import render_debug_video

    inp = Path(args.input)
    if not inp.is_absolute():
        inp = PROJECT_ROOT / inp

    det = CourtDetector(device=args.device)
    kps, score = det.detect_best_frame(inp)
    print(f"Court auto-detect: trustworthy={score.get('trustworthy')} "
          f"native_inliers={score.get('native_inliers')} reproj={score.get('mean_err_px')}px")
    if not score.get("trustworthy"):
        print("  (court NOT trustworthy — overlay may be off; in production this would ask for manual keypoints)")

    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    render_debug_video(inp, out, keypoints=kps, max_seconds=args.max_seconds)
    print(f"\n✓ Wrote {out}")


if __name__ == "__main__":
    main()
