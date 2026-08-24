"""cv/tools/visualize_court_detection.py — SEE the auto-detected court.

Runs the best-frame court auto-detector on each clip, draws the detected court
(numbered keypoints + court lines + the full 24-zone model warped through the
homography) onto that frame, saves a PNG, and builds an HTML contact sheet so you
can eyeball how spotlessly the court model lands on the real lines.

    python cv/tools/visualize_court_detection.py --all --open
    python cv/tools/visualize_court_detection.py "tests/Indoor Match 1 15.53.25.mp4"

Output: outputs/court_detection/<clip>.png + index.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.analysis.visualizer import (
    _build_h,
    _draw_court_lines,
    _draw_net,
    _draw_zone_overlay,
    _draw_keypoint_dots,
)

_ALL_CLIPS = [
    "tests/Indoor Match 1 15.53.25.mp4",
    "tests/Indoor Match 2 15.53.25.mp4",
    "tests/Outdoor Match 1 15.53.25.mp4",
    "tests/Outdoor Practice 15.53.25.mp4",
    "tests/tennis_test6.mov",
]
# Court ROIs (drawn once per camera). A clip whose name contains a key gets cropped
# to that court for detection — essential on multi-court (indoor) views.
_ROI_FOR = {
    "Indoor Match": "cv/court_rois/Indoor_Match.json",
    "Outdoor Match 1": "cv/court_rois/Outdoor_Match_1.json",
    "tennis_test6": "cv/court_rois/tennis_test6.json",
}
OUT_DIR = PROJECT_ROOT / "outputs" / "court_detection"


def annotate_clip(detector, video_path: Path, roi_polygon=None) -> dict | None:
    """Detect the court, draw it on the best frame, save a PNG. Return metadata."""
    kps, score, all_scores = detector.detect_best_frame(
        video_path, roi_polygon=roi_polygon, return_diagnostics=True)
    if "reason" in score:
        return {"video": video_path.name, "error": score["reason"]}

    frame_idx = int(score.get("_frame", 0))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"video": video_path.name, "error": f"could not read frame {frame_idx}"}

    H = _build_h(kps)
    if H is not None:
        H_inv = np.linalg.inv(H)
        _draw_zone_overlay(frame, H_inv)   # 24 zones warped onto the court
        _draw_net(frame, H_inv)
    _draw_court_lines(frame, kps)          # reconstructed court skeleton
    _draw_keypoint_dots(frame, kps)        # 14 keypoints

    trusted = bool(score.get("trustworthy"))
    verdict = "TRUSTWORTHY" if trusted else "NOT TRUSTED"
    trust_n = sum(1 for s in all_scores if s.get("geom_ok"))
    banner = (f"{video_path.name}   frame {frame_idx}   native inliers {score.get('native_inliers')}   "
              f"reproj {score.get('mean_err_px')}px   geom {'ok' if score.get('geom_ok') else 'BAD'}   {verdict}")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    colour = (80, 240, 120) if trusted else (60, 90, 240)
    cv2.putText(frame, banner, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{video_path.stem}.png"
    cv2.imwrite(str(png), frame)
    return {
        "video": video_path.name,
        "png": png.name,
        "frame": frame_idx,
        "native_inliers": score.get("native_inliers"),
        "mean_err": score.get("mean_err_px"),
        "geom_ok": bool(score.get("geom_ok")),
        "trustworthy": trusted,
        "trust_frames": f"{trust_n}/{len(all_scores)}",
        "roi_used": roi_polygon is not None,
    }


def write_html(results: list[dict], video_cards: list[dict] | None = None) -> Path:
    cards = []
    for r in results:
        if "error" in r:
            cards.append(f'<div class="card err"><h3>{r["video"]}</h3><p>⚠️ {r["error"]}</p></div>')
            continue
        badge = "ok" if r["trustworthy"] else "bad"
        label = "TRUSTWORTHY" if r["trustworthy"] else "NOT TRUSTED — needs manual"
        cards.append(f'''<div class="card">
      <img src="{r['png']}" loading="lazy">
      <div class="meta">
        <h3>{r['video']}</h3>
        <span class="badge {badge}">{label}</span>
        <ul>
          <li>native inliers: <b>{r['native_inliers']}</b> &nbsp; reproj: {r['mean_err']}px &nbsp; ROI crop: {'yes' if r.get('roi_used') else 'no'}</li>
          <li>court geometry: {'valid ✓' if r['geom_ok'] else 'invalid ✗'}</li>
          <li>chosen frame {r['frame']} &nbsp;(geometry-valid frames: {r['trust_frames']})</li>
        </ul>
      </div>
    </div>''')

    video_section = ""
    if video_cards:
        vids = "".join(f'''<div class="card">
      <video src="{v['file']}" controls loop muted playsinline></video>
      <div class="meta"><h3>{v['title']}</h3><p>{v['caption']}</p></div>
    </div>''' for v in video_cards)
        video_section = f'<h2>Full pipeline (court + zones + ball + players)</h2><div class="grid">{vids}</div>'

    html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>Court auto-detection</title>
<style>
  body{{background:#0f1115;color:#e6e6e6;font-family:-apple-system,system-ui,sans-serif;margin:0;padding:24px}}
  h1{{font-weight:650;margin:0 0 4px}} .sub{{color:#8a94a6;margin:0 0 24px}}
  h2{{margin:32px 0 12px;font-weight:600}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}}
  .card{{background:#171a21;border:1px solid #262b36;border-radius:12px;overflow:hidden}}
  .card img,.card video{{width:100%;display:block;background:#000}}
  .meta{{padding:14px 16px}} .meta h3{{margin:0 0 8px;font-size:15px}}
  .meta ul{{margin:8px 0 0;padding-left:16px;color:#aab3c2;font-size:13px;line-height:1.7}}
  .badge{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px}}
  .badge.ok{{background:#123d1e;color:#5fe08a}} .badge.bad{{background:#3d1212;color:#e05f5f}}
  .card.err{{padding:16px}} b{{color:#fff}}
</style></head><body>
  <h1>Court auto-detection — native-anchored best-frame selection</h1>
  <p class="sub">Yellow dots = reconstructed 14 keypoints. White lines = court skeleton.
  Shaded polygons = the 24-zone court model warped through the homography. Frames are chosen by how
  many <b>native</b> (un-reconstructed) keypoints agree on one court plane; low native inliers ⇒
  NOT TRUSTED ⇒ fall back to the manual editor.</p>
  <div class="grid">{"".join(cards)}</div>
  {video_section}
</body></html>'''
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "index.html"
    out.write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser(description="Visualize auto court detection")
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true", help="visualize all bundled test clips")
    ap.add_argument("--device", default=None)
    ap.add_argument("--open", action="store_true", help="open the HTML when done (macOS)")
    args = ap.parse_args()

    paths = list(args.videos)
    if args.all:
        paths = _ALL_CLIPS + paths
    if not paths:
        ap.error("provide at least one video path or --all")

    from cv.detection.court_detector import CourtDetector
    detector = CourtDetector(device=args.device)
    if detector.model is None:
        raise SystemExit("Court model failed to load")

    from cv.detection.court_roi import load_polygon
    results = []
    for p in paths:
        vp = Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
        if not vp.exists():
            results.append({"video": vp.name, "error": "not found"})
            continue
        roi = next((load_polygon(str(PROJECT_ROOT / rp)) for key, rp in _ROI_FOR.items()
                    if key in vp.name), None)
        print(f"Annotating {vp.name} ...{' [ROI crop]' if roi is not None else ''}")
        results.append(annotate_clip(detector, vp, roi_polygon=roi))

    video_cards = None
    if (OUT_DIR / "pipeline.mp4").exists():
        video_cards = [{
            "file": "pipeline.mp4",
            "title": "Outdoor Match 1 — full pipeline on the auto-detected court",
            "caption": "Court + 24-zone model + TrackNet ball trail on the auto-detected court. "
                       "Player boxes skipped — the custom YOLO weights need re-wiring (separate issue).",
        }]
    html = write_html(results, video_cards)
    print(f"\n✓ Contact sheet: {html}")
    if args.open:
        import subprocess
        subprocess.run(["open", str(html)], check=False)


if __name__ == "__main__":
    main()
