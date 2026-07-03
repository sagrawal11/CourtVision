"""cv/tools/evaluate_pipeline.py — score pipeline output against ground-truth labels.

Compares detected events (bounces, hits, point boundaries) and aggregate counts
against a hand-labeled annotation CSV (cv/training_data/<video>_annotations.csv),
reporting precision / recall / F1 per event type plus a point/winner/error count
comparison. This is the CV accuracy yardstick for the P2.x work.

Two modes:

  # Run the pipeline on a video (needs the .mp4, model weights, and the 14
  # confirmed court keypoints as a JSON list of [x, y] pairs):
  python cv/tools/evaluate_pipeline.py \
      --annotations "cv/training_data/Indoor Match 1 15.53.25_annotations.csv" \
      --video "tests/Indoor Match 1 15.53.25.mp4" \
      --keypoints keypoints.json

  # Compare a precomputed detections JSON (no pipeline run needed) — handy for
  # regression tracking or when the pipeline output is dumped elsewhere:
  python cv/tools/evaluate_pipeline.py \
      --annotations "cv/training_data/Indoor Match 1 15.53.25_annotations.csv" \
      --detections detections.json

The detections JSON shape (all keys optional):
  {"bounces": [f, ...], "hits": [f, ...], "point_starts": [f, ...],
   "point_ends": [f, ...], "n_points": int, "n_winners": int, "n_errors": int}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv.eval import labels  # noqa: E402

# Pipeline outcome strings that count as errors (vs the label vocab which splits
# net/unforced/forced — only the winners-vs-errors totals are comparable).
_PIPELINE_ERROR_OUTCOMES = {"error_net", "error_out"}


def _frames(events, event_type):
    return [e.frame for e in events if e.event_type == event_type]


def detected_from_points(points) -> dict:
    """Build a detections dict from pipeline PointRecords."""
    bounces, hits, starts, ends = [], [], [], []
    n_winners = n_errors = 0
    for p in points:
        starts.append(p.start_frame)
        if getattr(p, "end_frame", None) is not None:
            ends.append(p.end_frame)
        bounces += [b.frame_idx for b in p.bounces]
        hits += [h.frame_idx for h in p.shots]
        if p.outcome == "winner":
            n_winners += 1
        elif p.outcome in _PIPELINE_ERROR_OUTCOMES:
            n_errors += 1
    return {
        "bounces": bounces, "hits": hits, "point_starts": starts, "point_ends": ends,
        "n_points": len(points), "n_winners": n_winners, "n_errors": n_errors,
    }


def evaluate(csv_path: str, detected: dict, tol: int = 15, max_frame: int | None = None) -> dict:
    """Score a detections dict against the labeled CSV within ``tol`` frames.

    If ``max_frame`` is set, labels and detections are both restricted to frames
    < max_frame so a partial run (``--max-frames``) is compared fairly.
    """
    events = labels.parse_events(csv_path)
    if max_frame is not None:
        events = [e for e in events if e.frame < max_frame]

    completed = [p for p in labels.group_points(events) if p.outcome]
    labeled_counts = {
        "n_points": len(completed),
        "n_winners": sum(1 for p in completed if p.outcome == "winner"),
        "n_errors": sum(1 for p in completed if p.outcome in labels.ERROR_OUTCOMES),
    }

    event_report = {}
    for etype, key in (("bounce", "bounces"), ("hit", "hits"),
                       ("point_start", "point_starts"), ("point_end", "point_ends")):
        pred = detected.get(key, [])
        if max_frame is not None:
            pred = [f for f in pred if f < max_frame]
        tp, fp, fn = labels.match_frames(pred, _frames(events, etype), tol)
        event_report[etype] = labels.precision_recall_f1(tp, fp, fn)

    return {
        "tolerance_frames": tol,
        "max_frame": max_frame,
        "events": event_report,
        "counts": {
            "labeled": labeled_counts,
            "detected": {
                "n_points": detected.get("n_points"),
                "n_winners": detected.get("n_winners"),
                "n_errors": detected.get("n_errors"),
            },
        },
    }


def _run_pipeline(video: str, keypoints_path: str, frame_skip: int = 1,
                  max_frames: int | None = None) -> dict:
    from cv.pipeline import AnalyticsPipeline  # heavy import, only when running

    kps = json.loads(Path(keypoints_path).read_text())
    kps = [tuple(p) if p else None for p in kps]
    result = AnalyticsPipeline().process(
        video_path=video, court_keypoints=kps,
        frame_skip=frame_skip, max_frames=max_frames,
    )
    return detected_from_points(result._points)


def _print_report(report: dict) -> None:
    print(f"\n== Event detection (±{report['tolerance_frames']} frames) ==")
    print(f"{'event':<13}{'precision':>10}{'recall':>10}{'f1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    for etype, m in report["events"].items():
        print(f"{etype:<13}{m['precision']:>10}{m['recall']:>10}{m['f1']:>8}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")
    lab, det = report["counts"]["labeled"], report["counts"]["detected"]
    print("\n== Aggregate counts (labeled vs detected) ==")
    for k in ("n_points", "n_winners", "n_errors"):
        print(f"{k:<13}{lab[k]!s:>10}{det[k]!s:>10}")


def main():
    ap = argparse.ArgumentParser(description="Score pipeline output vs ground-truth labels")
    ap.add_argument("--annotations", required=True, help="ground-truth CSV")
    ap.add_argument("--video", help="video to run the pipeline on")
    ap.add_argument("--keypoints", help="JSON list of 14 [x,y] court keypoints (with --video)")
    ap.add_argument("--detections", help="precomputed detections JSON (instead of --video)")
    ap.add_argument("--tol", type=int, default=15, help="frame tolerance for event matching")
    ap.add_argument("--frame-skip", type=int, default=1, help="process every Nth frame (faster)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (fast iteration; labels are windowed to match)")
    ap.add_argument("--save-detections", metavar="PATH",
                    help="write the pipeline's raw detections to JSON so you can re-score "
                         "(different --tol/--detections) without re-running the pipeline")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()

    if args.detections:
        detected = json.loads(Path(args.detections).read_text())
    elif args.video:
        if not args.keypoints:
            ap.error("--keypoints is required with --video (pipeline needs the court homography)")
        detected = _run_pipeline(args.video, args.keypoints,
                                 frame_skip=args.frame_skip, max_frames=args.max_frames)
        if args.save_detections:
            Path(args.save_detections).write_text(json.dumps(detected))
            print(f"Detections saved: {args.save_detections} "
                  f"(re-score instantly with --detections {args.save_detections})")
    else:
        ap.error("provide either --detections or --video")

    report = evaluate(args.annotations, detected, tol=args.tol, max_frame=args.max_frames)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
