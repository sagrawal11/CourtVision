"""Parse annotation-app ground-truth CSVs and score detections against them.

The annotation app (annotation_collaboration/) exports one CSV per video to
cv/training_data/<video>_annotations.csv with columns:

    video_id, frame, event_type, stroke_type, point_end_type, player

  event_type      : point_start | point_end | hit | bounce
  stroke_type     : serve | forehand | backhand | overhead | volley   (hits)
  point_end_type  : winner | unforced_error | net_error | forced_error (point_end)

This module is stdlib-only on purpose so it can be imported and unit-tested
without the torch/OpenCV detector stack.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ERROR_OUTCOMES = {"unforced_error", "net_error", "forced_error"}

# Sort priority for events sharing a frame: a point_start opens before same-frame
# hits/bounces are attributed, and a point_end closes after them.
_PRIORITY = {"point_start": 0, "bounce": 1, "hit": 1, "point_end": 2}


@dataclass
class LabeledEvent:
    frame: int
    event_type: str
    stroke_type: str = ""
    point_end_type: str = ""
    player: str = ""


@dataclass
class LabeledPoint:
    start_frame: int
    end_frame: Optional[int] = None
    outcome: Optional[str] = None          # None => point never closed
    hits: List[LabeledEvent] = field(default_factory=list)
    bounces: List[LabeledEvent] = field(default_factory=list)

    @property
    def rally_length(self) -> int:
        return len(self.hits)


def parse_events(csv_path: str) -> List[LabeledEvent]:
    """Read every labeled event from a CSV, sorted by (frame, event priority)."""
    events: List[LabeledEvent] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            frame = row.get("frame", "").strip()
            etype = row.get("event_type", "").strip()
            if not frame or not etype:
                continue
            events.append(LabeledEvent(
                frame=int(frame),
                event_type=etype,
                stroke_type=(row.get("stroke_type") or "").strip(),
                point_end_type=(row.get("point_end_type") or "").strip(),
                player=(row.get("player") or "").strip(),
            ))
    events.sort(key=lambda e: (e.frame, _PRIORITY.get(e.event_type, 1)))
    return events


def group_points(events: List[LabeledEvent]) -> List[LabeledPoint]:
    """Fold a flat event list into points (point_start … point_end).

    A point_start opens a point; hits/bounces attribute to the open point; a
    point_end closes it with its outcome. A point_start with a still-open point
    closes the previous one as unfinished (outcome=None); a trailing open point
    is kept as unfinished too.
    """
    points: List[LabeledPoint] = []
    current: Optional[LabeledPoint] = None
    for ev in events:
        if ev.event_type == "point_start":
            if current is not None:
                points.append(current)
            current = LabeledPoint(start_frame=ev.frame)
        elif ev.event_type == "hit":
            if current is not None:
                current.hits.append(ev)
        elif ev.event_type == "bounce":
            if current is not None:
                current.bounces.append(ev)
        elif ev.event_type == "point_end":
            if current is not None:
                current.end_frame = ev.frame
                current.outcome = ev.point_end_type or None
                points.append(current)
                current = None
    if current is not None:
        points.append(current)
    return points


def summarize(csv_path: str) -> dict:
    """Ground-truth aggregate summary for one video's labels."""
    events = parse_events(csv_path)
    points = group_points(events)
    completed = [p for p in points if p.outcome]
    outcomes = Counter(p.outcome for p in completed)
    strokes = Counter(e.stroke_type for e in events if e.event_type == "hit" and e.stroke_type)
    rallies = [p.rally_length for p in completed]
    return {
        "n_points": len(points),
        "n_completed_points": len(completed),
        "n_hits": sum(1 for e in events if e.event_type == "hit"),
        "n_bounces": sum(1 for e in events if e.event_type == "bounce"),
        "outcomes": dict(outcomes),
        "n_winners": outcomes.get("winner", 0),
        "n_errors": sum(outcomes.get(o, 0) for o in ERROR_OUTCOMES),
        "strokes": dict(strokes),
        "avg_rally_length": round(sum(rallies) / len(rallies), 2) if rallies else 0.0,
    }


# ── Scoring: match detected event frames against labeled event frames ─────────

def match_frames(pred: List[int], true: List[int], tol: int) -> Tuple[int, int, int]:
    """Greedy 1-to-1 nearest match within ``tol`` frames.

    Returns (true_positives, false_positives, false_negatives). Each labeled
    frame can be matched at most once, so duplicate detections count as FPs.
    """
    true_sorted = sorted(true)
    used = [False] * len(true_sorted)
    tp = 0
    for pf in sorted(pred):
        best_i, best_d = -1, tol + 1
        for i, tf in enumerate(true_sorted):
            if used[i]:
                continue
            d = abs(pf - tf)
            if d <= tol and d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            used[best_i] = True
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }
