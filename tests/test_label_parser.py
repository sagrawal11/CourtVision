"""Tests for cv/eval/labels.py — the ground-truth annotation parser + scoring (P3.2).

Locks the ground-truth counts for the committed annotation CSVs (a regression
catch if the parser or the fixtures drift) and verifies the frame-matching /
precision-recall scoring used to compare pipeline output against labels.

The parser is loaded standalone (not `from cv.eval...`) to avoid importing the
cv package's torch-backed detectors, so this runs in CI without CV deps.

Run with: pytest tests/test_label_parser.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cv_eval_labels", _ROOT / "cv" / "eval" / "labels.py")
labels = importlib.util.module_from_spec(_spec)
# Register before exec so the dataclasses (which use `from __future__ import
# annotations`) can resolve their field types via sys.modules.
sys.modules["cv_eval_labels"] = labels
_spec.loader.exec_module(labels)

TRAINING_DATA = _ROOT / "cv" / "training_data"


# ── Parser / ground-truth summary ─────────────────────────────────────────────

# Hand-counted from the committed CSVs (awk over event_type / point_end_type).
EXPECTED = {
    "Indoor Match 1 15.53.25": {
        "n_points": 132, "n_completed_points": 131, "n_hits": 807, "n_bounces": 730,
        "n_winners": 34, "n_errors": 97,  # 36 net + 55 unforced + 6 forced
    },
    "Indoor Match 2 15.53.25": {
        "n_points": 151, "n_completed_points": 148, "n_hits": 672, "n_bounces": 601,
        "n_winners": 37, "n_errors": 111,  # 40 net + 69 unforced + 2 forced
    },
}


@pytest.mark.parametrize("video_id, exp", EXPECTED.items())
def test_summary_matches_ground_truth(video_id, exp):
    summary = labels.summarize(str(TRAINING_DATA / f"{video_id}_annotations.csv"))
    for key, value in exp.items():
        assert summary[key] == value, f"{video_id}: {key} expected {value}, got {summary[key]}"
    # Winners + errors must account for every completed point.
    assert summary["n_winners"] + summary["n_errors"] == summary["n_completed_points"]


def test_points_close_and_carry_outcomes():
    events = labels.parse_events(str(TRAINING_DATA / "Indoor Match 1 15.53.25_annotations.csv"))
    points = labels.group_points(events)
    completed = [p for p in points if p.outcome]
    # Every completed point has an end frame at/after its start and a known outcome.
    assert all(p.end_frame is not None and p.end_frame >= p.start_frame for p in completed)
    assert all(p.outcome in {"winner", *labels.ERROR_OUTCOMES} for p in completed)


def test_empty_csv_summarizes_to_zero(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("video_id,frame,event_type,stroke_type,point_end_type,player\n")
    summary = labels.summarize(str(csv))
    assert summary["n_points"] == 0 and summary["n_hits"] == 0


# ── Frame-matching / precision-recall scoring ─────────────────────────────────

def test_match_frames_within_tolerance():
    # Predictions near the truth (within tol=5) are TPs; the extra 999 is an FP.
    tp, fp, fn = labels.match_frames(pred=[10, 22, 999], true=[12, 20, 40], tol=5)
    assert (tp, fp, fn) == (2, 1, 1)


def test_match_frames_is_one_to_one():
    # Two predictions near a single labeled frame → only one TP, the other an FP.
    tp, fp, fn = labels.match_frames(pred=[10, 11], true=[10], tol=5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_precision_recall_f1():
    m = labels.precision_recall_f1(tp=8, fp=2, fn=2)
    assert m["precision"] == 0.8 and m["recall"] == 0.8 and m["f1"] == 0.8


def test_prf_handles_empty():
    assert labels.precision_recall_f1(0, 0, 0)["f1"] == 0.0
