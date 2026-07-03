"""Tests for the SAM-3D-Body pose scaffold (P2.5).

Verifies the gated-window cost-control logic (real) and that the estimator is
inert until a model is wired in. Loaded standalone to stay torch-free (CI-safe).

Run with: pytest tests/test_pose_scaffold.py
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("pose_estimator_uut", _ROOT / "cv" / "detection" / "pose_estimator.py")
pose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pose)


def test_empty_hits_no_windows():
    assert pose.gated_hit_windows([]) == []


def test_single_hit_window():
    assert pose.gated_hit_windows([100], window=3) == [(97, 103)]


def test_overlapping_hits_merge():
    # 100±3 = 97..103 and 102±3 = 99..105 overlap → one interval.
    assert pose.gated_hit_windows([100, 102], window=3) == [(97, 105)]


def test_distant_hits_stay_separate():
    assert pose.gated_hit_windows([100, 500], window=3) == [(97, 103), (497, 503)]


def test_window_never_goes_negative():
    assert pose.gated_hit_windows([1], window=5) == [(0, 6)]


def test_estimator_is_inert_until_model_wired():
    est = pose.PoseEstimator()
    assert est.is_available() is False
    # The heavy paths must not be silently usable while inert.
    with pytest.raises(NotImplementedError):
        est.estimate_pose_at(frame=None)
