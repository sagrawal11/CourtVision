"""Tests for P2.3 (net-error precision) and P2.4 (changeover robustness).

Both modules import cleanly without torch (numpy/cv2 only), so they're loaded
standalone to keep this CI-safe.

Run with: pytest tests/test_cv_outcome_changeover.py
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # dataclasses (PEP 563) need the module registered
    spec.loader.exec_module(mod)
    return mod


pd = _load("point_detector_uut", "cv/analysis/point_detector.py")
pi = _load("player_identity_uut", "cv/analysis/player_identity.py")


def _bounce(in_bounds):
    return SimpleNamespace(is_in_bounds=in_bounds)


# ── P2.3: net-error precision ─────────────────────────────────────────────────

def test_final_out_of_bounds_bounce_is_error_out():
    assert pd.classify_point_outcome([_bounce(True), _bounce(False)]) == "error_out"


def test_final_in_bounds_bounce_is_in_play():
    assert pd.classify_point_outcome([_bounce(True)]) == "in_play"


def test_no_bounce_ball_at_net_is_net_error():
    # Ball last seen at the net (court_y ~ 0.5) → confident net error.
    assert pd.classify_point_outcome([], last_ball_court_y=0.5) == "error_net"
    assert pd.classify_point_outcome([], last_ball_court_y=0.55) == "error_net"


def test_no_bounce_ball_away_from_net_is_not_net_error():
    # The fix: ball lost far from the net → tracker likely lost it, NOT a net
    # error. (Old code always returned error_net here.)
    assert pd.classify_point_outcome([], last_ball_court_y=0.9) == "in_play"


def test_no_bounce_no_ball_info_defaults_to_in_play():
    assert pd.classify_point_outcome([], last_ball_court_y=None) == "in_play"


# ── P2.4: changeover robustness ───────────────────────────────────────────────

def _tracker(raw, min_samples):
    t = pi.PlayerIdentityTracker(smooth_window=1, persist_frames=10_000,
                                 min_persist_samples=min_samples)
    t._raw_history = list(raw)
    t._frame_indices = list(range(len(raw)))
    return t


def test_real_changeover_is_detected():
    # Solid block on the near side, then a solid block on the far side.
    raw = [1] * 5 + [0] * 12
    assert _tracker(raw, min_samples=10)._smoothed_at(0) == 1  # sanity: smoothing identity
    assert _tracker(raw, min_samples=10).get_changeover_frames() == [5]


def test_lone_flip_during_detection_gap_is_ignored():
    # One flip then a long gap of missing detections → not enough corroboration.
    raw = [1] * 5 + [0] + [-1] * 15
    assert _tracker(raw, min_samples=10).get_changeover_frames() == []


def test_min_persist_samples_is_the_threshold():
    raw = [1] * 5 + [0] * 6  # only 5 confirming samples after the flip (j=6..10)
    assert _tracker(raw, min_samples=5).get_changeover_frames() == [5]
    assert _tracker(raw, min_samples=8).get_changeover_frames() == []
