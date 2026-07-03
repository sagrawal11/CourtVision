"""Tests for cv/eval/homography.py — the court-keypoint homography validator.

Loaded standalone to stay torch-free (CI-safe). Verifies a clean perspective
fit passes and a mismatched-order set fails.

Run with: pytest tests/test_homography_check.py
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cv_eval_homography", _ROOT / "cv" / "eval" / "homography.py")
hg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hg)

# Frontend→reference permutation (the mismatch that the remap fixes).
FRONTEND_TO_REFERENCE = [2, 5, 7, 3, 10, 13, 11, 8, 12, 9, 0, 4, 6, 1]


def test_identity_is_valid():
    # Reference points mapped to themselves (by index) → a perfect fit.
    res = hg.check(list(hg.REFERENCE_KEYPOINTS))
    assert res["valid"] is True
    assert res["inliers"] == 14
    assert res["mean_err_px"] < 1.0
    assert res["in_bounds"] == 14


def test_wrong_order_is_invalid():
    # Reference points shuffled by the frontend permutation, paired with the
    # reference by index → no clean homography (mirrors the real-app bug).
    shuffled = [hg.REFERENCE_KEYPOINTS[FRONTEND_TO_REFERENCE[i]] for i in range(14)]
    res = hg.check(shuffled)
    assert res["valid"] is False


def test_too_few_points_invalid():
    res = hg.check([None] * 14)
    assert res["valid"] is False and "reason" in res
