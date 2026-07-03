"""Tests for the court-keypoint frontend→reference remap (the homography bug fix).

Verifies reorder_frontend_to_reference inverts the documented frontend↔reference
keypoint mapping. This guards the remap logic's self-consistency; the mapping
itself must still be confirmed against a real frontend-placed court_config.

analysis_job is loaded standalone (its top-level imports are stdlib only), so
this runs in CI without the torch/CV stack.

Run with: pytest tests/test_keypoint_remap.py
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("analysis_job_under_test", _ROOT / "cv" / "analysis_job.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["analysis_job_under_test"] = _mod
_spec.loader.exec_module(_mod)

# Frontend index f → reference index r (the semantic mapping the remap must invert).
FRONTEND_TO_REFERENCE = [2, 5, 7, 3, 10, 13, 11, 8, 12, 9, 0, 4, 6, 1]


def test_remap_inverts_frontend_to_reference():
    # 14 distinct reference-order points.
    ref_order = [(r * 10.0, r * 10.0 + 1) for r in range(14)]
    # Simulate what the frontend stores: index f holds the point at reference perm[f].
    frontend_order = [ref_order[FRONTEND_TO_REFERENCE[f]] for f in range(14)]
    # The remap must recover the reference order.
    assert _mod.reorder_frontend_to_reference(frontend_order) == ref_order


def test_reorder_is_a_permutation():
    # Every frontend index is used exactly once (no drops/dupes).
    assert sorted(_mod._REORDER_TO_REFERENCE) == list(range(14))


def test_remap_preserves_none_entries():
    frontend_order = [None] * 14
    frontend_order[10] = (5.0, 6.0)  # frontend idx 10 → reference idx 0
    out = _mod.reorder_frontend_to_reference(frontend_order)
    assert out[0] == (5.0, 6.0)
    assert sum(1 for p in out if p is None) == 13
