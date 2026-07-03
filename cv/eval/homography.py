"""Homography validation for court keypoints (cv2 + numpy only, no torch).

REFERENCE_KEYPOINTS mirrors cv.detection.court_detector.CourtReference.key_points
(and REF_*_MIN/MAX its extent constants) so this module stays importable without
the detector stack. pipeline._build_homography maps keypoint[i] → key_points[i]
by index, so a candidate set is "valid" only when, in that index order, it fits a
clean perspective transform onto the reference court.

If the reference court model ever changes, update these constants (a test guards
that they still fit the identity case).
"""

from __future__ import annotations

import cv2
import numpy as np

# CourtReference.key_points order: 0-3 far/near doubles corners, 4-7 far/near
# singles corners, 8-11 far/near service-line singles, 12-13 far/near service center.
REFERENCE_KEYPOINTS = [
    (286, 561), (1379, 561), (286, 2935), (1379, 2935),
    (423, 561), (423, 2935), (1242, 561), (1242, 2935),
    (423, 1110), (1242, 1110), (423, 2386), (1242, 2386),
    (832, 1110), (832, 2386),
]
REF_X_MIN, REF_X_MAX, REF_Y_MIN, REF_Y_MAX = 286, 1379, 561, 2935

# Pass thresholds — the correct order fits ~11px mean / 14 inliers; a wrong order
# fits ~2000px / <6 inliers, so there's a wide margin.
_MIN_INLIERS = 12
_MAX_MEAN_ERR = 50.0


def check(src_keypoints, ransac_thresh: float = 50.0) -> dict:
    """Fit src pixel keypoints → REFERENCE_KEYPOINTS by index and report quality.

    Returns a dict with valid/inliers/mean_err_px/max_err_px/in_bounds (or
    valid=False + reason when a homography can't be fit).
    """
    src, dst = [], []
    for i, kp in enumerate(src_keypoints):
        if kp is not None and kp[0] is not None:
            src.append([float(kp[0]), float(kp[1])])
            dst.append([float(REFERENCE_KEYPOINTS[i][0]), float(REFERENCE_KEYPOINTS[i][1])])
    if len(src) < 4:
        return {"valid": False, "reason": f"only {len(src)} valid points (need ≥4)"}

    src_a, dst_a = np.array(src), np.array(dst)
    H, mask = cv2.findHomography(src_a, dst_a, cv2.RANSAC, ransac_thresh)
    if H is None:
        return {"valid": False, "reason": "findHomography returned None"}

    proj = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst_a, axis=1)
    nx = (proj[:, 0] - REF_X_MIN) / (REF_X_MAX - REF_X_MIN)
    ny = (proj[:, 1] - REF_Y_MIN) / (REF_Y_MAX - REF_Y_MIN)
    in_bounds = int(np.sum((nx >= -0.05) & (nx <= 1.05) & (ny >= -0.05) & (ny <= 1.05)))
    inliers = int(mask.sum())

    return {
        "valid": inliers >= _MIN_INLIERS and float(err.mean()) <= _MAX_MEAN_ERR,
        "n_points": len(src),
        "inliers": inliers,
        "mean_err_px": round(float(err.mean()), 1),
        "max_err_px": round(float(err.max()), 1),
        "in_bounds": in_bounds,
    }
