#!/usr/bin/env python3
"""
cv/tools/train_models.py — Train CatBoost models for bounce, hit, and stroke classification.

Reads annotation CSVs and matching feature .npz files from cv/training_data/,
trains three models, evaluates them with leave-one-video-out cross-validation,
and saves the best models to cv/models/.

Trained models
--------------
  cv/models/bounce_model.cbm        — binary: is this frame a ball bounce?
  cv/models/hit_model.cbm           — binary: is this frame a racket hit?
  cv/models/point_start_model.cbm   — binary: is this frame the start of a point (a serve)?
  cv/models/point_end_model.cbm     — binary: is this frame the end of a point?
  cv/models/stroke_model.cbm        — multiclass: forehand / backhand / serve / overhead / volley

Feature windows
---------------
  All models use a sliding window of ball (and player) positions centred on the
  candidate frame.  Missing detections are replaced by 0 with an accompanying
  validity flag, so the model learns to handle occlusion.

  Bounce      : ±12 frames of ball xy + y-velocity + y-acceleration
  Hit         : ±12 frames of ball xy + y-velocity + ±6 frames of player
                positions + ball-to-player distances
  Point start : same features as Hit  (point_start IS a serve hit)
  Point end   : same features as Bounce
  Stroke      : ±6 frames of ball xy + y-velocity + ±6 frames of player
                positions + relative ball-to-player offset

Negative sampling
-----------------
  All binary classifiers sample negatives only from IN-POINT frames — i.e.
  frames between an annotated point_start and the corresponding point_end.
  This focuses each model on distinguishing its target event from other
  in-rally events, and avoids wasting capacity on trivially-easy negatives
  from warmup / between-point / changeover periods.

  Per positive, NEG_RATIO random in-point frames (excluding the guard band
  around any positive of the same event type) are sampled as negatives.
  CatBoost class weights further penalise false negatives.

Usage
-----
    python cv/tools/train_models.py
    python cv/tools/train_models.py --data-dir cv/training_data --model-dir cv/models
    python cv/tools/train_models.py --skip-stroke   # skip if too few stroke labels
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

BOUNCE_HALF_WIN = 12   # frames before/after candidate for bounce features
HIT_HALF_WIN    = 12   # frames before/after candidate for hit features
PLAYER_HALF_WIN =  6   # frames before/after for player position features
STROKE_HALF_WIN =  6   # frames before/after candidate for stroke features

GUARD_BAND = 3         # frames around a positive event excluded from negatives
NEG_RATIO  = 6         # negative samples per positive for bounce/hit

MIN_EVENTS_TO_TRAIN = 20   # skip a model if fewer than this many labelled events

STROKE_LABELS = ["forehand", "backhand", "serve", "overhead", "volley"]
STROKE_TO_INT  = {s: i for i, s in enumerate(STROKE_LABELS)}


# ── Feature engineering ───────────────────────────────────────────────────────

def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 (tree models don't require normalisation)."""
    out = arr.copy()
    out[np.isnan(out)] = 0.0
    return out


def _valid_mask(arr: np.ndarray) -> np.ndarray:
    """Return 1.0 where arr is not NaN, 0.0 where it is."""
    return (~np.isnan(arr)).astype(np.float32)


def _window(arr: np.ndarray, center: int, half: int) -> np.ndarray:
    """
    Extract a fixed-length window of 2*half+1 values centred on `center`.
    Out-of-bounds positions are filled with 0.
    """
    n   = len(arr)
    out = np.zeros(2 * half + 1, dtype=np.float32)
    for i, fi in enumerate(range(center - half, center + half + 1)):
        if 0 <= fi < n:
            v = arr[fi]
            out[i] = 0.0 if np.isnan(v) else float(v)
    return out


def _valid_window(arr: np.ndarray, center: int, half: int) -> np.ndarray:
    """Same as _window but returns 1.0/0.0 validity flags."""
    n   = len(arr)
    out = np.zeros(2 * half + 1, dtype=np.float32)
    for i, fi in enumerate(range(center - half, center + half + 1)):
        if 0 <= fi < n and not np.isnan(arr[fi]):
            out[i] = 1.0
    return out


def _velocity(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Compute frame-to-frame differences, treating NaN as gaps.
    Returns array of same length as arr (first element = 0).
    """
    vel = np.zeros(n, dtype=np.float32)
    clean = _fill_nans(arr[:n])
    vel[1:] = np.diff(clean)
    # Zero out velocity across NaN gaps
    for i in range(1, n):
        if np.isnan(arr[i]) or np.isnan(arr[i - 1]):
            vel[i] = 0.0
    return vel


def bounce_features(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    fps: float,
    center: int,
) -> np.ndarray:
    """
    Feature vector for the bounce classifier at frame `center`.

    Layout (109 features):
      ball_x window    (2W+1 = 25)
      ball_y window    (25)
      ball validity    (25)
      y-velocity win   (25)
      y-accel window   (9 at half-window // 2 = 6 → 13... keep same W for simplicity)
      fps              (1)
    """
    W = BOUNCE_HALF_WIN
    vel_y  = _velocity(ball_y,  len(ball_y))
    accel_y = _velocity(vel_y, len(vel_y))

    return np.concatenate([
        _window(ball_x,  center, W),          # 25
        _window(ball_y,  center, W),          # 25
        _valid_window(ball_x, center, W),     # 25
        _window(vel_y,   center, W),          # 25
        _window(accel_y, center, W),          # 25
        np.array([fps], dtype=np.float32),    #  1
    ])  # total: 126


def hit_features(
    ball_x: np.ndarray, ball_y: np.ndarray,
    near_x: np.ndarray, near_y: np.ndarray,
    far_x:  np.ndarray, far_y:  np.ndarray,
    fps:    float,
    center: int,
) -> np.ndarray:
    """
    Feature vector for the hit classifier at frame `center`.

    Layout (175 features):
      ball_x/y window + validity + y-vel + y-accel  (126, same as bounce)
      near_x/y window + validity                    (3 × 13 = 39)
      far_x/y  window + validity                    (39)
      ball-to-near distance window                  (13)
      ball-to-far  distance window                  (13)
      fps                                           (1)
    """
    W  = HIT_HALF_WIN
    WP = PLAYER_HALF_WIN

    vel_y   = _velocity(ball_y, len(ball_y))
    accel_y = _velocity(vel_y,  len(vel_y))

    # Precompute ball-to-player distances per frame
    bx_c = _fill_nans(ball_x)
    by_c = _fill_nans(ball_y)
    nx_c = _fill_nans(near_x)
    ny_c = _fill_nans(near_y)
    fx_c = _fill_nans(far_x)
    fy_c = _fill_nans(far_y)

    dist_near = np.hypot(bx_c - nx_c, by_c - ny_c).astype(np.float32)
    dist_far  = np.hypot(bx_c - fx_c, by_c - fy_c).astype(np.float32)
    # Zero out distances where either party was not detected
    dist_near[np.isnan(ball_x) | np.isnan(near_x)] = 0.0
    dist_far [np.isnan(ball_x) | np.isnan(far_x )] = 0.0

    return np.concatenate([
        _window(ball_x,  center, W),          # 25
        _window(ball_y,  center, W),          # 25
        _valid_window(ball_x, center, W),     # 25
        _window(vel_y,   center, W),          # 25
        _window(accel_y, center, W),          # 25
        np.array([fps], dtype=np.float32),    #  1  → subtotal 126
        _window(near_x, center, WP),          # 13
        _window(near_y, center, WP),          # 13
        _valid_window(near_x, center, WP),    # 13
        _window(far_x,  center, WP),          # 13
        _window(far_y,  center, WP),          # 13
        _valid_window(far_x, center, WP),     # 13
        _window(dist_near, center, WP),       # 13
        _window(dist_far,  center, WP),       # 13
    ])  # total: 230


def stroke_features(
    ball_x: np.ndarray, ball_y: np.ndarray,
    near_x: np.ndarray, near_y: np.ndarray,
    far_x:  np.ndarray, far_y:  np.ndarray,
    fps:    float,
    center: int,
    player_is_near:  int,        # 1 = near player hit, 0 = far player hit
    player_is_lefty: int = 0,    # 1 = hitting player is left-handed (inference only)
) -> np.ndarray:
    """
    Feature vector for the stroke type classifier at frame `center`.

    X-axis mirroring (canonicalise everything to "right-handed near player")
    -----------------------------------------------------------------------
    Two independent factors flip what "ball to the right of body" means:

      1. Far vs near player: they face each other, so a far-player forehand
         has the ball at ball_x < player_x while a near-player forehand has
         ball_x > player_x.

      2. Lefty vs righty: a lefty's forehand side is the opposite side of
         their body from a righty's.

    Each factor independently flips the x sign; applying both is no net flip.

    During TRAINING we only train on righty-only matches, so player_is_lefty
    is always 0 and the lefty path is exercised only at INFERENCE time when
    the user tells us a player is left-handed.

    Layout:
      ball_x/y window + validity + y-vel    (4 × 13 = 52)
      player_x/y window + validity           (3 × 13 = 39)
      relative ball offset (bx-px, by-py)   (2 × 13 = 26)
      player_is_near flag + fps             (2)
    """
    W = STROKE_HALF_WIN

    vel_y = _velocity(ball_y, len(ball_y))

    # Pick the hitting player's arrays
    px = near_x if player_is_near else far_x
    py = near_y if player_is_near else far_y

    bx_c = _fill_nans(ball_x)
    by_c = _fill_nans(ball_y)
    px_c = _fill_nans(px)
    py_c = _fill_nans(py)

    # Combine the two flips: far player flips once, lefty flips once. Doing
    # both is no net flip (lefty far = righty near in canonical coords).
    far_sign   = 1.0 if player_is_near else -1.0
    lefty_sign = -1.0 if player_is_lefty else 1.0
    x_sign     = far_sign * lefty_sign

    rel_bx = (x_sign * (bx_c - px_c)).astype(np.float32)
    rel_by = (by_c - py_c).astype(np.float32)
    # Zero out relative positions where detection is missing
    rel_bx[np.isnan(ball_x) | np.isnan(px)] = 0.0
    rel_by[np.isnan(ball_y) | np.isnan(py)] = 0.0

    # Mirror the absolute ball x and player x so the model sees a consistent
    # left/right orientation in the raw position features too
    mirrored_ball_x = (x_sign * bx_c).astype(np.float32)
    mirrored_px     = (x_sign * px_c).astype(np.float32)

    return np.concatenate([
        _window(mirrored_ball_x, center, W),  # 13  (x mirrored for far player)
        _window(ball_y,          center, W),  # 13
        _valid_window(ball_x,    center, W),  # 13
        _window(vel_y,           center, W),  # 13
        _window(mirrored_px,     center, W),  # 13  (x mirrored for far player)
        _window(py,              center, W),  # 13
        _valid_window(px,        center, W),  # 13
        _window(rel_bx,          center, W),  # 13
        _window(rel_by, center, W),           # 13
        np.array([float(player_is_near), fps], dtype=np.float32),  # 2
    ])  # total: 119


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(
    data_dir: Path,
    exclude:  Optional[List[str]] = None,
) -> List[dict]:
    """
    Load all matched (CSV + npz) pairs from data_dir.

    Training assumption: every video in data_dir is a righty-only match.
    If you have lefty footage, exclude it from training — at inference time
    the user tells us each player's handedness and stroke_features mirrors
    the coordinates accordingly.

    Args:
        data_dir: directory containing annotation CSVs and feature npz files.
        exclude:  list of substrings; any annotation file whose stem contains
                  one of these is skipped (case-insensitive). Useful for
                  excluding incomplete or lefty matches without moving files.

    Returns a list of dicts, one per video:
        video_id, annotations (list of dicts), npz (arrays)
    """
    import csv

    csv_files = sorted(data_dir.glob("*_annotations.csv"))
    if not csv_files:
        print(f"No annotation CSVs found in {data_dir}")
        return []

    exclude_lc = [s.lower() for s in (exclude or [])]
    dataset = []
    for csv_path in csv_files:
        stem    = csv_path.stem.replace("_annotations", "")

        if any(ex in stem.lower() for ex in exclude_lc):
            print(f"  ⊝  {stem}   (excluded)")
            continue

        npz_path = data_dir / f"{stem}_features.npz"
        if not npz_path.exists():
            print(f"  ⚠  No matching features file for {csv_path.name} — run extract_features.py first. Skipping.")
            continue

        with open(csv_path, newline="") as f:
            annotations = list(csv.DictReader(f))

        npz = np.load(npz_path)
        total_frames = int(npz["total_frames"])

        n_in = len(annotations)
        annotations = _postprocess_annotations(annotations, total_frames)
        n_out = len(annotations)

        dataset.append({
            "video_id":    stem,
            "annotations": annotations,
            "npz":         npz,
        })

        msg = f"  ✓  {stem}  ({n_out} annotations, {total_frames} frames)"
        if n_in != n_out:
            msg += f"   [{n_in - n_out} dropped after cleanup]"
        print(msg)

    return dataset


# ── Annotation cleanup ───────────────────────────────────────────────────────

# Number of frames to push a point_end past a coincident bounce so the segment
# includes a few frames of "rally tail" rather than cutting at the exact bounce.
POINT_END_BOUNCE_OFFSET = 30


def _postprocess_annotations(annotations: List[dict], total_frames: int) -> List[dict]:
    """
    Apply two cleanup rules to a video's annotations:

      1. point_end push-back
         If a point_end shares its frame with a bounce (the common "winner
         lands here" pattern), push the point_end forward by 30 frames so the
         segment includes a few frames of rally tail. The 30-frame value is
         clamped against the total frame count.

      2. Drop orphan point_start events
         If a point_start has no matching point_end before the next
         point_start, the whole pair is invalid for point-segmentation
         training, so we drop the orphan start (the bounces / hits inside it
         are still kept; only the boundary marker is removed).
    """
    # Index bounces by frame for the push-back check
    bounce_frames = {int(a["frame"]) for a in annotations if a["event_type"] == "bounce"}

    # ── 1. Push back point_end frames that coincide with a bounce ──────────
    out: List[dict] = []
    for a in annotations:
        if a["event_type"] == "point_end" and int(a["frame"]) in bounce_frames:
            new_frame = min(int(a["frame"]) + POINT_END_BOUNCE_OFFSET, total_frames - 1)
            a = {**a, "frame": new_frame}
        out.append(a)

    # ── 2. Drop orphan point_start events ──────────────────────────────────
    # Walk through chronologically; a point_start is kept only if a point_end
    # appears before the next point_start.
    out_sorted = sorted(out, key=lambda a: int(a["frame"]))
    starts = [(i, int(a["frame"])) for i, a in enumerate(out_sorted) if a["event_type"] == "point_start"]
    ends   = [int(a["frame"]) for a in out_sorted if a["event_type"] == "point_end"]

    orphan_indices: set = set()
    for k, (i, sf) in enumerate(starts):
        next_start_frame = starts[k + 1][1] if k + 1 < len(starts) else float("inf")
        has_matching_end = any(sf <= ef < next_start_frame for ef in ends)
        if not has_matching_end:
            orphan_indices.add(i)

    return [a for i, a in enumerate(out_sorted) if i not in orphan_indices]


# ── Dataset builders ──────────────────────────────────────────────────────────

def _positive_frames(annotations: List[dict], event_type: str) -> List[int]:
    return [int(a["frame"]) for a in annotations if a["event_type"] == event_type]


def _guard_set(positives: List[int], band: int, n: int) -> set:
    """Frames that must not be used as negatives (near any positive)."""
    guard = set()
    for f in positives:
        for d in range(-band, band + 1):
            fi = f + d
            if 0 <= fi < n:
                guard.add(fi)
    return guard


def _in_point_frames(annotations: List[dict], total_frames: int) -> Set[int]:
    """
    Return the set of frame indices that fall within any annotated point.

    A frame F is in-point iff there exists a (start, end) pair from the
    annotations such that start ≤ F ≤ end. Assumes _postprocess_annotations
    has already dropped orphan point_start events so starts and ends pair
    1:1 in chronological order.

    Used to restrict negative sampling to in-rally frames only — the model
    never sees warmup / changeover / between-point frames as negatives,
    which keeps the decision boundary tuned for the things we actually
    care about classifying.
    """
    starts = sorted(int(a["frame"]) for a in annotations if a["event_type"] == "point_start")
    ends   = sorted(int(a["frame"]) for a in annotations if a["event_type"] == "point_end")
    in_pt: Set[int] = set()
    for s, e in zip(starts, ends):
        in_pt.update(range(s, min(e + 1, total_frames)))
    return in_pt


def _sample_negatives(
    rng:        np.random.Generator,
    n_frames:   int,
    half_win:   int,
    in_pt:      Set[int],
    guard:      Set[int],
    ball_x:     np.ndarray,
    n_positive: int,
    neg_ratio:  int = NEG_RATIO,
) -> List[int]:
    """
    Sample negative frames for binary event classification, restricted to
    in-point frames (so the model only learns to distinguish events from
    other in-rally moments, not from pre-match warmup).
    """
    candidates = [
        fi for fi in range(half_win, n_frames - half_win)
        if fi in in_pt
        and fi not in guard
        and not np.isnan(ball_x[fi])
    ]
    if not candidates:
        return []
    n_neg = min(len(candidates), n_positive * neg_ratio)
    return rng.choice(candidates, size=n_neg, replace=False).tolist()


def build_bounce_dataset(
    dataset: List[dict],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (N, F), y (N,), groups (N,) — groups = video index for LOVO CV.

    Negatives are sampled from in-point frames only — i.e. frames between an
    annotated point_start and the corresponding point_end. This focuses the
    model on the cases that matter (distinguishing bounces from other in-rally
    events) and ignores pre-match warmup / between-point dead time entirely.
    """
    X_list, y_list, g_list = [], [], []

    for gidx, entry in enumerate(dataset):
        npz    = entry["npz"]
        ball_x = npz["ball_x"].astype(np.float32)
        ball_y = npz["ball_y"].astype(np.float32)
        fps    = float(npz["fps"])
        n      = len(ball_x)

        positives = _positive_frames(entry["annotations"], "bounce")
        if not positives:
            continue

        guard = _guard_set(positives, GUARD_BAND, n)
        in_pt = _in_point_frames(entry["annotations"], n)

        for f in positives:
            X_list.append(bounce_features(ball_x, ball_y, fps, f))
            y_list.append(1)
            g_list.append(gidx)

        for f in _sample_negatives(rng, n, BOUNCE_HALF_WIN, in_pt, guard, ball_x, len(positives)):
            X_list.append(bounce_features(ball_x, ball_y, fps, f))
            y_list.append(0)
            g_list.append(gidx)

    if not X_list:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    return np.array(X_list), np.array(y_list), np.array(g_list)


def build_hit_dataset(
    dataset: List[dict],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns X, y, groups. Negatives restricted to in-point frames."""
    X_list, y_list, g_list = [], [], []

    for gidx, entry in enumerate(dataset):
        npz    = entry["npz"]
        ball_x = npz["ball_x"].astype(np.float32)
        ball_y = npz["ball_y"].astype(np.float32)
        near_x = npz["near_x"].astype(np.float32)
        near_y = npz["near_y"].astype(np.float32)
        far_x  = npz["far_x"].astype(np.float32)
        far_y  = npz["far_y"].astype(np.float32)
        fps    = float(npz["fps"])
        n      = len(ball_x)

        positives = _positive_frames(entry["annotations"], "hit")
        if not positives:
            continue

        guard = _guard_set(positives, GUARD_BAND, n)
        in_pt = _in_point_frames(entry["annotations"], n)

        for f in positives:
            X_list.append(hit_features(ball_x, ball_y, near_x, near_y, far_x, far_y, fps, f))
            y_list.append(1)
            g_list.append(gidx)

        for f in _sample_negatives(rng, n, HIT_HALF_WIN, in_pt, guard, ball_x, len(positives)):
            X_list.append(hit_features(ball_x, ball_y, near_x, near_y, far_x, far_y, fps, f))
            y_list.append(0)
            g_list.append(gidx)

    if not X_list:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    return np.array(X_list), np.array(y_list), np.array(g_list)


def build_point_start_dataset(
    dataset: List[dict],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Binary classifier: is frame F the start of a point?

    Positives  : annotated point_start frames (always coincide with a serve hit)
    Negatives  : in-point frames that are not within the guard band of a
                 point_start. The model effectively learns "serve-at-rally-start"
                 versus "any other in-rally frame" — which is the exact
                 distinction we need to detect rally boundaries.
    Features   : reuses hit_features, since point_start IS a hit (a serve)
                 and the serve has distinctive ball/player context the model
                 can pick up (back-court player, tossing ball arc, etc.).
    """
    X_list, y_list, g_list = [], [], []

    for gidx, entry in enumerate(dataset):
        npz    = entry["npz"]
        ball_x = npz["ball_x"].astype(np.float32)
        ball_y = npz["ball_y"].astype(np.float32)
        near_x = npz["near_x"].astype(np.float32)
        near_y = npz["near_y"].astype(np.float32)
        far_x  = npz["far_x"].astype(np.float32)
        far_y  = npz["far_y"].astype(np.float32)
        fps    = float(npz["fps"])
        n      = len(ball_x)

        positives = _positive_frames(entry["annotations"], "point_start")
        if not positives:
            continue

        guard = _guard_set(positives, GUARD_BAND, n)
        in_pt = _in_point_frames(entry["annotations"], n)

        for f in positives:
            X_list.append(hit_features(ball_x, ball_y, near_x, near_y, far_x, far_y, fps, f))
            y_list.append(1)
            g_list.append(gidx)

        for f in _sample_negatives(rng, n, HIT_HALF_WIN, in_pt, guard, ball_x, len(positives)):
            X_list.append(hit_features(ball_x, ball_y, near_x, near_y, far_x, far_y, fps, f))
            y_list.append(0)
            g_list.append(gidx)

    if not X_list:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    return np.array(X_list), np.array(y_list), np.array(g_list)


def build_point_end_dataset(
    dataset: List[dict],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Binary classifier: is frame F the end of a point?

    Positives  : annotated point_end frames. Note _postprocess_annotations
                 pushes these forward by 30 frames whenever they coincide with
                 a bounce, so the positive's feature window captures the
                 ~1s of trajectory AFTER the final bounce — which is exactly
                 the "ball going off court / no rally continues" signal.
    Negatives  : in-point frames not within guard band of any point_end.
    Features   : reuses bounce_features, since the most informative signal
                 for "this is the last ball-court interaction" lives in the
                 ball trajectory window. Net-shot endings (no bounce) come
                 with a zero/NaN ball trajectory tail which the model can
                 also pick up via the validity flags.
    """
    X_list, y_list, g_list = [], [], []

    for gidx, entry in enumerate(dataset):
        npz    = entry["npz"]
        ball_x = npz["ball_x"].astype(np.float32)
        ball_y = npz["ball_y"].astype(np.float32)
        fps    = float(npz["fps"])
        n      = len(ball_x)

        positives = _positive_frames(entry["annotations"], "point_end")
        if not positives:
            continue

        guard = _guard_set(positives, GUARD_BAND, n)
        in_pt = _in_point_frames(entry["annotations"], n)

        for f in positives:
            X_list.append(bounce_features(ball_x, ball_y, fps, f))
            y_list.append(1)
            g_list.append(gidx)

        for f in _sample_negatives(rng, n, BOUNCE_HALF_WIN, in_pt, guard, ball_x, len(positives)):
            X_list.append(bounce_features(ball_x, ball_y, fps, f))
            y_list.append(0)
            g_list.append(gidx)

    if not X_list:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    return np.array(X_list), np.array(y_list), np.array(g_list)


def build_stroke_dataset(
    dataset: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Returns X, y (int class indices), groups, label_names.

    Assumes all videos are righty-only (player_is_lefty = 0 everywhere).
    """
    X_list, y_list, g_list = [], [], []

    for gidx, entry in enumerate(dataset):
        npz     = entry["npz"]
        ball_x  = npz["ball_x"].astype(np.float32)
        ball_y  = npz["ball_y"].astype(np.float32)
        near_x  = npz["near_x"].astype(np.float32)
        near_y  = npz["near_y"].astype(np.float32)
        far_x   = npz["far_x"].astype(np.float32)
        far_y   = npz["far_y"].astype(np.float32)
        fps     = float(npz["fps"])
        frame_h = float(npz.get("frame_h", 720))

        for ann in entry["annotations"]:
            if ann["event_type"] != "hit":
                continue
            stroke = ann.get("stroke_type", "").strip().lower()
            if stroke not in STROKE_TO_INT:
                continue

            f = int(ann["frame"])

            # Prefer the annotated player field; otherwise infer from ball Y
            # (near = lower half of frame, far = upper half).
            player_field = ann.get("player", "").strip().lower()
            if player_field == "near":
                player_is_near = 1
            elif player_field == "far":
                player_is_near = 0
            else:
                by_at_f = ball_y[f] if (f < len(ball_y) and not np.isnan(ball_y[f])) else frame_h / 2
                player_is_near = int(by_at_f > frame_h * 0.5)

            X_list.append(stroke_features(
                ball_x, ball_y, near_x, near_y, far_x, far_y,
                fps, f, player_is_near, player_is_lefty=0,
            ))
            y_list.append(STROKE_TO_INT[stroke])
            g_list.append(gidx)

    if not X_list:
        return np.empty((0, 0)), np.empty(0), np.empty(0), STROKE_LABELS

    return np.array(X_list), np.array(y_list, dtype=np.int32), np.array(g_list), STROKE_LABELS


# ── Training + evaluation ─────────────────────────────────────────────────────

def _catboost_binary(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    scale_pos: float = 3.0,
) -> object:
    """Train a CatBoost binary classifier and print leave-one-video-out metrics."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  catboost not installed — run: pip install catboost")
        return None

    from sklearn.metrics import classification_report

    unique_groups = np.unique(groups)

    all_preds, all_true = [], []
    for test_g in unique_groups:
        tr_mask = groups != test_g
        te_mask = groups == test_g
        if tr_mask.sum() < 10 or te_mask.sum() < 2:
            continue

        clf = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            class_weights={0: 1.0, 1: scale_pos},
            eval_metric="F1",
            verbose=False,
            random_seed=42,
        )
        clf.fit(X[tr_mask], y[tr_mask])
        preds = clf.predict(X[te_mask])
        all_preds.extend(preds.tolist())
        all_true.extend(y[te_mask].tolist())

    if all_true:
        print(f"\n  [{model_name}] Leave-one-video-out evaluation:")
        print(classification_report(all_true, all_preds, target_names=["not_event", model_name],
                                    zero_division=0))
    else:
        print(f"\n  [{model_name}] Only one video — skipping cross-validation.")

    # Final model trained on all data
    clf_final = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        class_weights={0: 1.0, 1: scale_pos},
        eval_metric="F1",
        verbose=False,
        random_seed=42,
    )
    clf_final.fit(X, y)
    return clf_final


def _catboost_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_names: List[str],
    model_name: str,
) -> object:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  catboost not installed — run: pip install catboost")
        return None

    from sklearn.metrics import classification_report

    unique_groups = np.unique(groups)

    all_preds, all_true = [], []
    for test_g in unique_groups:
        tr_mask = groups != test_g
        te_mask = groups == test_g
        if tr_mask.sum() < 10 or te_mask.sum() < 1:
            continue

        clf = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            verbose=False,
            random_seed=42,
        )
        clf.fit(X[tr_mask], y[tr_mask])
        preds = clf.predict(X[te_mask]).flatten()
        all_preds.extend(preds.tolist())
        all_true.extend(y[te_mask].tolist())

    if all_true:
        present = sorted(set(all_true))
        names   = [label_names[i] for i in present]
        print(f"\n  [{model_name}] Leave-one-video-out evaluation:")
        print(classification_report(all_true, all_preds, labels=present,
                                    target_names=names, zero_division=0))
    else:
        print(f"\n  [{model_name}] Only one video — skipping cross-validation.")

    clf_final = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        verbose=False,
        random_seed=42,
    )
    clf_final.fit(X, y)
    return clf_final


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train bounce / hit / stroke models from annotated tennis videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir",          default=str(PROJECT_ROOT / "cv" / "training_data"))
    parser.add_argument("--model-dir",         default=str(PROJECT_ROOT / "cv" / "models"))
    parser.add_argument("--skip-bounce",       action="store_true")
    parser.add_argument("--skip-hit",          action="store_true")
    parser.add_argument("--skip-stroke",       action="store_true")
    parser.add_argument("--skip-point-start",  action="store_true")
    parser.add_argument("--skip-point-end",    action="store_true")
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="Substring; videos whose stem contains it are skipped. "
             "Repeat the flag to exclude multiple videos.",
    )
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*60}")
    print(f"  Tennis ML Model Trainer")
    print(f"  Data dir  : {data_dir}")
    print(f"  Model dir : {model_dir}")
    print(f"{'='*60}\n")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading dataset …")
    dataset = load_dataset(data_dir, exclude=args.exclude)

    if not dataset:
        print("\nNo data found.  Run these steps first:")
        print("  1. python cv/tools/extract_features.py --video <your_video.mp4>")
        print("  2. python cv/tools/annotate.py         --video <your_video.mp4>")
        sys.exit(1)

    print(f"\nLoaded {len(dataset)} video(s).")

    metrics: Dict[str, dict] = {}

    # ── Bounce model ──────────────────────────────────────────────────────────
    if not args.skip_bounce:
        print(f"\n{'─'*50}")
        print("Building bounce dataset …")
        X_b, y_b, g_b = build_bounce_dataset(dataset, rng)
        print(f"  Positives: {(y_b == 1).sum()}   Negatives: {(y_b == 0).sum()}")

        if (y_b == 1).sum() < MIN_EVENTS_TO_TRAIN:
            print(f"  Too few bounce labels ({(y_b==1).sum()} < {MIN_EVENTS_TO_TRAIN}) — skipping.")
        else:
            model_b = _catboost_binary(X_b, y_b, g_b, "bounce", scale_pos=4.0)
            if model_b is not None:
                out_path = str(model_dir / "bounce_model.cbm")
                model_b.save_model(out_path)
                print(f"\n  Saved → {out_path}")
                metrics["bounce"] = {"n_pos": int((y_b == 1).sum()), "n_neg": int((y_b == 0).sum())}

    # ── Hit model ─────────────────────────────────────────────────────────────
    if not args.skip_hit:
        print(f"\n{'─'*50}")
        print("Building hit dataset …")
        X_h, y_h, g_h = build_hit_dataset(dataset, rng)
        print(f"  Positives: {(y_h == 1).sum()}   Negatives: {(y_h == 0).sum()}")

        if (y_h == 1).sum() < MIN_EVENTS_TO_TRAIN:
            print(f"  Too few hit labels ({(y_h==1).sum()} < {MIN_EVENTS_TO_TRAIN}) — skipping.")
        else:
            model_h = _catboost_binary(X_h, y_h, g_h, "hit", scale_pos=4.0)
            if model_h is not None:
                out_path = str(model_dir / "hit_model.cbm")
                model_h.save_model(out_path)
                print(f"\n  Saved → {out_path}")
                metrics["hit"] = {"n_pos": int((y_h == 1).sum()), "n_neg": int((y_h == 0).sum())}

    # ── Stroke classifier ─────────────────────────────────────────────────────
    if not args.skip_stroke:
        print(f"\n{'─'*50}")
        print("Building stroke dataset …")
        X_s, y_s, g_s, label_names = build_stroke_dataset(dataset)
        if len(y_s) > 0:
            for i, name in enumerate(label_names):
                print(f"  {name:<15} {(y_s == i).sum()} samples")

        if len(y_s) < MIN_EVENTS_TO_TRAIN:
            print(f"  Too few stroke labels ({len(y_s)} < {MIN_EVENTS_TO_TRAIN}) — skipping.")
        else:
            model_s = _catboost_multiclass(X_s, y_s, g_s, label_names, "stroke")
            if model_s is not None:
                out_path = str(model_dir / "stroke_model.cbm")
                model_s.save_model(out_path)
                # Save label mapping alongside model
                label_path = str(model_dir / "stroke_labels.json")
                with open(label_path, "w") as f:
                    json.dump(label_names, f, indent=2)
                print(f"\n  Saved → {out_path}")
                print(f"  Saved → {label_path}")
                metrics["stroke"] = {"labels": label_names, "n": int(len(y_s))}

    # ── Point-start model ─────────────────────────────────────────────────────
    if not args.skip_point_start:
        print(f"\n{'─'*50}")
        print("Building point_start dataset …")
        X_ps, y_ps, g_ps = build_point_start_dataset(dataset, rng)
        print(f"  Positives: {(y_ps == 1).sum()}   Negatives: {(y_ps == 0).sum()}")

        if (y_ps == 1).sum() < MIN_EVENTS_TO_TRAIN:
            print(f"  Too few point_start labels ({(y_ps==1).sum()} < {MIN_EVENTS_TO_TRAIN}) — skipping.")
        else:
            model_ps = _catboost_binary(X_ps, y_ps, g_ps, "point_start", scale_pos=4.0)
            if model_ps is not None:
                out_path = str(model_dir / "point_start_model.cbm")
                model_ps.save_model(out_path)
                print(f"\n  Saved → {out_path}")
                metrics["point_start"] = {"n_pos": int((y_ps == 1).sum()), "n_neg": int((y_ps == 0).sum())}

    # ── Point-end model ───────────────────────────────────────────────────────
    if not args.skip_point_end:
        print(f"\n{'─'*50}")
        print("Building point_end dataset …")
        X_pe, y_pe, g_pe = build_point_end_dataset(dataset, rng)
        print(f"  Positives: {(y_pe == 1).sum()}   Negatives: {(y_pe == 0).sum()}")

        if (y_pe == 1).sum() < MIN_EVENTS_TO_TRAIN:
            print(f"  Too few point_end labels ({(y_pe==1).sum()} < {MIN_EVENTS_TO_TRAIN}) — skipping.")
        else:
            model_pe = _catboost_binary(X_pe, y_pe, g_pe, "point_end", scale_pos=4.0)
            if model_pe is not None:
                out_path = str(model_dir / "point_end_model.cbm")
                model_pe.save_model(out_path)
                print(f"\n  Saved → {out_path}")
                metrics["point_end"] = {"n_pos": int((y_pe == 1).sum()), "n_neg": int((y_pe == 0).sum())}

    # ── Summary ───────────────────────────────────────────────────────────────
    meta_path = model_dir / "model_info.json"
    with open(meta_path, "w") as f:
        json.dump({
            "trained_on_videos": [d["video_id"] for d in dataset],
            "models": metrics,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete.  Metadata → {meta_path}")
    print(f"  Models written  → {model_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
