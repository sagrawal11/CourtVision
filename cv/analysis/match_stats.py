"""
Match statistics aggregator for tennis match analysis.

Takes per-point data (from PointSegmenter) and per-point player labels
(from POITracker) and computes the full statistics suite expected by the
web frontend.

Output schema (stored in `match_data` table):
    {
        "total_points":    int,
        "poi_points_won":  int,
        "opp_points_won":  int,

        "poi_shots":       int,   # total groundstrokes + volleys
        "poi_winners":     int,
        "poi_errors":      int,   # unforced + forced combined
        "poi_in_play":     int,
        "poi_winner_pct":  float,

        "poi_serves_total":    int,
        "poi_first_serves_in": int,
        "poi_serve_1_pct":     float,   # 1st serve %
        "poi_aces":            int,

        "serve_zones": {           # how many serves landed in each zone
            "deuce_wide": int, "deuce_body": int, "deuce_T": int,
            "ad_wide": int,   "ad_body": int,   "ad_T": int,
        },

        "rally_lengths":    List[int],    # per-point rally length in shots
        "avg_rally_length": float,
    }
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional
import numpy as np

from cv.analysis.point_detector import PointRecord, BounceEvent


# -------------------------------------------------------------------------
# Serve zone classification
# -------------------------------------------------------------------------

def classify_serve_zone(
    court_x: float,
    court_y: float,
    serving_side: str,  # "near" | "far"
    deuce_end: bool,    # True = serving from deuce (right) court
) -> str:
    """
    Map a serve bounce (in normalised court coords 0-1) to a named zone.

    Court coordinate system:
        x: 0 = left doubles sideline, 1 = right doubles sideline
        y: 0 = far baseline (top), 1 = near baseline (bottom)

    Service box zones (receiver's perspective):
        - T:    ball lands near the center service line
        - Body: ball lands near the receiver's body
        - Wide: ball lands near the singles sideline
    """
    # Singles sideline x-range approximately 0.1 – 0.9 of the full court width
    # Center service line at x ≈ 0.5
    cx = court_x

    if deuce_end:
        # Deuce court: receiver is on the right side (ad side, x > 0.5)
        if cx < 0.3:
            zone = "T"
        elif cx < 0.6:
            zone = "Body"
        else:
            zone = "Wide"
        return f"deuce_{zone.lower()}"
    else:
        # Ad court: receiver is on the left side (deuce side, x < 0.5)
        if cx > 0.7:
            zone = "T"
        elif cx > 0.4:
            zone = "Body"
        else:
            zone = "Wide"
        return f"ad_{zone.lower()}"


# -------------------------------------------------------------------------
# Match Stats Aggregator
# -------------------------------------------------------------------------

@dataclasses.dataclass
class MatchStats:
    """Full match statistics for the target player (POI)."""

    # Point-level
    total_points: int = 0
    poi_points_won: int = 0
    opp_points_won: int = 0

    # Shot-level
    poi_shots: int = 0
    poi_winners: int = 0
    poi_errors: int = 0
    poi_in_play: int = 0

    # Serve
    poi_serves_total: int = 0
    poi_first_serves_in: int = 0
    poi_aces: int = 0

    # Per-zone serve counts
    serve_zones: Dict[str, int] = dataclasses.field(default_factory=lambda: {
        "deuce_t": 0, "deuce_body": 0, "deuce_wide": 0,
        "ad_t": 0,    "ad_body": 0,    "ad_wide": 0,
    })

    # Rally length
    rally_lengths: List[int] = dataclasses.field(default_factory=list)

    # Advanced Target Player Analytics (Phase 6)
    poi_serve_speed_avg: float = 0.0
    poi_forehand_speed_avg: float = 0.0
    poi_backhand_speed_avg: float = 0.0
    poi_forehands: int = 0
    poi_backhands: int = 0

    # Internal buffers for average calculation
    _serve_speeds: List[float] = dataclasses.field(default_factory=list, repr=False)
    _forehand_speeds: List[float] = dataclasses.field(default_factory=list, repr=False)
    _backhand_speeds: List[float] = dataclasses.field(default_factory=list, repr=False)
    
    def finalize_averages(self):
        if self._serve_speeds:
            self.poi_serve_speed_avg = round(float(np.mean(self._serve_speeds)), 1)
        if self._forehand_speeds:
            self.poi_forehand_speed_avg = round(float(np.mean(self._forehand_speeds)), 1)
        if self._backhand_speeds:
            self.poi_backhand_speed_avg = round(float(np.mean(self._backhand_speeds)), 1)

    @property
    def poi_winner_pct(self) -> float:
        total = self.poi_winners + self.poi_errors
        return round(self.poi_winners / total * 100, 1) if total > 0 else 0.0

    @property
    def poi_serve_1_pct(self) -> float:
        return round(self.poi_first_serves_in / self.poi_serves_total * 100, 1) \
            if self.poi_serves_total > 0 else 0.0

    @property
    def avg_rally_length(self) -> float:
        return round(float(np.mean(self.rally_lengths)), 1) if self.rally_lengths else 0.0

    def to_dict(self) -> dict:
        return {
            "total_points":        self.total_points,
            "poi_points_won":      self.poi_points_won,
            "opp_points_won":      self.opp_points_won,
            "poi_shots":           self.poi_shots,
            "poi_winners":         self.poi_winners,
            "poi_errors":          self.poi_errors,
            "poi_in_play":         self.poi_in_play,
            "poi_winner_pct":      self.poi_winner_pct,
            "poi_serves_total":    self.poi_serves_total,
            "poi_first_serves_in": self.poi_first_serves_in,
            "poi_serve_1_pct":     self.poi_serve_1_pct,
            "poi_aces":            self.poi_aces,
            "serve_zones":         self.serve_zones,
            "rally_lengths":       self.rally_lengths,
            "avg_rally_length":    self.avg_rally_length,
            "poi_serve_speed_avg": self.poi_serve_speed_avg,
            "poi_forehand_speed_avg": self.poi_forehand_speed_avg,
            "poi_backhand_speed_avg": self.poi_backhand_speed_avg,
            "poi_forehands":       self.poi_forehands,
            "poi_backhands":       self.poi_backhands,
        }


class MatchStatsAggregator:
    """
    Aggregates PointRecord list into a MatchStats object.

    Usage:
        agg = MatchStatsAggregator(poi_start_side="near")
        stats = agg.aggregate(points)
        result = stats.to_dict()
    """

    def __init__(self, poi_start_side: str = "near"):
        self.poi_start_side = poi_start_side

    def aggregate(self, points: List[PointRecord]) -> MatchStats:
        stats = MatchStats()
        stats.total_points = len(points)

        for point in points:
            is_poi_serving = (point.serve_player == self.poi_start_side)
            # Note: switch logic mirrors PointStateMachine's serve_side alternation
            # (already baked into point.serve_player by the state machine)

            # ── Serve stats ────────────────────────────────────────────
            if is_poi_serving:
                stats.poi_serves_total += 1
                if point.serve_bounce is not None:
                    stats.poi_first_serves_in += 1
                    # Classify serve zone
                    game_number = points.index(point)   # crude proxy for deuce/ad
                    deuce_end = (game_number % 2 == 0)  # alternates each point
                    zone = classify_serve_zone(
                        point.serve_bounce.court_x,
                        point.serve_bounce.court_y,
                        serving_side=point.serve_player,
                        deuce_end=deuce_end,
                    )
                    if zone in stats.serve_zones:
                        stats.serve_zones[zone] += 1

            # ── Shot + Point outcome ───────────────────────────────────
            rally_len = len(point.bounces)
            if rally_len:
                stats.rally_lengths.append(rally_len)
                stats.poi_shots += rally_len  # rough proxy until shot classifier runs

            if point.outcome == "winner":
                if is_poi_serving:
                    stats.poi_points_won += 1
                    stats.poi_winners += 1
                else:
                    stats.opp_points_won += 1

            elif point.outcome in ("error_out", "error_net"):
                # Determine who made the error
                # If it's the last shot of a rally, error belongs to the player
                # who last touched the ball. Without shot classifier, we infer:
                # odd-numbered bounces = POI hit last; even = OPP hit last
                poi_hit_last = (rally_len % 2 == (1 if is_poi_serving else 0))
                if poi_hit_last:
                    stats.poi_errors += 1
                    stats.opp_points_won += 1
                else:
                    stats.poi_points_won += 1

            elif point.outcome == "in_play":
                stats.poi_in_play += 1
                # Can't determine winner without shot classifier
                stats.poi_points_won += rally_len % 2   # rough estimate

            # ── Target Player Shot Mechanics (Phase 6) ──────────────────
            for shot in point.shots:
                if shot.player == self.poi_start_side:
                    # Increment hit counters
                    if shot.shot_type == "forehand":
                        stats.poi_forehands += 1
                        if shot.speed_kmh: stats._forehand_speeds.append(shot.speed_kmh)
                    elif shot.shot_type == "backhand":
                        stats.poi_backhands += 1
                        if shot.speed_kmh: stats._backhand_speeds.append(shot.speed_kmh)
                    elif shot.shot_type == "serve":
                        if shot.speed_kmh: stats._serve_speeds.append(shot.speed_kmh)

                    # Tally absolute winners/errors explicitly designated by HitDetector
                    if shot.is_winner:
                        stats.poi_winners += 1
                    if shot.is_error:
                        stats.poi_errors += 1

        stats.finalize_averages()
        return stats


# -------------------------------------------------------------------------
# Shot-level stats (Phase 3 — populated by shot classifier)
# -------------------------------------------------------------------------

def merge_shot_classifications(
    stats: MatchStats,
    shot_records: List[dict],
    poi_label: str = "poi",
) -> MatchStats:
    """
    Refine MatchStats once shot-level data is available from the shot classifier.

    Each shot_record should have:
        {
            "point_idx": int,
            "shot_idx":  int,        # within point
            "player":    "poi"|"opp",
            "shot_type": "forehand"|"backhand"|"serve"|"volley"|"smash",
            "outcome":   "winner"|"error"|"in_play",
            "frame":     int,
        }
    """
    # Reset shot-level counters that we'll recompute from ground truth
    stats.poi_shots   = 0
    stats.poi_winners = 0
    stats.poi_errors  = 0
    stats.poi_in_play = 0

    for shot in shot_records:
        if shot.get("player") != poi_label:
            continue
        stats.poi_shots += 1
        outcome = shot.get("outcome", "in_play")
        if outcome == "winner":
            stats.poi_winners += 1
        elif outcome == "error":
            stats.poi_errors += 1
        else:
            stats.poi_in_play += 1

    return stats
