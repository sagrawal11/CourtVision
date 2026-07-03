"""Tests for PointSegmenter._apply_dual_bounce_winner (P2.2).

A true winner bounces twice on the same court half within 2 seconds with no
intervening racket contact. These exercise the REAL method (not a copy of the
logic) so a regression in point_detector.py is caught.

Run with: pytest tests/test_dual_bounce.py
"""

from cv.analysis.point_detector import PointSegmenter, BounceEvent, HitEvent, PointRecord


def _segmenter():
    return PointSegmenter(fps=30, player_start_side="near")


def _point(bounces, shots):
    pt = PointRecord(
        point_idx=0, start_frame=0, end_frame=100, serve_player="near",
        outcome="in_play", error_player=None, bounces=[],
    )
    pt.bounces = bounces
    pt.shots = shots
    return pt


def _bounce(frame, court_y):
    return BounceEvent(frame_idx=frame, x=100.0, y=100.0, court_x=0.5, court_y=court_y, is_in_bounds=True)


def _hit(frame, player="near"):
    return HitEvent(frame_idx=frame, x=50.0, y=50.0, player=player,
                    court_x=0.5, court_y=0.8, speed_kmh=100.0, shot_type="forehand")


def test_two_bounces_same_half_no_hit_between_is_winner():
    pt = _point([_bounce(30, 0.2), _bounce(60, 0.3)], [_hit(10)])
    assert _segmenter()._apply_dual_bounce_winner(pt) is True
    assert pt.outcome == "winner"
    assert pt.shots[0].is_winner is True


def test_hit_between_bounces_is_not_a_winner():
    # A return between the two bounces means the ball was played back — not a winner.
    pt = _point([_bounce(30, 0.2), _bounce(60, 0.3)], [_hit(10), _hit(45, player="far")])
    assert _segmenter()._apply_dual_bounce_winner(pt) is False
    assert pt.outcome == "in_play"
    assert all(not h.is_winner for h in pt.shots)


def test_bounces_on_opposite_halves_not_a_winner():
    pt = _point([_bounce(30, 0.2), _bounce(60, 0.8)], [_hit(10)])
    assert _segmenter()._apply_dual_bounce_winner(pt) is False
    assert pt.outcome == "in_play"


def test_bounces_too_far_apart_in_time_not_a_winner():
    # > fps*2 (60 frames at 30fps) apart → second bounce is a separate event.
    pt = _point([_bounce(30, 0.2), _bounce(120, 0.3)], [_hit(10)])
    assert _segmenter()._apply_dual_bounce_winner(pt) is False
    assert pt.outcome == "in_play"


def test_single_bounce_is_noop():
    pt = _point([_bounce(30, 0.2)], [_hit(10)])
    assert _segmenter()._apply_dual_bounce_winner(pt) is False
    assert pt.outcome == "in_play"


def test_only_applies_to_in_play_points():
    pt = _point([_bounce(30, 0.2), _bounce(60, 0.3)], [_hit(10)])
    pt.outcome = "error_net"
    assert _segmenter()._apply_dual_bounce_winner(pt) is False
    assert pt.outcome == "error_net"
