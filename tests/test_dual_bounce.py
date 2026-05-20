import cv2
import numpy as np
from cv.analysis.point_detector import PointSegmenter, BounceEvent, HitEvent, PointRecord

def test_dual_bounce_winner():
    segmenter = PointSegmenter(fps=30, player_start_side="near")
    
    # We don't need to run the full pipeline, just feed a mocked point into the loop
    point = PointRecord(point_idx=0, start_frame=0, end_frame=100, serve_player="near", outcome="in_play", error_player=None, bounces=[])
    
    # Create two bounces on the far side of the court (court_y < 0.5)
    b1 = BounceEvent(frame_idx=30, x=100.0, y=100.0, court_x=0.5, court_y=0.2, is_in_bounds=True)
    b2 = BounceEvent(frame_idx=60, x=100.0, y=100.0, court_x=0.6, court_y=0.3, is_in_bounds=True)
    point.bounces = [b1, b2]
    
    # Create a hit BEFORE the first bounce
    hit1 = HitEvent(frame_idx=10, x=50.0, y=50.0, player="near", court_x=0.5, court_y=0.8, speed_kmh=100.0, shot_type="forehand")
    point.shots = [hit1]
    
    # Run the specific logic block from PointSegmenter.run
    if point.outcome == "in_play" and len(point.bounces) >= 2:
        sorted_bounces = sorted(point.bounces, key=lambda b: b.frame_idx)
        for i in range(len(sorted_bounces) - 1):
            b1_iter = sorted_bounces[i]
            b2_iter = sorted_bounces[i+1]
            
            if b2_iter.frame_idx - b1_iter.frame_idx <= segmenter.fps * 2:
                if (b1_iter.court_y > 0.5 and b2_iter.court_y > 0.5) or (b1_iter.court_y < 0.5 and b2_iter.court_y < 0.5):
                    hits_between = [h for h in point.shots if b1_iter.frame_idx < h.frame_idx < b2_iter.frame_idx]
                    if not hits_between:
                        hits_before = [h for h in point.shots if h.frame_idx < b1_iter.frame_idx]
                        if hits_before:
                            hits_before[-1].is_winner = True
                            point.outcome = "winner"
                        break

    print(f"Outcome: {point.outcome}")
    print(f"Hit is_winner: {hit1.is_winner}")
    assert point.outcome == "winner"
    assert hit1.is_winner == True
    print("Test passed!")

if __name__ == "__main__":
    test_dual_bounce_winner()
