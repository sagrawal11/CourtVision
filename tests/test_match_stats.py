from cv.analysis.match_stats import MatchStatsAggregator
from cv.analysis.point_detector import PointRecord

def test():
    agg = MatchStatsAggregator(poi_start_side="near")
    p1 = PointRecord(point_idx=0, start_frame=0, end_frame=10, serve_player="near", outcome="error_net", error_player="near", bounces=[])
    p2 = PointRecord(point_idx=1, start_frame=20, end_frame=30, serve_player="far", outcome="error_out", error_player="far", bounces=[])
    p3 = PointRecord(point_idx=2, start_frame=40, end_frame=50, serve_player="near", outcome="winner", error_player=None, bounces=[])
    stats = agg.aggregate([p1, p2, p3])
    
    print(f"POI points won: {stats.poi_points_won}")
    print(f"OPP points won: {stats.opp_points_won}")
    print(f"POI errors: {stats.poi_errors}")

if __name__ == "__main__":
    test()
