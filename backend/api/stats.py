from fastapi import APIRouter, HTTPException, Depends
import os
from supabase import create_client, Client
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id

router = APIRouter(prefix="/api/stats", tags=["stats"])

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")

supabase: Client = create_client(supabase_url, supabase_key)


def _aggregate_from_matches(matches: list) -> dict:
    """
    Roll up per-match stats JSONB into a season-level summary.
    Reads from matches.stats — no shots table round-trip needed.
    """
    totals = {
        "total_points": 0,
        "poi_points_won": 0,
        "poi_shots": 0,
        "poi_winners": 0,
        "poi_errors": 0,
        "poi_serves_total": 0,
        "poi_first_serves_in": 0,
        "poi_aces": 0,
        "poi_forehands": 0,
        "poi_backhands": 0,
    }
    serve_speeds, forehand_speeds, backhand_speeds, rally_avgs = [], [], [], []
    per_match = []

    for m in matches:
        s = m.get("stats") or {}
        for k in totals:
            totals[k] += s.get(k, 0)
        if s.get("avg_rally_length"):
            rally_avgs.append(s["avg_rally_length"])
        if s.get("poi_serve_speed_avg"):
            serve_speeds.append(s["poi_serve_speed_avg"])
        if s.get("poi_forehand_speed_avg"):
            forehand_speeds.append(s["poi_forehand_speed_avg"])
        if s.get("poi_backhand_speed_avg"):
            backhand_speeds.append(s["poi_backhand_speed_avg"])

        per_match.append({
            "match_id": m["id"],
            "match_date": m.get("match_date"),
            "opponent": m.get("opponent"),
            "total_points": s.get("total_points", 0),
            "poi_points_won": s.get("poi_points_won", 0),
            "poi_winners": s.get("poi_winners", 0),
            "poi_errors": s.get("poi_errors", 0),
            "avg_rally_length": s.get("avg_rally_length", 0),
            "poi_serve_1_pct": s.get("poi_serve_1_pct", 0),
        })

    def safe_avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    return {
        "total_matches": len(matches),
        "season_totals": totals,
        "avg_rally_length": safe_avg(rally_avgs),
        "avg_serve_speed_kmh": safe_avg(serve_speeds),
        "avg_forehand_speed_kmh": safe_avg(forehand_speeds),
        "avg_backhand_speed_kmh": safe_avg(backhand_speeds),
        "per_match": per_match,
    }


@router.get("/player/{player_id}")
async def get_player_stats(player_id: str, user_id: str = Depends(get_user_id)):
    """Season statistics for a player. Coaches can view any team member; players see their own."""
    if player_id != user_id:
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_response.data or user_response.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Access denied")

    matches_response = (
        supabase.table("matches")
        .select("id, match_date, opponent, stats")
        .eq("user_id", player_id)
        .eq("status", "completed")
        .order("match_date", desc=True)
        .execute()
    )
    return {"player_id": player_id, **_aggregate_from_matches(matches_response.data or [])}


@router.get("/my-stats")
async def get_my_stats(user_id: str = Depends(get_user_id)):
    """Current user's season statistics."""
    return await get_player_stats(user_id, user_id)
