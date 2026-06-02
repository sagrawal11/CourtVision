from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import os
from datetime import datetime, timezone
from supabase import create_client, Client
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id

router = APIRouter(prefix="/api/matches", tags=["matches"])

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")

supabase: Client = create_client(supabase_url, supabase_key)


class MatchCreate(BaseModel):
    playsight_link: Optional[str] = None   # Optional — use video file upload instead
    player_name: Optional[str] = None
    user_id: Optional[str] = None  # For coaches uploading matches for players
    match_date: Optional[str] = None  # ISO date string (YYYY-MM-DD)
    opponent: Optional[str] = None
    notes: Optional[str] = None


@router.get("/")
async def list_matches(user_id: str = Depends(get_user_id)):
    """
    List all matches for a user.
    Coaches see all team member matches, players see only their own.
    """
    # Get user role
    user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
    
    if not user_response.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_role = user_response.data.get("role")
    
    if user_role == "coach":
        # Get all team member matches
        # First get teams the coach belongs to
        teams_response = supabase.table("team_members").select("team_id").eq("user_id", user_id).execute()
        team_ids = [t["team_id"] for t in (teams_response.data or [])]
        
        # Get all team members
        if team_ids:
            members_response = supabase.table("team_members").select("user_id").in_("team_id", team_ids).execute()
            member_ids = [m["user_id"] for m in (members_response.data or [])]
            
            # Get matches for all team members
            matches_response = supabase.table("matches").select("*").in_("user_id", member_ids).order("created_at", desc=True).execute()
        else:
            matches_response = supabase.table("matches").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    else:
        # Player sees only their own matches
        matches_response = supabase.table("matches").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    
    return {"matches": matches_response.data or []}


@router.get("/{match_id}")
async def get_match(match_id: str, user_id: str = Depends(get_user_id)):
    """Get a specific match with all its data, including analysis stats and shots."""
    # Get match
    match_response = supabase.table("matches").select("*").eq("id", match_id).single().execute()
    
    if not match_response.data:
        raise HTTPException(status_code=404, detail="Match not found")
    
    match = match_response.data
    
    # Verify user has access (RLS should handle this, but double-check)
    if match["user_id"] != user_id:
        # Check if user is coach on same team
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_response.data or user_response.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Get match data blob (for debugging / raw output)
    match_data_response = supabase.table("match_data").select("stats_summary").eq("match_id", match_id).execute()
    match_data = match_data_response.data[0] if match_data_response.data else None
    
    # Get shots ordered by frame/timestamp
    shots_response = (
        supabase.table("shots")
        .select("id,frame,shot_type,speed_kmh,player,result,is_winner,is_error,video_timestamp,start_pos")
        .eq("match_id", match_id)
        .order("frame", desc=False)
        .execute()
    )
    
    return {
        "match": match,
        "stats": match.get("stats"),           # Full MatchStats dict (from analysis_job.py)
        "analysis_shots": match.get("analysis_shots"),  # Flat shots list for visualization
        "match_data": match_data,
        "shots": shots_response.data or []
    }


@router.get("/{match_id}/points")
async def get_match_points(match_id: str, user_id: str = Depends(get_user_id)):
    """Return all detected points for a match, ordered by point_idx."""
    match_response = supabase.table("matches").select("user_id").eq("id", match_id).single().execute()
    if not match_response.data:
        raise HTTPException(status_code=404, detail="Match not found")

    match_owner = match_response.data["user_id"]
    if match_owner != user_id:
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_response.data or user_response.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Access denied")

    points_response = (
        supabase.table("points")
        .select("*")
        .eq("match_id", match_id)
        .order("point_idx", desc=False)
        .execute()
    )
    return {"points": points_response.data or []}


class PointClassify(BaseModel):
    manual_outcome: str  # 'winner' | 'forced_error' | 'unforced_error' | 'ace' | 'double_fault'


@router.patch("/{match_id}/points/{point_idx}")
async def classify_point(
    match_id: str,
    point_idx: int,
    body: PointClassify,
    user_id: str = Depends(get_user_id),
):
    """Save a coach's manual classification for a single point."""
    valid = {"winner", "forced_error", "unforced_error", "ace", "double_fault"}
    if body.manual_outcome not in valid:
        raise HTTPException(status_code=422, detail=f"manual_outcome must be one of {sorted(valid)}")

    match_response = supabase.table("matches").select("user_id").eq("id", match_id).single().execute()
    if not match_response.data:
        raise HTTPException(status_code=404, detail="Match not found")

    match_owner = match_response.data["user_id"]
    if match_owner != user_id:
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_response.data or user_response.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Access denied")

    now_iso = datetime.now(timezone.utc).isoformat()
    update_response = (
        supabase.table("points")
        .update({"manual_outcome": body.manual_outcome, "reviewed_at": now_iso})
        .eq("match_id", match_id)
        .eq("point_idx", point_idx)
        .execute()
    )
    if not update_response.data:
        raise HTTPException(status_code=404, detail="Point not found")
    return {"point": update_response.data[0]}


@router.patch("/{match_id}/poi")
async def swap_poi(match_id: str, user_id: str = Depends(get_user_id)):
    """Flip poi_start_side between 'near' and 'far'. Stats reflect the change on next re-analysis."""
    match_response = supabase.table("matches").select("user_id, poi_start_side").eq("id", match_id).single().execute()
    if not match_response.data:
        raise HTTPException(status_code=404, detail="Match not found")

    match_owner = match_response.data["user_id"]
    if match_owner != user_id:
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_response.data or user_response.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Access denied")

    current = match_response.data.get("poi_start_side") or "near"
    new_side = "far" if current == "near" else "near"
    supabase.table("matches").update({"poi_start_side": new_side}).eq("id", match_id).execute()
    return {"poi_start_side": new_side}


@router.post("/")
async def create_match(match_data: MatchCreate, user_id: str = Depends(get_user_id)):
    """
    Create a new match.
    If user_id is provided and the authenticated user is a coach, use the provided user_id.
    Otherwise, use the authenticated user's ID.
    """
    # Determine which user_id to use for the match
    match_user_id = user_id
    
    # If user_id is provided in request, verify user is a coach and can upload for that player
    if match_data.user_id:
        # Get authenticated user's role
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_role = user_response.data.get("role")
        
        if user_role != "coach":
            raise HTTPException(status_code=403, detail="Only coaches can upload matches for other players")
        
        # Verify the target player is on one of the coach's teams
        # Get coach's teams
        teams_response = supabase.table("team_members").select("team_id").eq("user_id", user_id).execute()
        team_ids = [t["team_id"] for t in (teams_response.data or [])]
        
        if team_ids:
            # Check if target player is a member of any of these teams
            members_response = supabase.table("team_members").select("team_id").eq("user_id", match_data.user_id).in_("team_id", team_ids).execute()
            
            if not members_response.data:
                raise HTTPException(status_code=403, detail="Player must be a member of your team")
        
        match_user_id = match_data.user_id
    
    match_insert_data = {
        "user_id": match_user_id,
        "playsight_link": match_data.playsight_link,
        "player_name": match_data.player_name,
        "status": "pending"
    }
    
    # Add optional fields if provided
    if match_data.match_date:
        match_insert_data["match_date"] = match_data.match_date
    if match_data.opponent:
        match_insert_data["opponent"] = match_data.opponent
    if match_data.notes:
        match_insert_data["notes"] = match_data.notes
    
    match_response = supabase.table("matches").insert(match_insert_data).execute()
    
    if not match_response.data:
        raise HTTPException(status_code=500, detail="Failed to create match")
    
    return {"match": match_response.data[0]}
