"""
Video Upload & Processing API.

Manages the full lifecycle of a match video from upload to analysis.
Free-tier stack: Supabase Storage for video files, local subprocess for CV processing.

Upload Flow:
  1. POST /api/videos/prepare-upload              — Create match record, return storage path + signed upload URL
  2. (Browser uploads directly to Supabase Storage using signed URL)
  3. POST /api/videos/{id}/confirm-upload          — Verify upload, kick off local court_setup_job
  4. POST /api/videos/{id}/court-keypoints         — (Internal) Court job posts AI keypoints here
  5. PUT  /api/videos/{id}/court-keypoints         — User confirms keypoints, triggers full analysis
  6. GET  /api/videos/{id}/frame-url               — Get signed URL for frame image display
  7. GET  /api/videos/{id}/status                  — Poll overall status + court_setup_status
  8. POST /api/videos/{id}/generate-debug-video    — Trigger local debug video rendering job
  9. PATCH /api/videos/{id}/debug-video-ready      — (Internal) Job notifies backend when done
 10. GET  /api/videos/{id}/debug-video-url         — Get signed download URL for debug video

See docs/video_pipeline.md for the full sequence diagram.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client

sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id
from api.storage import (
    get_upload_path, get_frame_path,
    create_signed_upload_url, create_signed_download_url,
    delete_file, file_exists,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/videos", tags=["videos"])

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")
supabase: Client = create_client(supabase_url, supabase_key)

# Project root for launching local subprocesses
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----- Request / Response Models ---------------------------------------------

class PrepareUploadRequest(BaseModel):
    filename: str
    match_date: Optional[str] = None
    opponent: Optional[str] = None
    player_name: Optional[str] = None
    notes: Optional[str] = None
    player_user_id: Optional[str] = None  # For coaches uploading on behalf of a player


class ConfirmUploadRequest(BaseModel):
    storage_path: str   # Echo back so we can verify it matches what was issued


class CourtKeypointPayload(BaseModel):
    """14 (x, y) keypoint pairs. Null values indicate undetected points."""
    kp0_x: Optional[float] = None;  kp0_y: Optional[float] = None
    kp1_x: Optional[float] = None;  kp1_y: Optional[float] = None
    kp2_x: Optional[float] = None;  kp2_y: Optional[float] = None
    kp3_x: Optional[float] = None;  kp3_y: Optional[float] = None
    kp4_x: Optional[float] = None;  kp4_y: Optional[float] = None
    kp5_x: Optional[float] = None;  kp5_y: Optional[float] = None
    kp6_x: Optional[float] = None;  kp6_y: Optional[float] = None
    kp7_x: Optional[float] = None;  kp7_y: Optional[float] = None
    kp8_x: Optional[float] = None;  kp8_y: Optional[float] = None
    kp9_x: Optional[float] = None;  kp9_y: Optional[float] = None
    kp10_x: Optional[float] = None; kp10_y: Optional[float] = None
    kp11_x: Optional[float] = None; kp11_y: Optional[float] = None
    kp12_x: Optional[float] = None; kp12_y: Optional[float] = None
    kp13_x: Optional[float] = None; kp13_y: Optional[float] = None
    ai_suggested: bool = True


class PlayerIdentification(BaseModel):
    match_id: str
    frame_data: dict
    selected_player_coords: dict


# ----- Helpers ---------------------------------------------------------------

def _get_match_or_403(match_id: str, user_id: str) -> dict:
    resp = supabase.table("matches").select("*").eq("id", match_id).single().execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="Match not found")
    match = resp.data
    if match["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return match


# ----- Endpoints -------------------------------------------------------------

@router.post("/prepare-upload")
async def prepare_upload(
    req: PrepareUploadRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Step 1 of the upload flow.

    Creates a match record and returns a signed Supabase Storage URL that
    the browser uses to upload the video file directly — it never passes
    through our backend server.

    Returns:
        {
            "match_id": str,
            "storage_path": str,   # store this, needed for confirm-upload
            "upload_url": str,     # signed PUT URL — valid for 1 hour
        }
    """
    # Coach uploading for a player
    match_user_id = user_id
    if req.player_user_id:
        user_resp = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_resp.data or user_resp.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Only coaches can upload for other players")
        match_user_id = req.player_user_id

    # Create the match record first so we have an ID for the storage path
    insert_data = {
        "user_id": match_user_id,
        "video_filename": req.filename,
        "status": "pending",
        "court_setup_status": "pending",
        **({"player_name": req.player_name} if req.player_name else {}),
        **({"match_date": req.match_date} if req.match_date else {}),
        **({"opponent": req.opponent} if req.opponent else {}),
        **({"notes": req.notes} if req.notes else {}),
    }
    match_resp = supabase.table("matches").insert(insert_data).execute()
    if not match_resp.data:
        raise HTTPException(status_code=500, detail="Failed to create match record")

    match_id = match_resp.data[0]["id"]
    storage_path = get_upload_path(match_id, req.filename)

    # Save storage path to match record
    supabase.table("matches").update({"s3_temp_key": storage_path}).eq("id", match_id).execute()

    # Generate signed upload URL for direct browser→Supabase upload
    try:
        upload_url = create_signed_upload_url(storage_path)
    except Exception as e:
        logger.error(f"Failed to generate signed upload URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate upload URL")

    return {"match_id": match_id, "storage_path": storage_path, "upload_url": upload_url}


@router.post("/{match_id}/confirm-upload")
async def confirm_upload(
    match_id: str,
    req: ConfirmUploadRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Step 3 of the upload flow (after the browser finishes the direct upload).

    Verifies the file landed in storage, then launches the local court_setup_job
    as a background subprocess so this endpoint returns immediately.
    """
    match = _get_match_or_403(match_id, user_id)

    if match.get("s3_temp_key") != req.storage_path:
        raise HTTPException(status_code=400, detail="storage_path mismatch")

    if not file_exists(req.storage_path):
        raise HTTPException(status_code=400, detail="Video not found in storage — upload may have failed")

    supabase.table("matches").update({"status": "court_setup"}).eq("id", match_id).execute()

    # Launch local cv/court_setup_job.py in the background
    _trigger_local_court_setup(match_id, req.storage_path)

    return {"message": "Upload confirmed. Court setup starting.", "match_id": match_id}


@router.post("/{match_id}/court-keypoints")
async def receive_ai_court_keypoints(match_id: str, keypoints: CourtKeypointPayload):
    """
    Internal endpoint called by the local court_setup_job when detection completes.

    Stores AI-suggested keypoints and marks court_setup_status='ready'
    so the frontend court editor can render them for user review.
    """
    kp_data: dict = keypoints.model_dump()
    kp_data["match_id"] = match_id
    supabase.table("court_configs").upsert(kp_data, on_conflict="match_id").execute()
    supabase.table("matches").update({"court_setup_status": "ready"}).eq("id", match_id).execute()
    logger.info(f"AI court keypoints saved for match {match_id}")
    return {"message": "Court keypoints saved", "match_id": match_id}


@router.put("/{match_id}/court-keypoints")
async def confirm_court_keypoints(
    match_id: str,
    keypoints: CourtKeypointPayload,
    user_id: str = Depends(get_user_id),
):
    """
    Step 5 — called when the user clicks "Confirm Court" in the court editor.

    Saves the user-corrected keypoints to court_configs and triggers the full
    local analysis pipeline as a background subprocess.
    """
    _get_match_or_403(match_id, user_id)

    kp_data: dict = keypoints.model_dump()
    kp_data.update({
        "match_id": match_id,
        "ai_suggested": False,
        "confirmed_at": "now()",
        "confirmed_by": user_id,
    })
    supabase.table("court_configs").upsert(kp_data, on_conflict="match_id").execute()
    supabase.table("matches").update({
        "court_setup_status": "confirmed",
        "status": "processing",
    }).eq("id", match_id).execute()

    match = supabase.table("matches").select("s3_temp_key").eq("id", match_id).single().execute().data
    if match and match.get("s3_temp_key"):
        _trigger_local_full_analysis(match_id, match["s3_temp_key"])

    return {"message": "Court confirmed. Full analysis starting.", "match_id": match_id}


@router.get("/{match_id}/frame-url")
async def get_frame_url(match_id: str, user_id: str = Depends(get_user_id)):
    """
    Returns a 30-minute signed Supabase Storage URL for the court setup frame image.
    Used by the court editor UI to display the background frame.

    The frame is a JPEG uploaded by court_setup_job at:
    temp-uploads/{match_id}/frame_1000.jpg
    """
    _get_match_or_403(match_id, user_id)
    frame_path = get_frame_path(match_id)

    if not file_exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame not ready yet")

    try:
        url = create_signed_download_url(frame_path, expiry=1800)
        return {"url": url}
    except Exception as e:
        logger.error(f"Failed to generate frame URL for match {match_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate frame URL")


@router.get("/{match_id}/status")
async def get_processing_status(match_id: str, user_id: str = Depends(get_user_id)):
    """
    Poll endpoint for the frontend to check both overall status and
    the court editor readiness.
    """
    match = _get_match_or_403(match_id, user_id)
    return {
        "status": match.get("status"),
        "court_setup_status": match.get("court_setup_status"),
        "processed_at": match.get("processed_at"),
    }


@router.post("/identify-player")
async def identify_player(identification: PlayerIdentification, user_id: str = Depends(get_user_id)):
    """Store player identification data used by the CV backend to track a specific player."""
    match_response = supabase.table("matches").select("id").eq("id", identification.match_id).eq("user_id", user_id).execute()
    if not match_response.data:
        raise HTTPException(status_code=403, detail="Access denied")

    ident_response = supabase.table("player_identifications").insert({
        "match_id": identification.match_id,
        "frame_data": identification.frame_data,
        "selected_player_coords": identification.selected_player_coords,
    }).execute()
    if not ident_response.data:
        raise HTTPException(status_code=500, detail="Failed to store identification")
    return {"message": "Player identification stored", "identification": ident_response.data[0]}


class DebugVideoReadyPayload(BaseModel):
    storage_path: str


@router.post("/{match_id}/generate-debug-video")
async def generate_debug_video(
    match_id: str,
    user_id: str = Depends(get_user_id),
):
    """
    Step 8 — Triggered by "Generate Debug Video" button in the court editor.

    Requires the court keypoints to already be confirmed (court_setup_status='confirmed').
    Launches cv/debug_video_job.py as a background subprocess which:
      1. Fetches confirmed keypoints from Supabase
      2. Downloads the original video
      3. Renders an annotated debug video (court lines, zones, ball, players)
      4. Uploads the result to Supabase Storage
      5. PATCHes /api/videos/{match_id}/debug-video-ready

    NOTE: Requires the backend to be running locally (models not available on Render).
    For Render deployments this will trigger the job on the Render server, which will
    fail gracefully if models are not available.
    """
    match = _get_match_or_403(match_id, user_id)

    # Validate that keypoints are confirmed
    if match.get("court_setup_status") not in ("confirmed", "ready"):
        raise HTTPException(
            status_code=400,
            detail="Court keypoints must be confirmed before generating a debug video."
        )

    storage_path = match.get("s3_temp_key")
    if not storage_path:
        raise HTTPException(status_code=400, detail="No video found for this match.")

    # Clear any previous debug video status
    supabase.table("matches").update({"debug_video_status": "generating"}).eq("id", match_id).execute()

    _trigger_local_debug_video(match_id, storage_path)

    return {"message": "Debug video generation started.", "match_id": match_id}


@router.patch("/{match_id}/debug-video-ready")
async def debug_video_ready(match_id: str, payload: DebugVideoReadyPayload):
    """
    Step 9 — Internal callback from cv/debug_video_job.py when rendering is complete.

    Updates the match record with the storage path of the finished debug video
    and marks debug_video_status='ready' so the frontend can show a download link.
    """
    supabase.table("matches").update({
        "debug_video_status": "ready",
        "debug_video_path": payload.storage_path,
    }).eq("id", match_id).execute()
    logger.info(f"Debug video ready for match {match_id}: {payload.storage_path}")
    return {"message": "Debug video marked as ready.", "match_id": match_id}


@router.get("/{match_id}/debug-video-url")
async def get_debug_video_url(match_id: str, user_id: str = Depends(get_user_id)):
    """
    Step 10 — Returns a signed download URL for the rendered debug video.

    Poll this endpoint after triggering generation. Returns 404 until the
    job completes and the file is available in storage.
    """
    match = _get_match_or_403(match_id, user_id)

    debug_status = match.get("debug_video_status")
    debug_path = match.get("debug_video_path")

    if debug_status == "generating":
        raise HTTPException(status_code=202, detail="Debug video is still generating.")

    if debug_status != "ready" or not debug_path:
        raise HTTPException(status_code=404, detail="Debug video not available.")

    if not file_exists(debug_path):
        raise HTTPException(status_code=404, detail="Debug video file not found in storage.")

    try:
        url = create_signed_download_url(debug_path, expiry=3600)  # 1-hour link
        return {"url": url, "status": "ready"}
    except Exception as e:
        logger.error(f"Failed to generate debug video URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL.")


# ----- Local Processing Triggers ---------------------------------------------

def _trigger_local_court_setup(match_id: str, storage_path: str) -> None:
    """
    Launch cv/court_setup_job.py as a detached background subprocess.

    The job downloads the video from Supabase Storage, extracts frame 1000,
    runs CourtDetector, uploads the frame JPEG, and POSTs keypoints back to
    POST /api/videos/{match_id}/court-keypoints.

    In production, replace with boto3 Batch submit_job().
    """
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    script = str(PROJECT_ROOT / "cv" / "court_setup_job.py")
    python = sys.executable

    logger.info(f"Launching local court setup job: match={match_id}")
    subprocess.Popen(
        [python, script, "--storage-path", storage_path, "--match-id", match_id, "--backend-url", backend_url],
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _trigger_local_full_analysis(match_id: str, storage_path: str) -> None:
    """
    Launch cv/pipeline.py as a detached background subprocess for full analysis.
    TODO: Implement full analysis job output writing to Supabase.
    """
    logger.info(f"Full analysis triggered for match={match_id} (TODO: implement analysis result storage)")


def _trigger_local_debug_video(match_id: str, storage_path: str, max_seconds: float = 60.0) -> None:
    """
    Launch cv/debug_video_job.py as a detached background subprocess.

    The job fetches keypoints from Supabase, downloads the video, renders an
    annotated debug video (court lines, zones, ball, players), uploads the
    result to Supabase Storage, and PATCHes /debug-video-ready when done.

    max_seconds: Only render this many seconds of the video (default 60s for
                 fast verification — avoids processing hour-long match videos).
    """
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    script = str(PROJECT_ROOT / "cv" / "debug_video_job.py")
    python = sys.executable

    logger.info(f"Launching local debug video job: match={match_id} (max {max_seconds}s)")
    subprocess.Popen(
        [
            python, script,
            "--match-id", match_id,
            "--storage-path", storage_path,
            "--backend-url", backend_url,
            "--max-seconds", str(max_seconds),
        ],
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

