"""
Video Upload & Processing API.

Manages the full lifecycle of a match video from upload to analysis.
Free-tier stack: Supabase Storage for video files, local subprocess for CV processing.

Upload Flow:
  1. POST /api/videos/prepare-upload              — Create match record, return storage path + signed upload URL
  2. (Browser uploads directly to Supabase Storage using signed URL)
  3. POST /api/videos/{id}/confirm-upload          — Verify upload landed, save court keypoints + POI side, mark as processing
  4. POST /api/videos/{id}/generate-player-selection — Launch player_selection_job (extract frames + YOLO) for player picking
  5. PUT  /api/videos/{id}/court-keypoints         — User confirms keypoints, triggers full analysis
  6. GET  /api/videos/{id}/status                  — Poll overall status + court_setup_status
  7. POST /api/videos/{id}/generate-debug-video    — Trigger local debug video rendering job
  8. PATCH /api/videos/{id}/debug-video-ready      — (Internal) Job notifies backend when done
  9. GET  /api/videos/{id}/debug-video-url         — Get signed download URL for debug video

See docs/video-pipeline.md for the full sequence diagram.
"""

import json
import os
import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel
from supabase import create_client, Client

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from auth import get_user_id
from rate_limit import limiter
from api.storage import (
    file_exists,
    get_upload_path,
    create_signed_upload_url,
    create_signed_download_url,
    upload_file_from_path,
)
from services.playsight import (
    download_playsight_video,
    is_playsight_url,
    PlaySightImportError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/videos", tags=["videos"])

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
AWS_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")

# Shared secret for server→server internal callbacks (e.g. CV job notifications).
# Must be set in production; if unset, the callback is allowed but a warning is
# logged so local dev keeps working.
INTERNAL_JOB_SECRET = os.getenv("INTERNAL_JOB_SECRET")

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
    storage_path: str
    keypoints: dict[str, float | bool | None] | None = None   # Echo back so we can verify it matches what was issued
    poi_start_side: str = "near"


class ImportPlaysightRequest(BaseModel):
    """Coach pastes a PlaySight share link; we scrape, download, and stash it.

    The flow mirrors ``prepare-upload`` + the direct browser PUT, but is
    server-side so the coach never has to download/re-upload the .mp4.
    """

    playsight_url: str
    match_date: Optional[str] = None
    opponent: Optional[str] = None
    player_name: Optional[str] = None
    notes: Optional[str] = None
    player_user_id: Optional[str] = None  # For coaches importing on behalf of a player
    filename: Optional[str] = None         # Optional override; default = "playsight_<match_id>.mp4"


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
    logger.info(f"prepare-upload: creating match record for user={user_id}, file={req.filename}")
    match_resp = supabase.table("matches").insert(insert_data).execute()
    if not match_resp.data:
        raise HTTPException(status_code=500, detail="Failed to create match record")

    match_id = match_resp.data[0]["id"]
    storage_path = get_upload_path(match_id, req.filename)
    logger.info(f"prepare-upload: match_id={match_id}, storage_path={storage_path}")

    # Save storage path to match record
    supabase.table("matches").update({"s3_temp_key": storage_path}).eq("id", match_id).execute()

    # Generate signed upload URL for direct browser→Supabase upload
    try:
        upload_url = create_signed_upload_url(storage_path)
        logger.info(f"prepare-upload: signed URL generated OK (match_id={match_id})")
    except Exception as e:
        logger.error(f"prepare-upload: failed to generate signed URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate upload URL")

    return {"match_id": match_id, "storage_path": storage_path, "upload_url": upload_url}


@router.post("/import-playsight")
@limiter.limit("5/hour")
async def import_playsight(
    request: Request,
    req: ImportPlaysightRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Server-side ingest of a PlaySight share link.

    Replaces step 1+2 of the normal upload flow when the coach has a
    PlaySight share URL instead of a local .mp4. We:

      1. Validate the link looks like PlaySight.
      2. Create the match record (same shape as ``/prepare-upload``).
      3. Scrape the PlaySight page for its embedded HLS playlist URL.
      4. Use yt-dlp + ffmpeg to download and mux it into a temp .mp4.
      5. Push the .mp4 into Supabase Storage at the canonical
         ``temp-uploads/{match_id}/{filename}`` location.
      6. Return the same ``{match_id, storage_path}`` that the upload modal
         would otherwise get — the client can then jump straight into the
         player selection / court keypoint steps.

    Returns:
        {
            "match_id": str,
            "storage_path": str,
            "stream_url": str,       # debugging
            "size_bytes": int,
        }
    """
    if not is_playsight_url(req.playsight_url):
        raise HTTPException(
            status_code=400,
            detail="That URL doesn't look like a PlaySight share link.",
        )

    # Coach uploading for a player
    match_user_id = user_id
    if req.player_user_id:
        user_resp = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_resp.data or user_resp.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Only coaches can upload for other players")
        match_user_id = req.player_user_id

    # 1. Create the match row first so we have an ID for the storage path
    placeholder_filename = req.filename or "playsight_match.mp4"
    insert_data = {
        "user_id": match_user_id,
        "video_filename": placeholder_filename,
        "status": "pending",
        "court_setup_status": "pending",
        **({"player_name": req.player_name} if req.player_name else {}),
        **({"match_date": req.match_date} if req.match_date else {}),
        **({"opponent": req.opponent} if req.opponent else {}),
        **({"notes": req.notes} if req.notes else {}),
    }
    logger.info(f"import-playsight: creating match record for user={user_id}, url={req.playsight_url[:80]}")
    match_resp = supabase.table("matches").insert(insert_data).execute()
    if not match_resp.data:
        raise HTTPException(status_code=500, detail="Failed to create match record")

    match_id = match_resp.data[0]["id"]
    final_filename = req.filename or f"playsight_{match_id}.mp4"
    storage_path = get_upload_path(match_id, final_filename)
    supabase.table("matches").update({
        "s3_temp_key": storage_path,
        "video_filename": final_filename,
        "status": "importing",
    }).eq("id", match_id).execute()
    logger.info(f"import-playsight: match_id={match_id}, storage_path={storage_path}")

    # 2. Download into a temp file, then push to storage, then clean up.
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="playsight_")
    local_path = os.path.join(tmpdir, final_filename)

    try:
        result = download_playsight_video(req.playsight_url, local_path)
        logger.info(
            f"import-playsight: downloaded {result.size_bytes / 1_048_576:.1f} MB "
            f"from {result.stream_url[:80]}"
        )

        upload_file_from_path(storage_path, result.local_path, content_type="video/mp4")
        logger.info(f"import-playsight: uploaded to storage:{storage_path}")

        supabase.table("matches").update({
            "status": "pending",  # back to the normal "ready for court setup" state
        }).eq("id", match_id).execute()

        return {
            "match_id": match_id,
            "storage_path": storage_path,
            "stream_url": result.stream_url,
            "size_bytes": result.size_bytes,
        }

    except PlaySightImportError as e:
        logger.error(f"import-playsight: PlaySight import failed: {e}")  # full detail server-side only
        supabase.table("matches").update({
            "status": "failed",
            "analysis_error": f"PlaySight import failed: {e}",
        }).eq("id", match_id).execute()
        raise HTTPException(status_code=502, detail="PlaySight import failed. The video may be private or unavailable.")
    except Exception as e:
        logger.exception(f"import-playsight: unexpected error: {e}")
        supabase.table("matches").update({
            "status": "failed",
            "analysis_error": f"Unexpected error during PlaySight import: {e}",
        }).eq("id", match_id).execute()
        raise HTTPException(status_code=500, detail="Unexpected error during PlaySight import")
    finally:
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rmdir(tmpdir)
        except OSError:
            pass


@router.post("/{match_id}/confirm-upload")
async def confirm_upload(
    match_id: str,
    req: ConfirmUploadRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Step 3 of the upload flow (after the browser finishes the direct upload).

    Verifies the file landed in storage, saves the court keypoints the coach
    confirmed in the editor along with the POI start side, and marks the match
    as ready for CV processing.
    """
    logger.info(f"confirm-upload: match_id={match_id}, storage_path={req.storage_path}")

    match = _get_match_or_403(match_id, user_id)

    if match.get("s3_temp_key") != req.storage_path:
        raise HTTPException(status_code=400, detail="storage_path mismatch")

    if not file_exists(req.storage_path):
        logger.error(f"confirm-upload: file not found in storage: {req.storage_path}")
        raise HTTPException(status_code=400, detail="Video not found in storage — upload may have failed")

    logger.info(f"confirm-upload: file verified in storage, saving keypoints and marking as processing")
    
    # 1. Save the manually-confirmed keypoints from the frontend
    if req.keypoints:
        supabase.table("court_configs").upsert({
            "match_id": match_id,
            **req.keypoints
        }).execute()
        logger.info(f"confirm-upload: saved {len(req.keypoints)} keypoints for match {match_id}")
    
    # 2. Mark match as ready for CV processing and save POI side
    supabase.table("matches").update({
        "status": "processing",
        "court_setup_status": "confirmed",
        "poi_start_side": req.poi_start_side
    }).eq("id", match_id).execute()

    return {"message": "Upload confirmed, court setup saved, processing started", "match_id": match_id}

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
        "analyzed_at": match.get("analyzed_at"),
        "analysis_error": match.get("analysis_error"),
    }



@router.put("/{match_id}/court-keypoints")
async def confirm_court_keypoints(
    match_id: str,
    payload: dict,
    user_id: str = Depends(get_user_id),
):
    """
    Step 5 of the upload flow — user confirms court keypoints from the court editor.

    Saves the 14 keypoints to court_configs, marks the match as confirmed,
    and launches the full analysis pipeline as a background subprocess.

    Body: flat dict of kp0_x, kp0_y, ..., kp13_x, kp13_y + ai_suggested (bool).
    """
    match = _get_match_or_403(match_id, user_id)

    storage_path = match.get("s3_temp_key")
    if not storage_path:
        raise HTTPException(status_code=400, detail="No video found for this match.")

    poi_start_side = match.get("poi_start_side") or "near"

    # Save keypoints to court_configs
    supabase.table("court_configs").upsert({
        "match_id": match_id,
        **payload,
        "ai_suggested": False,
        "confirmed_at": "now()",
        "confirmed_by": user_id,
    }).execute()
    logger.info(f"court-keypoints: saved keypoints for match {match_id}")

    # Update match status
    supabase.table("matches").update({
        "court_setup_status": "confirmed",
        "status": "processing",
    }).eq("id", match_id).execute()

    # Dispatch the full analysis pipeline — SQS in production, subprocess in dev
    _dispatch_analysis(
        match_id=match_id,
        storage_path=storage_path,
        poi_start_side=poi_start_side,
    )

    return {"message": "Keypoints confirmed, full analysis started.", "match_id": match_id}


@router.post("/{match_id}/generate-player-selection")
async def generate_player_selection(match_id: str, user_id: str = Depends(get_user_id)):
    """
    Step 2a of the upload flow: triggers the background job to extract 5 frames
    and run YOLO player detection, returning clickable bounding boxes.
    """
    match = _get_match_or_403(match_id, user_id)
    
    # Update status to let frontend know generation is in progress
    supabase.table("matches").update({
        "status": "generating_frames"
    }).eq("id", match_id).execute()
    
    # Launch job in background
    cmd = [sys.executable, str(PROJECT_ROOT / "cv" / "player_selection_job.py"), "--match-id", match_id]
    subprocess.Popen(cmd)
    
    return {"message": "Player selection generation started"}


@router.get("/{match_id}/player-selection-data")
async def get_player_selection_data(match_id: str, user_id: str = Depends(get_user_id)):
    """
    Step 2b of the upload flow: fetches the JSON manifest containing the 
    5 base64-encoded annotated frames from Supabase Storage.
    """
    match = _get_match_or_403(match_id, user_id)
    
    storage_path = f"player_selection_frames/{match_id}.json"
    
    try:
        # Download the JSON manifest directly
        res = supabase.storage.from_("match-videos").download(storage_path)
        import json
        manifest = json.loads(res.decode("utf-8"))
        return manifest
    except Exception as e:
        logger.error(f"Failed to fetch player selection data: {e}")
        raise HTTPException(status_code=404, detail="Player selection data not ready or failed")


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
async def debug_video_ready(
    match_id: str,
    payload: DebugVideoReadyPayload,
    x_internal_secret: str | None = Header(default=None),
):
    """
    Step 9 — Internal callback from cv/debug_video_job.py when rendering is complete.

    Server→server only. Authenticated with a shared INTERNAL_JOB_SECRET header
    rather than a user JWT, since the caller is a background job, not a browser.

    Updates the match record with the storage path of the finished debug video
    and marks debug_video_status='ready' so the frontend can show a download link.
    """
    if INTERNAL_JOB_SECRET:
        if x_internal_secret != INTERNAL_JOB_SECRET:
            raise HTTPException(status_code=401, detail="Invalid internal secret")
    else:
        logger.warning(
            "debug-video-ready called without INTERNAL_JOB_SECRET configured — "
            "endpoint is unauthenticated. Set INTERNAL_JOB_SECRET in production."
        )

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


def _open_log(name: str, match_id: str):  # type: ignore[return]
    """Open a log file for a subprocess job. Returns the file object."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{match_id[:8]}_{ts}.log"
    logger.info(f"Subprocess log → {log_path}")
    return open(log_path, "w", buffering=1)  # line-buffered





def _dispatch_analysis(
    match_id: str,
    storage_path: str,
    poi_start_side: str = "near",
    frame_skip: int = 1,
) -> None:
    """Dispatch an analysis job — SQS in production, local subprocess in dev."""
    if SQS_QUEUE_URL:
        import boto3
        sqs = boto3.client("sqs", region_name=AWS_REGION)
        sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps({"match_id": match_id, "frame_skip": frame_skip}),
        )
        logger.info(f"Analysis job queued to SQS: match={match_id}")
    else:
        _trigger_local_full_analysis(match_id, storage_path, poi_start_side, frame_skip)


def _trigger_local_full_analysis(
    match_id: str,
    storage_path: str,
    poi_start_side: str = "near",
    frame_skip: int = 1,
) -> None:
    """
    Launch cv/analysis_job.py as a detached background subprocess.
    The job downloads the video, runs the full AnalyticsPipeline, and
    saves stats + shots to Supabase, updating match.status to 'completed'.
    Logs stdout+stderr to logs/analysis_{match_id[:8]}_{ts}.log.
    """
    script = str(PROJECT_ROOT / "cv" / "analysis_job.py")
    python = sys.executable
    log_fh = _open_log("analysis", match_id)

    logger.info(f"Launching analysis_job: match={match_id}, poi_side={poi_start_side}")
    subprocess.Popen(
        [
            python, script,
            "--match-id", match_id,
            "--frame-skip", str(frame_skip),
        ],
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )


def _trigger_local_debug_video(match_id: str, storage_path: str, max_seconds: float = 60.0) -> None:
    """
    Launch cv/debug_video_job.py as a detached background subprocess.
    Logs stdout+stderr to logs/debug_video_{match_id[:8]}_{ts}.log.
    """
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    script = str(PROJECT_ROOT / "cv" / "debug_video_job.py")
    python = sys.executable
    log_fh = _open_log("debug_video", match_id)

    logger.info(f"Launching debug_video_job: match={match_id} (max {max_seconds}s)")
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
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )

