"""
cv/analysis_job.py — Full CV analysis pipeline background job.

Triggered by:
    PUT /api/videos/{match_id}/court-keypoints  (after user confirms keypoints)

Flow:
    1. Fetch match record (storage path, poi_start_side) from Supabase
    2. Fetch confirmed court keypoints from court_configs
    3. Download original video from Supabase Storage
    4. Run cv/pipeline.py → AnalyticsPipeline.process()
    5. Save results to Supabase:
       a. matches → stats (JSONB), analysis_shots (JSONB), status='completed', analyzed_at
       b. shots   → bulk insert per-shot rows
       c. match_data → upsert full JSON blob for debugging
    6. On any error → matches.status='failed', matches.analysis_error=<message>

Usage (manual test):
    python cv/analysis_job.py --match-id <uuid>

Environment variables required (inherited from subprocess.Popen):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analysis_job")


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _sb():
    """Return an authenticated Supabase client."""
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


def fetch_match(match_id: str) -> dict:
    """Fetch the match row (raises on missing)."""
    sb = _sb()
    resp = sb.table("matches").select("*").eq("id", match_id).single().execute()
    if not resp.data:
        raise RuntimeError(f"Match {match_id} not found")
    return resp.data


# court_configs stores kp0..kp13 in the FRONTEND court editor's order (see the
# frontend upload-modal DEFAULT_KEYPOINTS: 0-3 bottom baseline, 4-6 bottom
# service line, 7-9 top service line, 10-13 top baseline). But
# pipeline._build_homography maps keypoint[i] -> CourtReference.key_points[i],
# which uses a DIFFERENT order. Feeding the raw frontend order pairs mismatched
# points and yields a geometrically wrong homography (verified: ~2100px mean
# reprojection error vs ~11px when correctly ordered). This lookup reorders
# frontend indices into the reference order the pipeline expects.
#
# _REORDER_TO_REFERENCE[r] = the frontend kp index whose physical intersection
# belongs at reference index r.
#
# NOTE: derived from the frontend editor's keypoint semantics. Confirm against a
# real frontend-placed court_config (place points in the actual court editor,
# then check the homography is valid) before fully trusting production stats.
_REORDER_TO_REFERENCE = [10, 13, 0, 3, 11, 1, 12, 2, 7, 9, 4, 6, 8, 5]


def reorder_frontend_to_reference(
    frontend_kps: List[Optional[Tuple[float, float]]],
) -> List[Optional[Tuple[float, float]]]:
    """Reorder court_configs kp0..13 (frontend editor order) into the
    CourtReference.key_points order pipeline._build_homography maps by index."""
    return [frontend_kps[_REORDER_TO_REFERENCE[r]] for r in range(14)]


def fetch_keypoints(match_id: str) -> Optional[List[Optional[Tuple[float, float]]]]:
    """Load confirmed court keypoints from court_configs (14 points), reordered
    from the frontend editor's order into the pipeline's reference order.

    Returns None when no court_config exists — the pipeline then auto-detects the
    court from the best-fitting frame. Manual keypoints are an override, not a
    requirement (auto-detect + manual fallback)."""
    sb = _sb()
    resp = sb.table("court_configs").select("*").eq("match_id", match_id).maybe_single().execute()
    if not resp or not resp.data:
        logger.info(f"No court_config for match {match_id} — pipeline will auto-detect the court")
        return None
    data = resp.data
    frontend_kps = []
    for i in range(14):
        x = data.get(f"kp{i}_x")
        y = data.get(f"kp{i}_y")
        frontend_kps.append(
            (float(x), float(y)) if (x is not None and y is not None) else None
        )
    kps = reorder_frontend_to_reference(frontend_kps)
    valid = sum(1 for k in kps if k is not None)
    logger.info(f"Court keypoints: {valid}/14 valid (reordered frontend→reference)")
    return kps


def download_video(storage_path: str, local_path: str) -> None:
    """Download the match video from Supabase Storage to a local file."""
    from api.storage import download_file
    logger.info(f"Downloading video from {storage_path} ...")
    download_file(storage_path, local_path)
    size_mb = os.path.getsize(local_path) / 1_048_576
    logger.info(f"Download complete — {size_mb:.1f} MB")


# Recognized top-level box types for ISO base-media (MP4) / QuickTime (MOV) files.
_VIDEO_CONTAINER_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot"}


def assert_is_video(local_path: str) -> None:
    """Reject anything that is not an MP4/MOV container before it reaches the pipeline.

    The upload path only ever produces ISO base-media video (a browser MP4 or an
    ffmpeg-muxed PlaySight stream), so a header mismatch means a bogus or hostile
    upload — fail fast instead of feeding arbitrary bytes to OpenCV/torch. This is
    a defence-in-depth check on top of the Storage bucket's size/MIME limits.
    """
    with open(local_path, "rb") as fh:
        header = fh.read(12)
    if len(header) < 12 or header[4:8] not in _VIDEO_CONTAINER_BOXES:
        raise ValueError(f"Uploaded file is not a recognized MP4/MOV video (header {header!r})")


def mark_processing(match_id: str) -> None:
    _sb().table("matches").update({
        "status": "processing",
        "analysis_error": None,
    }).eq("id", match_id).execute()


def save_results(match_id: str, result, points) -> None:
    """
    Persist AnalysisResult to Supabase:
      - matches row  → stats + analysis_shots + status + analyzed_at
      - points table → one row per detected point (coach classifies outcomes manually)
      - shots table  → bulk insert (result='in_play' — outcome set by coach via review UI)
      - match_data   → full JSON blob (fallback / debug)
    """
    sb = _sb()
    now_iso = datetime.now(timezone.utc).isoformat()

    stats_dict = result.match_stats or {}
    shots_list = result.shots or []

    # 1. Update matches
    logger.info(f"Saving stats to matches table ({len(shots_list)} shots)...")
    sb.table("matches").update({
        "status": "completed",
        "analyzed_at": now_iso,
        "stats": stats_dict,
        "analysis_shots": shots_list,
        "analysis_error": None,
    }).eq("id", match_id).execute()

    # 2. Bulk-insert points (boundaries only; manual_outcome left null for coach review)
    if points:
        logger.info(f"Inserting {len(points)} point rows into points table...")
        # Delete any previous points for this match (re-analysis)
        sb.table("points").delete().eq("match_id", match_id).execute()
        point_rows = [
            {
                "match_id": match_id,
                "point_idx": pt.point_idx,
                "start_frame": pt.start_frame,
                "end_frame": pt.end_frame,
                "start_timestamp_s": round(pt.start_frame / max(result.fps, 1), 3) if pt.start_frame is not None else None,
                "end_timestamp_s": round(pt.end_frame / max(result.fps, 1), 3) if pt.end_frame is not None else None,
                "serve_player": pt.serve_player,
                "rally_length": pt.rally_length,
            }
            for pt in points
        ]
        BATCH = 500
        for i in range(0, len(point_rows), BATCH):
            sb.table("points").insert(point_rows[i : i + BATCH]).execute()

    # 3. Bulk-insert shots
    #    result is always 'in_play' — coaches classify outcomes via the review UI.
    if shots_list:
        logger.info(f"Inserting {len(shots_list)} shot rows into shots table...")
        # Delete previous shots for this match
        sb.table("shots").delete().eq("match_id", match_id).execute()
        shot_rows = [
            {
                "match_id": match_id,
                "frame": s.get("frame"),
                "point_idx": s.get("point_idx"),
                "start_pos": {"x": s.get("x"), "y": s.get("y")} if s.get("x") is not None else None,
                "end_pos": None,
                "shot_type": s.get("shot_type"),
                "result": "in_play",
                "speed_kmh": s.get("speed_kmh"),
                "player": s.get("player"),
                "is_winner": False,
                "is_error": False,
                "video_timestamp": s.get("frame", 0) / max(result.fps, 1),
            }
            for s in shots_list
        ]
        BATCH = 500
        for i in range(0, len(shot_rows), BATCH):
            sb.table("shots").insert(shot_rows[i : i + BATCH]).execute()

    # 4. Upsert match_data (full JSON for debugging)
    logger.info("Upserting match_data blob...")
    try:
        from dataclasses import asdict
        full_dict = asdict(result)
        full_dict.pop("frames", None)
        sb.table("match_data").upsert({
            "match_id": match_id,
            "json_data": full_dict,
            "stats_summary": stats_dict,
        }).execute()
    except Exception as e:
        logger.warning(f"match_data upsert failed (non-fatal): {e}")


def mark_failed(match_id: str, error_msg: str) -> None:
    try:
        _sb().table("matches").update({
            "status": "failed",
            "analysis_error": str(error_msg)[:2000],
        }).eq("id", match_id).execute()
    except Exception as e:
        logger.error(f"Could not even mark match as failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full CV analysis pipeline job")
    parser.add_argument("--match-id", default=os.getenv("MATCH_ID"), help="Supabase match UUID")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (2-3 recommended for speed)")
    parser.add_argument("--max-frames", type=int, default=None, help="Hard cap on frames (for testing)")
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    if not args.match_id:
        parser.error("--match-id is required (or set MATCH_ID env var)")

    match_id = args.match_id
    logger.info(f"╔══ Analysis job started for match {match_id} ══╗")

    try:
        # 0. Mark match as processing
        mark_processing(match_id)

        # 1. Fetch match metadata
        match = fetch_match(match_id)
        storage_path = match.get("s3_temp_key")
        if not storage_path:
            raise RuntimeError("Match has no video (s3_temp_key is null)")

        poi_start_side = match.get("poi_start_side") or "near"
        logger.info(f"poi_start_side = {poi_start_side}")

        # 2. Fetch court keypoints
        keypoints = fetch_keypoints(match_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = str(Path(tmpdir) / "video.mp4")

            # 3. Download video, then verify it is actually an MP4/MOV container
            download_video(storage_path, video_path)
            assert_is_video(video_path)

            # 4. Run full pipeline
            logger.info("Starting AnalyticsPipeline...")
            from cv.pipeline import AnalyticsPipeline
            pipeline = AnalyticsPipeline(
                device=args.device,
                poi_start_side=poi_start_side,
            )
            result = pipeline.process(
                video_path=video_path,
                court_keypoints=keypoints,
                match_id=match_id,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
            )

            logger.info(
                f"Pipeline complete — "
                f"{len(result.frames)} frames, "
                f"{len(result.shots)} shots detected"
            )

            # Collect the PointRecord list from the segmenter for storage.
            # The pipeline stores them in result._points if available; fall back to [].
            points = getattr(result, "_points", [])

            # 5. Save results
            save_results(match_id, result, points)

    except Exception as e:
        logger.exception(f"Analysis job FAILED: {e}")
        mark_failed(match_id, str(e))
        sys.exit(1)

    logger.info(f"╚══ Analysis job complete for match {match_id} ══╝")


if __name__ == "__main__":
    main()
