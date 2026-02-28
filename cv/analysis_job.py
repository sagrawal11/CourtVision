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


def fetch_keypoints(match_id: str) -> List[Optional[Tuple[float, float]]]:
    """Load confirmed court keypoints from court_configs (14 points)."""
    sb = _sb()
    resp = sb.table("court_configs").select("*").eq("match_id", match_id).single().execute()
    if not resp.data:
        raise RuntimeError(f"No court_configs for match {match_id} — were keypoints confirmed?")
    data = resp.data
    kps = []
    for i in range(14):
        x = data.get(f"kp{i}_x")
        y = data.get(f"kp{i}_y")
        kps.append(
            (float(x), float(y)) if (x is not None and y is not None) else None
        )
    valid = sum(1 for k in kps if k is not None)
    logger.info(f"Court keypoints: {valid}/14 valid")
    return kps


def download_video(storage_path: str, local_path: str) -> None:
    """Download the match video from Supabase Storage to a local file."""
    from api.storage import download_file
    logger.info(f"Downloading video from {storage_path} ...")
    download_file(storage_path, local_path)
    size_mb = os.path.getsize(local_path) / 1_048_576
    logger.info(f"Download complete — {size_mb:.1f} MB")


def mark_processing(match_id: str) -> None:
    _sb().table("matches").update({
        "status": "processing",
        "analysis_error": None,
    }).eq("id", match_id).execute()


def save_results(match_id: str, result) -> None:
    """
    Persist AnalysisResult to Supabase:
      - matches row  → stats + analysis_shots + status + analyzed_at
      - shots table  → bulk insert
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

    # 2. Bulk-insert shots
    if shots_list:
        logger.info(f"Inserting {len(shots_list)} shot rows into shots table...")
        shot_rows = [
            {
                "match_id": match_id,
                "frame": s.get("frame"),
                "start_pos": {"x": s.get("x"), "y": s.get("y")},
                "end_pos": None,
                "shot_type": s.get("shot_type"),
                "result": "winner" if s.get("is_winner") else ("error" if s.get("is_error") else "in_play"),
                "speed_kmh": s.get("speed_kmh"),
                "player": s.get("player"),
                "is_winner": bool(s.get("is_winner")),
                "is_error": bool(s.get("is_error")),
                "video_timestamp": s.get("frame", 0) / max(result.fps, 1),
            }
            for s in shots_list
        ]
        # Insert in batches of 500 to avoid payload size limits
        BATCH = 500
        for i in range(0, len(shot_rows), BATCH):
            sb.table("shots").insert(shot_rows[i : i + BATCH]).execute()

    # 3. Upsert match_data (full JSON for debugging)
    logger.info("Upserting match_data blob...")
    try:
        from dataclasses import asdict
        full_dict = asdict(result)
        # Strip the large per-frame array — keep only summary data
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

            # 3. Download video
            download_video(storage_path, video_path)

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

            # 5. Save results
            save_results(match_id, result)

    except Exception as e:
        logger.exception(f"Analysis job FAILED: {e}")
        mark_failed(match_id, str(e))
        sys.exit(1)

    logger.info(f"╚══ Analysis job complete for match {match_id} ══╝")


if __name__ == "__main__":
    main()
