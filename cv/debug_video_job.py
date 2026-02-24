"""
cv/debug_video_job.py — Local subprocess job for generating a debug video.

Triggered by:
    POST /api/videos/{match_id}/generate-debug-video

Flow:
    1. Fetch confirmed keypoints from Supabase court_configs
    2. Download original video from Supabase Storage
    3. Run cv/analysis/visualizer.render_debug_video()
    4. Upload the result video to Supabase Storage
    5. PATCH /api/videos/{match_id}/debug-video-ready with the storage path

The generated video is stored at:
    debug-videos/{match_id}/debug.mp4

Usage (manual):
    python cv/debug_video_job.py \\
        --match-id {uuid} \\
        --storage-path "temp-uploads/{id}/video.mp4" \\
        --backend-url http://localhost:8000 \\
        --max-seconds 60

Environment variables required (inherited from backend subprocess.Popen):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import argparse
import logging
import os
import sys
import tempfile
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug_video_job")

DEBUG_VIDEO_BUCKET = "match-videos"


def fetch_keypoints(match_id: str) -> list[tuple[float | None, float | None]]:
    """
    Load the confirmed court keypoints from Supabase court_configs.
    Returns a list of 14 (x, y) tuples (values may be None if unset).
    """
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    resp = sb.table("court_configs").select("*").eq("match_id", match_id).single().execute()
    if not resp.data:
        raise RuntimeError(f"No court_configs found for match {match_id}")
    data = resp.data
    keypoints = []
    for i in range(14):
        x = data.get(f"kp{i}_x")
        y = data.get(f"kp{i}_y")
        keypoints.append((float(x) if x is not None else None,
                          float(y) if y is not None else None))
    return keypoints


def upload_debug_video(match_id: str, local_path: str) -> str:
    """
    Upload the rendered debug video to Supabase Storage.
    Returns the storage path string.
    """
    from api.storage import upload_file
    storage_path = f"debug-videos/{match_id}/debug.mp4"
    with open(local_path, "rb") as f:
        video_bytes = f.read()
    upload_file(storage_path, video_bytes, content_type="video/mp4")
    logger.info(f"Uploaded debug video to {storage_path}")
    return storage_path


def notify_backend(backend_url: str, match_id: str, storage_path: str) -> None:
    """Notify the backend that the debug video is ready."""
    url = f"{backend_url.rstrip('/')}/api/videos/{match_id}/debug-video-ready"
    resp = requests.patch(url, json={"storage_path": storage_path}, timeout=30)
    resp.raise_for_status()
    logger.info(f"Backend notified: {resp.json()}")


def main():
    parser = argparse.ArgumentParser(description="Generate debug annotation video (local job)")
    parser.add_argument("--match-id", default=os.getenv("MATCH_ID"), required=False)
    parser.add_argument("--storage-path", default=os.getenv("STORAGE_PATH"), required=False,
                        help="Supabase Storage path of the original video")
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://localhost:8000"))
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="Max seconds of video to render (default 60s for quick verification)")
    args = parser.parse_args()

    if not args.match_id or not args.storage_path:
        parser.error("--match-id and --storage-path are required")

    logger.info(f"Debug video job started for match={args.match_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "video.mp4")
        output_path = str(Path(tmpdir) / "debug.mp4")

        # 1. Fetch keypoints
        logger.info("Fetching confirmed keypoints from Supabase...")
        keypoints = fetch_keypoints(args.match_id)
        valid = sum(1 for kp in keypoints if kp[0] is not None)
        logger.info(f"  {valid}/14 keypoints available")

        # 2. Download original video
        logger.info("Downloading video from Supabase Storage...")
        from api.storage import download_file
        download_file(args.storage_path, video_path)
        logger.info(f"  Downloaded to {video_path}")

        # 3. Render debug video
        logger.info("Rendering debug video...")
        from cv.analysis.visualizer import render_debug_video
        render_debug_video(
            input_path=video_path,
            output_path=output_path,
            keypoints=keypoints,
            max_seconds=args.max_seconds,
            enable_ball_tracking=True,
            enable_player_tracking=True,
        )

        # 4. Upload to Supabase Storage
        storage_path = upload_debug_video(args.match_id, output_path)

    # 5. Notify backend (tmpdir fully cleaned up by this point)
    notify_backend(args.backend_url, args.match_id, storage_path)
    logger.info("Debug video job complete.")


if __name__ == "__main__":
    main()
