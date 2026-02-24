"""
Court Setup Job — runs locally as a background subprocess.

Triggered by the backend immediately after a video upload is confirmed.
Extracts frame ~1000 from the video, uploads it to Supabase Storage for
display in the court editor, runs the CourtDetector model, and POSTs the
AI-suggested keypoints back to the backend.

Usage (manual testing):
    python cv/court_setup_job.py \\
        --storage-path "temp-uploads/{match_id}/video.mov" \\
        --match-id "{match_id}" \\
        --backend-url "http://localhost:8000"

Environment variables required (inherited from backend via subprocess.Popen):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    BACKEND_URL   (optional, defaults to http://localhost:8000)
"""

import argparse
import logging
import os
import sys
import tempfile
import requests
from pathlib import Path

# Allow importing from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("court_setup_job")

# Frame to extract — ~1000 ensures the court is clearly visible and
# players have moved away from the baseline corners
SETUP_FRAME_NUMBER = 1000


def extract_frame(video_path: str, frame_number: int):
    """
    Extract a single frame from a video file.
    Falls back to the last available frame if the video is shorter than frame_number.

    Returns:
        (frame_bgr, total_frames) — BGR numpy array and total frame count
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(frame_number, max(0, total_frames - 1))
    logger.info(f"Video has {total_frames} frames — extracting frame {target}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {target} from {video_path}")

    return frame, total_frames


def run_court_detection(frame):
    """
    Run the CourtDetector model on a single frame.

    Returns:
        List of 14 (x, y) tuples — None values indicate undetected keypoints.
    """
    from cv.detection.court_detector import CourtDetector
    detector = CourtDetector()
    keypoints = detector.detect_court_in_frame(frame, apply_homography=True)
    detected = sum(1 for k in keypoints if k[0] is not None)
    logger.info(f"Detected {detected} / 14 keypoints")
    return keypoints


def upload_frame_to_storage(match_id: str, frame) -> None:
    """
    Encode the extracted frame as JPEG and upload it to Supabase Storage
    so the court editor UI can display it as the draggable background.

    Stored at: temp-uploads/{match_id}/frame_1000.jpg
    """
    from api.storage import upload_file, get_frame_path

    _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_path = get_frame_path(match_id)
    upload_file(frame_path, jpeg_buf.tobytes(), content_type="image/jpeg")
    logger.info(f"Uploaded frame to storage: {frame_path}")


def keypoints_to_payload(keypoints) -> dict:
    """
    Convert the flat list of 14 (x, y) tuples to the API payload format.
    None values are sent as null (stored as NULL in Postgres).
    """
    payload: dict = {"ai_suggested": True}
    for i, (x, y) in enumerate(keypoints):
        payload[f"kp{i}_x"] = float(x) if x is not None else None
        payload[f"kp{i}_y"] = float(y) if y is not None else None
    return payload


def post_keypoints(backend_url: str, match_id: str, payload: dict) -> None:
    """POST AI-detected keypoints to the backend API."""
    url = f"{backend_url.rstrip('/')}/api/videos/{match_id}/court-keypoints"
    logger.info(f"Posting keypoints to {url}")
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    logger.info(f"Keypoints accepted: {response.json()}")


def update_frame_count(match_id: str, frame_count: int) -> None:
    """Store the total frame count in the match record for informational use."""
    try:
        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        sb.table("matches").update({"frame_count": frame_count}).eq("id", match_id).execute()
    except Exception as e:
        logger.warning(f"Could not update frame_count: {e}")


def main():
    parser = argparse.ArgumentParser(description="Court Setup Job (local)")
    parser.add_argument("--storage-path", default=os.getenv("STORAGE_PATH"), required=False)
    parser.add_argument("--match-id", default=os.getenv("MATCH_ID"), required=False)
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://localhost:8000"))
    args = parser.parse_args()

    if not args.storage_path or not args.match_id:
        parser.error("--storage-path and --match-id are required (or set STORAGE_PATH / MATCH_ID env vars)")

    logger.info(f"Starting court setup for match={args.match_id}, path={args.storage_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "video.mp4")

        # 1. Download video from Supabase Storage
        logger.info("Downloading video from Supabase Storage...")
        from api.storage import download_file
        download_file(args.storage_path, video_path)

        # 2. Extract the setup frame
        frame, total_frames = extract_frame(video_path, SETUP_FRAME_NUMBER)

        # 3. Upload frame JPEG to storage for the court editor UI
        upload_frame_to_storage(args.match_id, frame)

        # 4. Run court detection
        keypoints = run_court_detection(frame)

        # 5. Store total frame count in match record
        update_frame_count(args.match_id, total_frames)

    # 6. POST keypoints back to backend (tmpdir cleaned up, frame still in storage)
    payload = keypoints_to_payload(keypoints)
    post_keypoints(args.backend_url, args.match_id, payload)

    logger.info("Court setup job complete.")


if __name__ == "__main__":
    main()
