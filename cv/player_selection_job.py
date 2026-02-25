"""
cv/player_selection_job.py — Background job for generating player selection frames.

Extracts 5 evenly spaced frames from a video, runs YOLO player detection on them,
and uploads the annotated frames (or metadata) to Supabase Storage so the frontend
can present a UI for the user to click their player of interest.

Triggered via `POST /api/videos/{match_id}/generate-player-selection-frames`.
Results are saved to `player_selection_frames/` in the Supabase bucket.

Usage:
    python cv/player_selection_job.py --match-id <uuid>
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import json
import base64

import cv2
import numpy as np
from supabase import create_client, Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.detection.player_detector import PlayerDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not supabase_url or not supabase_key:
    logger.error("Supabase credentials not found in env")
    sys.exit(1)
supabase: Client = create_client(supabase_url, supabase_key)

NUM_FRAMES = 5
BUCKET_NAME = "match-videos"


def _draw_player_boxes(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Draw simple boxes (no indices/labels) around detected players to guide the user."""
    out = frame.copy()
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        # Draw a sleek highlight box that feels interactive
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 200, 120), 4)
        
        # Add an inner border for contrast
        cv2.rectangle(out, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    args = parser.parse_args()

    match_id = args.match_id
    logger.info(f"Starting Player Selection Job for match {match_id}")

    # 1. Fetch match record to find video path
    res = supabase.table("matches").select("s3_temp_key").eq("id", match_id).execute()
    if not res.data:
        logger.error("Match not found")
        sys.exit(1)
        
    s3_key = res.data[0].get("s3_temp_key")
    if not s3_key:
        logger.error("Match has no associated video file")
        sys.exit(1)

    # 2. Get signed URL for video download
    #    (Using create_signed_url just to get a temporary read link, 
    #    so we don't have to download the whole gigabyte file locally if OpenCV can stream it)
    try:
        url_res = supabase.storage.from_(BUCKET_NAME).create_signed_url(s3_key, 3600)
        signed_url = url_res.get("signedURL")
    except Exception as e:
        logger.error(f"Failed to get signed URL: {e}")
        sys.exit(1)

    # OpenCV can read directly from http(s) streams (if compiled with FFmpeg, which it usually is)
    cap = cv2.VideoCapture(signed_url)
    if not cap.isOpened():
        logger.error("Failed to open video stream from Supabase")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Video opened. Total frames: {total_frames}, FPS: {fps}")

    # Calculate 5 evenly spaced frame indices, avoiding the very beginning and very end
    # where players might be walking on/off court
    margin = int(total_frames * 0.1) # 10% margin
    indices = np.linspace(margin, total_frames - margin, NUM_FRAMES, dtype=int)
    logger.info(f"Sampling frames at indices: {indices}")

    # Load player detector
    logger.info("Loading YOLO PlayerDetector...")
    try:
        detector = PlayerDetector(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu")
    except Exception as e:
        logger.error(f"Failed to load PlayerDetector: {e}")
        sys.exit(1)

    # We will bundle everything into a JSON artifact:
    # {
    #   "frames": [
    #     {
    #       "frame_index": 1234,
    #       "image_base64": "...",
    #       "boxes": [[x1,y1,x2,y2], ...]
    #     },
    #     ...
    #   ]
    # }
    selection_data = {"frames": []}

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {idx}")
            continue

        logger.info(f"Running detection on frame {idx}...")
        boxes_array = detector.run_human_detection(frame, bbox_thr=0.5)
        
        # run_human_detection returns np.ndarray of shape (N, 4) -> [x1, y1, x2, y2]
        valid_boxes = boxes_array.tolist() if len(boxes_array) > 0 else []
        
        # Only include frames where we actually detected players
        if len(valid_boxes) >= 1:
            # Draw boxes on frame for the user to see
            annotated_frame = _draw_player_boxes(frame, boxes_array)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            h, w = frame.shape[:2]
            selection_data["frames"].append({
                "frame_index": int(idx),
                "image_base64": f"data:image/jpeg;base64,{b64_str}",
                "boxes": [[float(x) for x in b] for b in valid_boxes],
                "width": int(w),
                "height": int(h)
            })
            logger.info(f"Frame {idx}: found {len(valid_boxes)} players")

    cap.release()

    if not selection_data["frames"]:
        logger.error("No players were detected in any sampled frame.")
        # Notify backend that generation failed
        # We could add an API endpoint for this, but for now we just exit
        sys.exit(1)

    # Upload the JSON manifest to Supabase Storage
    # E.g. videos/player_selection_frames/{match_id}.json
    output_key = f"player_selection_frames/{match_id}.json"
    logger.info(f"Uploading bundled JSON to generic bucket: {output_key}")
    
    json_bytes = json.dumps(selection_data).encode("utf-8")
    
    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            output_key,
            json_bytes,
            {"content-type": "application/json", "upsert": "true"}
        )
        logger.info("Upload successful")
    except Exception as e:
        # Sometimes the supabase Python client raises weird exceptions on upsert,
        # but the upload still succeeds. Let's try updating if upload fails.
        logger.warning(f"Upload threw error, falling back to update: {e}")
        try:
            supabase.storage.from_(BUCKET_NAME).update(
                output_key,
                json_bytes,
                {"content-type": "application/json"}
            )
            logger.info("Update successful")
        except Exception as e2:
            logger.error(f"Storage update failed: {e2}")
            sys.exit(1)

    # Update match status so frontend knows the frames are ready
    # status becomes "player_selection"
    try:
        supabase.table("matches").update({
            "status": "player_selection"
        }).eq("id", match_id).execute()
        logger.info("Match status updated to 'player_selection'")
    except Exception as e:
        logger.error(f"Failed to update match status: {e}")
        sys.exit(1)

    logger.info("Job completed successfully!")


if __name__ == "__main__":
    main()
