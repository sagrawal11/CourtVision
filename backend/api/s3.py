"""
S3 Utility Module.

Handles all interactions with Amazon S3 for temporary video storage.
Videos are stored under the 'temp-uploads/' prefix with a 48-hour lifecycle
policy configured on the bucket — they auto-delete even if our code doesn't
clean them up explicitly.

Bucket: courtvision-uploads  (configured via AWS_S3_BUCKET env var)
Key format: temp-uploads/{match_id}/{original_filename}
"""

import os
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ----- Configuration ---------------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "courtvision-uploads")
PRESIGNED_URL_EXPIRY = 3600  # 1 hour — enough time for large video uploads

# Create a single shared S3 client (thread-safe, reused across requests)
_s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)


# ----- Public API ------------------------------------------------------------

def generate_presigned_upload_url(match_id: str, filename: str) -> dict:
    """
    Generate a presigned S3 PUT URL for direct browser-to-S3 video upload.

    The browser uploads the file directly to S3 — it never touches our server.
    This avoids memory/bandwidth constraints on the backend and enables
    resumable uploads on the client side.

    Args:
        match_id: UUID of the match record (used as part of the S3 key)
        filename: Original filename from the user's device (e.g. "match.mov")

    Returns:
        {
            "upload_url": str,   # Presigned PUT URL (expires in 1 hour)
            "s3_key": str,       # Key to store in matches.s3_temp_key
            "bucket": str,
        }

    Raises:
        ClientError: If boto3 fails to generate the URL (credentials issue, etc.)
    """
    # Sanitise filename to avoid S3 key injection
    safe_filename = Path(filename).name  # strips directory components
    s3_key = f"temp-uploads/{match_id}/{safe_filename}"

    upload_url = _s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": AWS_S3_BUCKET,
            "Key": s3_key,
            "ContentType": "video/*",
        },
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )

    logger.info(f"Generated presigned upload URL for match {match_id}: {s3_key}")
    return {
        "upload_url": upload_url,
        "s3_key": s3_key,
        "bucket": AWS_S3_BUCKET,
    }


def generate_presigned_download_url(s3_key: str, expiry: int = 3600) -> str:
    """
    Generate a presigned GET URL so the frontend can display the video frame
    without making it publicly accessible.

    Args:
        s3_key: The S3 object key (e.g. "temp-uploads/{match_id}/video.mov")
        expiry: URL expiry in seconds (default 1 hour)

    Returns:
        Presigned GET URL string
    """
    url = _s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": AWS_S3_BUCKET, "Key": s3_key},
        ExpiresIn=expiry,
    )
    return url


def download_video(s3_key: str, local_path: str) -> None:
    """
    Download a video file from S3 to a local path.
    Used by AWS Batch processing jobs to pull the video before analysis.

    Args:
        s3_key: S3 object key
        local_path: Absolute path on local disk to download the file to
    """
    logger.info(f"Downloading s3://{AWS_S3_BUCKET}/{s3_key} → {local_path}")
    _s3_client.download_file(AWS_S3_BUCKET, s3_key, local_path)
    logger.info(f"Download complete: {local_path}")


def delete_video(s3_key: str) -> None:
    """
    Delete a temporary video from S3 after processing is complete.
    Also called if a match is deleted or processing fails permanently.

    Note: The bucket also has a 48-hour lifecycle policy as a safety net,
    so even if this call fails, the file will eventually be purged.

    Args:
        s3_key: S3 object key to delete
    """
    try:
        _s3_client.delete_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
        logger.info(f"Deleted temp video from S3: {s3_key}")
    except ClientError as e:
        # Log but don't raise — lifecycle policy will clean up eventually
        logger.warning(f"Failed to delete S3 object {s3_key}: {e}")


def object_exists(s3_key: str) -> bool:
    """
    Check if an S3 object exists (used to verify upload completed before
    triggering processing).

    Args:
        s3_key: S3 object key to check

    Returns:
        True if the object exists and is accessible
    """
    try:
        _s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False
