"""
Supabase Storage Utility Module.

Replaces the AWS S3 module for the free-tier development stack.
Handles all video and frame image storage using Supabase Storage buckets.

Bucket: match-videos
  temp-uploads/{match_id}/{original_filename}   ← uploaded video
  temp-uploads/{match_id}/frame_1000.jpg        ← extracted court setup frame

Free tier limits (generous for dev):
  - 1 GB total storage
  - 2 GB bandwidth/month
  - No egress costs within Supabase

See docs/architecture.md for the full storage strategy.
"""

import os
import logging
from pathlib import Path
from supabase import create_client, Client

logger = logging.getLogger(__name__)

STORAGE_BUCKET = "match-videos"

# Shared client — initialised lazily so imports don't crash if env vars aren't set
_supabase: Client | None = None


def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _supabase = create_client(url, key)
    return _supabase


# ----- Public API ------------------------------------------------------------

def get_upload_path(match_id: str, filename: str) -> str:
    """Return the canonical storage path for a match video upload."""
    safe_filename = Path(filename).name  # Strip any directory traversal
    return f"temp-uploads/{match_id}/{safe_filename}"


def get_frame_path(match_id: str) -> str:
    """Return the storage path for the extracted court setup frame JPEG."""
    return f"temp-uploads/{match_id}/frame_1000.jpg"


def create_signed_upload_url(storage_path: str) -> str:
    """
    Generate a short-lived signed URL the browser can use to upload
    a file directly to Supabase Storage without hitting our backend.

    Returns:
        Signed upload URL string (valid for 1 hour)
    """
    sb = _get_client()
    result = sb.storage.from_(STORAGE_BUCKET).create_signed_upload_url(storage_path)
    logger.debug(f"create_signed_upload_url raw result: {result}")
    # supabase-py returns different keys depending on version:
    #   older  → {"signedURL": "...", "token": "...", "path": "..."}
    #   newer  → {"signedUrl": "...", "token": "...", "path": "..."}
    url = result.get("signedURL") or result.get("signedUrl") or result.get("signed_url") or result.get("url")
    token = result.get("token")
    logger.info(f"create_signed_upload_url raw: url={url!r}, token={'yes' if token else 'no'}, all_keys={list(result.keys())}")

    if not url:
        raise ValueError(f"No signed URL in response. Keys received: {list(result.keys())}")

    # storage-py may return a relative path — prefix with Supabase project URL
    if url.startswith("/"):
        supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        url = f"{supabase_url}{url}"

    # Append the upload token as a query parameter if not already present
    if token and "token=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={token}"

    logger.info(f"Final upload URL prefix: {url[:80]}...")
    return url


def create_signed_download_url(storage_path: str, expiry: int = 3600) -> str:
    """
    Generate a short-lived signed URL for reading a file from Supabase Storage.

    Args:
        storage_path: Path within the bucket
        expiry: URL lifetime in seconds (default 1 hour)

    Returns:
        Signed download URL string
    """
    sb = _get_client()
    result = sb.storage.from_(STORAGE_BUCKET).create_signed_url(storage_path, expiry)
    logger.debug(f"create_signed_download_url raw result type={type(result)} keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    url = result.get("signedURL") or result.get("signedUrl") or result.get("url") if isinstance(result, dict) else str(result)
    if not url:
        raise ValueError(f"No signed URL in download response. Keys: {list(result.keys())}")
    return url


def download_file(storage_path: str, local_path: str) -> None:
    """
    Download a file from Supabase Storage to a local path.
    Used by the local court_setup_job to pull the video before processing.

    Args:
        storage_path: Path within the bucket (e.g. "temp-uploads/{id}/video.mp4")
        local_path: Absolute local path to write the file to
    """
    logger.info(f"Downloading storage:{storage_path} → {local_path}")
    sb = _get_client()
    data = sb.storage.from_(STORAGE_BUCKET).download(storage_path)
    Path(local_path).write_bytes(data)
    logger.info(f"Download complete ({len(data) // 1024} KB)")


def upload_file(storage_path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """
    Upload bytes to Supabase Storage.
    Used by the court_setup_job to store the extracted frame JPEG.

    Args:
        storage_path: Destination path within the bucket
        data: Raw bytes to upload
        content_type: MIME type (e.g. "image/jpeg")
    """
    sb = _get_client()
    sb.storage.from_(STORAGE_BUCKET).upload(
        path=storage_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    logger.info(f"Uploaded {len(data) // 1024} KB to storage:{storage_path}")


def delete_file(storage_path: str) -> None:
    """
    Delete a file from Supabase Storage after processing completes.
    The bucket does NOT have an automatic expiry policy, so explicit
    deletion after analysis is the primary cleanup mechanism.

    Args:
        storage_path: Path within the bucket to delete
    """
    try:
        sb = _get_client()
        sb.storage.from_(STORAGE_BUCKET).remove([storage_path])
        logger.info(f"Deleted storage:{storage_path}")
    except Exception as e:
        logger.warning(f"Failed to delete storage:{storage_path}: {e}")


def file_exists(storage_path: str) -> bool:
    """
    Check if a file exists in Supabase Storage.
    Used to verify an upload completed before triggering processing.

    Args:
        storage_path: Path within the bucket

    Returns:
        True if the file exists
    """
    try:
        sb = _get_client()
        # list() with a path filter — if result is non-empty, file exists
        parts = storage_path.rsplit("/", 1)
        prefix = parts[0] + "/" if len(parts) > 1 else ""
        name = parts[-1]
        files = sb.storage.from_(STORAGE_BUCKET).list(prefix)
        return any(f.get("name") == name for f in (files or []))
    except Exception:
        return False
