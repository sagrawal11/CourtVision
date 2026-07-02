"""
PlaySight integration service.

PlaySight does not currently expose a public API for downloading the raw match
video, but their share pages embed an Open Graph ``og:video`` meta tag so that
the link previews well in iMessage/Slack/etc. That meta tag contains the
underlying HLS playlist URL (typically an ``.m3u8``).

This module exposes a small set of helpers that:

    1. Validate that a URL looks like a PlaySight share link.
    2. Scrape the share page HTML with ``requests`` + ``BeautifulSoup`` to pull
       out the raw stream URL.
    3. Hand that URL to ``yt-dlp`` (which delegates to ``ffmpeg`` under the
       hood) to download every HLS segment and stitch them into a single
       ``.mp4`` file the CV pipeline can ingest with ``cv2.VideoCapture``.

The pipeline expects local ``.mp4`` files (see ``cv/pipeline.py``), so this
service deliberately returns a local file path — the caller is then
responsible for uploading the file to Supabase Storage and creating the
``matches`` row in exactly the same way as a direct browser upload would.

Usage from Python:

    from backend.services.playsight import download_playsight_video

    local_path = download_playsight_video(
        "https://my.playsight.com/share?svkey=...",
        output_path="/tmp/match.mp4",
    )

System requirements:
    * ``ffmpeg`` must be installed and on ``PATH`` (used by ``yt-dlp`` to
      mux the HLS segments).
    * ``requests``, ``beautifulsoup4``, and ``yt-dlp`` Python packages
      (declared in ``backend/requirements.txt``).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A modern desktop UA — PlaySight serves the full HTML (including og:video
# meta tags) only when it thinks a real browser is asking.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
)

_PLAYSIGHT_DOMAINS = ("playsight.com", "playsight.net")

# How long to wait for the share page HTML before giving up.
_HTTP_TIMEOUT_SEC = 15


# ---------------------------------------------------------------------------
# Errors / result types
# ---------------------------------------------------------------------------


class PlaySightImportError(RuntimeError):
    """Raised when we cannot turn a PlaySight share URL into a local .mp4."""


@dataclass
class PlaySightImportResult:
    """The outcome of a successful PlaySight import."""

    local_path: str  # absolute path to the downloaded .mp4 on the local FS
    stream_url: str  # the og:video URL we extracted (m3u8 or mp4)
    source_url: str  # the original share URL the caller passed in
    size_bytes: int  # size of the downloaded file


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_playsight_url(url: str) -> bool:
    """Return ``True`` if *url* is a PlaySight share link.

    Accepts ``playsight.com``/``playsight.net`` and any subdomain of them
    (``my.playsight.com``, ``share.playsight.com``, …). Uses the parsed
    *hostname* (not ``netloc``) and an exact domain/suffix match so that
    SSRF-style spoofs like ``playsight.com.evil.com`` or
    ``playsight.com@evil.com`` are rejected.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in _PLAYSIGHT_DOMAINS)


def _assert_public_url(url: str) -> None:
    """Raise PlaySightImportError unless *url* is http(s) and resolves only to
    public IP addresses.

    Blocks SSRF to loopback/private/link-local/reserved ranges (incl. the cloud
    metadata endpoint 169.254.169.254). Applied to both the share page we scrape
    and the media stream URL we hand to yt-dlp, since either host is influenced
    by attacker-controlled page content.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PlaySightImportError(f"Refusing non-http(s) URL: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise PlaySightImportError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise PlaySightImportError(f"Could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise PlaySightImportError(
                f"Refusing to fetch URL resolving to non-public address {ip} (host {host!r})"
            )


def validate_playsight_link(link: str) -> bool:
    """Backwards-compatible alias retained for older callers."""
    return is_playsight_url(link)


def extract_stream_url(playsight_url: str, *, session: Optional[requests.Session] = None) -> str:
    """Scrape *playsight_url* and return the embedded raw video stream URL.

    The function looks for ``<meta property="og:video:secure_url">`` first and
    falls back to ``<meta property="og:video">``. PlaySight share pages set
    these to an ``.m3u8`` HLS playlist URL.

    Args:
        playsight_url: A PlaySight share link.
        session: Optional pre-configured ``requests.Session`` (used in tests).

    Returns:
        The direct media stream URL.

    Raises:
        PlaySightImportError: If the page can't be fetched or doesn't include
        the expected metadata.
    """
    if not is_playsight_url(playsight_url):
        raise PlaySightImportError(
            f"URL does not look like a PlaySight share link: {playsight_url!r}"
        )

    headers = {
        "User-Agent": _DEFAULT_USER_AGENT,
        # Some CDNs serve different HTML to clients that don't advertise this.
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    http = session or requests
    _assert_public_url(playsight_url)
    logger.info("PlaySight: fetching share page %s", playsight_url)
    try:
        response = http.get(playsight_url, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise PlaySightImportError(
            f"Failed to fetch PlaySight share page: {exc}"
        ) from exc

    soup = BeautifulSoup(response.text, "html.parser")

    candidates = [
        soup.find("meta", property="og:video:secure_url"),
        soup.find("meta", property="og:video"),
        # Some PlaySight pages use itemprop instead of OpenGraph.
        soup.find("meta", itemprop="contentURL"),
    ]
    for tag in candidates:
        if tag and tag.get("content"):
            stream_url = tag["content"].strip()
            logger.info("PlaySight: found embedded stream URL: %s", stream_url)
            return stream_url

    # As a last resort, scan the raw HTML for an .m3u8 reference.
    m3u8_match = re.search(r"https?://[^\s\"'<>]+\.m3u8[^\s\"'<>]*", response.text)
    if m3u8_match:
        stream_url = m3u8_match.group(0)
        logger.info("PlaySight: recovered stream via regex fallback: %s", stream_url)
        return stream_url

    raise PlaySightImportError(
        "Could not locate the raw video stream in the PlaySight share page. "
        "The video may be private or PlaySight may have changed their share page."
    )


def download_playsight_video(
    playsight_url: str,
    output_path: str,
    *,
    overwrite: bool = True,
    session: Optional[requests.Session] = None,
) -> PlaySightImportResult:
    """Download a PlaySight share URL into a local ``.mp4`` file.

    This wraps :func:`extract_stream_url` and feeds the result to ``yt-dlp``.
    ``yt-dlp`` downloads all of the HLS chunks and uses the system ``ffmpeg``
    binary to stitch them into a single MP4. The MP4 is what our CV pipeline
    (``cv/pipeline.py``) ingests with ``cv2.VideoCapture``.

    Args:
        playsight_url: The PlaySight share link a coach pasted in.
        output_path: Where the final ``.mp4`` should land. The extension is
            normalised to ``.mp4`` because ``merge_output_format='mp4'``.
        overwrite: If ``True`` (default) any existing file at ``output_path``
            is replaced. If ``False`` and the file already exists,
            :class:`PlaySightImportError` is raised.
        session: Optional ``requests.Session`` for HTML scraping (tests).

    Returns:
        A :class:`PlaySightImportResult` describing the downloaded file.

    Raises:
        PlaySightImportError: If anything fails — scraping, downloading,
            ffmpeg muxing, or file verification.
    """
    if not output_path:
        raise PlaySightImportError("output_path is required")

    # 1. Sanity check: we need ffmpeg on PATH for yt-dlp to mux HLS into MP4.
    if shutil.which("ffmpeg") is None:
        raise PlaySightImportError(
            "ffmpeg is not installed or not on PATH. Install ffmpeg "
            "(e.g. `brew install ffmpeg`) so PlaySight HLS streams can be "
            "stitched into MP4."
        )

    # 2. Pull the direct stream URL out of the share page, then re-validate it:
    #    the URL comes from attacker-influenceable page content, so it must also
    #    be confirmed to point at a public address before yt-dlp fetches it.
    stream_url = extract_stream_url(playsight_url, session=session)
    _assert_public_url(stream_url)

    # 3. Normalise output_path to end in .mp4.
    base_path, ext = os.path.splitext(output_path)
    if ext.lower() != ".mp4":
        logger.debug("PlaySight: rewriting output extension %r → .mp4", ext)
    final_path = f"{base_path}.mp4"

    if os.path.exists(final_path):
        if not overwrite:
            raise PlaySightImportError(f"Refusing to overwrite existing file: {final_path}")
        os.remove(final_path)

    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)

    # 4. Hand off to yt-dlp. Import lazily so module import doesn't depend on
    #    the optional dep (e.g. for unit tests that only exercise scraping).
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in environments without yt-dlp
        raise PlaySightImportError(
            "yt-dlp is not installed. Run "
            "`pip install -r backend/requirements.txt` to install it."
        ) from exc

    ydl_opts = {
        "outtmpl": f"{base_path}.%(ext)s",
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        # Suppress yt-dlp's interactive UI when running as a backend service.
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        # HLS-specific: rely on ffmpeg which we just verified is installed.
        "hls_prefer_native": False,
    }

    logger.info("PlaySight: starting yt-dlp download → %s", final_path)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([stream_url])
    except Exception as exc:  # yt_dlp raises a zoo of exception types
        raise PlaySightImportError(f"yt-dlp failed to download stream: {exc}") from exc

    if not os.path.exists(final_path):
        # yt-dlp may have written a different extension if the format negotiation
        # picked something other than mp4. Search the directory for the result.
        candidates = [
            p for p in os.listdir(os.path.dirname(final_path) or ".")
            if p.startswith(os.path.basename(base_path) + ".")
        ]
        raise PlaySightImportError(
            "yt-dlp completed but no .mp4 was produced. "
            f"Candidates in output dir: {candidates}"
        )

    size_bytes = os.path.getsize(final_path)
    if size_bytes == 0:
        raise PlaySightImportError(f"Downloaded file is empty: {final_path}")

    logger.info(
        "PlaySight: download complete (%.1f MB) → %s",
        size_bytes / 1_048_576,
        final_path,
    )
    return PlaySightImportResult(
        local_path=final_path,
        stream_url=stream_url,
        source_url=playsight_url,
        size_bytes=size_bytes,
    )


# ---------------------------------------------------------------------------
# CLI for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download a PlaySight share video to MP4.")
    parser.add_argument("url", help="PlaySight share URL")
    parser.add_argument(
        "-o", "--output",
        default="playsight_match.mp4",
        help="Output .mp4 path (default: playsight_match.mp4)",
    )
    args = parser.parse_args()

    try:
        result = download_playsight_video(args.url, args.output)
    except PlaySightImportError as exc:
        print(f"[!] {exc}")
        raise SystemExit(1)

    print(f"[*] Saved {result.size_bytes / 1_048_576:.1f} MB to {result.local_path}")
    print(f"[*] Stream URL: {result.stream_url}")
