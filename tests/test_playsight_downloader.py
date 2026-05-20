"""
Tests for backend/services/playsight.py

These exercise the URL validation + HTML-scraping logic without hitting the
network. Run as:

    python tests/test_playsight_downloader.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make `backend/` importable as a top-level package the same way `main.py` does.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from services.playsight import (  # noqa: E402
    is_playsight_url,
    extract_stream_url,
    PlaySightImportError,
)


def test_is_playsight_url():
    """Basic URL validation should accept PlaySight hosts and reject everything else."""
    assert is_playsight_url("https://my.playsight.com/share?svkey=foo")
    assert is_playsight_url("https://share.playsight.com/video/abc")
    assert is_playsight_url("http://playsight.com/somepath")
    # Different host
    assert not is_playsight_url("https://example.com/foo")
    # Looks like PlaySight but wrong scheme
    assert not is_playsight_url("ftp://my.playsight.com/foo")
    # Garbage
    assert not is_playsight_url("")
    assert not is_playsight_url("not-a-url")
    assert not is_playsight_url(None)  # type: ignore[arg-type]
    print("[OK] test_is_playsight_url")


def test_extract_stream_url_og_video_secure_url():
    """Prefer og:video:secure_url over og:video when both are present."""
    html = """
        <html><head>
            <meta property="og:video:secure_url" content="https://cdn.playsight.com/secure.m3u8">
            <meta property="og:video" content="https://cdn.playsight.com/insecure.m3u8">
        </head><body></body></html>
    """
    fake_resp = MagicMock(text=html)
    fake_resp.raise_for_status = MagicMock()
    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=fake_resp)

    url = extract_stream_url("https://my.playsight.com/share?svkey=foo", session=fake_session)
    assert url == "https://cdn.playsight.com/secure.m3u8", url
    print("[OK] test_extract_stream_url_og_video_secure_url")


def test_extract_stream_url_falls_back_to_og_video():
    """If only og:video is present, that's still good enough."""
    html = """
        <html><head>
            <meta property="og:video" content="https://cdn.playsight.com/playlist.m3u8">
        </head><body></body></html>
    """
    fake_resp = MagicMock(text=html)
    fake_resp.raise_for_status = MagicMock()
    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=fake_resp)

    url = extract_stream_url("https://my.playsight.com/share?svkey=foo", session=fake_session)
    assert url == "https://cdn.playsight.com/playlist.m3u8"
    print("[OK] test_extract_stream_url_falls_back_to_og_video")


def test_extract_stream_url_regex_fallback():
    """If no meta tags match, regex-scan the raw HTML for an .m3u8 URL."""
    html = """
        <html><head></head><body>
            <script>var src = "https://cdn.playsight.com/abc/master.m3u8?token=xyz";</script>
        </body></html>
    """
    fake_resp = MagicMock(text=html)
    fake_resp.raise_for_status = MagicMock()
    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=fake_resp)

    url = extract_stream_url("https://my.playsight.com/share?svkey=foo", session=fake_session)
    assert url == "https://cdn.playsight.com/abc/master.m3u8?token=xyz", url
    print("[OK] test_extract_stream_url_regex_fallback")


def test_extract_stream_url_no_video_raises():
    """A page with no embedded video metadata should raise."""
    html = "<html><head><title>Sign in</title></head><body>Login required</body></html>"
    fake_resp = MagicMock(text=html)
    fake_resp.raise_for_status = MagicMock()
    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=fake_resp)

    try:
        extract_stream_url("https://my.playsight.com/share?svkey=foo", session=fake_session)
    except PlaySightImportError as exc:
        assert "Could not locate" in str(exc)
        print("[OK] test_extract_stream_url_no_video_raises")
        return
    raise AssertionError("expected PlaySightImportError")


def test_extract_stream_url_rejects_non_playsight():
    """We refuse to scrape anything outside PlaySight."""
    try:
        extract_stream_url("https://youtube.com/watch?v=foo")
    except PlaySightImportError as exc:
        assert "does not look like a PlaySight share link" in str(exc)
        print("[OK] test_extract_stream_url_rejects_non_playsight")
        return
    raise AssertionError("expected PlaySightImportError")


def main():
    test_is_playsight_url()
    test_extract_stream_url_og_video_secure_url()
    test_extract_stream_url_falls_back_to_og_video()
    test_extract_stream_url_regex_fallback()
    test_extract_stream_url_no_video_raises()
    test_extract_stream_url_rejects_non_playsight()
    print("\nAll PlaySight downloader tests passed.")


if __name__ == "__main__":
    main()
