"""Tests for the SSRF guard in backend/services/playsight.py (audit SSRF #6).

_assert_public_url must reject non-http(s) schemes and any host that resolves to
a private/loopback/link-local/reserved/metadata address, while allowing public
IPs. DNS resolution is mocked so these run offline.

Run with: pytest tests/test_playsight_ssrf.py
"""

import socket
from unittest.mock import patch

import pytest

from services.playsight import _assert_public_url, PlaySightImportError  # noqa: E402


def _addrinfo(ip):
    """Shape a socket.getaddrinfo() return with a single resolved IP."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


def test_rejects_non_http_scheme():
    with pytest.raises(PlaySightImportError):
        _assert_public_url("ftp://example.com/file")


def test_rejects_url_without_host():
    with pytest.raises(PlaySightImportError):
        _assert_public_url("https://")


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",        # loopback
        "10.0.0.5",         # private
        "192.168.1.10",     # private
        "169.254.169.254",  # cloud metadata (link-local)
        "0.0.0.0",          # unspecified
    ],
)
def test_rejects_internal_addresses(ip):
    with patch("services.playsight.socket.getaddrinfo", return_value=_addrinfo(ip)):
        with pytest.raises(PlaySightImportError):
            _assert_public_url("https://evil.example.com/stream")


def test_allows_public_address():
    with patch("services.playsight.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
        _assert_public_url("https://my.playsight.com/stream")  # no raise


def test_unresolvable_host_rejected():
    with patch("services.playsight.socket.getaddrinfo", side_effect=socket.gaierror("nope")):
        with pytest.raises(PlaySightImportError):
            _assert_public_url("https://does-not-exist.playsight.com/stream")
