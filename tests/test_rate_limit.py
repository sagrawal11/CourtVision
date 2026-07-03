"""Tests for the slowapi rate limiter wiring (audit RATE_LIMITING #10).

Covers the client-IP key function (forge-resistance behind the ALB) and the
end-to-end 429 behavior with per-IP bucket isolation. Run with:
pytest tests/test_rate_limit.py
"""

from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import rate_limit
from rate_limit import client_ip, limiter


def _req(headers=None, host="10.0.0.1"):
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (host, 12345),
    }
    return Request(scope)


def test_client_ip_uses_socket_when_no_forwarded_header():
    assert client_ip(_req(host="203.0.113.7")) == "203.0.113.7"


def test_client_ip_takes_last_forwarded_entry():
    # The ALB appends the true client IP last, so the key is the last entry.
    r = _req(headers={"X-Forwarded-For": "198.51.100.9, 203.0.113.7"})
    assert client_ip(r) == "203.0.113.7"


def test_forged_forwarded_header_cannot_move_the_key():
    # A client-forged single value is still the "last" entry only when no proxy
    # appends — but behind the ALB the real IP is appended after it. Simulate:
    forged_only = _req(headers={"X-Forwarded-For": "1.2.3.4"})
    behind_alb = _req(headers={"X-Forwarded-For": "1.2.3.4, 203.0.113.7"})
    assert client_ip(forged_only) == "1.2.3.4"
    assert client_ip(behind_alb) == "203.0.113.7"  # forged prefix ignored


# ── End-to-end limit behavior on a minimal app using the shared limiter ────────

def _make_app():
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/ping")
    @limiter.limit("2/minute")
    async def ping(request: Request):
        return {"ok": True}

    return TestClient(app)


def test_limit_triggers_429_after_threshold():
    client = _make_app()
    hdr = {"X-Forwarded-For": "203.0.113.10"}
    assert client.get("/ping", headers=hdr).status_code == 200
    assert client.get("/ping", headers=hdr).status_code == 200
    assert client.get("/ping", headers=hdr).status_code == 429  # 3rd over the 2/min cap


def test_separate_ips_have_separate_buckets():
    client = _make_app()
    a = {"X-Forwarded-For": "203.0.113.20"}
    b = {"X-Forwarded-For": "203.0.113.21"}
    client.get("/ping", headers=a)
    client.get("/ping", headers=a)
    assert client.get("/ping", headers=a).status_code == 429
    # A different client IP is unaffected.
    assert client.get("/ping", headers=b).status_code == 200
