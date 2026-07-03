"""Tests for the internal debug-video-ready callback auth (audit AUTH #3).

The callback is server→server and must be gated by INTERNAL_JOB_SECRET rather
than a user JWT. Run with: pytest tests/test_backend_callback.py
"""

import pytest
from fastapi.testclient import TestClient

from api import videos  # noqa: E402
import main  # noqa: E402

CALLBACK = "/api/videos/m1/debug-video-ready"
BODY = {"storage_path": "debug-videos/m1/debug.mp4"}


@pytest.fixture
def client(monkeypatch, make_supabase):
    # The success path issues a matches update; stub it out.
    monkeypatch.setattr(videos, "supabase", make_supabase(lambda *a: []))
    return TestClient(main.app, raise_server_exceptions=False)


def test_missing_secret_rejected(client, monkeypatch):
    monkeypatch.setattr(videos, "INTERNAL_JOB_SECRET", "topsecret")
    resp = client.patch(CALLBACK, json=BODY)
    assert resp.status_code == 401


def test_wrong_secret_rejected(client, monkeypatch):
    monkeypatch.setattr(videos, "INTERNAL_JOB_SECRET", "topsecret")
    resp = client.patch(CALLBACK, json=BODY, headers={"X-Internal-Secret": "guess"})
    assert resp.status_code == 401


def test_correct_secret_accepted(client, monkeypatch):
    monkeypatch.setattr(videos, "INTERNAL_JOB_SECRET", "topsecret")
    resp = client.patch(CALLBACK, json=BODY, headers={"X-Internal-Secret": "topsecret"})
    assert resp.status_code == 200
    assert resp.json()["match_id"] == "m1"
