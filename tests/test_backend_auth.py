"""Tests for backend/auth.py get_user_id — the JWT verification dependency.

Run with: pytest tests/test_backend_auth.py
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import auth  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_missing_authorization_header_401():
    with pytest.raises(HTTPException) as exc:
        _run(auth.get_user_id(authorization=None))
    assert exc.value.status_code == 401


def test_auth_not_configured_500(monkeypatch):
    monkeypatch.setattr(auth, "supabase_auth", None)
    with pytest.raises(HTTPException) as exc:
        _run(auth.get_user_id(authorization="Bearer sometoken"))
    assert exc.value.status_code == 500


def test_invalid_token_401(monkeypatch):
    fake = MagicMock()
    fake.auth.get_user.return_value = SimpleNamespace(user=None)
    monkeypatch.setattr(auth, "supabase_auth", fake)
    with pytest.raises(HTTPException) as exc:
        _run(auth.get_user_id(authorization="Bearer badtoken"))
    assert exc.value.status_code == 401


def test_provider_error_maps_to_401(monkeypatch):
    fake = MagicMock()
    fake.auth.get_user.side_effect = RuntimeError("network down")
    monkeypatch.setattr(auth, "supabase_auth", fake)
    with pytest.raises(HTTPException) as exc:
        _run(auth.get_user_id(authorization="Bearer whatever"))
    assert exc.value.status_code == 401


def test_valid_token_returns_user_id_and_strips_bearer(monkeypatch):
    fake = MagicMock()
    fake.auth.get_user.return_value = SimpleNamespace(user=SimpleNamespace(id="user-123"))
    monkeypatch.setattr(auth, "supabase_auth", fake)

    assert _run(auth.get_user_id(authorization="Bearer tok-abc")) == "user-123"
    # The "Bearer " prefix must be stripped before verification.
    fake.auth.get_user.assert_called_with("tok-abc")


def test_valid_token_without_bearer_prefix(monkeypatch):
    fake = MagicMock()
    fake.auth.get_user.return_value = SimpleNamespace(user=SimpleNamespace(id="user-9"))
    monkeypatch.setattr(auth, "supabase_auth", fake)

    assert _run(auth.get_user_id(authorization="rawtoken")) == "user-9"
    fake.auth.get_user.assert_called_with("rawtoken")
