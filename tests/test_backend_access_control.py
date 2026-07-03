"""Tests for backend/api/matches.py team-scoped access control.

Covers _verify_match_access and _coach_shares_team_with — the check that a
coach may only reach a player's match if they share a team (audit ACCESS_CONTROL
#4). Run with: pytest tests/test_backend_access_control.py
"""

import pytest
from fastapi import HTTPException

from api import matches  # noqa: E402


def _patch_supabase(monkeypatch, resolver, make_supabase):
    monkeypatch.setattr(matches, "supabase", make_supabase(resolver))


def test_owner_accesses_own_match_ok(monkeypatch, make_supabase):
    # Owner == caller: no DB lookups needed, must not raise.
    def resolver(table, filters, single):
        raise AssertionError("owner path should not query the DB")

    _patch_supabase(monkeypatch, resolver, make_supabase)
    matches._verify_match_access(match_owner="u1", user_id="u1")  # no raise


def test_non_coach_non_owner_denied(monkeypatch, make_supabase):
    def resolver(table, filters, single):
        if table == "users":
            return {"role": "player"}
        return []

    _patch_supabase(monkeypatch, resolver, make_supabase)
    with pytest.raises(HTTPException) as exc:
        matches._verify_match_access(match_owner="owner", user_id="someplayer")
    assert exc.value.status_code == 403


def test_coach_on_shared_team_allowed(monkeypatch, make_supabase):
    def resolver(table, filters, single):
        if table == "users":
            return {"role": "coach"}
        if table == "team_members":
            # First query: coach's teams (filter user_id only).
            if "team_id" not in filters:
                return [{"team_id": "t1"}]
            # Second query: is owner in those teams?
            return [{"user_id": "owner"}]
        return []

    _patch_supabase(monkeypatch, resolver, make_supabase)
    matches._verify_match_access(match_owner="owner", user_id="coach")  # no raise


def test_coach_not_sharing_team_denied(monkeypatch, make_supabase):
    def resolver(table, filters, single):
        if table == "users":
            return {"role": "coach"}
        if table == "team_members":
            if "team_id" not in filters:
                return [{"team_id": "t1"}]
            return []  # owner not in the coach's teams
        return []

    _patch_supabase(monkeypatch, resolver, make_supabase)
    with pytest.raises(HTTPException) as exc:
        matches._verify_match_access(match_owner="owner", user_id="coach")
    assert exc.value.status_code == 403


def test_coach_with_no_teams_denied(monkeypatch, make_supabase):
    def resolver(table, filters, single):
        if table == "users":
            return {"role": "coach"}
        return []  # coach belongs to no teams

    _patch_supabase(monkeypatch, resolver, make_supabase)
    assert matches._coach_shares_team_with("coach", "owner") is False
    with pytest.raises(HTTPException):
        matches._verify_match_access(match_owner="owner", user_id="coach")
