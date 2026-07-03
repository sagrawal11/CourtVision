"""Shared pytest setup for the backend test suite.

The backend modules read Supabase credentials at import time (``api/*.py`` raise
if they are unset, and ``create_client`` is called at module load). We set
harmless fake values BEFORE any backend import so the app can be imported
without real credentials — ``create_client`` does not open a network connection
at construction, and every test patches the client it exercises.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("SUPABASE_URL", "https://fake-project.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "fake-service-role-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "fake-anon-key")

# supabase-py validates the API-key format inside create_client(), so fake creds
# raise before we can patch. Replace the factory with a stub BEFORE any backend
# module runs its `from supabase import create_client` — each test then patches
# the specific module-level client it exercises.
import supabase  # noqa: E402

supabase.create_client = lambda *args, **kwargs: MagicMock()

# Make `backend/` importable as a top-level package the same way main.py does.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import pytest  # noqa: E402


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """A fluent no-op query whose terminal ``execute()`` defers to a resolver.

    Mirrors the subset of the supabase-py builder the backend uses
    (``select/update/insert/eq/in_/single/execute``). The resolver receives the
    table name, accumulated filters, and whether ``single()`` was called, and
    returns the ``.data`` payload for that call.
    """

    def __init__(self, table, resolver):
        self._table = table
        self._resolver = resolver
        self._filters = {}
        self._single = False

    def select(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def insert(self, *args, **kwargs):
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def in_(self, column, values):
        self._filters[column] = list(values)
        return self

    def single(self):
        self._single = True
        return self

    def execute(self):
        return _FakeResponse(self._resolver(self._table, self._filters, self._single))


class FakeSupabase:
    """Drop-in stand-in for the supabase client, driven by a ``resolver``."""

    def __init__(self, resolver):
        self._resolver = resolver

    def table(self, name):
        return _FakeQuery(name, self._resolver)


@pytest.fixture
def make_supabase():
    """Return a factory: ``make_supabase(resolver) -> FakeSupabase``."""
    return FakeSupabase
