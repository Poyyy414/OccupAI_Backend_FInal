"""
Shared pytest fixtures for backend tests.

Importing backend.main triggers real DB/model setup only inside the FastAPI
`lifespan` context manager, not at import time — so these tests can call its
pure helper functions directly without a live database connection. The
env_settings fixture below fully replaces _read_setting() with a fake, so
even settings lookups (which normally hit the admin_settings table in
Postgres) stay DB-free in tests.
"""
import pytest

import backend.main as m


@pytest.fixture
def env_settings(monkeypatch):
    """
    Lets a test control exactly what backend.main._read_setting() returns,
    instead of depending on whatever is currently saved in the admin_settings
    table (which an admin could have changed via the dashboard toggles).

    Usage: env_settings({"PRICE_OVERRIDE_ENABLED": "false"})
    """
    def _apply(overrides: dict):
        def fake_read_setting(key, default=""):
            return overrides.get(key, default)
        monkeypatch.setattr(m, "_read_setting", fake_read_setting)
    return _apply
