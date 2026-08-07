"""
Shared pytest fixtures for backend tests.

Importing backend.main triggers real DB/model setup only inside the FastAPI
`lifespan` context manager, not at import time — so these tests can call its
pure helper functions directly without a live database connection.
"""
import pytest

import backend.main as m


@pytest.fixture
def env_settings(monkeypatch):
    """
    Lets a test control exactly what backend.main._read_env_value() returns,
    instead of depending on whatever is currently in the repo's .env file
    (which an admin could have changed via the dashboard toggles).

    Usage: env_settings({"PRICE_OVERRIDE_ENABLED": "false"})
    """
    def _apply(overrides: dict):
        def fake_read_env_value(key, default=""):
            return overrides.get(key, default)
        monkeypatch.setattr(m, "_read_env_value", fake_read_env_value)
    return _apply
