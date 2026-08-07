"""
Tests for PWD/Senior discount validation in backend/main.py.

_discount_rate_for_type() is the single choke point both the admin payment
form and the driver GCash checkout go through, so a bug here silently
affects real money in two places at once — worth pinning down explicitly.
"""
import pytest
from fastapi import HTTPException

import backend.main as m

BOTH_ENABLED = {"DISCOUNT_PWD_ENABLED": "true", "DISCOUNT_SENIOR_ENABLED": "true"}


def test_none_discount_needs_no_id(env_settings):
    env_settings(BOTH_ENABLED)
    kind, rate = m._discount_rate_for_type("none")
    assert kind == "none"
    assert rate == 0.0


def test_pwd_discount_uses_configured_rate(env_settings):
    env_settings(BOTH_ENABLED)
    kind, rate = m._discount_rate_for_type("pwd", "PWD-2024-00123")
    assert kind == "pwd"
    assert rate == m.PWD_SENIOR_DISCOUNT_RATE


def test_senior_discount_uses_configured_rate(env_settings):
    env_settings(BOTH_ENABLED)
    kind, rate = m._discount_rate_for_type("senior", "SC-2024-00456")
    assert kind == "senior"
    assert rate == m.PWD_SENIOR_DISCOUNT_RATE


def test_pwd_discount_without_id_number_is_rejected(env_settings):
    env_settings(BOTH_ENABLED)
    with pytest.raises(HTTPException) as exc:
        m._discount_rate_for_type("pwd", None)
    assert exc.value.status_code == 400


def test_discount_rejected_when_admin_has_turned_it_off(env_settings):
    env_settings({"DISCOUNT_PWD_ENABLED": "false", "DISCOUNT_SENIOR_ENABLED": "true"})
    with pytest.raises(HTTPException) as exc:
        m._discount_rate_for_type("pwd", "PWD-2024-00123")
    assert exc.value.status_code == 400

    # Senior is still enabled, so it should go through fine.
    kind, rate = m._discount_rate_for_type("senior", "SC-2024-00456")
    assert kind == "senior"


@pytest.mark.parametrize("bad_id", ["", "   ", "ab", "abcdefgh", "!!!!"])
def test_malformed_id_numbers_are_rejected(env_settings, bad_id):
    env_settings(BOTH_ENABLED)
    with pytest.raises(HTTPException) as exc:
        m._discount_rate_for_type("senior", bad_id)
    assert exc.value.status_code == 400


def test_unknown_discount_type_is_rejected(env_settings):
    env_settings(BOTH_ENABLED)
    with pytest.raises(HTTPException) as exc:
        m._discount_rate_for_type("student")
    assert exc.value.status_code == 400
