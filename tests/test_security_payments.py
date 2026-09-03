import base64
import hashlib
import hmac
import json
import time
from datetime import datetime

import backend.main as m
from fastapi.testclient import TestClient


def test_paymongo_signature_requires_current_timestamp_and_correct_mode(monkeypatch):
    secret = "webhook-test-secret"
    body = json.dumps({"data": {"id": "evt_123"}}).encode("utf-8")
    timestamp = int(time.time())
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()
    header = f"t={timestamp},te={digest}"
    monkeypatch.setattr(m, "PAYMONGO_WEBHOOK_SECRET", secret)

    assert m._verify_paymongo_webhook_signature(body, header, livemode=False)
    assert not m._verify_paymongo_webhook_signature(body, f"t={timestamp},te=bad", livemode=False)
    assert not m._verify_paymongo_webhook_signature(
        body,
        f"t={timestamp - m.PAYMONGO_WEBHOOK_TOLERANCE_SECONDS - 1},te={digest}",
        livemode=False,
    )


def test_stream_token_is_purpose_limited(monkeypatch):
    monkeypatch.setattr(m, "_shared_token_is_revoked", lambda token: False)
    token = m._sign_stream_token(42, "driver")
    payload = m._verify_auth_token(token)

    assert payload["user_id"] == 42
    assert payload["purpose"] == "stream"
    assert payload["exp"] <= int(time.time()) + m.STREAM_TOKEN_TTL_SECONDS


def test_driver_history_reports_server_total_not_page_length(monkeypatch):
    def fake_query(sql, params=None):
        if "COUNT(*)" in sql:
            return [{"total": 7}]
        return [{
            "payment_id": 1,
            "vehicle_type": "car",
            "duration_type": "daily",
            "regular_price_php": 50,
            "discount_type": "none",
            "discount_amount_php": 0,
            "final_amount_php": 50,
            "payment_method": "cash",
            "notes": None,
            "paid_at_ph": datetime(2026, 9, 3, 12, 0),
        }]

    monkeypatch.setattr(m, "query", fake_query)
    result = m.api_driver_history(
        user_id=9,
        limit=1,
        _auth={"user_id": 9, "role": "driver"},
    )

    assert len(result["records"]) == 1
    assert result["total_sessions"] == 7
    assert result["returned_sessions"] == 1


def test_daily_duration_defaults_match_flat_vehicle_rates(monkeypatch):
    monkeypatch.setattr(m, "_read_setting", lambda key, default: default)

    rates = m._duration_pricing_settings()

    assert rates["daily_rate_php_car"] == m.FLAT_RATE_CAR
    assert rates["daily_rate_php_motorcycle"] == m.FLAT_RATE_MOTORCYCLE


def test_operational_routes_have_auth_dependencies():
    protected = {
        "/api/stream",
        "/api/stats",
        "/api/occupancy",
        "/api/predictions",
        "/api/ml/dashboard",
        "/api/insights",
    }
    routes = {route.path: route for route in m.app.routes if route.path in protected}
    assert set(routes) == protected
    for route in routes.values():
        dependency_calls = {dependency.call for dependency in route.dependant.dependencies}
        assert m.require_user in dependency_calls or m.require_stream_access in dependency_calls


def test_html_response_gets_nonce_csp_and_security_headers():
    with TestClient(m.app) as client:
        response = client.get('/login')

    assert response.status_code == 200
    assert '__OCCUPAI_CSP_NONCE__' not in response.text
    csp = response.headers['content-security-policy']
    assert "script-src 'self' 'nonce-" in csp
    script_policy = csp.split('script-src ', 1)[1].split(';', 1)[0]
    assert "'unsafe-inline'" not in script_policy
    event_policy = csp.split('script-src-attr ', 1)[1].split(';', 1)[0]
    assert "'unsafe-hashes'" in event_policy
    expected_handler_hash = "'sha256-" + base64.b64encode(
        hashlib.sha256(b"switchPanel('overview', this)").digest()
    ).decode('ascii') + "'"
    assert expected_handler_hash in event_policy
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['x-frame-options'] == 'DENY'
