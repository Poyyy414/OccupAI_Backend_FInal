import base64
import hashlib
import hmac
import json
import time
import urllib.request
from datetime import datetime
from types import SimpleNamespace

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


def test_password_reset_pages_and_logo_are_served():
    with TestClient(m.app) as client:
        forgot = client.get('/forgot-password')
        reset = client.get('/reset-password')
        logo = client.get('/static/occupai_logo.png')

    assert forgot.status_code == 200
    assert 'Forgot password?' in forgot.text
    assert '__OCCUPAI_CSP_NONCE__' not in forgot.text
    assert reset.status_code == 200
    assert 'Choose a new password' in reset.text
    assert logo.status_code == 200
    assert logo.headers['content-type'].startswith('image/png')


def test_forgot_password_response_does_not_enumerate_accounts(monkeypatch):
    monkeypatch.setattr(m, '_rate_limit', lambda *args, **kwargs: None)
    monkeypatch.setattr(m, '_issue_password_reset_token', lambda email: None)
    monkeypatch.setattr(m, '_audit_auth_event', lambda *args, **kwargs: None)

    with TestClient(m.app) as client:
        response = client.post('/auth/forgot-password', json={'email': 'known@example.com'})

    assert response.status_code == 200
    assert response.json() == {
        'ok': True,
        'message': 'If an account exists for that email, a password reset link has been sent.',
    }


def test_brevo_password_reset_delivery_uses_transactional_api(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 201

    def fake_post(url, **kwargs):
        captured['url'] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(m, 'BREVO_API_KEY', 'test-key-not-a-real-secret')
    monkeypatch.setattr(m, 'BREVO_SENDER_EMAIL', 'support@example.com')
    monkeypatch.setattr(m.httpx, 'post', fake_post)

    assert m._send_password_reset_email(
        'driver@gmail.com',
        'https://example.com/reset-password?token=safe-test-token',
    )
    assert captured['url'] == 'https://api.brevo.com/v3/smtp/email'
    assert captured['headers']['api-key'] == 'test-key-not-a-real-secret'
    assert captured['json']['sender']['email'] == 'support@example.com'
    assert captured['json']['to'] == [{'email': 'driver@gmail.com'}]
    assert 'safe-test-token' in captured['json']['htmlContent']
    assert 'textContent' not in captured['json']


def test_philippine_mobile_number_is_required_and_normalized():
    assert m._normalize_ph_mobile('0917 123 4567') == '09171234567'
    assert m._normalize_ph_mobile('+63 917 123 4567') == '09171234567'
    assert m._normalize_ph_mobile('9171234567') == '09171234567'

    for invalid in ('', '12345', '08171234567'):
        try:
            m._normalize_ph_mobile(invalid)
        except Exception as exc:
            assert getattr(exc, 'status_code', None) == 400
        else:
            raise AssertionError(f'{invalid!r} should not be accepted')


def test_gcash_checkout_sends_required_customer_mobile_to_paymongo(monkeypatch):
    captured = {}

    class FakePaymongoResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                'data': {
                    'id': 'cs_test',
                    'attributes': {'checkout_url': 'https://checkout.example/test'},
                }
            }).encode()

    def fake_urlopen(request, timeout):
        captured['body'] = json.loads(request.data.decode())
        captured['timeout'] = timeout
        return FakePaymongoResponse()

    monkeypatch.setattr(m, 'PAYMONGO_SECRET_KEY', 'sk_test_not_real')
    monkeypatch.setattr(m, '_rate_limit', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m, 'execute', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        m,
        'query',
        lambda *_args, **_kwargs: [{
            'full_name': 'Ada Lovelace',
            'email': 'ada@example.com',
        }],
    )
    monkeypatch.setattr(
        m,
        '_effective_duration_pricing',
        lambda: {
            'daily_rate_php_car': 50.0,
            'pricing_mode': 'base',
            'demand_pricing_enabled': False,
        },
    )
    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)

    result = m.gcash_create_checkout(
        m.GcashCheckoutPayload(mobile_number='0917 123 4567'),
        SimpleNamespace(base_url='https://example.com/'),
        {'user_id': 7, 'role': 'driver'},
    )

    billing = captured['body']['data']['attributes']['billing']
    assert billing == {
        'name': 'Ada Lovelace',
        'email': 'ada@example.com',
        'phone': '09171234567',
    }
    assert result['checkout_url'] == 'https://checkout.example/test'
