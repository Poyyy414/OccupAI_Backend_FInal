import base64
from pathlib import Path

import pytest
from fastapi import HTTPException

import backend.main as m


def _data_url(mime, payload):
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


@pytest.mark.parametrize(
    "mime,payload",
    [
        ("image/jpeg", b"\xff\xd8\xff\xe0test"),
        ("image/png", b"\x89PNG\r\n\x1a\ntest"),
        ("image/webp", b"RIFF\x04\x00\x00\x00WEBPtest"),
    ],
)
def test_discount_id_image_accepts_supported_magic_bytes(mime, payload):
    decoded, detected = m._decode_discount_id_image(_data_url(mime, payload))
    assert decoded == payload
    assert detected == mime


def test_discount_id_image_rejects_spoofed_mime_type():
    with pytest.raises(HTTPException) as error:
        m._decode_discount_id_image(_data_url("image/png", b"\xff\xd8\xff\xe0test"))
    assert error.value.status_code == 400


def test_approved_verification_is_claimed_for_exact_user_and_id(monkeypatch):
    captured = {}
    class FakeCursor:
        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params
        def fetchone(self):
            return {"verification_id": 91}
        def close(self):
            pass
    class FakeConnection:
        def cursor(self):
            return FakeCursor()
        def commit(self):
            captured["committed"] = True
        def rollback(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(m, "get_db", lambda: FakeConnection())
    result = m._claim_discount_verification(7, "pwd", "PWD-1234", 91, "checkout-ref")

    assert result == 91
    assert "status='approved'" in captured["sql"]
    assert captured["params"] == ("checkout-ref", 91, 7, "pwd", "PWD-1234")
    assert captured["committed"] is True


def test_unapproved_discount_cannot_start_checkout(monkeypatch):
    class FakeCursor:
        def execute(self, *_args, **_kwargs):
            pass
        def fetchone(self):
            return None
        def close(self):
            pass
    class FakeConnection:
        def cursor(self):
            return FakeCursor()
        def commit(self):
            pass
        def rollback(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(m, "get_db", lambda: FakeConnection())
    with pytest.raises(HTTPException) as error:
        m._claim_discount_verification(7, "senior", "SC-1234", 42, "checkout-ref")
    assert error.value.status_code == 403


def test_discount_verification_controls_exist_in_driver_and_admin_ui():
    root = Path(__file__).parents[1]
    driver = (root / "template" / "driver.html").read_text(encoding="utf-8")
    admin = (root / "template" / "dashboard.html").read_text(encoding="utf-8")

    assert 'capture="environment"' in driver
    assert "Submit ID for owner approval" in driver
    assert "discount_verification_id" in driver
    assert "PWD / Senior ID Review" in admin
    assert "reviewDiscountVerification" in admin


def test_discount_verification_routes_have_expected_auth():
    paths = {
        "/api/discount-verifications": m.require_user,
        "/api/discount-verifications/mine": m.require_user,
        "/api/admin/discount-verifications": m.require_admin,
        "/api/admin/discount-verifications/{verification_id}/image": m.require_admin,
        "/api/admin/discount-verifications/{verification_id}/review": m.require_admin,
    }
    routes = {route.path: route for route in m.app.routes if route.path in paths}
    assert set(routes) == set(paths)
    for path, route in routes.items():
        dependency_calls = {dependency.call for dependency in route.dependant.dependencies}
        assert paths[path] in dependency_calls
