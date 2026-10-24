import base64
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import backend.main as m


def _data_url(mime, payload):
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def test_admin_review_buttons_use_csp_safe_delegated_events():
    dashboard = (Path(__file__).parents[1] / "template" / "dashboard.html").read_text(
        encoding="utf-8"
    )

    assert 'data-discount-review-action="approve"' in dashboard
    assert 'data-discount-review-action="reject"' in dashboard
    assert "[data-discount-review-action]" in dashboard
    assert 'onclick="reviewDiscountVerification(' not in dashboard


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


def test_complete_pwd_approval_and_discounted_payment_workflow(monkeypatch):
    """Driver submits -> admin sees/approves -> checkout unlocks -> payment consumes approval."""
    state = {"verification": None, "next_result": None, "commits": 0}

    class WorkflowCursor:
        rowcount = 0

        def execute(self, sql, params=None):
            normalized = " ".join(sql.split())
            self.rowcount = 0
            state["next_result"] = None
            if "INSERT INTO discount_verifications" in normalized:
                state["verification"] = {
                    "verification_id": 501,
                    "user_id": 7,
                    "discount_type": params[1],
                    "id_number": params[2],
                    "status": "pending",
                    "submitted_at": m.datetime.now(m.PH_TZ),
                    "reviewed_at": None,
                    "review_note": None,
                    "checkout_ref": None,
                }
                state["next_result"] = dict(state["verification"])
                self.rowcount = 1
            elif "reviewed_by_user_id" in normalized:
                record = state["verification"]
                if record and record["status"] == "pending":
                    record["status"] = params[0]
                    record["reviewed_at"] = m.datetime.now(m.PH_TZ)
                    record["review_note"] = params[2]
                    state["next_result"] = dict(record)
                    self.rowcount = 1
            elif "SET checkout_ref=%s, reserved_at=NOW()" in normalized:
                record = state["verification"]
                if record and record["status"] == "approved":
                    record["checkout_ref"] = params[0]
                    state["next_result"] = {"verification_id": record["verification_id"]}
                    self.rowcount = 1
            elif "SET status='consumed', consumed_at=NOW()" in normalized:
                record = state["verification"]
                if record and record["status"] == "approved" and record["checkout_ref"] == params[1]:
                    record["status"] = "consumed"
                    state["next_result"] = {"verification_id": record["verification_id"]}
                    self.rowcount = 1

        def fetchone(self):
            return state["next_result"]

        def close(self):
            pass

    class WorkflowConnection:
        def cursor(self):
            return WorkflowCursor()

        def commit(self):
            state["commits"] += 1

        def rollback(self):
            pass

        def close(self):
            pass

    def fake_query(sql, params=None, **_kwargs):
        if "FROM discount_verifications dv" in sql:
            record = state["verification"]
            return [{**record, "full_name": "Test Driver", "email": "driver@example.com"}]
        if "SELECT full_name, email FROM users" in sql:
            return [{"full_name": "Test Driver", "email": "driver@example.com"}]
        return []

    class FakePaymongoResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "data": {
                    "id": "cs_test_pwd",
                    "attributes": {"checkout_url": "https://checkout.example/pwd"},
                }
            }).encode()

    monkeypatch.setattr(m, "get_db", lambda: WorkflowConnection())
    monkeypatch.setattr(m, "query", fake_query)
    monkeypatch.setattr(m, "execute", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m, "_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m, "_discount_rate_for_type", lambda kind, _id: (kind, 0.20))
    monkeypatch.setattr(m, "_check_discount_id_abuse", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m, "PAYMONGO_SECRET_KEY", "sk_test_not_real")
    monkeypatch.setattr(m, "_effective_duration_pricing", lambda: {
        "daily_rate_php_car": 50.0,
        "pricing_mode": "base",
        "demand_pricing_enabled": False,
    })
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: FakePaymongoResponse())

    submitted = m.submit_discount_verification(
        m.DiscountVerificationSubmitPayload(
            discount_type="pwd",
            id_number="PWD-2026-0001",
            image_data_url=_data_url("image/jpeg", b"\xff\xd8\xff\xe0valid-test-image"),
        ),
        SimpleNamespace(),
        {"user_id": 7, "role": "driver"},
    )
    assert submitted["status"] == "pending"

    pending = m.list_discount_verifications("pending", {"user_id": 1, "role": "admin"})
    assert pending["items"][0]["email"] == "driver@example.com"
    assert pending["items"][0]["status"] == "pending"

    approved = m.review_discount_verification(
        501,
        m.DiscountVerificationReviewPayload(action="approve"),
        SimpleNamespace(),
        {"user_id": 1, "role": "admin"},
    )
    assert approved["verification"]["status"] == "approved"

    checkout = m.gcash_create_checkout(
        m.GcashCheckoutPayload(
            mobile_number="09171234567",
            discount_type="pwd",
            discount_id_number="PWD-2026-0001",
            discount_verification_id=501,
        ),
        SimpleNamespace(base_url="https://example.com/"),
        {"user_id": 7, "role": "driver"},
    )
    assert checkout["checkout_url"] == "https://checkout.example/pwd"
    assert checkout["regular_price_php"] == 50.0
    assert checkout["amount_php"] == 40.0

    checkout_record = {
        "ref": state["verification"]["checkout_ref"],
        "checkout_id": "cs_test_pwd",
        "user_id": 7,
        "vehicle_type": "car",
        "duration_type": "daily",
        "regular_price_php": 50.0,
        "final_amount_php": 40.0,
        "discount_type": "pwd",
        "discount_id_number": "PWD-2026-0001",
        "discount_verification_id": 501,
        "reservation_id": None,
    }
    monkeypatch.setattr(m, "_claim_gcash_checkout", lambda *_args, **_kwargs: (checkout_record, "claimed"))
    monkeypatch.setattr(m, "_record_parking_payment", lambda *_args, **_kwargs: {
        "ok": True, "payment_id": 9001, "final_amount_php": 40.0,
    })

    _, payment_status, payment = m._finalize_gcash_checkout(checkout_record["ref"])
    assert payment_status == "paid"
    assert payment["final_amount_php"] == 40.0
    assert state["verification"]["status"] == "consumed"
