"""Reservation state and slot availability tests without a live database."""

from datetime import datetime, timedelta
import time

import pytest
from fastapi import HTTPException

import backend.main as m


def _camera(role, count, prefix, occupied=()):
    occupied = set(occupied)
    return {
        "camera_role": role,
        "occupied": len(occupied),
        "free": count - len(occupied),
        "total": count,
        "occupancy_pct": len(occupied) / count * 100,
        "yolo_count": len(occupied),
        "car_count": len(occupied) if role == "car" else 0,
        "motorcycle_count": len(occupied) if role == "motorcycle" else 0,
        "fps": 10,
        "timestamp": "2026-09-20 10:00:00",
        "zones": {f"{prefix}{i:02d}": f"{prefix}{i:02d}" in occupied for i in range(1, count + 1)},
        "received_at": time.monotonic(),
    }


def test_only_configured_slots_can_be_reserved():
    assert m._reservation_slot_role("cars:C01") == ("cars", "car")
    assert m._reservation_slot_role("motorcycles:M20") == ("motorcycles", "motorcycle")
    for bad in ("cars:C11", "motorcycles:M21", "cars:M01", "main:C01", ""):
        with pytest.raises(HTTPException) as error:
            m._reservation_slot_role(bad)
        assert error.value.status_code == 400


def test_slot_colors_prioritize_physical_occupancy_and_fail_closed(monkeypatch):
    old = dict(m.camera_states)
    try:
        m.camera_states.clear()
        m.camera_states["cars"] = _camera("car", 10, "C", occupied={"C02"})
        m.camera_states["motorcycles"] = _camera("motorcycle", 20, "M")
        monkeypatch.setattr(m, "_expire_reservations", lambda: None)
        monkeypatch.setattr(m, "query", lambda *a, **kw: [
            {"slot_id": "cars:C01", "status": "reserved", "user_id": 7},
            {"slot_id": "cars:C02", "status": "reserved", "user_id": 8},
        ])
        statuses = {slot["slot_id"]: slot for slot in m._reservation_slot_statuses(7)}
        assert len(statuses) == 30
        assert statuses["cars:C01"]["status"] == "reserved"
        assert statuses["cars:C01"]["mine"] is True
        assert statuses["cars:C02"]["status"] == "occupied"
        assert statuses["cars:C03"]["status"] == "available"
        m.camera_states["motorcycles"]["received_at"] = time.monotonic() - 999
        assert {s["slot_id"]: s["status"] for s in m._reservation_slot_statuses()}["motorcycles:M01"] == "unknown"
    finally:
        m.camera_states.clear()
        m.camera_states.update(old)


def test_paid_reservation_commits_payment_and_expiry_together(monkeypatch):
    now = datetime.now(m.PH_TZ)
    reservation = {
        "reservation_id": 12, "checkout_ref": "checkout-ref", "status": "pending_payment",
        "hold_expires_at": now + timedelta(minutes=10), "payment_status": "unpaid",
        "vehicle_type": "car",
        "duration_type": "weekly", "slot_id": "cars:C01", "user_id": 7,
    }

    class Cursor:
        def __init__(self):
            self.statements = []
            self.results = iter([reservation, {"payment_id": 23, "paid_at": now}])

        def execute(self, sql, params=None):
            self.statements.append((sql, params))

        def fetchone(self):
            return next(self.results)

        def close(self):
            pass

    class Connection:
        def __init__(self):
            self.cur = Cursor()
            self.committed = False

        def cursor(self):
            return self.cur

        def commit(self):
            self.committed = True

        def rollback(self):
            raise AssertionError("Unexpected rollback")

        def close(self):
            pass

    conn = Connection()
    monkeypatch.setattr(m, "get_db", lambda: conn)
    monkeypatch.setattr(m, "_invalidate_payment_revenue_dashboard_cache", lambda: None)
    checkout = {
        "reservation_id": 12, "ref": "checkout-ref", "checkout_id": "paymongo-id",
        "regular_price_php": 300, "final_amount_php": 300, "discount_type": "none",
        "discount_id_number": None,
    }
    result = m._finalize_paid_reservation(checkout)
    assert conn.committed
    assert result["slot_id"] == "cars:C01"
    assert result["expires_at"] == (now + timedelta(days=7)).isoformat()
    assert any("INSERT INTO parking_payments" in sql for sql, _ in conn.cur.statements)
    assert any("UPDATE parking_reservations SET status='reserved'" in sql for sql, _ in conn.cur.statements)
    assert any("UPDATE gcash_checkouts SET status='paid'" in sql for sql, _ in conn.cur.statements)


def test_repeated_payment_callback_does_not_insert_another_payment(monkeypatch):
    expires = datetime.now(m.PH_TZ) + timedelta(days=1)

    class Cursor:
        def __init__(self):
            self.statements = []

        def execute(self, sql, params=None):
            self.statements.append(sql)

        def fetchone(self):
            return {"reservation_id": 12, "checkout_ref": "checkout-ref",
                    "payment_status": "paid", "payment_id": 23,
                    "slot_id": "cars:C01", "expires_at": expires}

        def close(self):
            pass

    class Connection:
        def __init__(self):
            self.cur = Cursor()
            self.committed = False

        def cursor(self):
            return self.cur

        def commit(self):
            self.committed = True

        def rollback(self):
            raise AssertionError("Unexpected rollback")

        def close(self):
            pass

    conn = Connection()
    monkeypatch.setattr(m, "get_db", lambda: conn)
    result = m._finalize_paid_reservation({"reservation_id": 12, "ref": "checkout-ref"})
    assert result["payment_id"] == 23
    assert conn.committed
    assert not any("INSERT INTO parking_payments" in sql for sql in conn.cur.statements)


def test_camera_arrival_and_departure_transitions(monkeypatch):
    statements = []
    monkeypatch.setattr(m, "query", lambda *a, **kw: [{"slot_id": "cars:C01"}])
    monkeypatch.setattr(m, "execute", lambda sql, params=None: statements.append((sql, params)))
    m._update_reservation_arrival("cars", {"C01": True, "C02": True})
    assert any("status='occupied'" in sql for sql, _ in statements)
    assert all(params == ("cars:C01",) for _, params in statements)
    statements.clear()
    m._update_reservation_arrival("cars", {"C01": False})
    assert any("vacant_since=COALESCE" in sql for sql, _ in statements)
    assert any("status='completed'" in sql for sql, _ in statements)


def test_reservation_templates_have_three_slot_colors():
    driver = (m.TEMPLATE_DIR / "driver.html").read_text(encoding="utf-8")
    admin = (m.TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    assert "payForReservation()" in driver
    assert "My reservation" in driver
    assert ".space-cell.reserved" in driver
    assert ".lot-cell.reserved" in admin
    assert "Reserved</span>" in admin
