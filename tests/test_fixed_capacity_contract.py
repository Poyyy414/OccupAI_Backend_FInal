import time

import pytest
from fastapi import HTTPException

import backend.main as m
from backend.models import YoloUpdate


def test_active_and_prediction_capacity_ignore_camera_and_demand_state(monkeypatch):
    monkeypatch.setattr(m, "_last_slot_adjustment", {"n_slots": 999, "demand": "HIGH"})
    monkeypatch.setattr(m, "camera_states", {})

    assert m._active_slot_capacity() == 30
    assert m._prediction_capacity() == 30
    assert m._capacity_settings() == {
        "car_slots": 10,
        "motorcycle_slots": 20,
        "total_slots": 30,
        "editable": False,
        "source": "official_fixed_layout",
    }


def test_backend_rejects_wrong_camera_capacity():
    update = YoloUpdate(
        occupied=0,
        free=9,
        total=9,
        occupancy_pct=0,
        camera_id="cars",
        camera_role="car",
    )

    with pytest.raises(HTTPException, match="exactly 10"):
        m.yolo_update(update, x_cam_token=m.CAM_TOKEN)


def test_driver_summary_does_not_turn_offline_capacity_into_free_spaces(env_settings):
    env_settings({"PRICE_OVERRIDE_ENABLED": "false"})
    previous = dict(m.camera_states)
    try:
        m.camera_states.clear()
        result = m.api_driver_summary()

        assert result["total"] == 30
        assert result["configured_capacity"] == 30
        assert result["reporting_capacity"] == 0
        assert result["occupied"] is None
        assert result["available"] is None
        assert result["available_text"] == "--"
        assert result["combined_complete"] is False
        assert result["car"]["price_php"] == 50.0
        assert result["motorcycle"]["price_php"] == 25.0
        assert result["car"]["dynamic_pricing_available"] is False
    finally:
        m.camera_states.clear()
        m.camera_states.update(previous)


def test_calibration_required_camera_is_unknown_not_fresh():
    previous = dict(m.camera_states)
    try:
        now = time.monotonic()
        m.camera_states.clear()
        m.camera_states["cars"] = {
            "camera_role": "car",
            "occupied": 0,
            "free": 10,
            "total": 10,
            "occupancy_pct": 0.0,
            "yolo_count": 0,
            "car_count": 0,
            "motorcycle_count": 0,
            "fps": 10.0,
            "timestamp": "2026-09-06 12:00:00",
            "zones": {"C01": False},
            "reporting": False,
            "camera_status": "calibration_required",
            "calibration_required": True,
            "received_at": now,
        }

        result = m._aggregate_camera_states_locked(now)

        assert result["reporting_capacity"] == 0
        assert result["cameras"]["cars"]["status"] == "calibration_required"
        assert result["cameras"]["cars"]["occupied"] is None
        assert result["zones"]["cars:C01"] is None
    finally:
        m.camera_states.clear()
        m.camera_states.update(previous)
