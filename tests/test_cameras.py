import asyncio
import base64
import time

import pytest
from fastapi import HTTPException

import backend.main as m


def test_camera_snapshot_decoder_accepts_jpeg_and_rejects_other_data():
    jpeg = b"\xff\xd8\xfftest-frame\xff\xd9"
    assert m._decode_camera_snapshot(base64.b64encode(jpeg).decode()) == jpeg
    assert m._decode_camera_snapshot("") is None

    with pytest.raises(HTTPException) as invalid_base64:
        m._decode_camera_snapshot("not-base64!")
    assert invalid_base64.value.status_code == 422

    with pytest.raises(HTTPException) as not_jpeg:
        m._decode_camera_snapshot(base64.b64encode(b"not an image").decode())
    assert not_jpeg.value.status_code == 422


def test_driver_template_contains_both_secure_live_feeds():
    html = (m.TEMPLATE_DIR / "driver.html").read_text(encoding="utf-8")
    assert "carsCameraImage" in html
    assert "motorcyclesCameraImage" in html
    assert "'/api/stream?camera_id='" in html
    assert "Secure online feed" in html


def test_online_relay_emits_the_latest_camera_frame():
    jpeg = b"\xff\xd8\xffrelay-frame\xff\xd9"
    previous = dict(m.camera_frames)
    try:
        m.camera_frames["cars"] = {
            "jpeg": jpeg,
            "received_at": time.monotonic(),
            "version": 1,
        }
        response = m._relay_camera_stream("cars")

        async def first_chunk():
            iterator = response.body_iterator
            chunk = await iterator.__anext__()
            await iterator.aclose()
            return chunk

        chunk = asyncio.run(first_chunk())
        assert b"Content-Type: image/jpeg" in chunk
        assert jpeg in chunk
        assert response.media_type == "multipart/x-mixed-replace; boundary=occupaiframe"
    finally:
        m.camera_frames.clear()
        m.camera_frames.update(previous)


def _camera(*, role, occupied, free, total, car_count=0, motorcycle_count=0,
            zones=None, received_at=None):
    return {
        "camera_role": role,
        "occupied": occupied,
        "free": free,
        "total": total,
        "occupancy_pct": round(occupied / total * 100, 1) if total else 0.0,
        "yolo_count": car_count + motorcycle_count,
        "car_count": car_count,
        "motorcycle_count": motorcycle_count,
        "fps": 10.0,
        "timestamp": "2026-09-03 12:00:00",
        "zones": zones or {},
        "received_at": time.monotonic() if received_at is None else received_at,
    }


def test_camera_states_are_aggregated_and_zones_are_namespaced():
    previous = dict(m.camera_states)
    try:
        now = time.monotonic()
        m.camera_states.clear()
        m.camera_states.update({
            "cars": _camera(
                role="car", occupied=2, free=8, total=10,
                car_count=2, zones={"A1": True, "A2": False},
                received_at=now,
            ),
            "motorcycles": _camera(
                role="motorcycle", occupied=1, free=19, total=20,
                motorcycle_count=1, zones={"B1": True},
                received_at=now,
            ),
        })

        result = m._aggregate_camera_states_locked(now)

        assert result["occupied"] == 3
        assert result["free"] == 27
        assert result["total"] == 30
        assert result["configured_capacity"] == 30
        assert result["reporting_capacity"] == 30
        assert result["combined_complete"] is True
        assert result["car_count"] == 2
        assert result["motorcycle_count"] == 1
        assert result["zones"] == {
            "cars:A1": True,
            "cars:A2": False,
            "motorcycles:B1": True,
        }
        assert set(result["cameras"]) == {"cars", "motorcycles"}
    finally:
        m.camera_states.clear()
        m.camera_states.update(previous)


def test_stale_camera_makes_combined_occupancy_unknown():
    previous = dict(m.camera_states)
    try:
        now = time.monotonic()
        m.camera_states.clear()
        m.camera_states.update({
            "cars": _camera(
                role="car", occupied=2, free=8, total=10,
                received_at=now,
            ),
            "motorcycles": _camera(
                role="motorcycle", occupied=4, free=16, total=20,
                received_at=now - m.CAMERA_STATE_TTL_SECONDS - 1,
            ),
        })

        result = m._aggregate_camera_states_locked(now)

        assert result["occupied"] is None
        assert result["free"] is None
        assert result["occupancy_pct"] is None
        assert result["total"] == 30
        assert result["reporting_capacity"] == 10
        assert result["combined_complete"] is False
        assert result["combined_status"] == "partial"
        assert result["cameras"]["cars"]["online"] is True
        assert result["cameras"]["motorcycles"]["online"] is False
    finally:
        m.camera_states.clear()
        m.camera_states.update(previous)


def test_offline_lot_keeps_official_capacity_and_unknown_zones():
    previous = dict(m.camera_states)
    try:
        m.camera_states.clear()
        result = m._aggregate_camera_states_locked(time.monotonic())

        assert result["configured_capacity"] == 30
        assert result["reporting_capacity"] == 0
        assert result["occupied"] is None
        assert result["free"] is None
        assert result["combined_status"] == "unavailable"
        assert len(result["zones"]) == 30
        assert all(value is None for value in result["zones"].values())
        assert set(result["cameras"]) == {"cars", "motorcycles"}
    finally:
        m.camera_states.clear()
        m.camera_states.update(previous)
