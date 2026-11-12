import os

import numpy as np
import pytest


os.environ["CAMERA_ROLE"] = "car"
os.environ["CAMERA_ID"] = "cars"
os.environ["WEBCAM_INDEX"] = "0"
os.environ["STREAM_PORT"] = "8001"
os.environ["DEPLOY_MODE"] = "local"

from yolo_service import detector_v7 as detector


class _FakeBox:
    def __init__(self, class_id, coordinates):
        self.conf = np.array([0.9])
        self.cls = np.array([class_id])
        self.xyxy = np.array([coordinates])


class _FakeResult:
    def __init__(self):
        self.boxes = [
            _FakeBox(0, [10, 20, 110, 120]),
            _FakeBox(1, [30, 40, 80, 140]),
        ]


class _FakeModel:
    names = {0: "toy_car", 1: "toy_motorcycle"}

    def __init__(self):
        self.last_classes = None

    def predict(self, _frame, **kwargs):
        self.last_classes = kwargs["classes"]
        return [_FakeResult()]


def test_custom_model_contract_rejects_coco_classes():
    assert detector.validate_toy_model(_FakeModel()) == {
        0: "toy_car",
        1: "toy_motorcycle",
    }

    coco_model = type("CocoModel", (), {"names": {2: "car", 3: "motorcycle"}})()
    with pytest.raises(RuntimeError, match="exactly the classes"):
        detector.validate_toy_model(coco_model)


def test_yolo_boxes_filters_detections_to_camera_role(monkeypatch):
    model = _FakeModel()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(detector, "CAMERA_ROLE", "car")
    boxes, types = detector.yolo_boxes(model, frame)
    assert boxes == [[10, 20, 110, 120]]
    assert types == ["car"]
    assert model.last_classes == [0]

    monkeypatch.setattr(detector, "CAMERA_ROLE", "motorcycle")
    boxes, types = detector.yolo_boxes(model, frame)
    assert boxes == [[30, 40, 80, 140]]
    assert types == ["motorcycle"]
    assert model.last_classes == [1]


def test_detector_uses_stable_trained_model_path():
    assert detector.YOLO_MODEL_PATH.endswith(
        os.path.join("yolo_service", "models", "toy_vehicle_v2_best.pt")
    )
    assert os.path.isfile(detector.YOLO_MODEL_PATH)


def test_video_overlay_draws_reserved_yellow_but_occupied_red(monkeypatch):
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    zones = {
        "C01": (10, 10, 80, 100),
        "C02": (90, 10, 160, 100),
    }
    calls = []

    monkeypatch.setattr(detector, "_reserved_slots_snapshot", lambda: {"C01", "C02"})
    monkeypatch.setattr(
        detector,
        "_draw_rounded_rect",
        lambda _frame, _x1, _y1, _x2, _y2, color, thickness=1, fill=False:
            calls.append((color, fill)),
    )

    detector.draw_zones(frame, zones, {"C01": False, "C02": True})

    assert calls[0] == (detector._RESERVED_COLOR, True)
    assert calls[1] == (detector._OCC_COLOR, True)
    assert calls[2] == (detector._RESERVED_COLOR, False)
    assert calls[3] == (detector._OCC_COLOR, False)
