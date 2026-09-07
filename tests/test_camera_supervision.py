from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_camera_reader_failure_requests_supervised_restart():
    source = (ROOT / "yolo_service" / "detector_v7.py").read_text(encoding="utf-8")
    assert "CAMERA_READ_FAILURE_LIMIT" in source
    assert "_cam_reader_failed.set()" in source
    assert "raise RuntimeError(\"Camera capture stopped\")" in source


def test_launcher_restarts_workers_without_changing_camera_mapping():
    source = (ROOT / "start_two_cameras.ps1").read_text(encoding="utf-8")
    assert "CameraId = 'cars'; WebcamIndex = 0; Port = 8001" in source
    assert "CameraId = 'motorcycles'; WebcamIndex = 1; Port = 8002" in source
    assert "Restarting in $delay second(s)" in source
    assert "AUTO_CAMERA_RECALIBRATE'] = 'true'" in source
