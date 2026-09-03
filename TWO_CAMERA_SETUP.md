# Two-camera diorama setup

The system now supports two independent detector workers reporting to one backend:

- one worker for the car POV (`CAMERA_ROLE=car`)
- one worker for the motorcycle POV (`CAMERA_ROLE=motorcycle`)

The backend combines their fresh occupancy, capacity, vehicle counts, and zones. The
web admin live-feed panel shows a camera selector when both workers are online.

## 1. Start the backend

Set these values in the backend process environment before starting FastAPI:

```text
CAMERA_STATE_TTL_SECONDS=15
CAMERA_STREAM_URLS=cars=http://127.0.0.1:8001/stream,motorcycles=http://127.0.0.1:8002/stream
```

`CAMERA_STREAM_URL_CARS` and `CAMERA_STREAM_URL_MOTORCYCLES` may be used instead
of `CAMERA_STREAM_URLS`.

These URLs are resolved by the backend machine. The `127.0.0.1` examples work when
FastAPI and both detector workers run on the same computer. If FastAPI is hosted on
Render or another server, the URLs must point to secure streams reachable from that
server; a camera worker's localhost is not the server's localhost. Count updates can
still reach a hosted backend even when its video proxy cannot reach the local streams.

The backend must run as one process so both camera states are visible to the same
in-memory aggregator. If multiple backend instances are deployed, move camera state
to a shared store before using this feature across instances.

## 2. Start the car camera worker

Open a separate terminal in the repository root:

```powershell
$env:CAMERA_ID = "cars"
$env:CAMERA_ROLE = "car"
$env:WEBCAM_INDEX = "0"
$env:STREAM_PORT = "8001"
python yolo_service/detector_v7.py
```

## 3. Start the motorcycle camera worker

Open another terminal in the repository root:

```powershell
$env:CAMERA_ID = "motorcycles"
$env:CAMERA_ROLE = "motorcycle"
$env:WEBCAM_INDEX = "1"
$env:STREAM_PORT = "8002"
python yolo_service/detector_v7.py
```

The camera indexes are operating-system device indexes and can change when USB
devices are reconnected. Confirm the worker logs show the intended camera and that
each stream is reachable at ports 8001 and 8002.

## 4. Calibrate each POV independently

Each worker needs a layout matching its own camera angle. The `R1_*`, `R2_*`,
`R3_*`, `NO_PARK_RECTS`, and `EXCLUDED_SLOTS` environment values are read by each
worker, so configure them for that worker before launch. Do not put different values
for both cameras in one shared `.env` file; use separate process environments or
separate environment files.

With two workers running, inspect the generated camera-specific files:

```text
debug_zones_cars.jpg
debug_zones_motorcycles.jpg
```

The car worker ignores motorcycle classifications and the motorcycle worker ignores
car classifications. If the toy color/size classifier is not reliable for the
diorama, set `TOY_ONLY_DETECTION=false` and configure the YOLO vehicle classes for
the model instead.

## 5. Verify the integration

1. Open the admin dashboard and wait for both camera workers to send an update.
2. Confirm `/api/stats` reports `cameras.cars` and `cameras.motorcycles`.
3. Confirm combined `total`, `occupied`, `free`, `car_count`, and `motorcycle_count`.
4. Switch the live-feed selector and verify each POV changes to its own stream.
5. Stop one worker and wait at least 15 seconds; its state must disappear from the
   aggregate and the other camera must continue reporting.
6. Restart the stopped worker and verify it returns without restarting the backend.

The camera stream proxy is still an operational endpoint and should be protected
with authentication before exposing the dashboard or streams outside a trusted
network; this remains listed in `QA_TASK_LIST.md`.
