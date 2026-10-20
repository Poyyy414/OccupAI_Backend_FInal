# OccupAI Camera To Render Setup

## Why localhost works while Render says Waiting

The webcam and computer-vision detector run on the laptop. Render cannot see
the laptop camera, and `127.0.0.1` always means the current computer/server.
The detector must publish its CV state to the Render backend over HTTPS.

The backend already accepts authenticated detector updates at:

`POST /yolo/update`

## Configure the Render service

In Render → Environment, set a unique `CAM_TOKEN`. Keep this value private.
The local detector must use the exact same value.

## Configure the laptop detector

In the local, uncommitted `.env`, use the following values:

```text
DEPLOY_MODE=local
LOCAL_BACKEND_URL=http://127.0.0.1:8000
REMOTE_BACKEND_URL=https://occupai-backend-final-last.onrender.com
PUSH_LOCAL_BACKEND=true
PUSH_REMOTE_BACKEND=true
CAM_TOKEN=<the same unique CAM_TOKEN configured in Render>
```

Do not commit the real token. `REMOTE_BACKEND_URL` is optional; it was added so
the local and deployed destinations are not confused with one `BACKEND_URL`.

## Verify it

1. Start the local FastAPI backend if localhost is also needed.
2. Start both workers with `.\\start_two_cameras.ps1`.
3. At startup, confirm each worker prints both local and Render push targets.
4. Confirm the detector logs successful `/yolo/update` requests, not HTTP 401,
   403, timeout, or connection errors.
5. Open the Render dashboard and wait up to the camera-state timeout window.
   Overview should show detector live, occupied/free slots, and FPS.

If Render shows HTTP 401, the two `CAM_TOKEN` values do not match. If it shows
timeouts, check the laptop internet connection and Render URL. If only local
works, check `PUSH_REMOTE_BACKEND=true` and the printed target list.

## Two-camera local setup

The admin web Live Feed shows separate **Car Camera** and **Motorcycle Camera**
cards. Camera identity is strict: the launcher starts exactly one process for
each physical camera using these settings.

Motorcycle worker:

```text
CAMERA_ID=motorcycles
CAMERA_ROLE=motorcycle
WEBCAM_INDEX=1
STREAM_PORT=8002
```

Car worker:

```text
CAMERA_ID=cars
CAMERA_ROLE=car
WEBCAM_INDEX=0
STREAM_PORT=8001
```

Do not start a third or legacy `main` worker. The launcher prevents duplicate
launcher instances and fails if either required stream port is already in use.

Each worker loads its fixed normalized boxes from
`yolo_service/camera_layouts.json`: 10 car boxes and 20 motorcycle boxes. The
backend publishes a combined occupied/free percentage only while both workers
are fresh and reporting. Otherwise the combined occupancy is Unknown while
the official configured capacity remains 30.

## Online live video

The detector includes its latest annotated JPEG in each authenticated
`/yolo/update`. The hosted backend keeps the newest frame for each camera and
relays it through the authenticated `/api/stream?camera_id=cars` and
`/api/stream?camera_id=motorcycles` endpoints. The driver page therefore shows
both feeds online without exposing ports 8001/8002 or the laptop's local IP.

Keep both workers and the laptop internet connection running. The relay is a
near-live monitoring view whose update rate follows detector pushes; the local
direct MJPEG stream remains smoother. If `CAMERA_STREAM_URLS` is configured,
the backend prefers those direct URLs. Otherwise it automatically uses the
pushed-frame relay.
