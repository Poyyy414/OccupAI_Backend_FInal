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
2. Start the detector with `python .\\yolo_service\\detector_v7.py`.
3. At startup, confirm the log prints both local and Render push targets.
4. Confirm the detector logs successful `/yolo/update` requests, not HTTP 401,
   403, timeout, or connection errors.
5. Open the Render dashboard and wait up to the camera-state timeout window.
   Overview should show detector live, occupied/free slots, and FPS.

If Render shows HTTP 401, the two `CAM_TOKEN` values do not match. If it shows
timeouts, check the laptop internet connection and Render URL. If only local
works, check `PUSH_REMOTE_BACKEND=true` and the printed target list.

## Two-camera local setup

The admin web Live Feed shows separate **Car Camera** and **Motorcycle Camera**
cards. Run one detector process for each physical USB camera, using a different
camera ID and stream port. The current car camera can remain `main`; using
`car` is clearer once the second process is ready.

Example settings for the second process:

```text
CAMERA_ID=motorcycle
CAMERA_ROLE=motorcycle
WEBCAM_INDEX=1
STREAM_PORT=8002
```

For a clean two-camera setup, use these values for the car process:

```text
CAMERA_ID=car
CAMERA_ROLE=car
WEBCAM_INDEX=0
STREAM_PORT=8001
```

The backend combines fresh camera states for the overall total, while each
camera card keeps its own occupied, free, capacity, and percentage values. Do
not run the old `main` detector at the same time as the new `car` detector, or
the same car lot may be counted twice.

## Important stream limitation

Sending `/yolo/update` makes online occupancy and prediction data work. The
Render live video feed is separate: Render cannot proxy `localhost:8001` on the
laptop. To display video online, the camera stream must be hosted at a public
HTTPS URL and configured as `CAMERA_STREAM_URL_MAIN`, or the system needs a
separate authenticated frame-upload/stream relay.
