# Fixed two-camera diorama setup

OccupAI uses two dedicated USB camera workers and one official 30-space lot:

| Worker | `CAMERA_ID` | `CAMERA_ROLE` | Webcam | Stream | Fixed spaces |
|---|---|---|---:|---:|---:|
| Cars | `cars` | `car` | 0 | 8001 | 10 |
| Motorcycles | `motorcycles` | `motorcycle` | 1 | 8002 | 20 |

Demand labels and predictions never add, remove, resize, or move these boxes.

## Start the backend

Configure the backend process with:

```text
CAMERA_STATE_TTL_SECONDS=15
CAMERA_STREAM_URLS=cars=http://127.0.0.1:8001/stream,motorcycles=http://127.0.0.1:8002/stream
```

The USB feeds remain local when the workers and dashboard are on the same
computer. A hosted backend can receive occupancy updates without being able to
proxy a local-only video stream.

## Start both camera workers

From the repository root, run the single launcher:

```powershell
.\start_two_cameras.ps1
```

It starts exactly one car worker and one motorcycle worker, refuses to run a
second launcher, checks ports 8001/8002 before startup, and reports a worker's
exit code. Each detector also validates its required camera ID, webcam index,
and stream port.

## Calibrate normalized layouts

Edit `yolo_service/camera_layouts.json`. Each camera role has its own:

- `parking_area`: the allowed diorama area `[x1, y1, x2, y2]`;
- `exclusion_areas`: named plants, entrances, roads, or other keep-clear areas;
- `slots`: saved normalized rectangles that scale with frame resolution.

All coordinates are fractions from `0.0` to `1.0`, not pixels. The loader fails
closed if a box is outside the parking area, overlaps an exclusion, has invalid
coordinates, or if the car/motorcycle count is not exactly 10/20. To keep a
machine-specific file elsewhere, set `CAMERA_LAYOUT_FILE` for both workers.

Use the generated `debug_zones_cars.jpg` and
`debug_zones_motorcycles.jpg` files to verify alignment. The committed layout
is a safe normalized starting point; physical camera placement still requires
calibration against the real diorama.

Lighting changes may re-warm occupancy detection, but never alter geometry. If
scene-change detection indicates that a camera moved, that camera reports
`calibration_required`; its spaces become Unknown until the JSON coordinates
are recalibrated and the worker is restarted.

## Offline and partial behavior

- Official configured capacity always remains 10 car + 20 motorcycle = 30.
- A fresh camera reports its own occupancy and reporting capacity.
- A stale, blocked, moved, or disconnected camera reports Unknown, never Free.
- Combined occupied/free/rate values exist only while both cameras are fresh.
- With one fresh camera, `reporting_capacity` is 10 or 20 and
  `combined_status` is `partial`; this is never presented as complete lot data.
- Dynamic customer pricing uses the official capacity of 30. If complete live
  occupancy is unavailable, normal PHP 50 car / PHP 25 motorcycle base prices
  are used without a low-demand discount.

## Verification

1. Confirm `/api/stats` always returns `configured_capacity: 30`, car capacity
   10, and motorcycle capacity 20.
2. Confirm both cameras are online, `reporting_capacity: 30`, and
   `combined_complete: true` before trusting combined occupancy.
3. Stop either worker and wait past `CAMERA_STATE_TTL_SECONDS`. Its values and
   combined occupancy must display `--`/Unknown while configured capacity stays
   30.
4. Restart it and confirm combined occupancy returns only after both feeds are
   fresh.
5. Move a camera enough to trigger scene-change detection and confirm the UI
   shows recalibration required without creating any boxes.
