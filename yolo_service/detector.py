"""
yolo_service/detector.py - OccupAI high-FPS detector

Run:
  python -m yolo_service.detector

The detector creates virtual parking slots even when the parking land has no
painted slot lines. It finds the largest ground/land surface in the camera
image, builds a perspective grid on top of that area, then marks each virtual
slot occupied when a YOLO vehicle box overlaps the slot.
"""
import base64
import importlib
import math
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int_set(name, default):
    raw = os.getenv(name)
    if not raw:
        return set(default)
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out or set(default)


# Core config
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CAM_TOKEN = os.getenv("CAM_TOKEN", "occupai_cam_2027")
STREAM_PORT = _env_int("STREAM_PORT", 8001)
WEBCAM_IDX = _env_int("WEBCAM_INDEX", 0)
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")

FEED_W = _env_int("FEED_W", 640)
FEED_H = _env_int("FEED_H", 480)
IMGSZ = _env_int("YOLO_IMGSZ", 320)
JPEG_Q = _env_int("JPEG_Q", 50)
PUSH_EVERY = _env_int("PUSH_EVERY", 15)

CONF_THRESH = _env_float("CONF_THRESH", 0.20)
IOU_THRESH = _env_float("IOU_THRESH", 0.20)
VEHICLE_CLASS_IDS = _env_int_set("VEHICLE_CLASS_IDS", {2, 3, 5, 7})

# Automatic virtual slot config
AUTO_GRID = _env_bool("AUTO_GRID", True)
AUTO_GRID_REFRESH_FRAMES = _env_int("AUTO_GRID_REFRESH_FRAMES", 150)
AUTO_SLOT_TOTAL = _env_int("AUTO_SLOT_TOTAL", _env_int("LOT_CAPACITY", 8))
AUTO_SLOT_ROWS = _env_int("AUTO_SLOT_ROWS", 0)
AUTO_SLOT_COLS = _env_int("AUTO_SLOT_COLS", 0)
SLOT_GAP_FRAC = _env_float("SLOT_GAP_FRAC", 0.035)
DRAW_SLOT_LABELS = _env_bool("DRAW_SLOT_LABELS", AUTO_SLOT_TOTAL <= 20)

# Land detection config
LAND_ROI_TOP = _env_float("LAND_ROI_TOP", 0.12)
LAND_MIN_AREA_PCT = _env_float("LAND_MIN_AREA_PCT", 0.18)
LAND_CLUSTER_K = max(2, _env_int("LAND_CLUSTER_K", 3))
LAND_MAX_SAMPLES = _env_int("LAND_MAX_SAMPLES", 8000)
LAND_VEHICLE_PAD = _env_int("LAND_VEHICLE_PAD", 10)
GRID_SMOOTHING = min(max(_env_float("GRID_SMOOTHING", 0.25), 0.0), 1.0)

# Old two-row fallback geometry
TOP_Y1 = 0.05
TOP_Y2 = 0.42
BOT_Y1 = 0.58
BOT_Y2 = 0.95
PAD_X1 = 0.02
PAD_X2 = 0.98
FALLBACK_COLS = 4

HEADERS = {"x-cam-token": CAM_TOKEN, "Content-Type": "application/json"}


_stream_frame = None
_stream_lock = threading.Lock()

_yolo_boxes = []
_yolo_lock = threading.Lock()

_pushing = False
_push_lock = threading.Lock()

_grid_result = {"seq": 0, "quad": None}
_grid_lock = threading.Lock()


def build_fixed_zones(w, h):
    zones = {}
    xs = int(w * PAD_X1)
    xe = int(w * PAD_X2)
    sw = max(1, (xe - xs) // FALLBACK_COLS)
    ty1 = int(h * TOP_Y1)
    ty2 = int(h * TOP_Y2)
    by1 = int(h * BOT_Y1)
    by2 = int(h * BOT_Y2)
    for i in range(FALLBACK_COLS):
        x1 = xs + i * sw
        x2 = x1 + sw
        zones[f"Z{i + 1:02d}"] = rect_to_poly(x1, ty1, x2, ty2)
        zones[f"Z{FALLBACK_COLS + i + 1:02d}"] = rect_to_poly(x1, by1, x2, by2)
    return zones


def rect_to_poly(x1, y1, x2, y2):
    return [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]


def fallback_lot_quad(w, h):
    return np.array(
        [
            [w * 0.08, h * 0.12],
            [w * 0.92, h * 0.12],
            [w * 0.98, h * 0.96],
            [w * 0.02, h * 0.96],
        ],
        dtype=np.float32,
    )


def order_quad(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    quad = np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ],
        dtype=np.float32,
    )
    if cv2.contourArea(quad.astype(np.float32)) <= 0:
        return None
    return quad


def parse_manual_lot_polygon(w, h):
    raw = os.getenv("LOT_POLYGON", "").strip()
    if not raw:
        return None
    pts = []
    for pair in raw.split(";"):
        bits = pair.strip().split(",")
        if len(bits) != 2:
            return None
        try:
            x = float(bits[0])
            y = float(bits[1])
        except ValueError:
            return None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x *= w
            y *= h
        pts.append([x, y])
    return order_quad(pts)


def _mask_vehicle_boxes(mask, boxes, y_offset=0):
    h, w = mask.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1 - LAND_VEHICLE_PAD)
        x2 = min(w - 1, x2 + LAND_VEHICLE_PAD)
        y1 = max(0, y1 - y_offset - LAND_VEHICLE_PAD)
        y2 = min(h - 1, y2 - y_offset + LAND_VEHICLE_PAD)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)


def detect_land_quad(frame, boxes):
    h, w = frame.shape[:2]
    top = int(max(0, min(h - 2, h * LAND_ROI_TOP)))
    roi = frame[top:h]
    if roi.size == 0:
        return None

    valid = np.full(roi.shape[:2], 255, dtype=np.uint8)
    _mask_vehicle_boxes(valid, boxes, y_offset=top)

    lab = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2LAB)
    pixels = lab[valid > 0].reshape(-1, 3).astype(np.float32)
    if len(pixels) < 1000:
        return None

    if len(pixels) > LAND_MAX_SAMPLES:
        idx = np.linspace(0, len(pixels) - 1, LAND_MAX_SAMPLES).astype(np.int32)
        sample = pixels[idx]
    else:
        sample = pixels

    k = min(LAND_CLUSTER_K, len(sample))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, _, centers = cv2.kmeans(
            sample,
            k,
            None,
            criteria,
            2,
            cv2.KMEANS_PP_CENTERS,
        )
    except cv2.error:
        return None

    flat = lab.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dist, axis=1).reshape(roi.shape[:2])
    counts = np.bincount(labels[valid > 0].reshape(-1), minlength=k)
    if counts.size == 0:
        return None

    land_idx = int(np.argmax(counts))
    mask = np.where(labels == land_idx, 255, 0).astype(np.uint8)
    mask[valid == 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < (w * h * LAND_MIN_AREA_PCT):
        return None

    contour = contour.reshape(-1, 2).astype(np.float32)
    contour[:, 1] += top
    hull = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2)
    quad = order_quad(hull)
    if quad is None:
        return None

    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad


def interp_quad(quad, u, v):
    tl, tr, br, bl = quad
    left = tl * (1.0 - v) + bl * v
    right = tr * (1.0 - v) + br * v
    return left * (1.0 - u) + right * u


def grid_shape_for_quad(quad, total):
    if total <= 0:
        return 0, 0
    if AUTO_SLOT_ROWS > 0 and AUTO_SLOT_COLS > 0:
        return AUTO_SLOT_ROWS, AUTO_SLOT_COLS
    if AUTO_SLOT_ROWS > 0:
        return AUTO_SLOT_ROWS, int(math.ceil(total / AUTO_SLOT_ROWS))
    if AUTO_SLOT_COLS > 0:
        return int(math.ceil(total / AUTO_SLOT_COLS)), AUTO_SLOT_COLS

    top_w = np.linalg.norm(quad[1] - quad[0])
    bot_w = np.linalg.norm(quad[2] - quad[3])
    left_h = np.linalg.norm(quad[3] - quad[0])
    right_h = np.linalg.norm(quad[2] - quad[1])
    width = max(1.0, (top_w + bot_w) / 2.0)
    height = max(1.0, (left_h + right_h) / 2.0)
    aspect = max(0.5, min(4.0, width / height))

    rows = max(1, int(round(math.sqrt(total / aspect))))
    cols = int(math.ceil(total / rows))
    return rows, cols


def build_slots_from_quad(quad, total):
    rows, cols = grid_shape_for_quad(quad, total)
    zones = {}
    if rows <= 0 or cols <= 0:
        return zones

    slot_num = 1
    gap_u = min(0.2, SLOT_GAP_FRAC) / cols
    gap_v = min(0.2, SLOT_GAP_FRAC) / rows
    for r in range(rows):
        for c in range(cols):
            if slot_num > total:
                break
            u0 = c / cols + gap_u
            u1 = (c + 1) / cols - gap_u
            v0 = r / rows + gap_v
            v1 = (r + 1) / rows - gap_v
            pts = np.array(
                [
                    interp_quad(quad, u0, v0),
                    interp_quad(quad, u1, v0),
                    interp_quad(quad, u1, v1),
                    interp_quad(quad, u0, v1),
                ]
            )
            zones[f"S{slot_num:02d}"] = np.rint(pts).astype(int).tolist()
            slot_num += 1
    return zones


def blend_quad(old, new, alpha):
    if old is None:
        return new
    return old * (1.0 - alpha) + new * alpha


def zone_polygon(zone):
    pts = np.asarray(zone, dtype=np.float32)
    if pts.shape == (4, 2):
        return pts
    if pts.size == 4:
        x1, y1, x2, y2 = pts.reshape(-1)
        return np.asarray(rect_to_poly(x1, y1, x2, y2), dtype=np.float32)
    return None


def box_center_in_zone(zone, box):
    poly = zone_polygon(zone)
    if poly is None:
        return False
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def compute_polygon_box_iou(zone, box):
    poly = zone_polygon(zone)
    if poly is None:
        return 0.0
    x1, y1, x2, y2 = map(float, box[:4])
    box_poly = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    slot_area = abs(cv2.contourArea(poly))
    box_area = max(0.0, (x2 - x1) * (y2 - y1))
    if slot_area <= 0 or box_area <= 0:
        return 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(poly, box_poly)
    except cv2.error:
        return 0.0
    union = slot_area + box_area - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def zone_occupied(zone_coords, boxes):
    for box in boxes:
        if box_center_in_zone(zone_coords, box):
            return True
        if compute_polygon_box_iou(zone_coords, box) > IOU_THRESH:
            return True
    return False


def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def slots_payload(zones, zone_status):
    payload = []
    for name, pts in zones.items():
        payload.append(
            {
                "id": name,
                "polygon": [[int(x), int(y)] for x, y in pts],
                "occupied": bool(zone_status.get(name, False)),
            }
        )
    return payload


def push_to_backend(occupied, free, total, pct, fps, zone_status, frame, boxes, slots):
    global _pushing
    with _push_lock:
        if _pushing:
            return
        _pushing = True
    try:
        frame_b64 = encode_frame(frame)
        requests.post(
            f"{BACKEND_URL}/yolo/update",
            json={
                "occupied": occupied,
                "free": free,
                "total": total,
                "occupancy_pct": pct,
                "lot_full": free == 0 and total > 0,
                "fps": fps,
                "yolo_count": len(boxes),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_b64": frame_b64,
                "yolo_boxes": [list(map(int, b[:4])) for b in boxes],
                "slots": slots,
                "zones": zone_status,
            },
            headers=HEADERS,
            timeout=2,
        )
    except Exception as e:
        print(f"[push] {e}")
    finally:
        with _push_lock:
            _pushing = False


def yolo_thread(model, frame_queue):
    global _yolo_boxes
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except Exception:
            continue

        res = model(frame, imgsz=IMGSZ, verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            for r in res.boxes:
                conf = float(r.conf[0])
                cls_id = int(r.cls[0]) if r.cls is not None else -1
                if conf < CONF_THRESH or cls_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                boxes.append([x1, y1, x2, y2])

        with _yolo_lock:
            _yolo_boxes = boxes


def land_grid_thread(grid_queue):
    while True:
        try:
            frame, boxes = grid_queue.get(timeout=1)
        except Exception:
            continue

        quad = detect_land_quad(frame, boxes)
        with _grid_lock:
            _grid_result["seq"] += 1
            _grid_result["quad"] = quad


def load_yolo_model(path):
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install it in the Python environment "
            "used to run the detector, e.g. .\\.venv311\\Scripts\\python.exe -m pip install ultralytics"
        ) from exc
    return ultralytics.YOLO(path)


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--occupaiframe")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        while True:
            try:
                with _stream_lock:
                    frame = _stream_frame
                if frame is None:
                    time.sleep(0.033)
                    continue
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                jpg = buf.tobytes()
                self.wfile.write(
                    b"--occupaiframe\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(jpg)).encode()
                    + b"\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )
                self.wfile.flush()
                time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break


def start_mjpeg_server():
    HTTPServer(("0.0.0.0", STREAM_PORT), MJPEGHandler).serve_forever()


def draw_lot_outline(frame, quad, source):
    if quad is None:
        return frame
    pts = np.rint(quad).astype(np.int32)
    cv2.polylines(frame, [pts], True, (255, 180, 60), 2)
    label = f"LOT: {source.upper()}"
    x = int(np.min(pts[:, 0]))
    y = max(18, int(np.min(pts[:, 1])) - 8)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 60), 1)
    return frame


def draw_zones(frame, zones, zone_status):
    overlay = frame.copy()
    zone_draw = []
    for name, pts in zones.items():
        poly = np.asarray(pts, dtype=np.int32)
        occ = zone_status.get(name, False)
        color = (0, 50, 220) if occ else (0, 200, 80)
        cv2.fillPoly(overlay, [poly], color)
        zone_draw.append((name, poly, color, occ))

    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    for name, poly, color, occ in zone_draw:
        cv2.polylines(frame, [poly], True, color, 1)

        if not DRAW_SLOT_LABELS:
            continue

        center = poly.mean(axis=0).astype(int)
        cv2.putText(frame, name, (center[0] - 16, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        cv2.putText(
            frame,
            "OCC" if occ else "FREE",
            (center[0] - 18, center[1] + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
        )
    return frame


def draw_boxes(frame, boxes):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 200, 255), -1)
    return frame


def draw_hud(frame, free, occupied, total, fps, grid_source):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"OccupAI | Free:{free}  Occ:{occupied}  Tot:{total}  FPS:{fps:.1f}  Grid:{grid_source}",
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 229, 160),
        1,
    )
    return frame


def detection_loop():
    global _stream_frame

    print("Loading YOLO...")
    model = load_yolo_model(MODEL_PATH)
    model(np.zeros((FEED_H, FEED_W, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)
    print(f"YOLO ready | model={MODEL_PATH} | imgsz={IMGSZ} | feed={FEED_W}x{FEED_H}")

    frame_queue = queue.Queue(maxsize=1)
    threading.Thread(target=yolo_thread, args=(model, frame_queue), daemon=True, name="yolo-infer").start()

    print(f"Opening camera {WEBCAM_IDX}...")
    cap = cv2.VideoCapture(WEBCAM_IDX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FEED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FEED_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.8)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {WEBCAM_IDX}")
        print("Set WEBCAM_INDEX=<n> in .env for the camera you want to use.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manual_quad = parse_manual_lot_polygon(actual_w, actual_h)
    lot_quad = manual_quad if manual_quad is not None else fallback_lot_quad(actual_w, actual_h)
    zones = build_slots_from_quad(lot_quad, AUTO_SLOT_TOTAL) if AUTO_GRID else build_fixed_zones(actual_w, actual_h)
    grid_source = "manual" if manual_quad is not None else ("auto-init" if AUTO_GRID else "fixed")
    last_grid_update = -AUTO_GRID_REFRESH_FRAMES
    last_applied_grid_seq = 0
    grid_queue = None

    if AUTO_GRID and manual_quad is None:
        grid_queue = queue.Queue(maxsize=1)
        threading.Thread(target=land_grid_thread, args=(grid_queue,), daemon=True, name="land-grid").start()

    print(f"Camera {WEBCAM_IDX}: {actual_w}x{actual_h}")
    print(f"Virtual slots: {len(zones)} | AUTO_GRID={AUTO_GRID} | source={grid_source}")
    print(f"Vehicle class IDs: {sorted(VEHICLE_CLASS_IDS)}")
    print(f"Backend: {BACKEND_URL} | stream: http://localhost:{STREAM_PORT}/stream")
    print("Ctrl+C to stop\n")

    frame_idx = 0
    fps_t = time.time()
    fps_n = 0
    fps_val = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame_idx += 1
        fps_n += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps_val = fps_n / (now - fps_t)
            fps_n = 0
            fps_t = now

        if frame_queue.empty():
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        with _yolo_lock:
            boxes = list(_yolo_boxes)

        if grid_queue is not None and frame_idx - last_grid_update >= AUTO_GRID_REFRESH_FRAMES:
            try:
                grid_queue.put_nowait((frame.copy(), list(boxes)))
            except queue.Full:
                pass
            last_grid_update = frame_idx

        if grid_queue is not None:
            with _grid_lock:
                grid_seq = _grid_result["seq"]
                detected_quad = _grid_result["quad"]
                if detected_quad is not None:
                    detected_quad = detected_quad.copy()

            if grid_seq != last_applied_grid_seq:
                last_applied_grid_seq = grid_seq
            else:
                detected_quad = None

            if detected_quad is not None:
                lot_quad = blend_quad(lot_quad, detected_quad, GRID_SMOOTHING)
                zones = build_slots_from_quad(lot_quad, AUTO_SLOT_TOTAL)
                grid_source = "auto"
            elif grid_source == "auto-init":
                grid_source = "fallback"

        zone_status = {name: zone_occupied(coords, boxes) for name, coords in zones.items()}
        occupied = int(sum(zone_status.values()))
        total = len(zones)
        free = max(0, total - occupied)
        pct = round(occupied / total * 100, 1) if total else 0.0

        annotated = frame.copy()
        annotated = draw_lot_outline(annotated, lot_quad, grid_source)
        annotated = draw_zones(annotated, zones, zone_status)
        annotated = draw_boxes(annotated, boxes)
        annotated = draw_hud(annotated, free, occupied, total, fps_val, grid_source)

        with _stream_lock:
            _stream_frame = annotated

        if frame_idx % PUSH_EVERY == 0:
            with _push_lock:
                push_busy = _pushing
            if not push_busy:
                slot_data = slots_payload(zones, zone_status)
                threading.Thread(
                    target=push_to_backend,
                    args=(occupied, free, total, pct, round(fps_val, 1), zone_status, annotated, boxes, slot_data),
                    daemon=True,
                ).start()

    cap.release()


if __name__ == "__main__":
    print("\nOccupAI Detector")
    print(f"Camera index: {WEBCAM_IDX}")
    print(f"MJPEG stream: http://localhost:{STREAM_PORT}/stream\n")
    threading.Thread(target=start_mjpeg_server, daemon=True).start()
    detection_loop()
