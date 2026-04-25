"""
yolo_service/detector.py
Auto grid detection — 4 top + 4 bottom = 8 slots
Camera: top-down bird's eye view
"""
import cv2
import numpy as np
import threading
import time
import base64
import os
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL  = os.getenv("BACKEND_URL", "http://localhost:8000")
CAM_TOKEN    = os.getenv("CAM_TOKEN",   "occupai_cam_2027")
WEBCAM_IDX   = int(os.getenv("WEBCAM_INDEX", "0"))
STREAM_PORT  = int(os.getenv("STREAM_PORT", "8001"))

FEED_W       = 480
FEED_H       = 360

IMGSZ        = 256
YOLO_SKIP    = 2
JPEG_Q       = 50
PUSH_EVERY   = 10

# Detect any object that could be a toy/model car
# cls 2=car, 3=motorcycle, 5=bus, 7=truck
# For diorama toy cars, we also lower confidence since toys look different
VEHICLE_CLS  = {2, 3, 5, 7}
CONF_THRESH  = 0.20          # lower = catches toy cars better
IOU_THRESH   = 0.20          # how much overlap = zone is occupied

HEADERS = {"x-cam-token": CAM_TOKEN, "Content-Type": "application/json"}

# ══════════════════════════════════════════════════════════════════════════════
#   GRID CONFIG — 4 top + 4 bottom with center aisle
#
#   Frame layout (top-down view):
#
#   y=0  ┌─────────────────────────────────┐
#        │  TOP ROW:  Z1 | Z2 | Z3 | Z4   │  top_y1 → top_y2
#        ├─────────────────────────────────┤
#        │         CENTER AISLE            │  aisle (ignored)
#        ├─────────────────────────────────┤
#        │  BOT ROW:  Z5 | Z6 | Z7 | Z8   │  bot_y1 → bot_y2
#   y=H  └─────────────────────────────────┘
#
#   Adjust these percentages to match your diorama camera view:
# ══════════════════════════════════════════════════════════════════════════════

# Vertical zone boundaries (as fraction of frame height)
TOP_Y1   = 0.05   # top row starts  at  5% from top
TOP_Y2   = 0.42   # top row ends    at 42% from top
BOT_Y1   = 0.58   # bottom row starts at 58% from top
BOT_Y2   = 0.95   # bottom row ends   at 95% from top

# Horizontal padding (as fraction of frame width)
PAD_X1   = 0.02   # left  padding
PAD_X2   = 0.98   # right padding

COLS     = 4      # number of slots per row


def build_zones(w, h):
    """
    Build 8 zone rectangles from frame dimensions.
    Returns dict: { "Z1": (x1,y1,x2,y2), ... "Z8": ... }
    """
    zones = {}
    x_start = int(w * PAD_X1)
    x_end   = int(w * PAD_X2)
    slot_w  = (x_end - x_start) // COLS

    # Top row — Z1..Z4
    ty1 = int(h * TOP_Y1)
    ty2 = int(h * TOP_Y2)
    for i in range(COLS):
        x1 = x_start + i * slot_w
        x2 = x1 + slot_w
        zones[f"Z{i+1}"] = (x1, ty1, x2, ty2)

    # Bottom row — Z5..Z8
    by1 = int(h * BOT_Y1)
    by2 = int(h * BOT_Y2)
    for i in range(COLS):
        x1 = x_start + i * slot_w
        x2 = x1 + slot_w
        zones[f"Z{COLS+i+1}"] = (x1, by1, x2, by2)

    return zones


# ── Shared frame for MJPEG stream ─────────────────────────────────────────────
_stream_frame = None
_stream_lock  = threading.Lock()

# ── Push guard ────────────────────────────────────────────────────────────────
_pushing   = False
_push_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#   MJPEG SERVER
# ══════════════════════════════════════════════════════════════════════════════
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != '/stream':
            self.send_response(404); self.end_headers(); return

        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=--occupaiframe')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        while True:
            try:
                with _stream_lock:
                    frame = _stream_frame
                if frame is None:
                    time.sleep(0.05); continue

                _, buf = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                jpg = buf.tobytes()
                self.wfile.write(
                    b'--occupaiframe\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' +
                    jpg + b'\r\n'
                )
                self.wfile.flush()
                time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break


def start_mjpeg_server():
    server = HTTPServer(('0.0.0.0', STREAM_PORT), MJPEGHandler)
    print(f"MJPEG stream → http://localhost:{STREAM_PORT}/stream")
    server.serve_forever()


# ══════════════════════════════════════════════════════════════════════════════
#   HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter: return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def box_center_in_zone(zone, box):
    """Check if the CENTER of a detected box falls inside the zone."""
    zx1, zy1, zx2, zy2 = zone
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def zone_occupied(zone_coords, boxes):
    """
    Zone is occupied if:
    - Any box center falls inside the zone, OR
    - IoU overlap exceeds threshold
    Using center check as primary (better for top-down)
    """
    for box in boxes:
        if box_center_in_zone(zone_coords, box):
            return True
        if compute_iou(zone_coords, tuple(box)) > IOU_THRESH:
            return True
    return False


def encode_frame(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def push_to_backend(occupied, free, total, pct, fps, zones, frame_b64):
    global _pushing
    with _push_lock:
        if _pushing: return
        _pushing = True
    try:
        requests.post(f"{BACKEND_URL}/yolo/update",
            json={
                "occupied":      occupied,
                "free":          free,
                "total":         total,
                "occupancy_pct": pct,
                "lot_full":      total > 0 and free == 0,
                "fps":           fps,
                "yolo_count":    occupied,
                "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_b64":  frame_b64,
                "yolo_boxes":    [],
                "slots":         [],
                "zones":         zones,
            },
            headers=HEADERS, timeout=3)
    except Exception as e:
        print(f"[push] {e}")
    with _push_lock:
        _pushing = False


# ══════════════════════════════════════════════════════════════════════════════
#   DRAW ZONES
# ══════════════════════════════════════════════════════════════════════════════
def draw_zones(frame, zones, zone_status):
    """Draw grid zones with color overlay — green=free, red=occupied."""
    for name, (zx1, zy1, zx2, zy2) in zones.items():
        occ   = zone_status.get(name, False)
        color = (0, 50, 200) if occ else (0, 200, 80)

        # subtle fill overlay
        ov = frame.copy()
        cv2.rectangle(ov, (zx1, zy1), (zx2, zy2), color, -1)
        cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)

        # border
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)

        # zone label — centered
        label  = f"{name}"
        status = "OCC" if occ else "FREE"
        lx = zx1 + (zx2 - zx1) // 2 - 15
        ly = zy1 + (zy2 - zy1) // 2

        cv2.putText(frame, label,  (lx, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, status, (lx, ly + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # draw aisle label
    aisle_y = int(frame.shape[0] * ((TOP_Y2 + BOT_Y1) / 2))
    cv2.putText(frame, "── AISLE ──",
                (frame.shape[1]//2 - 50, aisle_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    return frame


def draw_boxes(frame, yolo_boxes):
    """Draw raw YOLO detection boxes in yellow."""
    for (x1, y1, x2, y2) in yolo_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
        # draw center dot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)
    return frame


def draw_hud(frame, free, occupied, total, fps):
    """Draw top HUD bar."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame,
        f"OccupAI | Free:{free}  Occ:{occupied}  Tot:{total}  FPS:{fps:.1f}",
        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 229, 160), 1)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#   DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def detection_loop():
    global _stream_frame

    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    model(np.zeros((FEED_H, FEED_W, 3), dtype=np.uint8),
          imgsz=IMGSZ, verbose=False)
    print(f"YOLO ready — {IMGSZ}px  |  feed {FEED_W}x{FEED_H}")

    cap = cv2.VideoCapture(WEBCAM_IDX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FEED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FEED_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(1.0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam", WEBCAM_IDX); return

    # Build zones from actual frame size
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zones    = build_zones(actual_w, actual_h)

    print(f"Camera: {actual_w}x{actual_h}")
    print(f"Zones built: {list(zones.keys())}")
    print(f"Backend: {BACKEND_URL}")
    print("Press Ctrl+C to stop\n")

    frame_idx  = 0
    yolo_boxes = []
    fps_t = time.time(); fps_n = 0; fps_val = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005); continue

        if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
            frame = cv2.resize(frame, (actual_w, actual_h))

        frame_idx += 1
        fps_n     += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps_val = fps_n / (now - fps_t)
            fps_n   = 0; fps_t = now

        # ── YOLO inference every YOLO_SKIP frames ────────────────────────────
        if frame_idx % YOLO_SKIP == 0:
            res = model(frame, imgsz=IMGSZ, verbose=False)[0]
            yolo_boxes = []
            if res.boxes is not None:
                for r in res.boxes:
                    cls  = int(r.cls[0])
                    conf = float(r.conf[0])
                    # For diorama: accept ALL objects with decent confidence
                    # since toy cars may not be classified as vehicles
                    if conf > CONF_THRESH:
                        x1, y1, x2, y2 = map(int, r.xyxy[0])
                        yolo_boxes.append([x1, y1, x2, y2])

        # ── Zone status ───────────────────────────────────────────────────────
        zone_status = {n: zone_occupied(c, yolo_boxes) for n, c in zones.items()}
        occupied    = sum(zone_status.values())
        total       = len(zones)
        free        = total - occupied
        pct         = round(occupied / total * 100, 1) if total else 0.0

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = frame.copy()
        annotated = draw_zones(annotated, zones, zone_status)
        annotated = draw_boxes(annotated, yolo_boxes)
        annotated = draw_hud(annotated, free, occupied, total, fps_val)

        # ── Update MJPEG stream ───────────────────────────────────────────────
        with _stream_lock:
            _stream_frame = annotated.copy()

        # ── Push to backend every PUSH_EVERY frames ───────────────────────────
        if frame_idx % PUSH_EVERY == 0:
            fb64 = encode_frame(annotated)
            threading.Thread(
                target=push_to_backend,
                args=(occupied, free, total, pct,
                      round(fps_val, 1), zone_status, fb64),
                daemon=True
            ).start()

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n╔══════════════════════════════════════════╗")
    print("║  OccupAI Detector — 8 slot grid (4x2)   ║")
    print(f"║  MJPEG stream → http://localhost:{STREAM_PORT}/stream  ║")
    print("╚══════════════════════════════════════════╝\n")

    threading.Thread(target=start_mjpeg_server, daemon=True).start()
    detection_loop()