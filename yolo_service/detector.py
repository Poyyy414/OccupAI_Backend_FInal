"""
yolo_service/detector.py — OccupAI High-FPS Detector
=====================================================
FIXES:
  • High FPS: YOLO runs on separate thread, main loop never blocks
  • Zone status pushed correctly as {Z1: true/false} dict
  • WEBCAM_INDEX from .env (run find_camera.py to find your index)
  • Reduced JPEG quality slightly for faster encoding
  • PUSH_EVERY reduced to push stats more often

INSTALL (run once):
  pip install opencv-python ultralytics requests python-dotenv

SETUP:
  1. Add to .env:  WEBCAM_INDEX=0  (or 1, 2 — whichever is your webcam)
  2. Run: python -m yolo_service.detector
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
BACKEND_URL  = os.getenv("BACKEND_URL",    "http://localhost:8000")
CAM_TOKEN    = os.getenv("CAM_TOKEN",      "occupai_cam_2027")
STREAM_PORT  = int(os.getenv("STREAM_PORT",  "8001"))
WEBCAM_IDX   = int(os.getenv("WEBCAM_INDEX", "0"))

FEED_W       = 640    # wider for better detection
FEED_H       = 480
IMGSZ        = 320    # larger = more accurate, smaller = faster; 320 is good balance
JPEG_Q       = 60
PUSH_EVERY   = 5      # push stats every 5 frames (~6x/sec at 30fps)

CONF_THRESH  = 0.20
IOU_THRESH   = 0.20

HEADERS = {"x-cam-token": CAM_TOKEN, "Content-Type": "application/json"}

# ── Grid (fraction of frame) ──────────────────────────────────────────────────
TOP_Y1 = 0.05;  TOP_Y2 = 0.42
BOT_Y1 = 0.58;  BOT_Y2 = 0.95
PAD_X1 = 0.02;  PAD_X2 = 0.98
COLS   = 4


def build_zones(w, h):
    zones = {}
    xs = int(w*PAD_X1);  xe = int(w*PAD_X2);  sw = (xe-xs)//COLS
    ty1 = int(h*TOP_Y1); ty2 = int(h*TOP_Y2)
    by1 = int(h*BOT_Y1); by2 = int(h*BOT_Y2)
    for i in range(COLS):
        x1 = xs + i*sw
        zones[f"Z{i+1}"]      = (x1, ty1, x1+sw, ty2)
        zones[f"Z{COLS+i+1}"] = (x1, by1, x1+sw, by2)
    return zones


# ── Shared state ──────────────────────────────────────────────────────────────
_stream_frame  = None
_stream_lock   = threading.Lock()

# YOLO runs on its own thread — results shared here
_yolo_boxes    = []
_yolo_lock     = threading.Lock()

_pushing       = False
_push_lock     = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO inference thread — runs continuously, updates _yolo_boxes
#  This keeps the main capture loop at full camera FPS
# ══════════════════════════════════════════════════════════════════════════════
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
                if float(r.conf[0]) > CONF_THRESH:
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    boxes.append([x1, y1, x2, y2])
        with _yolo_lock:
            _yolo_boxes = boxes


# ══════════════════════════════════════════════════════════════════════════════
#  MJPEG server
# ══════════════════════════════════════════════════════════════════════════════
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
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
                    time.sleep(0.033); continue
                _, buf = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                jpg = buf.tobytes()
                self.wfile.write(
                    b'--occupaiframe\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpg)).encode() +
                    b'\r\n\r\n' + jpg + b'\r\n')
                self.wfile.flush()
                time.sleep(0.033)    # ~30fps to browser
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break


def start_mjpeg_server():
    HTTPServer(('0.0.0.0', STREAM_PORT), MJPEGHandler).serve_forever()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def box_center_in_zone(zone, box):
    zx1,zy1,zx2,zy2 = zone
    cx = (box[0]+box[2])/2;  cy = (box[1]+box[3])/2
    return zx1<=cx<=zx2 and zy1<=cy<=zy2

def compute_iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0

def zone_occupied(zone_coords, boxes):
    for box in boxes:
        if box_center_in_zone(zone_coords, box): return True
        if compute_iou(zone_coords, tuple(box)) > IOU_THRESH: return True
    return False

def encode_frame(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return base64.b64encode(buf.tobytes()).decode('utf-8')

def push_to_backend(occupied, free, total, pct, fps, zone_status, frame_b64):
    global _pushing
    with _push_lock:
        if _pushing: return
        _pushing = True
    try:
        requests.post(f"{BACKEND_URL}/yolo/update", json={
            "occupied":      occupied,
            "free":          free,
            "total":         total,
            "occupancy_pct": pct,
            "lot_full":      free == 0 and total > 0,
            "fps":           fps,
            "yolo_count":    occupied,
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_b64":  frame_b64,
            "yolo_boxes":    [],
            "slots":         [],
            # FIX: zones must be {name: bool} — e.g. {"Z1": true, "Z2": false}
            "zones":         zone_status,
        }, headers=HEADERS, timeout=2)
    except Exception as e:
        print(f"[push] {e}")
    finally:
        with _push_lock:
            _pushing = False


# ══════════════════════════════════════════════════════════════════════════════
#  Draw
# ══════════════════════════════════════════════════════════════════════════════
def draw_zones(frame, zones, zone_status):
    for name,(zx1,zy1,zx2,zy2) in zones.items():
        occ   = zone_status.get(name, False)
        color = (0,50,220) if occ else (0,200,80)
        ov    = frame.copy()
        cv2.rectangle(ov,(zx1,zy1),(zx2,zy2),color,-1)
        cv2.addWeighted(ov,0.22,frame,0.78,0,frame)
        cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),color,2)
        lx = zx1+(zx2-zx1)//2-15
        ly = zy1+(zy2-zy1)//2
        cv2.putText(frame,name,(lx,ly-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
        cv2.putText(frame,"OCC" if occ else "FREE",(lx,ly+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1)
    # Aisle label
    ay = int(frame.shape[0]*((TOP_Y2+BOT_Y1)/2))
    cv2.putText(frame,"── AISLE ──",(frame.shape[1]//2-50,ay),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,80,80),1)
    return frame

def draw_boxes(frame, boxes):
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,255),1)
        cv2.circle(frame,((x1+x2)//2,(y1+y2)//2),3,(0,200,255),-1)
    return frame

def draw_hud(frame, free, occupied, total, fps):
    cv2.rectangle(frame,(0,0),(frame.shape[1],28),(0,0,0),-1)
    cv2.putText(frame,
        f"OccupAI | Free:{free}  Occ:{occupied}  Tot:{total}  FPS:{fps:.1f}",
        (6,18),cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,229,160),1)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  Main detection loop — camera runs at full FPS, YOLO on separate thread
# ══════════════════════════════════════════════════════════════════════════════
def detection_loop():
    global _stream_frame

    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    # Warmup
    model(np.zeros((FEED_H,FEED_W,3),dtype=np.uint8),imgsz=IMGSZ,verbose=False)
    print(f"YOLO ready | imgsz={IMGSZ} | feed={FEED_W}x{FEED_H}")

    # Frame queue for YOLO thread (maxsize=1 → always process latest frame)
    import queue
    fq = queue.Queue(maxsize=1)
    threading.Thread(target=yolo_thread, args=(model,fq), daemon=True,
                     name="yolo-infer").start()

    print(f"Opening camera {WEBCAM_IDX}...")
    cap = cv2.VideoCapture(WEBCAM_IDX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FEED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FEED_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    time.sleep(0.8)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {WEBCAM_IDX}")
        print("  → Run find_camera.py to find the right index")
        print("  → Set WEBCAM_INDEX=<n> in your .env")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zones    = build_zones(actual_w, actual_h)
    print(f"Camera {WEBCAM_IDX}: {actual_w}x{actual_h} | zones: {list(zones.keys())}")
    print(f"Backend: {BACKEND_URL} | Ctrl+C to stop\n")

    frame_idx  = 0
    fps_t = time.time(); fps_n = 0; fps_val = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005); continue

        frame_idx += 1
        fps_n     += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps_val = fps_n / (now - fps_t)
            fps_n   = 0; fps_t = now

        # Feed latest frame to YOLO thread (non-blocking — drop if busy)
        try:
            fq.put_nowait(frame.copy())
        except Exception:
            pass   # YOLO still processing last frame — skip this one

        # Get current YOLO results (always available from last inference)
        with _yolo_lock:
            boxes = list(_yolo_boxes)

        # Zone status
        zone_status = {n: zone_occupied(c, boxes) for n, c in zones.items()}
        occupied    = sum(zone_status.values())
        total       = len(zones)
        free        = total - occupied
        pct         = round(occupied/total*100, 1) if total else 0.0

        # Annotate and update stream
        annotated = frame.copy()
        annotated = draw_zones(annotated, zones, zone_status)
        annotated = draw_boxes(annotated, boxes)
        annotated = draw_hud(annotated, free, occupied, total, fps_val)

        with _stream_lock:
            _stream_frame = annotated

        # Push stats to backend
        if frame_idx % PUSH_EVERY == 0:
            fb64 = encode_frame(annotated)
            threading.Thread(
                target=push_to_backend,
                args=(occupied,free,total,pct,round(fps_val,1),zone_status,fb64),
                daemon=True
            ).start()

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n╔══════════════════════════════════════════╗")
    print(f"║  OccupAI Detector  (cam={WEBCAM_IDX})              ║")
    print(f"║  MJPEG → http://localhost:{STREAM_PORT}/stream   ║")
    print("╚══════════════════════════════════════════╝\n")
    threading.Thread(target=start_mjpeg_server, daemon=True).start()
    detection_loop()