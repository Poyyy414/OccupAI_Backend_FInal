"""
yolo_service/detector.py  — OccupAI v7.2
=========================================
Fixes in v7.2:
  - Watches slot_state.check_and_clear_bg_reset() every frame
    → re-warms MOG2 in background when layout changes (demand switch)
    → fixes all-red / false OCC after BUSY/HIGH layout kicks in
  - EXCLUDED_SLOTS re-read live from .env every frame via slot_state
  - HUD shows demand label immediately after first adjuster cycle
  - All R1/R2/R3 layout from .env (no hardcoded positions)
"""

import cv2
import numpy as np
import threading
import time
import base64
import os
import sys
import queue
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.slot_adjuster import (SlotState, SlotAdjusterThread,
                                   DemandLevel, build_layout)


# ══════════════════════════════════════════════════════════════════════════════
#   ENV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ei(k, d):
    try: return int(os.getenv(k, str(d)))
    except: return d

def _ef(k, d):
    try: return float(os.getenv(k, str(d)))
    except: return d

def _eb(k, d):
    return os.getenv(k, str(d)).strip().lower() not in {"0","false","no","off"}

def _es(k, d=""):
    raw = os.getenv(k, d) or ""
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


# ══════════════════════════════════════════════════════════════════════════════
#   CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CAM_TOKEN   = os.getenv("CAM_TOKEN",   "occupai_cam_2027")
WEBCAM_IDX  = _ei("WEBCAM_INDEX", 0)
STREAM_PORT = _ei("STREAM_PORT",  8001)
STREAM_FPS  = max(1, _ei("STREAM_FPS", 20))
MODELS_DIR  = os.getenv("MODELS_DIR",
              os.path.join(os.path.dirname(__file__), '..', 'backend', 'models'))

FEED_W = 640
FEED_H = 480
CAMERA_FPS  = max(1, _ei("CAMERA_FPS", 30))
IMGSZ       = _ei("IMGSZ",      224)
YOLO_SKIP   = _ei("YOLO_SKIP",    8)
YOLO_ASYNC  = _eb("YOLO_ASYNC", True)
YOLO_INTERVAL = max(0.05, _ef("YOLO_INTERVAL", 0.35))
JPEG_Q      = _ei("JPEG_Q",      42)
PUSH_EVERY  = _ei("PUSH_EVERY",  60)
CONF_THRESH = _ef("CONF_THRESH", 0.15)
IOU_THRESH  = _ef("IOU_THRESH",  0.15)
DRAW_DETECTOR_BOXES = _eb("DRAW_DETECTOR_BOXES", False)

SLOT_CORNER_RADIUS = max(0, _ei("SLOT_CORNER_RADIUS", 14))
EXCLUDED_SLOTS     = _es("EXCLUDED_SLOTS")

BG_HISTORY     = _ei("BG_HISTORY",    200)
BG_VAR_THRESH  = _ei("BG_THRESH",      30)
BG_LEARN_RATE  = _ef("BG_LEARN_RATE", 0.002)
OCC_DIFF_RATIO = _ef("OCC_DIFF_RATIO", 0.07)

TOY_MIN_F = _ef("TOY_MIN_AREA_FRAC", 0.0008)
TOY_MAX_F = _ef("TOY_MAX_AREA_FRAC", 0.10)
TOY_COLOR_RANGES = [
    (np.array([35,  100,  80]), np.array([85,  255, 255])),
    (np.array([90,  100,  80]), np.array([130, 255, 255])),
    (np.array([15,  100, 100]), np.array([35,  255, 255])),
    (np.array([0,   130, 100]), np.array([15,  255, 255])),
    (np.array([160, 130, 100]), np.array([180, 255, 255])),
]

STATIC_N          = _ei("STATIC_CHECK_EVERY",    35)
STATIC_THRESH     = _ef("STATIC_DIFF_THRESH",    3.5)
REDETECT_INTERVAL = _ei("REDETECT_INTERVAL",      45)
REDETECT_THRESH   = _ef("REDETECT_SCENE_THRESH", 45.0)
EDGE_MARGIN_F     = _ef("EDGE_MARGIN_F",          0.18)
MIN_SLOTS         = _ei("MIN_SLOTS", 4)

# MOG2 warmup frames when re-warming after layout change
BG_REWARM_N = _ei("BG_REWARM_N", 20)

HEADERS = {"x-cam-token": CAM_TOKEN, "Content-Type": "application/json"}

_stream_frame = None
_stream_lock  = threading.Lock()
_pushing      = False
_push_lock    = threading.Lock()
_cam_q: queue.Queue = queue.Queue(maxsize=1)
_cam_ok      = True
_cam_ok_lock = threading.Lock()

slot_state = SlotState()


# ══════════════════════════════════════════════════════════════════════════════
#   DB CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_db_history():
    try:
        r = requests.get(f"{BACKEND_URL}/api/parking_logs_recent",
                         headers=HEADERS, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[db_fetch] {e}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
#   BACKGROUND CAPTURE + REWARM
# ══════════════════════════════════════════════════════════════════════════════

def grab_background(cap, bg_sub, n=30):
    """Capture n frames to warm up MOG2. Returns averaged background image."""
    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if ret and f is not None:
            frames.append(f.astype(np.float32))
            bg_sub.apply(f, learningRate=1.0)
        time.sleep(0.04)
    if not frames:
        _, f = cap.read()
        return f
    return np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)


def rewarm_bg_sub(cap, bg_sub, n=20):
    """
    Re-warm MOG2 in-place after layout change.
    Called in a background thread — doesn't block detection.
    Resets the model so it learns the new 'empty' background.
    """
    print(f"[bg-rewarm] Re-warming MOG2 with {n} frames...")
    bg_sub.__init__()   # reset MOG2 state
    # Re-create is safer than __init__ on OpenCV objects:
    # caller should replace bg_sub with a new instance
    for _ in range(n):
        ret, f = cap.read()
        if ret and f is not None:
            bg_sub.apply(f, learningRate=1.0)
        time.sleep(0.04)
    print("[bg-rewarm] ✓ MOG2 re-warmed")


# ══════════════════════════════════════════════════════════════════════════════
#   DEBUG IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def _save_debug_zones(frame_or_bg, slots):
    dbg = frame_or_bg.copy() if frame_or_bg is not None else \
          np.zeros((FEED_H, FEED_W, 3), dtype=np.uint8)
    for i, (name, (x1,y1,x2,y2)) in enumerate(slots.items()):
        col = _COLORS[i % len(_COLORS)]
        _draw_rounded_rect(dbg, x1, y1, x2, y2, col, 2)
        cv2.putText(dbg, name, (x1+3, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    cv2.imwrite("debug_zones.jpg", dbg)


# ══════════════════════════════════════════════════════════════════════════════
#   CAMERA HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def is_blocked(frame, prev_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None: return False, gray
    diff = float(np.mean(np.abs(
        gray.astype(np.float32) - prev_gray.astype(np.float32))))
    return diff < STATIC_THRESH, gray


def edge_region_diff(fg, rg):
    h, w = fg.shape
    my = int(h * EDGE_MARGIN_F); mx = int(w * EDGE_MARGIN_F)
    def strips(g):
        return np.concatenate([g[:my,:].ravel(), g[-my:,:].ravel(),
                                g[my:-my,:mx].ravel(), g[my:-my,-mx:].ravel()])
    return float(np.mean(np.abs(
        strips(fg).astype(np.float32) - strips(rg).astype(np.float32))))


# ══════════════════════════════════════════════════════════════════════════════
#   OCCUPANCY
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua else 0.0

def occ_bg(zone, fg):
    x1,y1,x2,y2 = zone
    roi = fg[y1:y2, x1:x2]
    if roi.size == 0: return False
    return np.count_nonzero(roi)/roi.size >= OCC_DIFF_RATIO

def occ_yolo(zone, boxes):
    x1,y1,x2,y2 = zone
    for b in boxes:
        cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
        if x1<=cx<=x2 and y1<=cy<=y2: return True
        if _iou(zone,tuple(b)) > IOU_THRESH: return True
    return False

def find_toy_boxes(frame):
    h,w=frame.shape[:2]; fa=h*w
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=np.zeros((h,w),dtype=np.uint8)
    for lo,hi in TOY_COLOR_RANGES:
        mask=cv2.bitwise_or(mask,cv2.inRange(hsv,lo,hi))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, k,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for cnt in cnts:
        ca=cv2.contourArea(cnt)
        if not(fa*TOY_MIN_F<ca<fa*TOY_MAX_F): continue
        x,y,bw,bh=cv2.boundingRect(cnt)
        if 0.20<bw/max(bh,1)<5.0:
            boxes.append([x,y,x+bw,y+bh])
    return boxes

def yolo_boxes(model, frame):
    res=model(frame,imgsz=IMGSZ,verbose=False)[0]
    out=[]
    if res.boxes is not None:
        for r in res.boxes:
            if float(r.conf[0])>CONF_THRESH:
                x1,y1,x2,y2=map(int,r.xyxy[0])
                out.append([x1,y1,x2,y2])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#   DRAW
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = [
    (0,210,80),(0,185,235),(200,100,255),(255,175,0),(0,130,255),
    (255,200,0),(0,230,205),(255,100,155),(80,255,80),(255,80,80),
    (80,200,255),(255,255,80),(180,100,255),(100,255,180),(255,140,60),
    (60,140,255),(200,255,60),(255,60,200),(60,255,200),(200,60,255),
]
_OCC_COLOR = (30,30,220)
_NAME_CI   = {}
_DEMAND_COLORS = {
    DemandLevel.LOW:    (0,200,100),
    DemandLevel.NORMAL: (0,229,160),
    DemandLevel.BUSY:   (0,180,255),
    DemandLevel.HIGH:   (0,100,255),
}


def _draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=1, fill=False):
    r = min(SLOT_CORNER_RADIUS, max(0,(x2-x1)//2), max(0,(y2-y1)//2))
    if r <= 1:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,-1 if fill else thickness)
        return
    if fill:
        cv2.rectangle(img,(x1+r,y1),(x2-r,y2),color,-1)
        cv2.rectangle(img,(x1,y1+r),(x2,y2-r),color,-1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img,(cx,cy),r,color,-1)
        return
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),  0,0,90,color,thickness)
    cv2.ellipse(img,(x1+r,y2-r),(r,r), 90,0,90,color,thickness)


def _slot_color(name, occ):
    if occ: return _OCC_COLOR
    if name not in _NAME_CI:
        _NAME_CI[name] = len(_NAME_CI) % len(_COLORS)
    return _COLORS[_NAME_CI[name]]


def draw_zones(frame, zones, zone_status):
    overlay = frame.copy()
    drawn = []
    for name,(x1,y1,x2,y2) in zones.items():
        occ = zone_status.get(name, False)
        col = _slot_color(name, occ)
        _draw_rounded_rect(overlay,x1,y1,x2,y2,col,fill=True)
        drawn.append((name,x1,y1,x2,y2,occ,col))

    if drawn:
        cv2.addWeighted(overlay,0.14,frame,0.86,0,frame)

    for name,x1,y1,x2,y2,occ,col in drawn:
        zw,zh = x2-x1, y2-y1
        _draw_rounded_rect(frame,x1,y1,x2,y2,col,3 if occ else 2)
        fs = max(0.22, min(0.42, zw/130))
        cx,cy = x1+zw//2, y1+zh//2
        for txt,dy in [(name,-7),("OCC" if occ else "FREE",+9)]:
            (tw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,fs,1)
            tx=cx-tw//2
            cv2.putText(frame,txt,(tx,cy+dy),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),2)
            cv2.putText(frame,txt,(tx,cy+dy),cv2.FONT_HERSHEY_SIMPLEX,fs,col,1)
    return frame


def draw_boxes(frame, boxes, col):
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,1)
        cv2.circle(frame,((x1+x2)//2,(y1+y2)//2),4,col,-1)
    return frame


def draw_hud(frame, free, occ, total, fps,
             cam_ok=True, rescanning=False, rewarming=False,
             demand=DemandLevel.NORMAL):
    w = frame.shape[1]
    cv2.rectangle(frame,(0,0),(w,32),(0,0,0),-1)
    if rescanning:
        txt="OccupAI | Re-scanning..."; col=(0,200,255)
    elif rewarming:
        txt="OccupAI | Calibrating background..."; col=(0,200,255)
    elif not cam_ok:
        txt="OccupAI | CAMERA BLOCKED"; col=(0,60,255)
    else:
        txt=(f"OccupAI | Free:{free}  Occ:{occ}  Tot:{total}"
             f"  FPS:{fps:.1f}  [{demand}]")
        col=_DEMAND_COLORS.get(demand,(0,229,160))
    cv2.putText(frame,txt,(6,21),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)
    if cam_ok and not rescanning and not rewarming:
        reason=slot_state.adjustment_reason
        if reason and len(reason)>2:
            cv2.rectangle(frame,(0,32),(w,52),(15,15,15),-1)
            cv2.putText(frame,reason[:90],(6,46),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,(180,180,180),1)
    return frame


def draw_blocked(frame):
    h,w=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
    cv2.addWeighted(ov,0.60,frame,0.40,0,frame)
    for msg,y,fs,th in [
        ("CAMERA BLOCKED",h//2-22,1.1,3),
        ("Unblock camera to resume detection",h//2+18,0.5,1),
    ]:
        (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,fs,th)
        cv2.putText(frame,msg,(w//2-tw//2,y),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,60,255),th)
    return frame


def draw_scanning(frame, msg="LOADING LAYOUT..."):
    h,w=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
    cv2.addWeighted(ov,0.50,frame,0.50,0,frame)
    (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)
    cv2.putText(frame,msg,(w//2-tw//2,h//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,255),2)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#   CAMERA READER
# ══════════════════════════════════════════════════════════════════════════════

def camera_reader(cap, tw, th):
    while True:
        ret,f=cap.read()
        if not ret or f is None: time.sleep(0.005); continue
        if f.shape[1]!=tw or f.shape[0]!=th: f=cv2.resize(f,(tw,th))
        if _cam_q.full():
            try: _cam_q.get_nowait()
            except queue.Empty: pass
        _cam_q.put(f)


def detector_worker(model, in_q, result, stop_event):
    while not stop_event.is_set():
        try:
            frame = in_q.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            yolo_b = yolo_boxes(model, frame)
            toy_b = find_toy_boxes(frame)
            with result["lock"]:
                result["yolo"] = yolo_b
                result["toy"] = toy_b
                result["ts"] = time.time()
        except Exception as e:
            print(f"[detector-worker] {e}")


# ══════════════════════════════════════════════════════════════════════════════
#   MJPEG SERVER
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path!='/stream':
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header('Content-Type',
            'multipart/x-mixed-replace; boundary=--occupaiframe')
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()
        while True:
            try:
                with _stream_lock: f=_stream_frame
                if f is None: time.sleep(0.05); continue
                _,buf=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,JPEG_Q])
                jpg=buf.tobytes()
                self.wfile.write(
                    b'--occupaiframe\r\nContent-Type: image/jpeg\r\n'
                    b'Content-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'
                    +jpg+b'\r\n')
                self.wfile.flush()
                time.sleep(1.0/STREAM_FPS)
            except(BrokenPipeError,ConnectionResetError): break
            except Exception: break

def start_mjpeg_server():
    HTTPServer(('0.0.0.0',STREAM_PORT),MJPEGHandler).serve_forever()


# ══════════════════════════════════════════════════════════════════════════════
#   BACKEND PUSH
# ══════════════════════════════════════════════════════════════════════════════

def encode_frame(frame):
    _,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,JPEG_Q])
    return base64.b64encode(buf.tobytes()).decode()

def push_to_backend(occupied,free,total,pct,fps,zone_status,snapshot_frame):
    global _pushing
    with _push_lock:
        if _pushing: return
        _pushing=True
    try:
        fb64 = encode_frame(snapshot_frame)
        adj=slot_state.summary()
        requests.post(f"{BACKEND_URL}/yolo/update",json={
            "occupied":occupied,"free":free,"total":total,
            "occupancy_pct":pct,"lot_full":total>0 and free==0,
            "fps":fps,"yolo_count":occupied,
            "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_b64":fb64,"yolo_boxes":[],"slots":[],"zones":zone_status,
            "demand_level":adj.get("demand","NORMAL"),
            "forecast_veh":adj.get("forecast_veh",0),
            "adjustment_reason":adj.get("reason",""),
        },headers=HEADERS,timeout=3)
    except Exception as e: print(f"[push] {e}")
    finally:
        with _push_lock: _pushing=False


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def detection_loop():
    global _stream_frame, _cam_ok

    print("[yolo] Loading YOLOv8n...")
    model=YOLO("yolov8n.pt")
    model(np.zeros((FEED_H,FEED_W,3),dtype=np.uint8),imgsz=IMGSZ,verbose=False)
    print(f"[yolo] Ready  imgsz={IMGSZ}  conf≥{CONF_THRESH}  async={YOLO_ASYNC}")

    cap=cv2.VideoCapture(WEBCAM_IDX,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FEED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FEED_H)
    cap.set(cv2.CAP_PROP_FPS,CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    time.sleep(1.0)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_IDX}"); return

    actual_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam]  {actual_w}×{actual_h}  index={WEBCAM_IDX}")

    # ── MOG2 (shared reference so rewarm can replace it) ──────────────────────
    bg_sub_ref = [cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESH, detectShadows=False)]

    print("\n" + "═"*58)
    print("  OccupAI v7.2 — Row Layout + AI Adjuster")
    print("  Warming up background model (3s)...")
    print("═"*58)
    time.sleep(3.0)
    grab_background(cap, bg_sub_ref[0], n=25)

    # ── Build base layout ──────────────────────────────────────────────────────
    base_slots = build_layout(actual_w, actual_h)
    slot_state.set_base_slots(base_slots)
    print(f"[layout] ✓ {len(slot_state.active_slots)} active slots "
          f"(excluded: {_es('EXCLUDED_SLOTS')})")

    ret0,bg_frame=cap.read()
    if bg_frame is not None:
        _save_debug_zones(bg_frame, slot_state.active_slots)

    # ── AI Adjuster thread ─────────────────────────────────────────────────────
    adjuster=SlotAdjusterThread(
        slot_state=slot_state, models_dir=MODELS_DIR,
        db_fn=_fetch_db_history, frame_w=actual_w, frame_h=actual_h,
        backend_url=BACKEND_URL, cam_token=CAM_TOKEN,
    )
    adjuster.start()
    print(f"[adjuster] Started — cycles every {SlotAdjusterThread.ADJUST_INTERVAL}s")

    # ── Reference frame for scene change ──────────────────────────────────────
    ret_r,ref_f=cap.read()
    ref_gray=cv2.cvtColor(ref_f,cv2.COLOR_BGR2GRAY) if ret_r else None

    threading.Thread(target=camera_reader,args=(cap,actual_w,actual_h),
                     daemon=True,name="cam-reader").start()

    det_q = queue.Queue(maxsize=1)
    det_stop = threading.Event()
    det_result = {"lock": threading.Lock(), "yolo": [], "toy": [], "ts": 0.0}
    last_yolo_submit = 0.0
    if YOLO_ASYNC:
        threading.Thread(target=detector_worker,
                         args=(model,det_q,det_result,det_stop),
                         daemon=True,name="detector-worker").start()

    frame_idx=0; yolo_b=[]; toy_b=[]
    fps_t=time.time(); fps_n=0; fps_val=0.0
    prev_gray=None; rescanning=False; rewarming=False
    last_check_t=time.time(); last_n_slots=len(slot_state.active_slots)

    while True:
        try: frame=_cam_q.get(timeout=0.1)
        except queue.Empty: continue

        frame_idx+=1; fps_n+=1
        now=time.time()
        if now-fps_t>=1.0:
            fps_val=fps_n/(now-fps_t); fps_n=0; fps_t=now

        # ── Camera blocked check ───────────────────────────────────────────────
        if frame_idx%STATIC_N==0:
            blocked,prev_gray=is_blocked(frame,prev_gray)
            with _cam_ok_lock: _cam_ok=not blocked

        with _cam_ok_lock: cam_ok=_cam_ok

        # ── MOG2 rewarm when layout changes (fixes false OCC after switch) ────
        if slot_state.check_and_clear_bg_reset() and not rewarming:
            rewarming=True
            print("[detector] Layout changed — re-warming MOG2...")
            def _do_rewarm():
                nonlocal rewarming
                new_bs=cv2.createBackgroundSubtractorMOG2(
                    history=BG_HISTORY,varThreshold=BG_VAR_THRESH,
                    detectShadows=False)
                grab_background(cap, new_bs, n=BG_REWARM_N)
                bg_sub_ref[0]=new_bs
                rewarming=False
                print("[detector] ✓ MOG2 re-warmed after layout change")
            threading.Thread(target=_do_rewarm,daemon=True,name="bg-rewarm").start()

        # ── Scene change (camera physically moved) ────────────────────────────
        if (not rescanning and not rewarming and cam_ok
                and ref_gray is not None
                and (now-last_check_t)>=REDETECT_INTERVAL):
            last_check_t=now
            cg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ed=edge_region_diff(cg,ref_gray)
            print(f"[redetect] edge diff={ed:.1f}  thresh={REDETECT_THRESH}")
            if ed>REDETECT_THRESH:
                print("[redetect] Camera moved — re-warming MOG2...")
                rescanning=True
                def _redo():
                    nonlocal rescanning,ref_gray
                    try:
                        new_bs=cv2.createBackgroundSubtractorMOG2(
                            history=BG_HISTORY,varThreshold=BG_VAR_THRESH,
                            detectShadows=False)
                        grab_background(cap,new_bs,n=20)
                        bg_sub_ref[0]=new_bs
                        new_slots=build_layout(actual_w,actual_h)
                        if len(new_slots)>=MIN_SLOTS:
                            slot_state.set_base_slots(new_slots)
                        ret2,rf2=cap.read()
                        ref_gray=cv2.cvtColor(rf2,cv2.COLOR_BGR2GRAY) if ret2 else ref_gray
                        print(f"[redetect] ✓ Rebuilt: {len(slot_state.active_slots)} slots")
                    except Exception as e: print(f"[redetect] err: {e}")
                    finally: rescanning=False
                threading.Thread(target=_redo,daemon=True,name="redetect").start()

        # ── MOG2 foreground ───────────────────────────────────────────────────
        fg=bg_sub_ref[0].apply(frame,learningRate=BG_LEARN_RATE)
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,k,iterations=1)

        # ── YOLO + toy blobs ──────────────────────────────────────────────────
        if YOLO_ASYNC:
            if (cam_ok and not rescanning and not rewarming
                    and now-last_yolo_submit >= YOLO_INTERVAL):
                last_yolo_submit = now
                if det_q.full():
                    try: det_q.get_nowait()
                    except queue.Empty: pass
                try: det_q.put_nowait(frame.copy())
                except queue.Full: pass
            with det_result["lock"]:
                yolo_b = list(det_result["yolo"])
                toy_b = list(det_result["toy"])
        elif frame_idx%YOLO_SKIP==0 and cam_ok and not rescanning and not rewarming:
            yolo_b=yolo_boxes(model,frame)
            toy_b=find_toy_boxes(frame)

        # ── Active slots (AI may have updated) ────────────────────────────────
        active=slot_state.active_slots
        n_slots=len(active)
        if n_slots!=last_n_slots:
            _save_debug_zones(frame,active)
            last_n_slots=n_slots
            print(f"[detector] Slot count → {n_slots}  debug_zones.jpg updated")

        # ── Occupancy ─────────────────────────────────────────────────────────
        if cam_ok and not rescanning and not rewarming and active:
            all_b=yolo_b+toy_b
            zone_status={name:(occ_bg(c,fg) or occ_yolo(c,all_b))
                         for name,c in active.items()}
        else:
            zone_status={name:False for name in active}

        occupied=sum(zone_status.values())
        free=n_slots-occupied
        pct=round(occupied/n_slots*100,1) if n_slots else 0.0
        demand=slot_state.demand

        # ── Annotate ──────────────────────────────────────────────────────────
        ann=frame.copy()
        if not cam_ok:
            draw_blocked(ann); zone_status={}; occupied=free=0
        elif rescanning or rewarming:
            draw_scanning(ann, "CALIBRATING..." if rewarming else "RE-SCANNING...")
        else:
            draw_zones(ann,active,zone_status)
            if DRAW_DETECTOR_BOXES:
                draw_boxes(ann,yolo_b,(0,230,255))
                draw_boxes(ann,toy_b,(0,255,128))

        draw_hud(ann,free,occupied,n_slots,fps_val,
                 cam_ok=cam_ok,rescanning=rescanning,
                 rewarming=rewarming,demand=demand)

        with _stream_lock: _stream_frame=ann

        if frame_idx%PUSH_EVERY==0:
            threading.Thread(target=push_to_backend,
                args=(occupied,free,n_slots,pct,round(fps_val,1),zone_status,ann.copy()),
                daemon=True).start()

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
if __name__=='__main__':
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  OccupAI Detector v7.2                                   ║")
    print("║  Layout  : R1/R2/R3 from .env                           ║")
    print("║  Keepout : NO_PARK_RECTS clears entrance geometry       ║")
    print("║  AI      : demand-driven slot packing (30s cycles)      ║")
    print("║  MOG2    : auto re-warms on layout change (no false OCC)║")
    print(f"║  Camera  : index {WEBCAM_IDX}                                    ║")
    print(f"║  Stream  : http://localhost:{STREAM_PORT}/stream              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  TEST LEVELS (edit .env, restart detector):             ║")
    print("║  FORCE_DEMAND_LEVEL=LOW    → base layout               ║")
    print("║  FORCE_DEMAND_LEVEL=NORMAL → base layout               ║")
    print("║  FORCE_DEMAND_LEVEL=BUSY   → +columns, entrance clear  ║")
    print("║  FORCE_DEMAND_LEVEL=HIGH   → +columns/subrows          ║")
    print("║  FORCE_DEMAND_LEVEL=       → AI decides every 30s       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    threading.Thread(target=start_mjpeg_server,daemon=True,
                     name="mjpeg-server").start()
    print(f"[mjpeg] Streaming on :{STREAM_PORT}\n")
    detection_loop()
