"""
backend/main.py — OccupAI FastAPI Backend
Run: uvicorn backend.main:app --reload --port 8000
"""
import os
import bcrypt
import uvicorn
import threading
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.db     import get_db, query, execute
from backend.models import UserRegister, UserLogin, YoloUpdate, PushFrame

load_dotenv()

CAM_TOKEN    = os.getenv("CAM_TOKEN",   "occupai_cam_2027")
DEPLOY_MODE  = os.getenv("DEPLOY_MODE", "local")
STREAM_PORT  = os.getenv("STREAM_PORT", "8001")

BASE_DIR     = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "template"

app = FastAPI(title="OccupAI API", version="1.0.0")

app.mount("/static", StaticFiles(directory=str(TEMPLATE_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    error = traceback.format_exc()
    print(f"ERROR:\n{error}")
    return JSONResponse(status_code=500, content={"error": str(exc), "detail": error})

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "occupied":      0,
    "free":          0,
    "total":         0,
    "occupancy_pct": 0.0,
    "lot_full":      False,
    "fps":           0.0,
    "timestamp":     "",
    "yolo_count":    0,
    "yolo_boxes":    [],
    "slots":         [],
    "zones":         {},
}
snap       = {"frame_b64": "", "timestamp": ""}
history    = deque(maxlen=100)
state_lock = threading.Lock()
snap_lock  = threading.Lock()


# ══════════════════════════════
#   PAGE ROUTES
# ══════════════════════════════
@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(str(TEMPLATE_DIR / "login.html"))

@app.get("/login", response_class=FileResponse)
def login_page():
    return FileResponse(str(TEMPLATE_DIR / "login.html"))

@app.get("/register", response_class=FileResponse)
def register_page():
    return FileResponse(str(TEMPLATE_DIR / "register.html"))

@app.get("/dashboard", response_class=FileResponse)
def dashboard_page():
    return FileResponse(str(TEMPLATE_DIR / "dashboard.html"))


# ══════════════════════════════
#   HEALTH
# ══════════════════════════════
@app.get("/status")
def status():
    return {
        "status":  "ok",
        "mode":    DEPLOY_MODE,
        "time":    datetime.utcnow().isoformat(),
        "stream":  f"http://localhost:{STREAM_PORT}/stream"
    }


# ══════════════════════════════
#   YOLO ENDPOINTS
# ══════════════════════════════
@app.post("/yolo/update")
def yolo_update(data: YoloUpdate, x_cam_token: str = Header(...)):
    if x_cam_token != CAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with state_lock:
        state.update({
            "occupied":      data.occupied,
            "free":          data.free,
            "total":         data.total,
            "occupancy_pct": round(data.occupancy_pct, 1),
            "lot_full":      data.lot_full,
            "fps":           data.fps,
            "timestamp":     ts,
            "yolo_count":    data.yolo_count,
            "yolo_boxes":    data.yolo_boxes,
            "slots":         data.slots,
            "zones":         data.zones,
        })
    history.append({
        "time":     ts,
        "occupied": data.occupied,
        "total":    data.total,
        "pct":      round(data.occupancy_pct, 1)
    })
    try:
        execute("""INSERT INTO parking_logs
            (occupied, free, total, occupancy_pct, lot_full)
            VALUES (%s, %s, %s, %s, %s)""",
            (data.occupied, data.free, data.total,
             round(data.occupancy_pct, 1), data.lot_full))
    except Exception as e:
        print(f"DB log error: {e}")
    return {"ok": True}


@app.post("/yolo/push-frame")
def push_frame(data: PushFrame, x_cam_token: str = Header(...)):
    if x_cam_token != CAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with snap_lock:
        snap["frame_b64"] = data.frame
        snap["timestamp"] = ts
    return {"ok": True}


# ══════════════════════════════
#   PUBLIC API
# ══════════════════════════════
@app.get("/api/stats")
def api_stats():
    with state_lock:
        return dict(state)

@app.get("/api/snapshot")
def api_snapshot():
    with snap_lock:
        return {
            "image":     snap["frame_b64"],
            "timestamp": snap["timestamp"]
        }

@app.get("/api/history")
def api_history():
    return list(history)

@app.get("/api/stream-url")
def api_stream_url():
    """Returns the MJPEG stream URL so the dashboard knows where to point."""
    return {"url": f"http://localhost:{STREAM_PORT}/stream"}

@app.get("/api/predictions")
def api_predictions():
    try:
        rows = query("""
            SELECT EXTRACT(HOUR FROM logged_at) AS hour,
                   AVG(occupancy_pct) AS avg_pct
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '7 days'
            GROUP BY hour ORDER BY hour
        """)
        hourly = {str(int(r["hour"])): round(float(r["avg_pct"]), 1) for r in rows}
        for h in range(24):
            hourly.setdefault(str(h), 0.0)
        peak_hour = max(hourly, key=lambda h: hourly[h])
        peak_val  = hourly[peak_hour]
        return {
            "hourly_est": hourly,
            "peak_hour":  int(peak_hour),
            "peak_label": f"{peak_hour}:00 ({peak_val:.0f}%)",
            "busy_days":  ["Mon", "Tue", "Wed"],
            "quiet_days": ["Sat", "Sun"]
        }
    except Exception as e:
        print(f"predictions error: {e}")
        return {
            "hourly_est": {str(h): 0.0 for h in range(24)},
            "peak_hour":  8, "peak_label": "N/A",
            "busy_days":  [], "quiet_days": []
        }

@app.get("/api/occupancy")
def api_occupancy():
    with state_lock:
        return {
            "occupied":      state["occupied"],
            "free":          state["free"],
            "total":         state["total"],
            "occupancy_pct": state["occupancy_pct"],
            "zones":         state["zones"],
        }


# ══════════════════════════════
#   AUTH
# ══════════════════════════════
@app.post("/auth/register")
def register(data: UserRegister):
    pw_hash = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    try:
        rows = query("SELECT user_id FROM users WHERE email=%s", (data.email,))
        if rows:
            raise HTTPException(status_code=400, detail="Email already registered")
        result = query("""
            INSERT INTO users (first_name, last_name, email, password_hash, role)
            VALUES (%s, %s, %s, %s, 'driver') RETURNING user_id""",
            (data.first_name, data.last_name, data.email, pw_hash))
        new_id = result[0]["user_id"]
        execute("INSERT INTO drivers(user_id) VALUES(%s)", (new_id,))
        return {"ok": True, "user_id": new_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/login")
def login(data: UserLogin):
    try:
        rows = query("""
            SELECT user_id, first_name, last_name, full_name,
                   email, password_hash, role, is_active
            FROM users WHERE email=%s""", (data.email,))
        if not rows:
            raise HTTPException(status_code=404, detail="Email not found")
        user = rows[0]
        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="Account disabled")
        if not bcrypt.checkpw(data.password.encode(), user["password_hash"].encode()):
            raise HTTPException(status_code=401, detail="Incorrect password")
        execute("UPDATE users SET last_login=%s WHERE user_id=%s",
                (datetime.utcnow(), user["user_id"]))
        return {
            "ok":         True,
            "user_id":    user["user_id"],
            "first_name": user["first_name"],
            "last_name":  user["last_name"],
            "full_name":  user["full_name"],
            "email":      user["email"],
            "role":       user["role"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║   OccupAI FastAPI Backend v1.0       ║")
    print("║   http://localhost:8000              ║")
    print("╚══════════════════════════════════════╝\n")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)