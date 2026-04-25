from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class UserRegister(BaseModel):
    first_name: str
    last_name:  str
    email:      str
    password:   str

class UserLogin(BaseModel):
    email:    str
    password: str

class YoloUpdate(BaseModel):
    occupied:      int
    free:          int      = 0
    total:         int
    occupancy_pct: float    = 0.0
    lot_full:      bool     = False
    fps:           float    = 0.0
    yolo_count:    int      = 0
    timestamp:     str      = ""
    snapshot_b64:  str      = ""
    yolo_boxes:    list     = []
    slots:         list     = []
    zones:         dict     = {}   # {"Z1": True, "Z2": False, ...}

class PushFrame(BaseModel):
    frame: str              # base64 JPEG

class StatsResponse(BaseModel):
    occupied:      int
    free:          int
    total:         int
    occupancy_pct: float
    lot_full:      bool
    fps:           float
    timestamp:     str
    yolo_count:    int
    running:       bool

class SnapshotResponse(BaseModel):
    frame_b64: str
    timestamp: str

class PredictionResponse(BaseModel):
    hourly_est: dict
    peak_hour:  int
    peak_label: str
    busy_days:  list
    quiet_days: list

class StatusResponse(BaseModel):
    status:   str
    version:  str
    location: str

# Aliases
RegisterRequest = UserRegister
LoginRequest    = UserLogin