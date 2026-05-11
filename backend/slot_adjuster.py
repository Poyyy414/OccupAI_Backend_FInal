"""
backend/slot_adjuster.py  — OccupAI Slot Adjuster v1.4
=======================================================
Current behavior:
  - Demand levels build different slot counts from model predictions.
  - NO_PARK_RECTS keeps entrances clear by geometry, not hardcoded slot IDs.
  - UNIFORM_SLOT_SIZE makes every generated box the same width/height per layout.
"""

import time
import threading
import os
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import joblib
import keras
from dotenv import load_dotenv

PH_TZ = ZoneInfo("Asia/Manila")

def _now_ph():
    return datetime.now(PH_TZ)

def _reload_env():
    load_dotenv(override=True)


# ══════════════════════════════════════════════════════════════════════════════
#   KERAS CUSTOM LAYER
# ══════════════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable(package="occupai")
class SoftAttention(keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = keras.layers.Dense(units, activation="tanh")
        self.V = keras.layers.Dense(1)

    def call(self, x):
        w = keras.ops.softmax(self.V(self.W(x)), axis=1)
        return keras.ops.sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
#   CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LOT_CAPACITY = 44

NB1_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_morning_peak", "is_lunch_peak", "is_afternoon_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h",
    "roll_3h", "roll_7h", "roll_24h",
    "moto_ratio", "car_ratio", "ebike_ratio",
]

OCC_FEATURES_FALLBACK = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_morning_peak", "is_lunch_peak", "is_afternoon_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "vehicles_last_hour", "occupancy_pct_lag1",
]


# ══════════════════════════════════════════════════════════════════════════════
#   DEMAND LEVEL
# ══════════════════════════════════════════════════════════════════════════════

class DemandLevel:
    LOW    = "LOW"
    NORMAL = "NORMAL"
    BUSY   = "BUSY"
    HIGH   = "HIGH"

    @classmethod
    def all(cls):
        return {cls.LOW, cls.NORMAL, cls.BUSY, cls.HIGH}


# ══════════════════════════════════════════════════════════════════════════════
#   LIVE ENV READERS
# ══════════════════════════════════════════════════════════════════════════════

def _ef(k, d):
    try: return float(os.getenv(k, str(d)))
    except: return d

def _ei(k, d):
    try: return int(os.getenv(k, str(d)))
    except: return d

def _eb(k, d):
    return os.getenv(k, str(d)).strip().lower() not in {"0","false","no","off"}

def _es(k, d=""):
    raw = os.getenv(k, d) or ""
    return {s.strip().upper() for s in raw.split(",") if s.strip()}

def _get_force_level():
    val = (os.getenv("FORCE_DEMAND_LEVEL") or "").strip().upper()
    return val if val in DemandLevel.all() else None

def _get_excluded():
    return _es("EXCLUDED_SLOTS")

def _get_no_park_rects():
    """
    Generic geometry-based keep-clear areas, not slot-name hardcoding.
    Format: NO_PARK_RECTS=x1,y1,x2,y2;...
    Values are frame fractions from 0.0 to 1.0.
    """
    raw = os.getenv("NO_PARK_RECTS", "").strip()
    rects = []
    for chunk in raw.split(";"):
        if not chunk.strip():
            continue
        try:
            x1, y1, x2, y2 = [float(v.strip()) for v in chunk.split(",")]
        except ValueError:
            continue
        rects.append((
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        ))
    return rects

def _point_in_rects(x, y, rects):
    for x1, y1, x2, y2 in rects:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def _get_thresholds():
    return {
        "busy_occ":  _ef("BUSY_OCC_THRESH",  0.40),
        "busy_fore": _ef("BUSY_FORE_THRESH",  0.25),
        "high_occ":  _ef("HIGH_OCC_THRESH",   0.70),
        "high_fore": _ef("HIGH_FORE_THRESH",  0.65),
        "low_occ":   _ef("LOW_OCC_THRESH",    0.35),
        "low_fore":  _ef("LOW_FORE_THRESH",   0.30),
    }

def _get_packing():
    return {
        "busy_col": _ei("BUSY_DEMAND_COL_BONUS",  1),
        "high_col": _ei("HIGH_DEMAND_COL_BONUS",  1),
        "high_row": _ei("HIGH_DEMAND_ROW_BONUS",  1),
        "busy_target": _ei("BUSY_TARGET_SLOTS",  0),
        "high_target": _ei("HIGH_TARGET_SLOTS",  0),
        "uniform":  _eb("UNIFORM_SLOT_SIZE", True),
    }

def _get_demand_counts(demand):
    """
    Optional per-demand row counts from .env.
    Example: HIGH_R1_N=7, HIGH_R2_N=2, HIGH_R3_N=7.
    LOW uses the NORMAL guide layout when no LOW-specific values are present.
    """
    rc = _get_row_config()
    prefix = "NORMAL" if demand == DemandLevel.LOW else demand
    fallback_prefix = "NORMAL" if demand == DemandLevel.LOW else ""

    result = {}
    found = False
    for row_key, env_suffix in (("r1", "R1_N"), ("r2", "R2_N"), ("r3", "R3_N")):
        env_key = f"{prefix}_{env_suffix}"
        raw = os.getenv(env_key)
        if raw is None and fallback_prefix:
            env_key = f"{fallback_prefix}_{env_suffix}"
            raw = os.getenv(env_key)
        if raw is not None:
            found = True
        result[row_key] = max(0, _ei(env_key, rc[row_key]["n"]))

    return result if found else None

def _limit_slot_count(slots, target):
    if target <= 0 or len(slots) <= target:
        return slots
    return dict(list(slots.items())[:target])

def _get_row_config():
    """
    Returns row config from .env.
    NO_PARK_RECTS handles keep-clear areas like entrances without slot IDs.
    """
    return {
        "r1": {
            "x1":    _ef("R1_X1",   0.03),
            "y1":    _ef("R1_Y1",   0.04),
            "x2":    _ef("R1_X2",   0.97),
            "y2":    _ef("R1_Y2",   0.36),
            "n":     _ei("R1_N",    7),
            "wfrac": _ef("R1_WFRAC",0.82),
            "hfrac": _ef("R1_HFRAC",0.82),
        },
        "r2": {
            "x1":    _ef("R2_X1",   0.03),
            "y1":    _ef("R2_Y1",   0.40),
            "x2":    _ef("R2_X2",   0.97),
            "y2":    _ef("R2_Y2",   0.62),
            "n":     _ei("R2_N",    2),
            "wfrac": _ef("R2_WFRAC",0.82),
            "hfrac": _ef("R2_HFRAC",0.82),
        },
        "r3": {
            "x1":    _ef("R3_X1",   0.03),
            "y1":    _ef("R3_Y1",   0.66),
            "x2":    _ef("R3_X2",   0.97),
            "y2":    _ef("R3_Y2",   0.97),
            "n":     _ei("R3_N",    5),
            "wfrac": _ef("R3_WFRAC",0.82),
            "hfrac": _ef("R3_HFRAC",0.82),
        },
        "pad": _ei("SLOT_PAD", 5),
    }


# ══════════════════════════════════════════════════════════════════════════════
#   ROW BUILDER  — with hard X2 cap
# ══════════════════════════════════════════════════════════════════════════════

def _build_row_slots(fw, fh, row_cfg, n_cols, n_sub_rows, start_idx, pad,
                     slot_size=None, no_park_rects=None):
    """
    Build slots for one row.
    row_cfg: dict with x1,y1,x2,y2,wfrac,hfrac (all as fractions).
    n_cols:  base columns + col_bonus.
    x2 from row_cfg is the HARD CAP — slots never go past it.
    """
    rx1 = int(fw * row_cfg["x1"])
    rx2 = int(fw * row_cfg["x2"])   # ← hard entrance boundary
    ry1 = int(fh * row_cfg["y1"])
    ry2 = int(fh * row_cfg["y2"])
    rw  = rx2 - rx1
    rh  = ry2 - ry1

    col_w  = rw / max(1, n_cols)
    sub_h  = rh / max(1, n_sub_rows)
    if slot_size:
        slot_w, slot_h = slot_size
    else:
        slot_w = max(10, int(col_w * row_cfg["wfrac"]))
        slot_h = max(10, int(sub_h * row_cfg["hfrac"]))
    no_park_rects = no_park_rects or []

    slots = []
    idx   = start_idx
    for sr in range(n_sub_rows):
        cy = int(ry1 + sr * sub_h + sub_h / 2)
        for c in range(n_cols):
            cx = int(rx1 + c * col_w + col_w / 2)
            if _point_in_rects(cx / max(1, fw), cy / max(1, fh), no_park_rects):
                idx += 1
                continue
            x1 = cx - slot_w // 2 + pad
            x2 = cx + slot_w // 2 - pad
            y1 = cy - slot_h // 2 + pad
            y2 = cy + slot_h // 2 - pad

            # Clamp to row bounds; NO_PARK_RECTS handles keep-clear areas.
            x1 = max(rx1, x1)
            x2 = min(rx2, x2)
            y1 = max(ry1, y1)
            y2 = min(ry2, y2)

            # Skip degenerate boxes.
            if x2 - x1 < 10 or y2 - y1 < 10:
                idx += 1
                continue

            slots.append((f"Z{idx}", (x1, y1, x2, y2)))
            idx += 1

    return slots, idx


def build_layout(fw, fh, col_bonus=0, sub_rows_r1=1, sub_rows_r3=1,
                 row_counts=None):
    """
    Build full slot layout. col_bonus adds columns to ALL rows,
    but the x2 cap in each row config prevents entrance overlap.
    row_counts can override per-row counts for guided demand layouts.
    EXCLUDED_SLOTS applied on top.
    """
    rc   = _get_row_config()
    pad  = rc["pad"]
    excl = _get_excluded()
    no_park_rects = _get_no_park_rects()
    pk = _get_packing()

    row_plan = [("r1", sub_rows_r1), ("r2", 1), ("r3", sub_rows_r3)]
    slot_size = None
    if pk["uniform"]:
        widths = []
        heights = []
        for row_key, sub_rows in row_plan:
            row_cfg = rc[row_key]
            requested_cols = (row_counts or {}).get(row_key, row_cfg["n"] + col_bonus)
            if requested_cols <= 0:
                continue
            n_cols = max(1, requested_cols)
            rw = max(1, int(fw * row_cfg["x2"]) - int(fw * row_cfg["x1"]))
            rh = max(1, int(fh * row_cfg["y2"]) - int(fh * row_cfg["y1"]))
            widths.append(max(10, int((rw / n_cols) * row_cfg["wfrac"])))
            heights.append(max(10, int((rh / max(1, sub_rows)) * row_cfg["hfrac"])))
        if widths and heights:
            slot_size = (min(widths), min(heights))

    all_slots = []
    idx = 1
    for row_key, sub_rows in row_plan:
        row_cfg = rc[row_key]
        n_cols = (row_counts or {}).get(row_key, row_cfg["n"] + col_bonus)
        s, idx  = _build_row_slots(
            fw, fh, row_cfg, n_cols, sub_rows, idx, pad,
            slot_size=slot_size,
            no_park_rects=no_park_rects,
        )
        all_slots += s

    result = {name: coords for name, coords in all_slots
              if name.upper() not in excl}
    return result


# ══════════════════════════════════════════════════════════════════════════════
#   GRID ADJUSTER
# ══════════════════════════════════════════════════════════════════════════════

class GridAdjuster:
    def __init__(self, fw, fh):
        self.fw = fw
        self.fh = fh

    def compute_demand(self, occ_pct, forecast_veh):
        t    = _get_thresholds()
        occ  = occ_pct / 100.0
        fore = forecast_veh / LOT_CAPACITY
        if occ >= t["high_occ"]  or fore >= t["high_fore"]:  return DemandLevel.HIGH
        if occ >= t["busy_occ"]  or fore >= t["busy_fore"]:  return DemandLevel.BUSY
        if occ <= t["low_occ"]  and fore <= t["low_fore"]:   return DemandLevel.LOW
        return DemandLevel.NORMAL

    def slots_for_demand(self, demand):
        guided_counts = _get_demand_counts(demand)
        if guided_counts is not None:
            return build_layout(self.fw, self.fh, row_counts=guided_counts)

        pk = _get_packing()
        if demand == DemandLevel.HIGH:
            slots = build_layout(self.fw, self.fh,
                                 col_bonus=pk["high_col"],
                                 sub_rows_r1=1 + pk["high_row"],
                                 sub_rows_r3=1 + pk["high_row"])
            return _limit_slot_count(slots, pk["high_target"])
        if demand == DemandLevel.BUSY:
            slots = build_layout(self.fw, self.fh,
                                 col_bonus=pk["busy_col"])
            return _limit_slot_count(slots, pk["busy_target"])
        return build_layout(self.fw, self.fh)


# ══════════════════════════════════════════════════════════════════════════════
#   SHARED SLOT STATE
# ══════════════════════════════════════════════════════════════════════════════

class SlotState:
    def __init__(self):
        self._lock             = threading.Lock()
        self._base_slots       = {}
        self._active_slots     = {}
        self.demand            = DemandLevel.NORMAL
        self.forecast_veh      = 0.0
        self.current_occ       = 0.0
        self.last_adjusted     = None
        self.adjustment_reason = "Starting up..."
        self._needs_bg_reset   = False
        self._last_demand      = None

    def set_base_slots(self, slots):
        excl = _get_excluded()
        with self._lock:
            self._base_slots   = dict(slots)
            self._active_slots = {n: c for n, c in slots.items()
                                  if n.upper() not in excl}

    def update_slots(self, new_slots, demand, forecast_veh, current_occ, reason):
        excl = _get_excluded()
        with self._lock:
            filtered               = {n: c for n, c in new_slots.items()
                                       if n.upper() not in excl}
            self._active_slots     = filtered
            self.adjustment_reason = reason
            if demand != self._last_demand:
                self._needs_bg_reset = True
                self._last_demand    = demand
            self.demand       = demand
            self.forecast_veh = forecast_veh
            self.current_occ  = current_occ
            self.last_adjusted = _now_ph()

    @property
    def active_slots(self):
        excl = _get_excluded()
        with self._lock:
            return {n: c for n, c in self._active_slots.items()
                    if n.upper() not in excl}

    @property
    def base_slots(self):
        with self._lock:
            return dict(self._base_slots)

    def check_and_clear_bg_reset(self):
        with self._lock:
            if self._needs_bg_reset:
                self._needs_bg_reset = False
                return True
            return False

    def summary(self):
        with self._lock:
            return {
                "demand":        self.demand,
                "forecast_veh":  round(self.forecast_veh, 1),
                "current_occ":   round(self.current_occ,  1),
                "n_slots":       len(self._active_slots),
                "last_adjusted": (self.last_adjusted.strftime("%H:%M:%S")
                                  if self.last_adjusted else None),
                "reason":        self.adjustment_reason,
            }


# ══════════════════════════════════════════════════════════════════════════════
#   FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_nb1_features(history, lag_counts):
    now = _now_ph()
    h, dow, mon = now.hour, now.weekday(), now.month
    f = {
        "hour": h, "day_of_week": dow, "month": mon,
        "is_weekend":        int(dow >= 5),
        "is_morning_peak":   int(7 <= h <= 9),
        "is_lunch_peak":     int(11 <= h <= 13),
        "is_afternoon_peak": int(16 <= h <= 18),
        "hour_sin":  np.sin(2*np.pi*h/24),  "hour_cos":  np.cos(2*np.pi*h/24),
        "dow_sin":   np.sin(2*np.pi*dow/7), "dow_cos":   np.cos(2*np.pi*dow/7),
        "month_sin": np.sin(2*np.pi*mon/12),"month_cos": np.cos(2*np.pi*mon/12),
        "lag_1h":  lag_counts[0] if len(lag_counts) > 0 else 0,
        "lag_2h":  lag_counts[1] if len(lag_counts) > 1 else 0,
        "lag_3h":  lag_counts[2] if len(lag_counts) > 2 else 0,
        "lag_24h": lag_counts[3] if len(lag_counts) > 3 else 0,
        "roll_3h":  np.mean(lag_counts[:3])  if lag_counts else 0,
        "roll_7h":  np.mean(lag_counts[:7])  if lag_counts else 0,
        "roll_24h": np.mean(lag_counts[:24]) if lag_counts else 0,
        "moto_ratio": 0.941, "car_ratio": 0.049, "ebike_ratio": 0.010,
    }
    return np.array([[f[k] for k in NB1_FEATURES]], dtype=np.float32)


def _build_occ_features(current_occ, lag_occ, veh_lh, feats_list):
    now = _now_ph()
    h, dow, mon = now.hour, now.weekday(), now.month
    b = {
        "hour": h, "day_of_week": dow, "month": mon,
        "is_weekend":        int(dow >= 5),
        "is_morning_peak":   int(7 <= h <= 9),
        "is_lunch_peak":     int(11 <= h <= 13),
        "is_afternoon_peak": int(16 <= h <= 18),
        "hour_sin":  np.sin(2*np.pi*h/24),  "hour_cos":  np.cos(2*np.pi*h/24),
        "dow_sin":   np.sin(2*np.pi*dow/7), "dow_cos":   np.cos(2*np.pi*dow/7),
        "vehicles_last_hour":  veh_lh,
        "occupancy_pct_lag1":  lag_occ,
    }
    fl = feats_list if feats_list else OCC_FEATURES_FALLBACK
    return np.array([[b.get(k, 0) for k in fl]], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#   ADJUSTER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class SlotAdjusterThread(threading.Thread):

    ADJUST_INTERVAL = 30

    def __init__(self, slot_state, models_dir, db_fn,
                 frame_w, frame_h, backend_url, cam_token):
        super().__init__(daemon=True, name="slot-adjuster")
        self.state       = slot_state
        self.models_dir  = models_dir
        self.db_fn       = db_fn
        self.fw          = frame_w
        self.fh          = frame_h
        self.backend_url = backend_url
        self.headers     = {"x-cam-token": cam_token,
                            "Content-Type": "application/json"}
        self.adjuster    = GridAdjuster(frame_w, frame_h)
        self._nb1_model    = None
        self._occ_model    = None
        self._nb1_scaler_X = None
        self._nb1_scaler_y = None
        self._occ_scaler_X = None
        self._occ_scaler_y = None
        self._occ_features = None

    def _p(self, f):
        return os.path.join(self.models_dir, f)

    def _load_models(self):
        print("[adjuster] Loading ML models...")
        co = {"SoftAttention": SoftAttention}
        for attr, fname in [("_nb1_model", "spatio_temporal.keras"),
                             ("_occ_model", "occupancy_model.keras")]:
            try:
                setattr(self, attr,
                        keras.models.load_model(self._p(fname),
                                                custom_objects=co))
                print(f"[adjuster] ✓ {fname}")
            except Exception as e:
                print(f"[adjuster] ✗ {fname}: {e}")

        for attr, fname in [
            ("_nb1_scaler_X", "scaler_nb1_X.pkl"),
            ("_nb1_scaler_y", "scaler_nb1_y.pkl"),
            ("_occ_scaler_X", "scaler_occ_X.pkl"),
            ("_occ_scaler_y", "scaler_occ_y.pkl"),
            ("_occ_features", "occ_features.pkl"),
        ]:
            try:
                setattr(self, attr, joblib.load(self._p(fname)))
                print(f"[adjuster] ✓ {fname}")
            except Exception as e:
                print(f"[adjuster] ✗ {fname}: {e}")
        print("[adjuster] Models ready.")

    def _forecast(self, history):
        if not self._nb1_model or not self._nb1_scaler_X:
            return LOT_CAPACITY * 0.5
        lags = [r.get("occupied", 0) for r in history[:24]]
        while len(lags) < 24: lags.append(0)
        try:
            X = _build_nb1_features(history, lags)
            e = self._nb1_scaler_X.n_features_in_
            if X.shape[1] > e:   X = X[:, :e]
            elif X.shape[1] < e: X = np.pad(X, ((0,0),(0,e-X.shape[1])))
            Xs = self._nb1_scaler_X.transform(X)
            y  = float(self._nb1_model.predict(
                       Xs.reshape(1,1,-1), verbose=0).flatten()[0])
            return max(0., min(LOT_CAPACITY,
                   float(self._nb1_scaler_y.inverse_transform([[y]])[0][0])))
        except Exception as e:
            print(f"[adjuster] forecast err: {e}")
            return LOT_CAPACITY * 0.5

    def _pred_occ(self, history, cur_occ):
        if not self._occ_model or not self._occ_scaler_X:
            return cur_occ
        lag = history[0].get("occupancy_pct", cur_occ) if history else cur_occ
        vlh = history[0].get("occupied", 0) if history else 0
        try:
            X = _build_occ_features(cur_occ, lag, vlh, self._occ_features)
            e = self._occ_scaler_X.n_features_in_
            if X.shape[1] > e:   X = X[:, :e]
            elif X.shape[1] < e: X = np.pad(X, ((0,0),(0,e-X.shape[1])))
            Xs = self._occ_scaler_X.transform(X)
            y  = float(self._occ_model.predict(
                       Xs.reshape(1,1,-1), verbose=0).flatten()[0])
            return max(0., min(100.,
                   float(self._occ_scaler_y.inverse_transform([[y]])[0][0])))
        except Exception as e:
            print(f"[adjuster] occ err: {e}")
            return cur_occ

    def _run_cycle(self):
        _reload_env()
        force = _get_force_level()

        if force:
            demand       = force
            forecast_veh = 0.0
            cur_occ      = 0.0
            reason       = f"FORCE_DEMAND_LEVEL={force}"
        else:
            history      = self.db_fn()
            cur_occ      = history[0].get("occupancy_pct", 0.) if history else 0.
            forecast_veh = self._forecast(history)
            pred_occ     = self._pred_occ(history, cur_occ)

            if (_eb("CV_PEAK_TEST", False) or _eb("TEST_PEAK_ENABLED", False)) \
               and _now_ph().hour == _ei("TEST_PEAK_HOUR", 23) % 24:
                demand = DemandLevel.HIGH
                reason = f"CV_PEAK_TEST hour={_ei('TEST_PEAK_HOUR',23)%24:02d}"
            else:
                demand = self.adjuster.compute_demand(pred_occ, forecast_veh)
                fp = forecast_veh / LOT_CAPACITY * 100
                msgs = {
                    DemandLevel.HIGH:   f"HIGH: occ={pred_occ:.0f}% fore={forecast_veh:.0f}veh ({fp:.0f}%)",
                    DemandLevel.BUSY:   f"BUSY: occ={pred_occ:.0f}% fore={forecast_veh:.0f}veh",
                    DemandLevel.NORMAL: f"NORMAL: occ={pred_occ:.0f}% fore={forecast_veh:.0f}veh",
                    DemandLevel.LOW:    f"LOW: occ={pred_occ:.0f}% fore={forecast_veh:.0f}veh",
                }
                reason = msgs[demand]

        new_slots = self.adjuster.slots_for_demand(demand)
        if not new_slots:
            print("[adjuster] Empty layout — skipping")
            return

        self.state.update_slots(new_slots, demand, forecast_veh, cur_occ, reason)
        print(f"[adjuster] {demand}  slots={len(new_slots)}  {reason[:80]}")

        try:
            import requests as req
            req.post(f"{self.backend_url}/yolo/slot_adjustment",
                     json=self.state.summary(),
                     headers=self.headers, timeout=2)
        except Exception:
            pass

    def run(self):
        self._load_models()
        print("[adjuster] First cycle running immediately...")
        try:
            self._run_cycle()
        except Exception as e:
            print(f"[adjuster] first cycle error: {e}")

        while True:
            time.sleep(self.ADJUST_INTERVAL)
            try:
                self._run_cycle()
            except Exception as e:
                print(f"[adjuster] cycle error: {e}")
