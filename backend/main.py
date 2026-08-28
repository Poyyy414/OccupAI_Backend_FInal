"""
backend/main.py — OccupAI FastAPI Backend  v2.2
Run: uvicorn backend.main:app --reload --port 8000

CHANGES in v2.2:
  - All timestamps now Philippine Time (Asia/Manila, UTC+8)
  - /api/predictions now includes weekday_revenue and today_revenue_forecast
  - Revenue forecast added to predictions endpoint
"""
import os, re, math, bcrypt, uvicorn, joblib, threading, warnings, time, json, base64, secrets, hmac, hashlib
import psycopg2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
from backend.db     import get_db, query, execute
from backend.models import UserRegister, UserLogin, YoloUpdate, PushFrame

warnings.filterwarnings("ignore")
load_dotenv(override=True)

PH_TZ        = ZoneInfo("Asia/Manila")

_CAM_TOKEN_DEFAULT = "occupai_cam_2027"
_ADMIN_PASSWORD_DEFAULT = "12345678"

CAM_TOKEN    = os.getenv("CAM_TOKEN",   _CAM_TOKEN_DEFAULT)
DEPLOY_MODE  = os.getenv("DEPLOY_MODE", "local")
STREAM_PORT  = int(os.getenv("STREAM_PORT", "8001"))
LOT_CAPACITY = int(os.getenv("LOT_CAPACITY", "44"))
ADMIN_EMAIL  = os.getenv("ADMIN_EMAIL", "jpcambiado@gbox.ncf.edu.ph").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", _ADMIN_PASSWORD_DEFAULT)

if CAM_TOKEN == _CAM_TOKEN_DEFAULT:
    print(
        "[SECURITY WARNING] CAM_TOKEN is not set — using the hardcoded default value. "
        "Anyone who reads the source can push fake camera data. Set CAM_TOKEN in your "
        "environment (e.g. Render dashboard) to a private random value."
    )
if ADMIN_PASSWORD == _ADMIN_PASSWORD_DEFAULT:
    print(
        "[SECURITY WARNING] ADMIN_PASSWORD is not set — using the hardcoded default "
        "'12345678'. The admin account is only as secure as this value. Set ADMIN_PASSWORD "
        "in your environment (e.g. Render dashboard) to a strong private password."
    )
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "")
if not AUTH_SECRET_KEY:
    AUTH_SECRET_KEY = secrets.token_hex(32)
    print(
        "[auth] WARNING: AUTH_SECRET_KEY is not set. Using a random secret generated for this "
        "process — all login sessions will be invalidated on every restart/deploy. "
        "Set AUTH_SECRET_KEY in your environment for stable sessions."
    )

AUTH_TOKEN_TTL_SECONDS = 60 * 60 * 24  # 24h


def _sign_auth_token(user_id: int, role: str) -> str:
    payload = json.dumps({
        "user_id": user_id,
        "role": role,
        "exp": int(time.time()) + AUTH_TOKEN_TTL_SECONDS,
    }, separators=(",", ":")).encode()
    raw = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    sig = hmac.new(AUTH_SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def _decode_auth_token(token: str):
    """Signature + expiry check only — does not consider revocation."""
    try:
        raw, sig = token.split(".", 1)
        expected = hmac.new(AUTH_SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        padded = raw + "=" * (-len(raw) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


_revoked_tokens_lock = threading.Lock()
_revoked_tokens = {}  # token string -> exp unix timestamp, so "logout" can actually invalidate a token early


def _prune_revoked_tokens():
    now = int(time.time())
    with _revoked_tokens_lock:
        for t in [t for t, exp in _revoked_tokens.items() if exp < now]:
            _revoked_tokens.pop(t, None)


def _revoke_token(token: str):
    payload = _decode_auth_token(token)
    if not payload:
        return
    with _revoked_tokens_lock:
        _revoked_tokens[token] = int(payload.get("exp", 0))
    _prune_revoked_tokens()


def _verify_auth_token(token: str):
    payload = _decode_auth_token(token)
    if not payload:
        return None
    with _revoked_tokens_lock:
        if token in _revoked_tokens:
            return None
    return payload


_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 15 * 60    # failures older than this don't count toward the limit
_LOGIN_LOCKOUT_SECONDS = 5 * 60    # how long an account stays locked once the limit is hit

_login_attempts_lock = threading.Lock()
_login_attempts = {}  # email -> {"count": int, "window_start": float, "locked_until": float}


def _login_lockout_seconds_left(email: str):
    now = time.time()
    with _login_attempts_lock:
        entry = _login_attempts.get(email)
        if not entry:
            return 0
        if entry["locked_until"] and entry["locked_until"] > now:
            return int(entry["locked_until"] - now)
        if entry["locked_until"] and entry["locked_until"] <= now:
            _login_attempts.pop(email, None)
        return 0


def _login_record_failure(email: str):
    now = time.time()
    with _login_attempts_lock:
        entry = _login_attempts.get(email)
        if not entry or now - entry["window_start"] > _LOGIN_WINDOW_SECONDS:
            entry = {"count": 0, "window_start": now, "locked_until": 0.0}
        entry["count"] += 1
        if entry["count"] >= _LOGIN_MAX_ATTEMPTS:
            entry["locked_until"] = now + _LOGIN_LOCKOUT_SECONDS
        _login_attempts[email] = entry


def _login_record_success(email: str):
    with _login_attempts_lock:
        _login_attempts.pop(email, None)


_rate_limit_lock = threading.Lock()
_rate_limit_buckets = {}  # (bucket_name, client_key) -> [timestamps]


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request, bucket_name: str, max_calls: int, window_seconds: int):
    key = (bucket_name, _client_key(request))
    now = time.time()
    cutoff = now - window_seconds
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.get(key, [])
        bucket = [t for t in bucket if t > cutoff]
        if len(bucket) >= max_calls:
            _rate_limit_buckets[key] = bucket
            raise HTTPException(429, "Too many requests. Please slow down and try again shortly.")
        bucket.append(now)
        _rate_limit_buckets[key] = bucket


def _auth_payload_from_header(authorization: Optional[str]):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return _verify_auth_token(token)


def require_user(authorization: str = Header(None)):
    payload = _auth_payload_from_header(authorization)
    if not payload:
        raise HTTPException(401, "Authentication required")
    return payload


def require_admin(authorization: str = Header(None)):
    payload = require_user(authorization)
    if payload.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return payload


def _check_cam_token(x_cam_token: Optional[str]):
    if not x_cam_token or not hmac.compare_digest(x_cam_token, CAM_TOKEN):
        raise HTTPException(401, "Unauthorized")
DB_LOG_CONFIRM_SECONDS = max(0.0, float(os.getenv("DB_LOG_CONFIRM_SECONDS", "20")))
DB_LOG_MIN_INTERVAL_SECONDS = max(1.0, float(os.getenv("DB_LOG_MIN_INTERVAL_SECONDS", "20")))

BASE_DIR     = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "template"
MODEL_DIR    = Path(__file__).resolve().parent / "models"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
TRAINING_DATA_PATH = BASE_DIR / "parking_data_training.csv"

INTERNAL_STREAM = f"http://127.0.0.1:{STREAM_PORT}/stream"

FLAT_RATE_CAR        = float(os.getenv("FLAT_RATE_CAR", os.getenv("FLAT_RATE", "50")) or 50)
FLAT_RATE_MOTORCYCLE = float(os.getenv("FLAT_RATE_MOTORCYCLE", "25") or 25)
PWD_SENIOR_DISCOUNT_RATE = float(os.getenv("PWD_SENIOR_DISCOUNT_RATE", "0.20") or 0.20)
OCC_LOW_THRESH  = 7.0
OCC_HIGH_THRESH = 20.0

PAYMONGO_SECRET_KEY = os.getenv("PAYMONGO_SECRET_KEY", "")
PAYMONGO_PUBLIC_KEY = os.getenv("PAYMONGO_PUBLIC_KEY", "")

DEFAULT_MODEL_METRICS = {}


def _load_model_metrics():
    metrics = {k: dict(v) for k, v in DEFAULT_MODEL_METRICS.items()}
    try:
        if METRICS_PATH.exists():
            with METRICS_PATH.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            for key, value in loaded.items():
                base = metrics.get(key, {})
                base.update(value)
                metrics[key] = base
    except Exception as e:
        print(f"[ML] Could not load model metrics: {e}")
    return metrics


MODEL_METRICS = _load_model_metrics()


def _training_metrics(model_key):
    return dict(MODEL_METRICS.get(model_key, {}))


def _metric_score_pct(model_key, fallback=None):
    metrics = MODEL_METRICS.get(model_key, {})
    value = metrics.get("accuracy_pct")
    if value is None and metrics.get("r2") is not None:
        value = float(metrics["r2"]) * 100
    if value is None:
        value = fallback
    return round(float(value), 2) if value is not None else None


def _metric_score_label(model_key):
    metrics = MODEL_METRICS.get(model_key, {})
    if metrics.get("accuracy_pct") is not None:
        return "Accuracy"
    if metrics.get("r2") is not None:
        return "R2 score"
    return "Training score"


# ══════════════════════════════════════════════════════════════════
#  SoftAttention
# ══════════════════════════════════════════════════════════════════
import keras
from keras.layers import Dense, Layer

@keras.saving.register_keras_serializable(package="occupai")
class SoftAttention(Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, activation="tanh")
        self.V = Dense(1)

    def call(self, x):
        w = keras.ops.softmax(self.V(self.W(x)), axis=1)
        return keras.ops.sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


# ══════════════════════════════════════════════════════════════════
#  Feature lists
# ══════════════════════════════════════════════════════════════════
NB1_SEQ_LEN  = 24

NB1_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_morning_peak", "is_lunch_peak", "is_afternoon_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h",
    "roll_3h", "roll_7h", "roll_24h",
    "moto_ratio", "car_ratio", "ebike_ratio",
]  # 23 features

NB2_OCC_SEQ_LEN = 24

NB2_OCC_FEATURES_FALLBACK = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_morning_peak", "is_lunch_peak", "is_afternoon_peak", "is_peak_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "moto_ratio", "car_ratio", "ebike_ratio",
    "veh_lag_1h", "veh_lag_2h", "veh_lag_3h", "veh_lag_6h", "veh_lag_24h",
    "veh_roll_3h", "veh_roll_6h", "veh_roll_24h",
    "occ_lag_24h", "occ_roll_24h",
]


# ══════════════════════════════════════════════════════════════════
#  Feature Engineering
# ══════════════════════════════════════════════════════════════════
def _add_calendar(df):
    dt = df["datetime"]
    if "hour"              not in df: df["hour"]              = dt.dt.hour
    if "day_of_week"       not in df: df["day_of_week"]       = dt.dt.dayofweek
    if "month"             not in df: df["month"]             = dt.dt.month
    if "is_weekend"        not in df: df["is_weekend"]        = (dt.dt.dayofweek >= 5).astype(int)
    if "is_morning_peak"   not in df: df["is_morning_peak"]   = df["hour"].between(7,  9).astype(int)
    if "is_lunch_peak"     not in df: df["is_lunch_peak"]     = df["hour"].between(11, 13).astype(int)
    if "is_afternoon_peak" not in df: df["is_afternoon_peak"] = df["hour"].between(16, 18).astype(int)
    if "is_peak_hour"      not in df: df["is_peak_hour"]      = (
        df["is_morning_peak"] | df["is_lunch_peak"] | df["is_afternoon_peak"]
    ).astype(int)
    if "hour_sin"  not in df: df["hour_sin"]  = np.sin(2*math.pi*df["hour"]/24)
    if "hour_cos"  not in df: df["hour_cos"]  = np.cos(2*math.pi*df["hour"]/24)
    if "dow_sin"   not in df: df["dow_sin"]   = np.sin(2*math.pi*df["day_of_week"]/7)
    if "dow_cos"   not in df: df["dow_cos"]   = np.cos(2*math.pi*df["day_of_week"]/7)
    if "month_sin" not in df: df["month_sin"] = np.sin(2*math.pi*(df["month"]-1)/12)
    if "month_cos" not in df: df["month_cos"] = np.cos(2*math.pi*(df["month"]-1)/12)
    return df


def _engineer_nb1(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = _add_calendar(df)
    v = df["vehicles_hour"]

    if "lag_1h"  not in df: df["lag_1h"]  = v.shift(1).fillna(0)
    if "lag_2h"  not in df: df["lag_2h"]  = v.shift(2).fillna(0)
    if "lag_3h"  not in df: df["lag_3h"]  = v.shift(3).fillna(0)
    if "lag_24h" not in df: df["lag_24h"] = v.shift(24).fillna(0)

    if "roll_3h"  not in df: df["roll_3h"]  = v.shift(1).rolling(3,  min_periods=1).mean().fillna(0)
    if "roll_7h"  not in df: df["roll_7h"]  = v.shift(1).rolling(7,  min_periods=1).mean().fillna(0)
    if "roll_24h" not in df: df["roll_24h"] = v.shift(1).rolling(24, min_periods=1).mean().fillna(0)

    if "moto_ratio"  not in df: df["moto_ratio"]  = 0.88
    if "car_ratio"   not in df: df["car_ratio"]   = 0.02
    if "ebike_ratio" not in df: df["ebike_ratio"] = 0.01

    return df


def _engineer_nb2_occ(df, occ_feats=None):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = _add_calendar(df)
    v = df["vehicles_hour"]

    if "true_occ_pct" not in df:
        df["true_occ_pct"] = (v / LOT_CAPACITY * 100).clip(0, 100)
    o = df["true_occ_pct"]

    for sh, names in [
        (1,  ["veh_lag_1h",  "lag_1h"]),
        (2,  ["veh_lag_2h",  "lag_2h"]),
        (3,  ["veh_lag_3h",  "lag_3h"]),
        (6,  ["veh_lag_6h",  "lag_6h"]),
        (24, ["veh_lag_24h", "lag_24h"]),
    ]:
        val = v.shift(sh).fillna(0)
        for n in names:
            if n not in df: df[n] = val

    for win, names in [
        (3,  ["veh_roll_3h",  "roll_3h"]),
        (6,  ["veh_roll_6h",  "roll_6h"]),
        (7,  ["veh_roll_7h",  "roll_7h"]),
        (24, ["veh_roll_24h", "roll_24h"]),
    ]:
        val = v.shift(1).rolling(win, min_periods=1).mean().fillna(0)
        for n in names:
            if n not in df: df[n] = val

    for sh, names in [
        (1,  ["occ_lag_1h"]),
        (2,  ["occ_lag_2h"]),
        (3,  ["occ_lag_3h"]),
        (6,  ["occ_lag_6h"]),
        (24, ["occ_lag_24h"]),
    ]:
        val = o.shift(sh).fillna(0)
        for n in names:
            if n not in df: df[n] = val

    for win, names in [
        (3,  ["occ_roll_3h"]),
        (6,  ["occ_roll_6h"]),
        (24, ["occ_roll_24h"]),
    ]:
        val = o.shift(1).rolling(win, min_periods=1).mean().fillna(0)
        for n in names:
            if n not in df: df[n] = val

    if "moto_ratio"  not in df: df["moto_ratio"]  = 0.88
    if "car_ratio"   not in df: df["car_ratio"]   = 0.02
    if "ebike_ratio" not in df: df["ebike_ratio"] = 0.01

    return df


def _last_sequence(X, seq_len):
    if len(X) < seq_len:
        raise ValueError(f"Need {seq_len} rows, got {len(X)}")
    return X[-seq_len:][np.newaxis, :, :]


# ══════════════════════════════════════════════════════════════════
#  ML Engine
# ══════════════════════════════════════════════════════════════════
class _MLEngine:
    def __init__(self):
        self._nb1         = {}
        self._occ         = None
        # NOTE: dynamic pricing Random Forest (pricing_model.pkl / pricing_features.pkl)
        # was removed from loading — predict_price() has always used the formula-based
        # _dynamic_price_formula(), never this model, so it sat in memory unused. The
        # files are still on disk under backend/models/ and pricing_rf's historical
        # training metrics are still surfaced via _metric_score_pct/_training_metrics
        # in predict_price() for reference — only the live model load was dropped.
        self._rev         = None
        self._scX         = None
        self._scY         = None
        self._occ_scX     = None
        self._occ_scY     = None
        self._occ_feats   = None
        self._rev_feats   = None
        self._ready       = False

    def load(self):
        co = {"SoftAttention": SoftAttention}

        for name, fname in [
            ("Spatio-Temporal",   "spatio_temporal.keras"),
            ("LSTM",              "lstm_forecast.keras"),
            ("CNN-GRU+Attention", "cnn_gru_attention.keras"),
        ]:
            p = MODEL_DIR / fname
            if p.exists():
                try:
                    self._nb1[name] = keras.models.load_model(str(p), custom_objects=co)
                    print(f"[ML] OK {fname}  input={self._nb1[name].input_shape}")
                except Exception as e:
                    print(f"[ML] FAIL {fname}: {e}")
            else:
                print(f"[ML] missing {fname}")

        for attr, fname in [
            ("_occ",         "occupancy_model.keras"),
            ("_rev",         "revenue_model.pkl"),
            ("_scX",         "scaler_nb1_X.pkl"),
            ("_scY",         "scaler_nb1_y.pkl"),
            ("_occ_scX",     "scaler_occ_X.pkl"),
            ("_occ_scY",     "scaler_occ_y.pkl"),
            ("_occ_feats",   "occ_features.pkl"),
            ("_rev_feats",   "rev_features.pkl"),
        ]:
            p = MODEL_DIR / fname
            if p.exists():
                try:
                    if fname.endswith(".keras"):
                        setattr(self, attr, keras.models.load_model(str(p)))
                    else:
                        setattr(self, attr, joblib.load(str(p)))
                    val = getattr(self, attr)
                    extra = ""
                    if hasattr(val, "n_features_in_"):
                        extra = f"  expects={val.n_features_in_} features"
                    elif isinstance(val, list):
                        extra = f"  ({len(val)} features)"
                    print(f"[ML] OK {fname}{extra}")
                except Exception as e:
                    print(f"[ML] FAIL {fname}: {e}")

        self._load_nb1_fallback_scalers()
        self._ready = bool(self._nb1)
        if self._occ_feats is not None:
            print(f"[ML] occ_features.pkl has {len(self._occ_feats)} features: {self._occ_feats}")
        print(f"[ML] Ready={self._ready}  models={list(self._nb1)}")

    def _load_nb1_fallback_scalers(self):
        if self._scX is not None and self._scY is not None:
            return
        if not self._nb1 or not TRAINING_DATA_PATH.exists():
            return
        try:
            from sklearn.preprocessing import MinMaxScaler

            train_df = pd.read_csv(TRAINING_DATA_PATH)
            train_df = _engineer_nb1(train_df)
            feats = [f for f in NB1_FEATURES if f in train_df.columns]
            X = train_df[feats].values

            model_feats = list(self._nb1.values())[0].input_shape[-1]
            if X.shape[1] > model_feats:
                X = X[:, :model_feats]
            elif X.shape[1] < model_feats:
                X = np.pad(X, ((0, 0), (0, model_feats - X.shape[1])))

            if self._scX is None:
                self._scX = MinMaxScaler().fit(X)
                print(f"[ML] OK fallback NB1 X scaler from {TRAINING_DATA_PATH.name}  expects={self._scX.n_features_in_} features")

            if self._scY is None:
                self._scY = MinMaxScaler().fit(train_df[["vehicles_hour"]].values)
                print(f"[ML] OK fallback NB1 y scaler from {TRAINING_DATA_PATH.name}")
        except Exception as e:
            print(f"[ML] FAIL fallback NB1 scalers: {e}")

    def predict_vehicles(self, history_df, capacity=None):
        if not self._nb1:
            raise RuntimeError("NB1 models not loaded")
        capacity = max(1, int(capacity or _active_slot_capacity(LOT_CAPACITY)))
        df = _engineer_nb1(history_df)
        feats = [f for f in NB1_FEATURES if f in df.columns]
        X = df[feats].values

        if self._scX is not None:
            expected_n = self._scX.n_features_in_
            if X.shape[1] != expected_n:
                print(f"[ML] NB1 feature mismatch: have {X.shape[1]}, scaler wants {expected_n}")
                if X.shape[1] > expected_n:
                    X = X[:, :expected_n]
                else:
                    X = np.pad(X, ((0,0),(0, expected_n - X.shape[1])))
            Xs = self._scX.transform(X)
        else:
            from sklearn.preprocessing import MinMaxScaler
            Xs = MinMaxScaler().fit(X).transform(X)

        model_feats = list(self._nb1.values())[0].input_shape[-1]
        if Xs.shape[1] != model_feats:
            if Xs.shape[1] > model_feats:
                Xs = Xs[:, :model_feats]
            else:
                Xs = np.pad(Xs, ((0,0),(0, model_feats - Xs.shape[1])))

        Xi = _last_sequence(Xs, NB1_SEQ_LEN)
        preds = {}
        for name, mdl in self._nb1.items():
            try:
                y_s = float(mdl.predict(Xi, verbose=0).flatten()[0])
                if self._scY is not None:
                    y_v = float(self._scY.inverse_transform([[y_s]])[0][0])
                else:
                    mn  = float(df["vehicles_hour"].min())
                    mx  = float(df["vehicles_hour"].max())
                    y_v = y_s * (mx - mn) + mn
                preds[name] = max(0.0, round(y_v, 2))
            except Exception as e:
                print(f"[ML] {name}: {e}")

        if not preds:
            raise RuntimeError("All NB1 predictions failed")

        primary = preds.get("Spatio-Temporal", next(iter(preds.values())))
        occ_pct = round(min(primary / capacity * 100, 100.0), 1)
        last_db_ts = pd.to_datetime(df["datetime"].iloc[-1])
        now_ph = datetime.now(PH_TZ)
        data_age_hours = (now_ph - last_db_ts.replace(tzinfo=PH_TZ)).total_seconds() / 3600
        if data_age_hours > 2:
            next_ts = now_ph + timedelta(hours=1)
        else:
            next_ts = last_db_ts + timedelta(hours=1)
        data_stale = data_age_hours > 2
        status  = "LOW" if occ_pct < 30 else ("HIGH" if occ_pct >= 70 else "MEDIUM")

        return {
            "predicted_vehicles": primary,
            "predicted_occ_pct":  occ_pct,
            "slot_capacity":      capacity,
            "occupancy_status":   status,
            "prediction_for":     next_ts.strftime("%Y-%m-%d %H:%M"),
            "data_stale":         data_stale,
            "model_used":         "Spatio-Temporal",
            "confidence_pct":     _metric_score_pct("spatio_temporal"),
            "score_label":        _metric_score_label("spatio_temporal"),
            "training_metrics":   _training_metrics("spatio_temporal"),
            "all_models":         preds,
            "all_model_metrics": {
                "Spatio-Temporal":   _training_metrics("spatio_temporal"),
                "LSTM":              _training_metrics("lstm_forecast"),
                "CNN-GRU+Attention": _training_metrics("cnn_gru_attention"),
            },
        }

    def predict_occupancy(self, history_df, capacity=None):
        if self._occ is None:
            raise RuntimeError("Occupancy model not loaded")
        capacity = max(1, int(capacity or _active_slot_capacity(LOT_CAPACITY)))

        feats = self._occ_feats if self._occ_feats is not None else NB2_OCC_FEATURES_FALLBACK
        df = _engineer_nb2_occ(history_df, feats)
        feats = [f for f in feats if f in df.columns]
        X = df[feats].values

        if self._occ_scX is not None:
            expected_n = self._occ_scX.n_features_in_
            if X.shape[1] != expected_n:
                print(f"[ML] OCC feature mismatch: have {X.shape[1]}, scaler wants {expected_n}")
                if X.shape[1] > expected_n:
                    X = X[:, :expected_n]
                else:
                    X = np.pad(X, ((0,0),(0, expected_n - X.shape[1])))
            Xs = self._occ_scX.transform(X)
        else:
            from sklearn.preprocessing import MinMaxScaler
            Xs = MinMaxScaler().fit_transform(X)

        Xi  = _last_sequence(Xs, NB2_OCC_SEQ_LEN)
        y_s = float(self._occ.predict(Xi, verbose=0).flatten()[0])
        occ = float(self._occ_scY.inverse_transform([[y_s]])[0][0]) if self._occ_scY else y_s * 100.0
        occ = round(max(0.0, min(100.0, occ)), 1)
        occupied = min(capacity, max(0, round(occ / 100 * capacity)))

        return {
            "predicted_occ_pct": occ,
            "occupancy_status":  "LOW" if occ < 30 else ("HIGH" if occ >= 70 else "MEDIUM"),
            "slot_capacity":     capacity,
            "occupied_slots":    occupied,
            "free_slots":        max(0, capacity - occupied),
            "model_used":        "Occupancy-BiLSTM",
            "confidence_pct":    _metric_score_pct("occupancy_bilstm"),
            "score_label":       _metric_score_label("occupancy_bilstm"),
            "training_metrics":  _training_metrics("occupancy_bilstm"),
        }

    def predict_price(self, row, lot_capacity=None):
        vehicles = row.get("vehicles_hour", row.get("occupied", 0.0))
        when = row.get("datetime") or row.get("logged_at") or datetime.now(PH_TZ)
        result = _dynamic_price_formula(vehicles, lot_capacity, when)
        result.update({
            "model_used":       "Dynamic Pricing Formula",
            "confidence_pct":   _metric_score_pct("pricing_rf"),
            "score_label":      _metric_score_label("pricing_rf"),
            "training_metrics": _training_metrics("pricing_rf"),
        })
        return result

    def predict_revenue(self, row):
        if self._rev is None or self._rev_feats is None:
            raise RuntimeError("Revenue model not loaded")
        X   = np.array([[row.get(f, 0.0) for f in self._rev_feats]])
        rev = max(0.0, float(self._rev.predict(X)[0]))
        return {
            "predicted_daily_revenue_php": round(rev, 2),
            "model_used":                  "Revenue-GBR",
            "confidence_pct":              _metric_score_pct("revenue_gbr"),
            "score_label":                 _metric_score_label("revenue_gbr"),
            "training_metrics":            _training_metrics("revenue_gbr"),
        }


ml = _MLEngine()


# ══════════════════════════════════════════════════════════════════
#  Insight cache
# ══════════════════════════════════════════════════════════════════
_insight_cache: dict = {}
_insight_lock = threading.Lock()


def _live_status_from_state(snapshot=None):
    if snapshot is None:
        with state_lock:
            snapshot = dict(state)

    total = int(snapshot.get("total") or 0)
    occupied = int(snapshot.get("occupied") or 0)
    free = int(snapshot.get("free") if snapshot.get("free") is not None else max(0, total - occupied))
    pct = round(float(snapshot.get("occupancy_pct") or (occupied / total * 100 if total else 0.0)), 1)

    def _pct_word(p):
        if p >= 90: return "almost completely full"
        if p >= 70: return "very busy"
        if p >= 50: return "moderately busy"
        if p >= 25: return "lightly used"
        return "mostly empty"

    if snapshot.get("lot_full"):
        text = (
            "The parking lot is completely full right now. "
            "No available spaces remain. Consider redirecting incoming vehicles."
        )
    elif total == 0:
        text = "Waiting for the camera detector to connect."
    else:
        text = (
            f"The parking lot is currently {_pct_word(pct)}. "
            f"{free} space{'s are' if free != 1 else ' is'} available out of {total} total. "
            f"{occupied} vehicle{'s are' if occupied != 1 else ' is'} parked right now."
        )

    return {
        "live_status": text,
        "live": {
            "occupied": occupied,
            "free": free,
            "total": total,
            "occupancy_pct": pct,
            "zones": snapshot.get("zones") or {},
        },
    }


def _run_insights_now():
    now  = datetime.now(PH_TZ)
    hour = now.hour
    out  = {"generated_at": now.strftime("%Y-%m-%d %H:%M:%S")}

    with state_lock:
        s = dict(state)
    hist = list(history)
    df   = _db_history()

    def _hour_lbl(h):
        return f"{h%12 or 12}:00 {'AM' if h<12 else 'PM'}"

    def _pct_word(p):
        if p >= 90: return "almost completely full"
        if p >= 70: return "very busy"
        if p >= 50: return "moderately busy"
        if p >= 25: return "lightly used"
        return "mostly empty"

    def _trend(hist):
        if len(hist) < 2: return "Not enough data yet to describe a trend."
        pts   = [float(r["pct"]) for r in hist[-5:]]
        delta = pts[-1] - pts[0]
        if delta >  15: return "Occupancy has been rising quickly in the last few minutes."
        if delta >   5: return "Occupancy is gradually increasing."
        if delta < -15: return "Occupancy has been dropping quickly — spaces are opening up."
        if delta <  -5: return "Occupancy is slowly decreasing."
        return "Occupancy has been stable recently."

    occ      = s.get("occupancy_pct", 0)
    free     = s.get("free", 0)
    total    = s.get("total", 0)
    occupied = s.get("occupied", 0)
    active_capacity = _active_slot_capacity()

    out.update(_live_status_from_state(s))
    out["active_slot_capacity"] = active_capacity

    out["trend"] = _trend(hist)

    if ml._ready and len(df) >= 24:
        try:
            r   = ml.predict_vehicles(df, capacity=active_capacity)
            n   = int(round(r["predicted_vehicles"]))
            pf  = r["prediction_for"]
            st  = r["occupancy_status"]
            cf  = r.get("confidence_pct")
            score_label = r.get("score_label", "Training score")
            score_text = f" ({score_label}: {cf:.0f}%)" if cf is not None else ""
            desc = {
                "LOW":    "quiet — most spaces should be free",
                "MEDIUM": "moderately busy — about half the lot may be used",
                "HIGH":   "very busy — the lot could fill up",
            }
            stale_note = " (based on older data)" if r.get("data_stale") else ""
            time_label = pf[-5:]
            out["vehicle_forecast"] = (
                f"Around {time_label}, the system expects roughly "
                f"{n} vehicle{'s' if n!=1 else ''} to be in the lot. "
                f"It will likely be {desc.get(st,'uncertain')}.{score_text}{stale_note}"
            )
        except Exception as e:
            out["vehicle_forecast"] = f"Vehicle forecast temporarily unavailable. ({e})"
    else:
        out["vehicle_forecast"] = (
            "Vehicle forecast needs 24+ hours of history. "
            "Keep the detector running and it will start predicting soon."
        )

    if ml._ready and ml._occ is not None and len(df) >= 24:
        try:
            r  = ml.predict_occupancy(df, capacity=active_capacity)
            pf = r["free_slots"]
            cap = r.get("slot_capacity", active_capacity)
            st = r["occupancy_status"]
            urgency = {
                "LOW":    "You should have plenty of space available.",
                "MEDIUM": "Expect moderate traffic — some spaces may run low.",
                "HIGH":   "The lot is predicted to fill up. Consider preparing overflow parking.",
            }
            out["occupancy_forecast"] = (
                f"In the next hour the lot will be {_pct_word(r['predicted_occ_pct'])}, "
                f"with roughly {pf} of {cap} space{'s' if cap!=1 else ''} free. "
                f"{urgency.get(st,'')}"
            )
        except Exception as e:
            out["occupancy_forecast"] = f"Occupancy forecast temporarily unavailable. ({e})"
    else:
        out["occupancy_forecast"] = (
            "Occupancy forecast will appear after 24+ hours of operation."
        )

    try:
        price_vehicle_count = occupied if total else (float(df["vehicles_hour"].iloc[-1]) if not df.empty and "vehicles_hour" in df.columns else 0.0)
        r     = _dynamic_price_formula(price_vehicle_count, active_capacity)
        price = r["recommended_price_php"]
        chg   = r["price_change_pct"]
        ctx   = r.get("pricing_context", {})
        flat_rate = float(r.get("flat_rate_php") or _current_flat_rate())
        occ_formula = ctx.get("occupancy_pct", 0)
        occ_mult = ctx.get("occupancy_multiplier", 1.0)
        day_mult = ctx.get("day_multiplier", 1.0)
        day_rule = ctx.get("day_rule", "Weekday")
        if r.get("pricing_reason") == "manual_admin_override":
            formula_note = (
                f"Live context is {occ_formula:.0f}% occupancy and "
                f"{day_mult:.2f}x {day_rule.lower()} factor."
            )
            out["pricing"] = f"Manual admin rate is PHP {price:.0f}/hr. Dynamic pricing is paused. {formula_note}"
        elif abs(chg) < 5:
            formula_note = (
                f"Formula uses {occ_formula:.0f}% occupancy, {occ_mult:.2f}x occupancy multiplier, "
                f"and {day_mult:.2f}x {day_rule.lower()} multiplier."
            )
            out["pricing"] = f"Demand is normal. PHP {price:.0f}/hr is appropriate. {formula_note}"
        elif chg > 0:
            formula_note = (
                f"Formula uses {occ_formula:.0f}% occupancy, {occ_mult:.2f}x occupancy multiplier, "
                f"and {day_mult:.2f}x {day_rule.lower()} multiplier."
            )
            out["pricing"] = (
                f"Demand is higher than usual. Raising the rate to PHP {price:.0f}/hr "
                f"(+{chg:.0f}% above the PHP {flat_rate:.0f} flat rate) is recommended. {formula_note}"
            )
        else:
            formula_note = (
                f"Formula uses {occ_formula:.0f}% occupancy, {occ_mult:.2f}x occupancy multiplier, "
                f"and {day_mult:.2f}x {day_rule.lower()} multiplier."
            )
            out["pricing"] = (
                f"Demand is lower than usual. Offering PHP {price:.0f}/hr "
                f"({abs(chg):.0f}% below the PHP {flat_rate:.0f} standard) could attract more drivers. {formula_note}"
            )
        if r.get("pwd_senior_price_php") is not None:
            out["pricing"] += f" PWD/Senior discounted rate: PHP {float(r['pwd_senior_price_php']):.0f}/hr."
        out["pricing_details"] = r
    except Exception as e:
        out["pricing"] = f"Pricing recommendation temporarily unavailable. ({e})"

    try:
        weekday_revenue, today_revenue = _training_weekday_revenue_forecast(active_capacity)
        source = "training data"
        if not any(weekday_revenue.values()):
            weekday_revenue, today_revenue = _logged_weekday_revenue_forecast(active_capacity)
            source = "deduplicated live logs"

        if today_revenue is not None:
            out["revenue_forecast"] = (
                f"Today's estimated revenue is approximately PHP {today_revenue:,.2f}. "
                f"This uses the occupancy pricing formula and {source}, so repeated camera refreshes "
                f"do not inflate the total."
            )
        else:
            out["revenue_forecast"] = "Revenue forecast needs training data or at least one day of live logs."
    except Exception as e:
        out["revenue_forecast"] = f"Revenue forecast temporarily unavailable. ({e})"

    try:
        rows = query("""
            SELECT EXTRACT(HOUR FROM logged_at AT TIME ZONE 'Asia/Manila') AS hour,
                   AVG(occupied) AS avg_veh
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '7 days'
            GROUP BY hour ORDER BY hour
        """)
        training_hourly = _training_hourly_vehicle_avg()
        if training_hourly:
            hourly = training_hourly
            source = "historical training data"
        elif rows and len(rows) >= 3:
            hourly = {int(r["hour"]): float(r["avg_veh"]) for r in rows}
            source = "the last 7 days"
        else:
            hourly = {}

        if hourly:
            peak_h  = max(hourly, key=lambda h: hourly[h])
            quiet_h = min(hourly, key=lambda h: hourly[h])
            now_desc = (
                "You are currently in a peak period — expect the lot to stay busy."
                if abs(hour - peak_h) <= 1
                else "Traffic should be relatively normal at this hour."
            )
            out["peak_hours"] = (
                f"Based on {source}, the busiest time is around "
                f"{_hour_lbl(peak_h)} and the quietest is around "
                f"{_hour_lbl(quiet_h)}. {now_desc}"
            )
        else:
            out["peak_hours"] = "Peak hour analysis needs at least 7 days of data."
    except Exception as e:
        out["peak_hours"] = f"Peak hour analysis not available. ({e})"

    actions = []
    if s.get("lot_full"):
        actions.append("Activate overflow parking immediately.")
    elif occ >= 80:
        actions.append("The lot is almost full — consider opening overflow parking soon.")
    try:
        price_vehicle_count = occupied if total else (float(df["vehicles_hour"].iloc[-1]) if not df.empty and "vehicles_hour" in df.columns else 0.0)
        r = _dynamic_price_formula(price_vehicle_count, active_capacity)
        if r.get("pricing_reason") == "manual_admin_override":
            actions.append("Manual parking rate is active. Driver-facing price is controlled by admin settings.")
        elif r["price_change_pct"] > 10:
            actions.append("Consider raising the parking rate. Formula-based occupancy demand is high.")
        elif r["price_change_pct"] < -10:
            actions.append("Consider a promotional rate. Formula-based occupancy demand is low.")
    except Exception:
        pass

    out["admin_action_items"] = actions
    out["admin_action"] = (
        " ".join(actions) if actions
        else "✅ No immediate action needed. The lot is operating normally."
    )

    with _insight_lock:
        _insight_cache.clear()
        _insight_cache.update(out)

    print(f"[insights] Refreshed at {out['generated_at']}")


def _insight_scheduler():
    time.sleep(15)
    while True:
        try:
            _run_insights_now()
        except Exception as e:
            print(f"[insights] Scheduler error: {e}")
        time.sleep(60 * 60)


# ══════════════════════════════════════════════════════════════════
#  Lifespan
# ══════════════════════════════════════════════════════════════════
def _ensure_core_tables():
    try:
        execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGSERIAL PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                full_name TEXT GENERATED ALWAYS AS (BTRIM(first_name || ' ' || last_name)) STORED,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'driver',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                last_login TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower ON users (LOWER(email))"
        )
        execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        execute("""
            CREATE TABLE IF NOT EXISTS parking_logs (
                log_id BIGSERIAL PRIMARY KEY,
                occupied INTEGER NOT NULL,
                free INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL,
                occupancy_pct NUMERIC(6,2) NOT NULL DEFAULT 0,
                lot_full BOOLEAN NOT NULL DEFAULT FALSE,
                logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    except Exception as e:
        print(f"[DB] core table setup warning: {e}")


def _ensure_parking_log_indexes():
    try:
        execute(
            "CREATE INDEX IF NOT EXISTS idx_parking_logs_logged_at_desc "
            "ON parking_logs (logged_at DESC)"
        )
    except Exception as e:
        print(f"[DB] parking_logs index setup warning: {e}")


def _ensure_payment_table():
    try:
        execute("""
            CREATE TABLE IF NOT EXISTS parking_payments (
                payment_id BIGSERIAL PRIMARY KEY,
                user_id INTEGER,
                vehicle_type TEXT NOT NULL DEFAULT 'car',
                regular_price_php NUMERIC(10,2) NOT NULL,
                discount_type TEXT NOT NULL DEFAULT 'none',
                discount_rate NUMERIC(6,4) NOT NULL DEFAULT 0,
                discount_amount_php NUMERIC(10,2) NOT NULL DEFAULT 0,
                final_amount_php NUMERIC(10,2) NOT NULL,
                payment_method TEXT NOT NULL DEFAULT 'cash',
                notes TEXT,
                paid_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        execute(
            "CREATE INDEX IF NOT EXISTS idx_parking_payments_paid_at_desc "
            "ON parking_payments (paid_at DESC)"
        )
        execute("ALTER TABLE parking_payments ADD COLUMN IF NOT EXISTS user_id INTEGER")
        execute("ALTER TABLE parking_payments ADD COLUMN IF NOT EXISTS vehicle_type TEXT NOT NULL DEFAULT 'car'")
        execute("ALTER TABLE parking_payments ADD COLUMN IF NOT EXISTS discount_id_number TEXT")
    except Exception as e:
        print(f"[DB] payment table setup warning: {e}")


def _ensure_gcash_checkout_table():
    try:
        execute("""
            CREATE TABLE IF NOT EXISTS gcash_checkouts (
                ref TEXT PRIMARY KEY,
                checkout_id TEXT,
                user_id INTEGER,
                vehicle_type TEXT NOT NULL DEFAULT 'car',
                regular_price_php NUMERIC(10,2) NOT NULL,
                discount_type TEXT NOT NULL DEFAULT 'none',
                final_amount_php NUMERIC(10,2) NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        execute("ALTER TABLE gcash_checkouts ADD COLUMN IF NOT EXISTS vehicle_type TEXT NOT NULL DEFAULT 'car'")
        execute("ALTER TABLE gcash_checkouts ADD COLUMN IF NOT EXISTS discount_id_number TEXT")
        execute("ALTER TABLE gcash_checkouts ADD COLUMN IF NOT EXISTS error_message TEXT")
    except Exception as e:
        print(f"[DB] gcash checkout table setup warning: {e}")


def _ensure_admin_settings_table():
    # Backs the admin-toggle settings (discounts, layout override, price
    # override) — these used to live in a local .env file, which Render
    # wipes on every restart (including OOM crash-restarts), silently
    # resetting toggles back to their code defaults. Postgres survives
    # restarts, so this is the durable store for them now.
    try:
        execute("""
            CREATE TABLE IF NOT EXISTS admin_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    except Exception as e:
        print(f"[DB] admin_settings table setup warning: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_core_tables()
    try:
        _ensure_default_admin()
        print(f"[auth] Default admin ready: {ADMIN_EMAIL}")
    except Exception as e:
        print(f"[auth] Default admin setup warning: {e}")
    _ensure_parking_log_indexes()
    _ensure_payment_table()
    _ensure_gcash_checkout_table()
    _ensure_admin_settings_table()
    print("[OccupAI] Loading ML models...")
    try:
        ml.load()
    except Exception as e:
        print(f"[OccupAI] ML warning: {e}")
    t = threading.Thread(target=_insight_scheduler, daemon=True, name="insight-scheduler")
    t.start()
    print("[OccupAI] Insight scheduler started (15s warmup, then hourly).")
    yield
    print("[OccupAI] Shutdown.")


# ══════════════════════════════════════════════════════════════════
#  App
# ══════════════════════════════════════════════════════════════════
app = FastAPI(title="OccupAI API", version="2.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(TEMPLATE_DIR)), name="static")

_configured_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _configured_origins:
    _cors_origins = _configured_origins
elif DEPLOY_MODE == "local":
    # Local dev: keep it permissive so testing from any localhost port just works.
    _cors_origins = ["*"]
else:
    print(
        "[SECURITY WARNING] ALLOWED_ORIGINS is not set outside local dev — falling back to "
        "the app's own BACKEND_URL only. Set ALLOWED_ORIGINS (comma-separated) in your "
        "environment to control this explicitly."
    )
    _cors_origins = [os.getenv("BACKEND_URL", "").strip()] if os.getenv("BACKEND_URL") else ["*"]

app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    import traceback
    print(f"[unhandled] {request.method} {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ══════════════════════════════════════════════════════════════════
#  Shared state
# ══════════════════════════════════════════════════════════════════
state = {
    "occupied": 0, "free": 0, "total": 0, "occupancy_pct": 0.0,
    "lot_full": False, "fps": 0.0, "timestamp": "",
    "yolo_count": 0, "yolo_boxes": [], "slots": [], "zones": {},
    "car_count": 0, "motorcycle_count": 0,
}
history    = deque(maxlen=200)
state_lock = threading.Lock()
db_log_lock = threading.Lock()
db_log_state = {
    "signature": None,
    "first_seen": 0.0,
    "last_insert": 0.0,
    "zone_occupied_since": {},
    "confirmed_zones": {},
}


def _layout_capacity_from_env(mode=None):
    level = (mode or _read_setting("FORCE_DEMAND_LEVEL", "") or "").strip().upper()
    if level not in {"NORMAL", "BUSY", "HIGH"}:
        return 0
    total = 0
    for row in (1, 2, 3):
        raw = os.getenv(f"{level}_R{row}_N", os.getenv(f"R{row}_N", "0"))
        try:
            total += max(0, int(float(raw or 0)))
        except (TypeError, ValueError):
            pass
    return total


def _active_slot_capacity(default=None):
    with state_lock:
        live_total = int(state.get("total") or 0)
    if live_total > 0:
        return live_total

    adjustment = globals().get("_last_slot_adjustment") or {}
    try:
        adjusted_total = int(adjustment.get("n_slots") or adjustment.get("total") or 0)
        if adjusted_total > 0:
            return adjusted_total
    except (TypeError, ValueError):
        pass

    slots = adjustment.get("slots") or []
    if slots:
        return len(slots)

    env_total = _layout_capacity_from_env()
    if env_total > 0:
        return env_total

    return int(default or LOT_CAPACITY)


def _parking_log_signature(data: YoloUpdate):
    zones = data.zones or {}
    zone_state = tuple(sorted((str(name), bool(value)) for name, value in zones.items()))
    return (
        int(data.occupied or 0),
        int(data.free or 0),
        int(data.total or 0),
        zone_state,
    )


def _should_insert_parking_log(data: YoloUpdate):
    now = time.monotonic()
    zones = data.zones or {}

    with db_log_lock:
        if zones:
            active_names = {str(name) for name in zones}
            for name in list(db_log_state["zone_occupied_since"]):
                if name not in active_names:
                    db_log_state["zone_occupied_since"].pop(name, None)
            for name in list(db_log_state["confirmed_zones"]):
                if name not in active_names:
                    db_log_state["confirmed_zones"].pop(name, None)

            stable_times = []
            for raw_name, raw_occupied in zones.items():
                name = str(raw_name)
                if bool(raw_occupied):
                    since = db_log_state["zone_occupied_since"].setdefault(name, now)
                    stable_for_zone = now - since
                    stable_times.append(stable_for_zone)
                    db_log_state["confirmed_zones"][name] = (
                        stable_for_zone >= DB_LOG_CONFIRM_SECONDS
                    )
                else:
                    db_log_state["zone_occupied_since"].pop(name, None)
                    db_log_state["confirmed_zones"][name] = False

            confirmed_zone_state = tuple(
                sorted(
                    (name, bool(value))
                    for name, value in db_log_state["confirmed_zones"].items()
                    if name in active_names
                )
            )
            confirmed_occupied = sum(1 for _, occupied in confirmed_zone_state if occupied)
            total = int(data.total or len(active_names) or 0)
            confirmed_free = max(0, total - confirmed_occupied)
            confirmed_pct = round(confirmed_occupied / total * 100, 1) if total else 0.0
            stable_for = max(stable_times) if stable_times else 0.0

            if int(data.occupied or 0) > 0 and confirmed_occupied == 0:
                return False, stable_for, None

            signature = (confirmed_occupied, confirmed_free, total, confirmed_zone_state)
            if now - db_log_state["last_insert"] < DB_LOG_MIN_INTERVAL_SECONDS:
                return False, stable_for, None

            db_log_state["last_insert"] = now
            return True, stable_for, {
                "occupied": confirmed_occupied,
                "free": confirmed_free,
                "total": total,
                "occupancy_pct": confirmed_pct,
                "lot_full": total > 0 and confirmed_free == 0,
                "signature": signature,
            }

        signature = _parking_log_signature(data)
        if signature != db_log_state["signature"]:
            db_log_state["signature"] = signature
            db_log_state["first_seen"] = now
            return False, 0.0, None

        stable_for = now - db_log_state["first_seen"]
        if stable_for < DB_LOG_CONFIRM_SECONDS:
            return False, stable_for, None

        if now - db_log_state["last_insert"] < DB_LOG_MIN_INTERVAL_SECONDS:
            return False, stable_for, None

        db_log_state["last_insert"] = now
        return True, stable_for, {
            "occupied": int(data.occupied or 0),
            "free": int(data.free or 0),
            "total": int(data.total or 0),
            "occupancy_pct": round(float(data.occupancy_pct or 0.0), 1),
            "lot_full": bool(data.lot_full),
            "signature": signature,
        }


# ══════════════════════════════════════════════════════════════════
#  Page routes
# ══════════════════════════════════════════════════════════════════
@app.get("/",          response_class=FileResponse)
def root():            return FileResponse(str(TEMPLATE_DIR / "login.html"))
@app.get("/login",     response_class=FileResponse)
def login_page():      return FileResponse(str(TEMPLATE_DIR / "login.html"))
@app.get("/register",  response_class=FileResponse)
def register_page():   return FileResponse(str(TEMPLATE_DIR / "register.html"))
@app.get("/dashboard", response_class=FileResponse)
def dashboard_page():  return FileResponse(str(TEMPLATE_DIR / "dashboard.html"))
@app.get("/driver",    response_class=FileResponse)
def driver_page():     return FileResponse(str(TEMPLATE_DIR / "driver.html"))
@app.get("/analytics", response_class=FileResponse)
def analytics_page():
    page = TEMPLATE_DIR / "analytics.html"
    if not page.exists():
        page = TEMPLATE_DIR / "dashboard.html"
    return FileResponse(str(page))


# ══════════════════════════════════════════════════════════════════
#  Health
# ══════════════════════════════════════════════════════════════════
@app.get("/status")
def status():
    return {
        "status":        "ok",
        "mode":          DEPLOY_MODE,
        "time_ph":       datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S PHT"),
        "stream_proxy":  f"http://localhost:8000/api/stream",
        "stream_direct": f"http://localhost:{STREAM_PORT}/stream",
        "ml_ready":      ml._ready,
        "ml_models":     list(ml._nb1.keys()),
        "lot_capacity":  LOT_CAPACITY,
        "active_slot_capacity": _active_slot_capacity(),
    }


# ══════════════════════════════════════════════════════════════════
#  MJPEG proxy
# ══════════════════════════════════════════════════════════════════
@app.get("/api/stream")
async def stream_proxy():
    async def _pipe():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", INTERNAL_STREAM) as resp:
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        yield chunk
        except Exception as e:
            print(f"[proxy] {e}")

    return StreamingResponse(
        _pipe(),
        media_type="multipart/x-mixed-replace; boundary=--occupaiframe",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ══════════════════════════════════════════════════════════════════
#  YOLO endpoints
# ══════════════════════════════════════════════════════════════════
@app.post("/yolo/update")
def yolo_update(data: YoloUpdate, x_cam_token: str = Header(...)):
    _check_cam_token(x_cam_token)
    # PH time timestamp
    ts = datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S")
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
            "car_count":     data.car_count,
            "motorcycle_count": data.motorcycle_count,
            "yolo_boxes":    data.yolo_boxes,
            "slots":         data.slots,
            "zones":         data.zones,
        })
    history.append({
        "time": ts, "occupied": data.occupied,
        "total": data.total, "pct": round(data.occupancy_pct, 1),
    })
    db_logged, stable_for, db_snapshot = _should_insert_parking_log(data)
    if db_logged:
        try:
            execute(
                "INSERT INTO parking_logs (occupied,free,total,occupancy_pct,lot_full) "
                "VALUES (%s,%s,%s,%s,%s)",
                (
                    db_snapshot["occupied"],
                    db_snapshot["free"],
                    db_snapshot["total"],
                    db_snapshot["occupancy_pct"],
                    db_snapshot["lot_full"],
                ),
            )
        except Exception as e:
            print(f"[DB] {e}")
            db_logged = False
    return {
        "ok": True,
        "db_logged": db_logged,
        "stable_for_sec": round(stable_for, 1),
        "db_confirm_sec": DB_LOG_CONFIRM_SECONDS,
        "db_snapshot": db_snapshot,
    }


@app.post("/yolo/push-frame")
def push_frame(data: PushFrame, x_cam_token: str = Header(...)):
    _check_cam_token(x_cam_token)
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════
#  Live state
# ══════════════════════════════════════════════════════════════════
@app.get("/api/stats")
def api_stats():
    with state_lock: return dict(state)

@app.get("/api/history")
def api_history():
    return list(history)

@app.get("/api/occupancy")
def api_occupancy():
    with state_lock:
        return {k: state[k] for k in ("occupied","free","total","occupancy_pct","zones")}


# ══════════════════════════════════════════════════════════════════
#  DB history — pulls 168h (7 days) for better lag features
# ══════════════════════════════════════════════════════════════════
def _db_history(hours: int = 168) -> pd.DataFrame:
    try:
        limit = max(1, min(int(hours), 10000))
        rows = query(
            "SELECT logged_at AS datetime, occupied AS vehicles_hour "
            "FROM parking_logs ORDER BY logged_at DESC LIMIT %s",
            (limit,),
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["datetime"]      = pd.to_datetime(df["datetime"])
        df["vehicles_hour"] = df["vehicles_hour"].astype(float)
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        print(f"[ML] DB: {e}")
        return pd.DataFrame()


def _training_hourly_vehicle_avg():
    if not TRAINING_DATA_PATH.exists():
        return {}
    try:
        df = pd.read_csv(TRAINING_DATA_PATH, usecols=["hour", "vehicles_hour"])
        hourly = df.groupby("hour")["vehicles_hour"].mean().to_dict()
        return {int(h): float(v) for h, v in hourly.items()}
    except Exception as e:
        print(f"[predictions] training hourly fallback unavailable: {e}")
        return {}


def _training_hourly_occ_pct(capacity=None):
    capacity = max(1, int(capacity or _active_slot_capacity(LOT_CAPACITY)))
    hourly = _training_hourly_vehicle_avg()
    return {
        str(h): round(min(max(hourly.get(h, 0.0) / capacity * 100, 0.0), 100.0), 1)
        for h in range(24)
    }


def _historical_peak_context(now=None):
    now = now or datetime.now(PH_TZ)
    capacity = max(1, int(_active_slot_capacity(LOT_CAPACITY)))
    hourly = _training_hourly_vehicle_avg()
    if not hourly:
        return {
            "current_hour": now.hour,
            "is_peak_window": False,
            "source": "none",
        }

    peak_h = max(hourly, key=lambda h: hourly[h])
    current_h = now.hour
    current_demand = float(hourly.get(current_h, 0.0))
    peak_demand = max(float(v) for v in hourly.values()) or 1.0
    demand_ratio = current_demand / peak_demand
    ranked_hours = sorted(hourly, key=lambda h: hourly[h], reverse=True)
    top_hours = sorted(ranked_hours[:3])
    premium_pct = max(0.0, min(0.30, (demand_ratio - 0.45) * 0.60))
    flat_rate = _current_flat_rate()
    demand_floor = round(flat_rate * (1.0 + premium_pct), 2) if premium_pct > 0 else None

    return {
        "current_hour": current_h,
        "peak_hour": int(peak_h),
        "peak_label": f"{peak_h % 12 or 12}:00 {'AM' if peak_h < 12 else 'PM'}",
        "is_peak_window": current_h in top_hours,
        "peak_window_hours": top_hours,
        "expected_vehicles": round(current_demand, 2),
        "expected_occ_pct": round(min(current_demand / capacity * 100, 100.0), 1),
        "slot_capacity": capacity,
        "demand_ratio_to_peak": round(demand_ratio, 3),
        "demand_price_floor_php": demand_floor,
        "source": "historical_training_data",
    }


def _coerce_datetime(value, fallback=None):
    fallback = fallback or datetime.now(PH_TZ)
    if isinstance(value, datetime):
        return value.astimezone(PH_TZ) if value.tzinfo else value.replace(tzinfo=PH_TZ)
    try:
        parsed = pd.to_datetime(value)
        if pd.isna(parsed):
            return fallback
        dt = parsed.to_pydatetime()
        return dt.astimezone(PH_TZ) if dt.tzinfo else dt.replace(tzinfo=PH_TZ)
    except Exception:
        return fallback


def _occupancy_price_multiplier(occupancy_pct):
    pct = float(occupancy_pct or 0.0)
    if pct >= 90:
        return 1.80
    if pct >= 70:
        return 1.60
    if pct >= 50:
        return 1.40
    if pct >= 30:
        return 1.20
    if pct >= 10:
        return 1.00
    return 0.80


def _day_price_multiplier(when):
    dt = _coerce_datetime(when)
    if dt.weekday() == 0:
        return 1.10, "Monday"
    if dt.weekday() in (5, 6):
        return 0.90, "Weekend"
    return 1.00, "Weekday"


def _with_pwd_senior_discount(result):
    rate = max(0.0, min(1.0, float(PWD_SENIOR_DISCOUNT_RATE)))
    price = float(result.get("recommended_price_php") or 0.0)
    discount_amount = round(price * rate, 2)
    discounted_price = round(max(0.0, price - discount_amount), 2)
    result.update({
        "pwd_senior_discount_rate": rate,
        "pwd_senior_discount_pct": round(rate * 100, 1),
        "pwd_senior_discount_amount_php": discount_amount,
        "pwd_senior_price_php": discounted_price,
        "pwd_senior_note": f"PWD and senior citizens get {rate * 100:.0f}% off the active parking rate.",
    })
    return result


def _dynamic_price_formula(vehicles_hour=0.0, lot_capacity=None, when=None, vehicle_type="car"):
    vt = _normalize_vehicle_type(vehicle_type)
    capacity = max(1, int(lot_capacity or _active_slot_capacity(LOT_CAPACITY)))
    vehicles = max(0.0, float(vehicles_hour or 0.0))
    occupancy_pct = round(min(100.0, (vehicles / capacity) * 100.0), 2)
    occupancy_multiplier = _occupancy_price_multiplier(occupancy_pct)
    day_multiplier, day_rule = _day_price_multiplier(when or datetime.now(PH_TZ))
    flat_rate = _current_flat_rate(vt)

    manual_price = _manual_price_override(vt)
    if manual_price is not None:
        change_pct = round((manual_price - flat_rate) / flat_rate * 100.0, 1)
        return _with_pwd_senior_discount({
            "vehicle_type": vt,
            "recommended_price_php": manual_price,
            "base_model_price_php": manual_price,
            "flat_rate_php": flat_rate,
            "price_change_pct": change_pct,
            "pricing_reason": "manual_admin_override",
            "pricing_context": {
                "vehicles_hour": round(vehicles, 2),
                "lot_capacity": capacity,
                "occupancy_pct": occupancy_pct,
                "occupancy_multiplier": occupancy_multiplier,
                "occupancy_price_php": round(flat_rate * occupancy_multiplier, 2),
                "day_multiplier": day_multiplier,
                "day_rule": day_rule,
                "manual_override": True,
            },
            "formula": "price = admin_manual_price",
            "price_note": "Admin-set parking rate.",
        })

    occupancy_price = round(flat_rate * occupancy_multiplier, 2)
    final_price = round(occupancy_price * day_multiplier, 2)
    change_pct = round((final_price - flat_rate) / flat_rate * 100.0, 1)
    if change_pct > 10:
        note = "Peak-demand parking rate."
    elif change_pct < -10:
        note = "Lower-demand parking rate."
    else:
        note = "Standard parking rate."
    return _with_pwd_senior_discount({
        "vehicle_type": vt,
        "recommended_price_php": final_price,
        "base_model_price_php": occupancy_price,
        "flat_rate_php": flat_rate,
        "price_change_pct": change_pct,
        "pricing_reason": "occupancy_formula",
        "pricing_context": {
            "vehicles_hour": round(vehicles, 2),
            "lot_capacity": capacity,
            "occupancy_pct": occupancy_pct,
            "occupancy_multiplier": occupancy_multiplier,
            "occupancy_price_php": occupancy_price,
            "day_multiplier": day_multiplier,
            "day_rule": day_rule,
        },
        "formula": (
            "price = flat_rate * occupancy_multiplier * day_multiplier; "
            "occupancy_pct = vehicles_hour / lot_capacity * 100"
        ),
        "price_note": note,
    })


def _training_weekday_revenue_forecast(capacity=None):
    capacity = max(1, int(capacity or _active_slot_capacity(LOT_CAPACITY)))
    ordered_days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    pandas_day_labels = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }

    if not TRAINING_DATA_PATH.exists():
        return {day: 0.0 for day in ordered_days}, None

    try:
        df = pd.read_csv(
            TRAINING_DATA_PATH,
            usecols=["datetime", "day_of_week", "vehicles_hour"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["vehicles_hour"] = pd.to_numeric(df["vehicles_hour"], errors="coerce").fillna(0.0)
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce")
        df = df.dropna(subset=["datetime", "day_of_week"])

        revenues = []
        for row in df.itertuples(index=False):
            vehicles = max(0.0, float(row.vehicles_hour or 0.0))
            price = _dynamic_price_formula(vehicles, capacity, row.datetime)["recommended_price_php"]
            revenues.append(vehicles * price)

        df["formula_revenue"] = revenues
        daily = (
            df.groupby([df["datetime"].dt.date, "day_of_week"], as_index=False)["formula_revenue"]
            .sum()
        )

        weekday_revenue = {day: None for day in ordered_days}
        for dow, group in daily.groupby("day_of_week"):
            label = pandas_day_labels.get(int(dow))
            if label:
                weekday_revenue[label] = round(float(group["formula_revenue"].mean()), 2)

        for day in ordered_days:
            if weekday_revenue[day] is None:
                valid = [v for v in weekday_revenue.values() if v is not None and v > 0]
                weekday_revenue[day] = round(sum(valid) / len(valid), 2) if valid else 0.0

        today_label = datetime.now(PH_TZ).strftime("%a")
        return weekday_revenue, weekday_revenue.get(today_label)
    except Exception as e:
        print(f"[predictions revenue] training formula fallback unavailable: {e}")
        return {day: 0.0 for day in ordered_days}, None


def _logged_weekday_revenue_forecast(capacity=None):
    capacity = max(1, int(capacity or _active_slot_capacity(LOT_CAPACITY)))
    ordered_days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    sql_day_labels = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}

    try:
        rows = query("""
            SELECT DATE(logged_at AT TIME ZONE 'Asia/Manila') AS date,
                   EXTRACT(DOW FROM logged_at AT TIME ZONE 'Asia/Manila') AS dow,
                   EXTRACT(HOUR FROM logged_at AT TIME ZONE 'Asia/Manila') AS hour,
                   AVG(occupied) AS avg_vehicles
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '30 days'
            GROUP BY date, dow, hour
            ORDER BY date, hour
        """)
        if not rows:
            return {day: 0.0 for day in ordered_days}, None

        daily_totals = defaultdict(float)
        daily_dow = {}
        for row in rows:
            date_key = str(row["date"])
            dow = int(row["dow"])
            hour = int(row["hour"])
            vehicles = max(0.0, float(row["avg_vehicles"] or 0.0))
            when = pd.to_datetime(f"{date_key} {hour:02d}:00:00")
            price = _dynamic_price_formula(vehicles, capacity, when)["recommended_price_php"]
            daily_totals[date_key] += vehicles * price
            daily_dow[date_key] = dow

        by_day = defaultdict(list)
        for date_key, total in daily_totals.items():
            by_day[daily_dow.get(date_key, 0)].append(total)

        weekday_revenue = {day: None for day in ordered_days}
        for dow, totals in by_day.items():
            label = sql_day_labels.get(int(dow))
            if label and totals:
                weekday_revenue[label] = round(sum(totals) / len(totals), 2)

        for day in ordered_days:
            if weekday_revenue[day] is None:
                valid = [v for v in weekday_revenue.values() if v is not None and v > 0]
                weekday_revenue[day] = round(sum(valid) / len(valid), 2) if valid else 0.0

        today_label = datetime.now(PH_TZ).strftime("%a")
        return weekday_revenue, weekday_revenue.get(today_label)
    except Exception as e:
        print(f"[predictions revenue] logged formula fallback unavailable: {e}")
        return {day: 0.0 for day in ordered_days}, None


# ══════════════════════════════════════════════════════════════════
#  ML endpoints
# ══════════════════════════════════════════════════════════════════
def _month_key(dt):
    return dt.strftime("%Y-%m")


def _recent_month_keys(now, count=6):
    keys = []
    year = now.year
    month = now.month
    for _ in range(count):
        keys.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(keys))


_PAYMENT_METHODS = {"cash", "gcash", "card", "other"}


_DISCOUNT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{3,63}$")


def _discount_rate_for_type(discount_type, id_number=None):
    kind = str(discount_type or "none").strip().lower()
    if kind in {"pwd", "senior"}:
        settings = _discount_settings()
        if not settings[f"{kind}_enabled"]:
            raise HTTPException(400, f"{kind.upper()} discount is currently turned off by the admin.")
        cleaned = str(id_number or "").strip()
        if not cleaned:
            raise HTTPException(400, f"{kind.upper()} ID number is required to apply this discount.")
        if not _DISCOUNT_ID_RE.match(cleaned) or not any(c.isdigit() for c in cleaned):
            raise HTTPException(
                400,
                f"{kind.upper()} ID number looks invalid — use at least 4 characters "
                "(letters, numbers, spaces or dashes) and include at least one digit.",
            )
        return kind, max(0.0, min(1.0, float(PWD_SENIOR_DISCOUNT_RATE)))
    if kind in {"none", "", "regular"}:
        return "none", 0.0
    raise HTTPException(400, "discount_type must be none, pwd, or senior")


def _current_payment_regular_price(vehicle_type="car"):
    with state_lock:
        snapshot = dict(state)
    total = int(snapshot.get("total") or _active_slot_capacity(LOT_CAPACITY))
    occupied = int(snapshot.get("occupied") or 0)
    timestamp = snapshot.get("timestamp") or ""
    vehicles = occupied if total > 0 and timestamp else 0
    return float(_dynamic_price_formula(vehicles, total, vehicle_type=vehicle_type)["recommended_price_php"])


def _row_float(row, key):
    return round(float(row.get(key) or 0.0), 2)


def _row_int(row, key):
    return int(row.get(key) or 0)


def _empty_payment_dashboard(error=None):
    now = datetime.now(PH_TZ)
    active_capacity = max(1, int(_active_slot_capacity(LOT_CAPACITY)))
    daily = []
    for i in range(6, -1, -1):
        day = now.date() - timedelta(days=i)
        daily.append({
            "date": day.isoformat(),
            "label": day.strftime("%a"),
            "revenue_php": 0.0,
            "discount_php": 0.0,
            "transaction_count": 0,
        })
    monthly = []
    for key in _recent_month_keys(now, 6):
        month_dt = datetime.strptime(key + "-01", "%Y-%m-%d")
        monthly.append({
            "month": key,
            "label": month_dt.strftime("%b"),
            "revenue_php": 0.0,
            "discount_php": 0.0,
            "transaction_count": 0,
        })
    out = {
        "generated_at_ph": now.strftime("%Y-%m-%d %H:%M:%S"),
        "active_slot_capacity": active_capacity,
        "max_total_count": active_capacity,
        "max_occupied_count": 0,
        "today_revenue_php": 0.0,
        "week_revenue_php": 0.0,
        "month_revenue_php": 0.0,
        "today_profit_php": 0.0,
        "week_profit_php": 0.0,
        "month_profit_php": 0.0,
        "today_discount_php": 0.0,
        "week_discount_php": 0.0,
        "month_discount_php": 0.0,
        "today_transaction_count": 0,
        "week_transaction_count": 0,
        "month_transaction_count": 0,
        "log_count": 0,
        "total_revenue_php": 0.0,
        "avg_transaction_php": 0.0,
        "revenue_by_vehicle_type": {},
        "revenue_by_payment_method": {},
        "revenue_by_discount_type": {},
        "daily_revenue": daily,
        "monthly_revenue": monthly,
        "recent_payments": [],
        "suggested_regular_price_php": round(_current_payment_regular_price("car"), 2),
        "suggested_regular_price_php_car": round(_current_payment_regular_price("car"), 2),
        "suggested_regular_price_php_motorcycle": round(_current_payment_regular_price("motorcycle"), 2),
        "revenue_basis": "Actual recorded parking payments. Profit is gross because expenses are not tracked.",
    }
    if error:
        out["error"] = str(error)
    return out


def _payment_revenue_dashboard():
    now = datetime.now(PH_TZ)
    active_capacity = max(1, int(_active_slot_capacity(LOT_CAPACITY)))
    with state_lock:
        live_total = int(state.get("total") or 0)
        live_occupied = int(state.get("occupied") or 0)

    summary_rows = query("""
        SELECT
            COALESCE(SUM(final_amount_php) FILTER (
                WHERE DATE(paid_at AT TIME ZONE 'Asia/Manila') = DATE(NOW() AT TIME ZONE 'Asia/Manila')
            ), 0) AS today_revenue,
            COALESCE(SUM(final_amount_php) FILTER (
                WHERE DATE_TRUNC('week', paid_at AT TIME ZONE 'Asia/Manila') = DATE_TRUNC('week', NOW() AT TIME ZONE 'Asia/Manila')
            ), 0) AS week_revenue,
            COALESCE(SUM(final_amount_php) FILTER (
                WHERE DATE_TRUNC('month', paid_at AT TIME ZONE 'Asia/Manila') = DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Manila')
            ), 0) AS month_revenue,
            COALESCE(SUM(discount_amount_php) FILTER (
                WHERE DATE(paid_at AT TIME ZONE 'Asia/Manila') = DATE(NOW() AT TIME ZONE 'Asia/Manila')
            ), 0) AS today_discount,
            COALESCE(SUM(discount_amount_php) FILTER (
                WHERE DATE_TRUNC('week', paid_at AT TIME ZONE 'Asia/Manila') = DATE_TRUNC('week', NOW() AT TIME ZONE 'Asia/Manila')
            ), 0) AS week_discount,
            COALESCE(SUM(discount_amount_php) FILTER (
                WHERE DATE_TRUNC('month', paid_at AT TIME ZONE 'Asia/Manila') = DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Manila')
            ), 0) AS month_discount,
            COUNT(*) FILTER (
                WHERE DATE(paid_at AT TIME ZONE 'Asia/Manila') = DATE(NOW() AT TIME ZONE 'Asia/Manila')
            ) AS today_count,
            COUNT(*) FILTER (
                WHERE DATE_TRUNC('week', paid_at AT TIME ZONE 'Asia/Manila') = DATE_TRUNC('week', NOW() AT TIME ZONE 'Asia/Manila')
            ) AS week_count,
            COUNT(*) FILTER (
                WHERE DATE_TRUNC('month', paid_at AT TIME ZONE 'Asia/Manila') = DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Manila')
            ) AS month_count,
            COUNT(*) AS total_count,
            COALESCE(SUM(final_amount_php), 0) AS total_revenue
        FROM parking_payments
    """)
    summary = summary_rows[0] if summary_rows else {}

    vehicle_rows = query("""
        SELECT vehicle_type,
               COALESCE(SUM(final_amount_php), 0) AS revenue,
               COUNT(*) AS transaction_count
        FROM parking_payments
        GROUP BY vehicle_type
    """)
    payment_method_rows = query("""
        SELECT payment_method,
               COALESCE(SUM(final_amount_php), 0) AS revenue,
               COUNT(*) AS transaction_count
        FROM parking_payments
        GROUP BY payment_method
    """)
    discount_rows = query("""
        SELECT discount_type,
               COALESCE(SUM(final_amount_php), 0) AS revenue,
               COALESCE(SUM(discount_amount_php), 0) AS discount,
               COUNT(*) AS transaction_count
        FROM parking_payments
        GROUP BY discount_type
    """)

    daily_rows = query("""
        SELECT
            DATE(paid_at AT TIME ZONE 'Asia/Manila') AS bucket_date,
            COALESCE(SUM(final_amount_php), 0) AS revenue,
            COALESCE(SUM(discount_amount_php), 0) AS discount,
            COUNT(*) AS transaction_count
        FROM parking_payments
        WHERE paid_at >= NOW() - INTERVAL '7 days'
        GROUP BY bucket_date
        ORDER BY bucket_date
    """)
    monthly_rows = query("""
        SELECT
            DATE_TRUNC('month', paid_at AT TIME ZONE 'Asia/Manila')::date AS bucket_month,
            COALESCE(SUM(final_amount_php), 0) AS revenue,
            COALESCE(SUM(discount_amount_php), 0) AS discount,
            COUNT(*) AS transaction_count
        FROM parking_payments
        WHERE paid_at >= DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '5 months'
        GROUP BY bucket_month
        ORDER BY bucket_month
    """)
    recent_rows = query("""
        SELECT payment_id, vehicle_type, regular_price_php, discount_type,
               discount_amount_php, final_amount_php, payment_method, paid_at,
               discount_id_number
        FROM parking_payments
        ORDER BY paid_at DESC
        LIMIT 10
    """)

    daily_map = {}
    for row in daily_rows:
        when = _coerce_datetime(row.get("bucket_date"))
        daily_map[when.date().isoformat()] = row

    daily = []
    for i in range(6, -1, -1):
        day = now.date() - timedelta(days=i)
        key = day.isoformat()
        row = daily_map.get(key, {})
        daily.append({
            "date": key,
            "label": day.strftime("%a"),
            "revenue_php": _row_float(row, "revenue"),
            "discount_php": _row_float(row, "discount"),
            "transaction_count": _row_int(row, "transaction_count"),
        })

    monthly_map = {}
    for row in monthly_rows:
        when = _coerce_datetime(row.get("bucket_month"))
        monthly_map[_month_key(when)] = row

    monthly = []
    for key in _recent_month_keys(now, 6):
        row = monthly_map.get(key, {})
        month_dt = datetime.strptime(key + "-01", "%Y-%m-%d")
        monthly.append({
            "month": key,
            "label": month_dt.strftime("%b"),
            "revenue_php": _row_float(row, "revenue"),
            "discount_php": _row_float(row, "discount"),
            "transaction_count": _row_int(row, "transaction_count"),
        })

    recent = []
    for row in recent_rows:
        paid_at = _coerce_datetime(row.get("paid_at"))
        recent.append({
            "payment_id": row.get("payment_id"),
            "vehicle_type": row.get("vehicle_type") or "car",
            "regular_price_php": _row_float(row, "regular_price_php"),
            "discount_type": row.get("discount_type") or "none",
            "discount_amount_php": _row_float(row, "discount_amount_php"),
            "final_amount_php": _row_float(row, "final_amount_php"),
            "payment_method": row.get("payment_method") or "cash",
            "discount_id_number": row.get("discount_id_number"),
            "paid_at_ph": paid_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    today_revenue = _row_float(summary, "today_revenue")
    week_revenue = _row_float(summary, "week_revenue")
    month_revenue = _row_float(summary, "month_revenue")
    total_revenue = _row_float(summary, "total_revenue")
    total_count = _row_int(summary, "total_count")

    by_vehicle = {
        (row.get("vehicle_type") or "car"): {
            "revenue_php": _row_float(row, "revenue"),
            "transaction_count": _row_int(row, "transaction_count"),
        }
        for row in vehicle_rows
    }
    by_payment_method = {
        (row.get("payment_method") or "cash"): {
            "revenue_php": _row_float(row, "revenue"),
            "transaction_count": _row_int(row, "transaction_count"),
        }
        for row in payment_method_rows
    }
    by_discount = {
        (row.get("discount_type") or "none"): {
            "revenue_php": _row_float(row, "revenue"),
            "discount_php": _row_float(row, "discount"),
            "transaction_count": _row_int(row, "transaction_count"),
        }
        for row in discount_rows
    }

    return {
        "generated_at_ph": now.strftime("%Y-%m-%d %H:%M:%S"),
        "active_slot_capacity": active_capacity,
        "max_total_count": max(active_capacity, live_total, int(LOT_CAPACITY)),
        "max_occupied_count": live_occupied,
        "today_revenue_php": today_revenue,
        "week_revenue_php": week_revenue,
        "month_revenue_php": month_revenue,
        "today_profit_php": today_revenue,
        "week_profit_php": week_revenue,
        "month_profit_php": month_revenue,
        "today_discount_php": _row_float(summary, "today_discount"),
        "week_discount_php": _row_float(summary, "week_discount"),
        "month_discount_php": _row_float(summary, "month_discount"),
        "today_transaction_count": _row_int(summary, "today_count"),
        "week_transaction_count": _row_int(summary, "week_count"),
        "month_transaction_count": _row_int(summary, "month_count"),
        "log_count": total_count,
        "total_revenue_php": total_revenue,
        "avg_transaction_php": round(total_revenue / total_count, 2) if total_count else 0.0,
        "revenue_by_vehicle_type": by_vehicle,
        "revenue_by_payment_method": by_payment_method,
        "revenue_by_discount_type": by_discount,
        "daily_revenue": daily,
        "monthly_revenue": monthly,
        "recent_payments": recent,
        "suggested_regular_price_php": round(_current_payment_regular_price("car"), 2),
        "suggested_regular_price_php_car": round(_current_payment_regular_price("car"), 2),
        "suggested_regular_price_php_motorcycle": round(_current_payment_regular_price("motorcycle"), 2),
        "revenue_basis": "Actual recorded parking payments. Profit is gross because expenses are not tracked.",
    }


def _record_parking_payment(payload):
    vehicle_type = _normalize_vehicle_type(payload.vehicle_type, default="car")
    regular_price = _parse_price(
        payload.regular_price_php
        if payload.regular_price_php is not None
        else _current_payment_regular_price(vehicle_type)
    )
    discount_id_number = (getattr(payload, "discount_id_number", None) or "").strip()[:64] or None
    discount_type, discount_rate = _discount_rate_for_type(payload.discount_type, discount_id_number)
    discount_amount = round(regular_price * discount_rate, 2)
    final_amount = round(max(0.0, regular_price - discount_amount), 2)
    payment_method = (payload.payment_method or "cash").strip().lower()
    if payment_method not in _PAYMENT_METHODS:
        raise HTTPException(400, "payment_method must be one of: " + ", ".join(sorted(_PAYMENT_METHODS)))
    notes = (payload.notes or "").strip()[:500] or None
    paid_at = _coerce_datetime(payload.paid_at) if payload.paid_at else datetime.now(PH_TZ)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO parking_payments (
                vehicle_type, regular_price_php, discount_type, discount_rate,
                discount_amount_php, final_amount_php, payment_method, notes, paid_at, user_id,
                discount_id_number
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING payment_id, vehicle_type, regular_price_php, discount_type,
                      discount_rate, discount_amount_php, final_amount_php,
                      payment_method, notes, paid_at, discount_id_number
            """,
            (
                vehicle_type, regular_price, discount_type, discount_rate,
                discount_amount, final_amount, payment_method, notes, paid_at,
                payload.user_id, discount_id_number,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        paid_at_ph = _coerce_datetime(row["paid_at"]).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "ok": True,
            "payment_id": row["payment_id"],
            "vehicle_type": row["vehicle_type"],
            "regular_price_php": _row_float(row, "regular_price_php"),
            "discount_type": row["discount_type"],
            "discount_rate": float(row["discount_rate"] or 0.0),
            "discount_amount_php": _row_float(row, "discount_amount_php"),
            "final_amount_php": _row_float(row, "final_amount_php"),
            "payment_method": row["payment_method"],
            "notes": row["notes"],
            "discount_id_number": row["discount_id_number"],
            "paid_at_ph": paid_at_ph,
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()


def _format_price_result(result):
    ctx = result.get("pricing_context", {})
    if result.get("pricing_reason") == "manual_admin_override":
        note = "Admin-set parking rate. Manual pricing is active."
    else:
        note = (
            f"{result.get('price_note') or 'Standard parking rate.'} "
            f"Based on {ctx.get('occupancy_pct', 0):.0f}% occupancy and {ctx.get('day_rule', 'weekday').lower()} factor."
        )
    return {
        "vehicle_type": result.get("vehicle_type", "car"),
        "price_php": round(float(result["recommended_price_php"]), 2),
        "flat_rate_php": result.get("flat_rate_php"),
        "pwd_senior_price_php": result.get("pwd_senior_price_php"),
        "pwd_senior_discount_rate": result.get("pwd_senior_discount_rate", PWD_SENIOR_DISCOUNT_RATE),
        "pwd_senior_discount_pct": result.get("pwd_senior_discount_pct", round(PWD_SENIOR_DISCOUNT_RATE * 100, 1)),
        "pwd_senior_discount_amount_php": result.get("pwd_senior_discount_amount_php"),
        "pwd_senior_note": result.get("pwd_senior_note"),
        "price_note": note,
        "price_source": "occupancy_formula",
        "price_formula": result.get("formula"),
        "price_context": result.get("pricing_context"),
    }


def _driver_price_summary(vehicles_hour=None, lot_capacity=None):
    if vehicles_hour is None:
        try:
            df = _db_history()
            if not df.empty and "vehicles_hour" in df.columns:
                vehicles_hour = float(df["vehicles_hour"].iloc[-1] or 0.0)
        except Exception as e:
            print(f"[driver summary] price history fallback: {e}")

    car_result = _dynamic_price_formula(vehicles_hour or 0.0, lot_capacity, vehicle_type="car")
    moto_result = _dynamic_price_formula(vehicles_hour or 0.0, lot_capacity, vehicle_type="motorcycle")
    car = _format_price_result(car_result)
    moto = _format_price_result(moto_result)

    confidence = _metric_score_pct("pricing_rf")
    score_label = _metric_score_label("pricing_rf")

    # Top-level fields stay car-based for backward compatibility with older callers.
    return {
        **car,
        "car": car,
        "motorcycle": moto,
        "confidence_pct": confidence,
        "score_label": score_label,
    }


@app.get("/api/driver/summary")
def api_driver_summary():
    with state_lock:
        snapshot = dict(state)

    live_total = int(snapshot.get("total") or 0)
    total = live_total or int(_active_slot_capacity(LOT_CAPACITY))
    timestamp = snapshot.get("timestamp") or ""
    camera_online = live_total > 0 and bool(timestamp)

    if camera_online:
        occupied = max(0, int(snapshot.get("occupied") or 0))
        free = max(0, int(snapshot.get("free") if snapshot.get("free") is not None else total - occupied))
        occupancy_pct = round(float(snapshot.get("occupancy_pct") or (occupied / max(total, 1) * 100)), 1)
    else:
        occupied = None
        free = None
        occupancy_pct = None

    price = _driver_price_summary(occupied if camera_online else None, total)
    status_text = "Live parking availability" if camera_online else "Waiting for camera update"
    if free == 0 and camera_online:
        status_text = "Parking area is currently full"

    return {
        "available": free,
        "available_text": "--" if free is None else str(free),
        "occupied": occupied,
        "total": total,
        "occupancy_pct": occupancy_pct,
        "lot_full": bool(snapshot.get("lot_full")) if camera_online else False,
        "camera_online": camera_online,
        "last_update": timestamp,
        "status_text": status_text,
        "generated_at_ph": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        **price,
    }


@app.get("/api/ml/metrics")
def ml_metrics():
    return {
        "source": str(METRICS_PATH),
        "metrics": MODEL_METRICS,
        "generated_at_ph": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/ml/predict/vehicles")
def ml_predict_vehicles():
    if not ml._ready: raise HTTPException(503, "ML not loaded")
    df = _db_history()
    if len(df) < NB1_SEQ_LEN:
        raise HTTPException(422, f"Need {NB1_SEQ_LEN} rows, have {len(df)}")
    try: return ml.predict_vehicles(df, capacity=_active_slot_capacity())
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/predict/occupancy")
def ml_predict_occupancy():
    if not ml._ready: raise HTTPException(503, "ML not loaded")
    df = _db_history()
    if len(df) < NB2_OCC_SEQ_LEN:
        raise HTTPException(422, f"Need {NB2_OCC_SEQ_LEN} rows, have {len(df)}")
    try: return ml.predict_occupancy(df, capacity=_active_slot_capacity())
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/predict/price")
def ml_predict_price():
    df = _db_history()
    if df.empty: raise HTTPException(422, "No history")
    try:
        row = df.iloc[-1].to_dict()
        return _dynamic_price_formula(row.get("vehicles_hour", 0.0), _active_slot_capacity(), row.get("datetime"))
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/predict/revenue")
def ml_predict_revenue():
    try:
        active_capacity = max(1, int(_active_slot_capacity(LOT_CAPACITY)))
        weekday_revenue, today_forecast = _training_weekday_revenue_forecast(active_capacity)
        source = "training_data"
        if not any(weekday_revenue.values()):
            weekday_revenue, today_forecast = _logged_weekday_revenue_forecast(active_capacity)
            source = "deduplicated_live_logs"
        if today_forecast is None:
            raise HTTPException(422, "Need training data or live logs for revenue forecast")
        return {
            "predicted_daily_revenue_php": round(float(today_forecast), 2),
            "weekday_revenue": weekday_revenue,
            "model_used": "Revenue Formula Forecast",
            "source": source,
            "active_slot_capacity": active_capacity,
            "confidence_pct": _metric_score_pct("revenue_gbr"),
            "score_label": _metric_score_label("revenue_gbr"),
            "training_metrics": _training_metrics("revenue_gbr"),
        }
    except HTTPException:
        raise
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/dashboard")
def ml_dashboard():
    df  = _db_history()
    active_capacity = _active_slot_capacity()
    out = {
        "ml_ready": ml._ready,
        "timestamp": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S PHT"),
        "model_metrics": MODEL_METRICS,
        "active_slot_capacity": active_capacity,
    }
    if ml._ready and len(df) >= NB1_SEQ_LEN:
        for key, fn in [
            ("vehicles",  lambda: ml.predict_vehicles(df, capacity=active_capacity)),
            ("occupancy", lambda: ml.predict_occupancy(df, capacity=active_capacity)),
        ]:
            try:   out[key] = fn()
            except Exception as e: out[key] = {"error": str(e)}
        try:
            row = df.iloc[-1].to_dict()
            out["price"] = _dynamic_price_formula(row.get("vehicles_hour", 0.0), active_capacity, row.get("datetime"))
        except Exception as e:
            out["price"] = {"error": str(e)}
    else:
        out["note"] = f"Need {NB1_SEQ_LEN}+ rows (have {len(df)})"
    with state_lock:
        out["live"] = {k: state[k] for k in ("occupied","free","total","occupancy_pct","zones")}
    return out


# ══════════════════════════════════════════════════════════════════
#  Predictions — with PH time + revenue forecast per weekday
# ══════════════════════════════════════════════════════════════════
@app.get("/api/predictions")
def api_predictions():
    try:
        active_capacity = max(1, int(_active_slot_capacity(LOT_CAPACITY)))
        # Hourly avg occupancy in PH time (last 7 days)
        rows = query("""
            SELECT EXTRACT(HOUR FROM logged_at AT TIME ZONE 'Asia/Manila') AS hour,
                   AVG(occupancy_pct) AS avg_pct
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '7 days'
            GROUP BY hour ORDER BY hour
        """)
        training_hourly = _training_hourly_occ_pct(active_capacity)
        if any(training_hourly.values()):
            hourly = training_hourly
        else:
            hourly = {str(int(r["hour"])): round(float(r["avg_pct"]), 1) for r in rows}
            for h in range(24):
                hourly.setdefault(str(h), 0.0)
        peak = max(hourly, key=lambda h: hourly[h])

        weekday_revenue, today_forecast = _training_weekday_revenue_forecast(active_capacity)
        if not any(weekday_revenue.values()):
            weekday_revenue, today_forecast = _logged_weekday_revenue_forecast(active_capacity)

        days_with_revenue = {d: v for d, v in weekday_revenue.items() if v and v > 0}
        if days_with_revenue:
            avg_rev = sum(days_with_revenue.values()) / len(days_with_revenue)
            busy_days  = sorted([d for d, v in days_with_revenue.items() if v >= avg_rev],
                                key=lambda d: -days_with_revenue[d])
            quiet_days = sorted([d for d, v in weekday_revenue.items() if v < avg_rev],
                                key=lambda d: weekday_revenue.get(d, 0))
        else:
            busy_days  = []
            quiet_days = []

        return {
            "hourly_est":             hourly,
            "peak_hour":              int(peak),
            "peak_label":             f"{peak}:00 ({hourly[peak]:.0f}%)",
            "busy_days":              busy_days,
            "quiet_days":             quiet_days,
            "weekday_revenue":        weekday_revenue,
            "today_revenue_forecast": today_forecast,
            "active_slot_capacity":   active_capacity,
            "generated_at_ph":        datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        print(f"[predictions] {e}")
        return {
            "hourly_est":             {str(h): 0.0 for h in range(24)},
            "peak_hour":              8,
            "peak_label":             "N/A",
            "busy_days":              [],
            "quiet_days":             [],
            "weekday_revenue":        {},
            "today_revenue_forecast": None,
            "generated_at_ph":        datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }

# ══════════════════════════════════════════════════════════════════
#  ADD THIS ENDPOINT to backend/main.py
#  Place it after the existing /api/predictions endpoint
#  (around line where api_predictions() ends)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/revenue/dashboard")
def api_revenue_dashboard():
    try:
        return _payment_revenue_dashboard()
    except Exception as e:
        print(f"[revenue dashboard] {e}")
        return _empty_payment_dashboard(e)


@app.get("/api/payments/recent")
def api_payments_recent(limit: int = 10, offset: int = 0, _admin=Depends(require_admin)):
    safe_limit = max(1, min(int(limit), 100))
    safe_offset = max(0, int(offset))

    total_rows = query("SELECT COUNT(*) AS total FROM parking_payments")
    total = _row_int(total_rows[0], "total") if total_rows else 0

    rows = query(
        """
        SELECT payment_id, vehicle_type, regular_price_php, discount_type,
               discount_amount_php, final_amount_php, payment_method, paid_at,
               discount_id_number
        FROM parking_payments
        ORDER BY paid_at DESC
        LIMIT %s OFFSET %s
        """,
        (safe_limit, safe_offset),
    )
    items = []
    for row in rows:
        paid_at = _coerce_datetime(row.get("paid_at"))
        items.append({
            "payment_id": row.get("payment_id"),
            "vehicle_type": row.get("vehicle_type") or "car",
            "regular_price_php": _row_float(row, "regular_price_php"),
            "discount_type": row.get("discount_type") or "none",
            "discount_amount_php": _row_float(row, "discount_amount_php"),
            "final_amount_php": _row_float(row, "final_amount_php"),
            "payment_method": row.get("payment_method") or "cash",
            "discount_id_number": row.get("discount_id_number"),
            "paid_at_ph": paid_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return {"items": items, "total": total, "limit": safe_limit, "offset": safe_offset}


class PaymentRecordPayload(BaseModel):
    vehicle_type: Optional[str] = "car"
    regular_price_php: Optional[float] = None
    discount_type: Optional[str] = "none"
    discount_id_number: Optional[str] = None
    payment_method: Optional[str] = "cash"
    notes: Optional[str] = None
    paid_at: Optional[str] = None
    user_id: Optional[int] = None


@app.post("/api/payments")
def api_record_payment(payload: PaymentRecordPayload, request: Request, _admin=Depends(require_admin)):
    _rate_limit(request, "payments", max_calls=30, window_seconds=60)
    return _record_parking_payment(payload)


# ══════════════════════════════════════════════════════════════════

@app.get("/api/driver/history")
def api_driver_history(user_id: int, period: str = "all", limit: int = 50, _auth=Depends(require_user)):
    """Returns parking payment history for a specific driver."""
    if _auth.get("role") != "admin" and int(_auth.get("user_id") or 0) != int(user_id):
        raise HTTPException(403, "You can only view your own payment history")
    try:
        period_filter = ""
        if period == "today":
            period_filter = "AND DATE(paid_at AT TIME ZONE 'Asia/Manila') = CURRENT_DATE"
        elif period == "week":
            period_filter = "AND paid_at >= DATE_TRUNC('week', NOW() AT TIME ZONE 'Asia/Manila')"
        elif period == "month":
            period_filter = "AND paid_at >= DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Manila')"

        rows = query(f"""
            SELECT
                payment_id,
                vehicle_type,
                regular_price_php,
                discount_type,
                discount_amount_php,
                final_amount_php,
                payment_method,
                notes,
                paid_at AT TIME ZONE 'Asia/Manila' AS paid_at_ph
            FROM parking_payments
            WHERE user_id = %s {period_filter}
            ORDER BY paid_at DESC
            LIMIT %s
        """, (user_id, limit))

        records = []
        for r in rows:
            paid_at = r["paid_at_ph"]
            records.append({
                "payment_id": r["payment_id"],
                "vehicle_type": r["vehicle_type"] or "car",
                "regular_price_php": float(r["regular_price_php"]),
                "discount_type": r["discount_type"],
                "discount_amount_php": float(r["discount_amount_php"]),
                "final_amount_php": float(r["final_amount_php"]),
                "payment_method": r["payment_method"],
                "slot_label": "--",
                "duration_minutes": 0,
                "notes": r["notes"] or "",
                "paid_date": paid_at.strftime("%b %d, %Y") if paid_at else "--",
                "paid_time": paid_at.strftime("%I:%M %p") if paid_at else "--",
            })

        total_spent = sum(r["final_amount_php"] for r in records)
        return {
            "records": records,
            "total_spent_php": round(total_spent, 2),
            "total_sessions": len(records),
            "period": period,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  GCash Payment via PayMongo
# ══════════════════════════════════════════════════════════════════
class GcashCheckoutPayload(BaseModel):
    vehicle_type: Optional[str] = "car"
    discount_type: Optional[str] = "none"
    discount_id_number: Optional[str] = None
    description: Optional[str] = "OccupAI Parking Payment"
    user_id: Optional[int] = None


def _paymongo_headers():
    encoded = base64.b64encode(f"{PAYMONGO_SECRET_KEY}:".encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


@app.post("/api/gcash/create-checkout")
def gcash_create_checkout(payload: GcashCheckoutPayload, request: Request):
    _rate_limit(request, "gcash-checkout", max_calls=10, window_seconds=60)
    if not PAYMONGO_SECRET_KEY:
        raise HTTPException(503, "PayMongo is not configured. Set PAYMONGO_SECRET_KEY in .env")

    vehicle_type = _normalize_vehicle_type(payload.vehicle_type, default="car")
    regular_price = _current_payment_regular_price(vehicle_type)
    discount_id_number = (payload.discount_id_number or "").strip()[:64] or None
    discount_type, discount_rate = _discount_rate_for_type(payload.discount_type, discount_id_number)
    discount_amount = round(regular_price * discount_rate, 2)
    final_amount = round(max(1.0, regular_price - discount_amount), 2)
    amount_centavos = round(final_amount * 100)

    base_url = str(request.base_url).rstrip("/")

    import urllib.request
    import urllib.error

    ref = secrets.token_urlsafe(24)
    execute(
        """
        INSERT INTO gcash_checkouts (ref, user_id, vehicle_type, regular_price_php, discount_type, final_amount_php, discount_id_number)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (ref, payload.user_id, vehicle_type, regular_price, discount_type, final_amount, discount_id_number),
    )

    body = json.dumps({
        "data": {
            "attributes": {
                "line_items": [{
                    "name": f"Parking Fee ({vehicle_type.title()})",
                    "amount": amount_centavos,
                    "currency": "PHP",
                    "quantity": 1,
                    "description": payload.description or "OccupAI Parking Payment",
                }],
                "payment_method_types": ["gcash"],
                "success_url": f"{base_url}/api/gcash/success?ref={ref}",
                "cancel_url": f"{base_url}/driver",
                "description": payload.description or "OccupAI Parking Payment",
            }
        }
    }).encode()

    req = urllib.request.Request(
        "https://api.paymongo.com/v1/checkout_sessions",
        data=body,
        headers=_paymongo_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        print(f"[PayMongo] {e.code}: {error_body}")
        raise HTTPException(502, f"PayMongo error: {e.code}")

    checkout_url = result["data"]["attributes"]["checkout_url"]
    checkout_id = result["data"]["id"]
    execute("UPDATE gcash_checkouts SET checkout_id=%s WHERE ref=%s", (checkout_id, ref))

    return {
        "checkout_url": checkout_url,
        "checkout_id": checkout_id,
        "amount_php": final_amount,
        "discount_type": discount_type,
        "discount_amount_php": discount_amount,
    }


def _gcash_result_page(ok: bool, message: str):
    color = "#22c996" if ok else "#ef4444"
    icon = "&#10003;" if ok else "&#10007;"
    title = "Payment Successful!" if ok else "Payment Not Confirmed"
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + title + """</title>
<style>
body{font-family:"DM Sans",system-ui,sans-serif;background:#07100d;color:#f1f5f9;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.box{background:rgba(241,245,249,.08);border-radius:16px;padding:48px 36px;text-align:center;max-width:420px}
.check{font-size:64px;margin-bottom:16px;color:""" + color + """}
h1{font-size:24px;margin-bottom:8px;color:""" + color + """}
p{color:#8794a6;margin-bottom:24px}
a{display:inline-block;background:#22c996;color:#07100d;padding:12px 32px;border-radius:8px;
text-decoration:none;font-weight:600}
</style></head><body><div class="box">
<div class="check">""" + icon + """</div>
<h1>""" + title + """</h1>
<p>""" + message + """</p>
<a href="/driver">Back to Driver View</a>
</div></body></html>"""
    return HTMLResponse(content=html)


def _gcash_pending_page(ref: str, next_attempt: int):
    """Auto-refreshing page shown while PayMongo is still confirming a GCash payment.

    GCash confirmation frequently lags a few seconds past the redirect, so instead of
    hard-failing we reload the success URL a few times before giving up.
    """
    retry_url = f"/api/gcash/success?ref={ref}&attempt={next_attempt}"
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Confirming Payment…</title>
<meta http-equiv="refresh" content="4;url=""" + retry_url + """">
<style>
body{font-family:"DM Sans",system-ui,sans-serif;background:#07100d;color:#f1f5f9;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.box{background:rgba(241,245,249,.08);border-radius:16px;padding:48px 36px;text-align:center;max-width:420px}
.spin{width:56px;height:56px;margin:0 auto 20px;border:5px solid rgba(34,201,150,.25);
border-top-color:#22c996;border-radius:50%;animation:s 1s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
h1{font-size:22px;margin-bottom:8px;color:#22c996}
p{color:#8794a6;margin-bottom:24px}
a{display:inline-block;background:#22c996;color:#07100d;padding:12px 32px;border-radius:8px;
text-decoration:none;font-weight:600}
</style></head><body><div class="box">
<div class="spin"></div>
<h1>Confirming your payment…</h1>
<p>Please wait a moment while GCash confirms your payment. This can take a few seconds.</p>
<a href=\"""" + retry_url + """\">Check now</a>
</div></body></html>"""
    return HTMLResponse(content=html)


# Number of times the success page will auto-refresh while waiting for GCash to
# confirm. Combined with the per-request polling below this gives PayMongo a
# generous window (well over a minute) before we show a hard failure.
_GCASH_MAX_PAGE_RETRIES = 6


def _checkout_is_paid(attributes: dict) -> bool:
    """True if a PayMongo checkout session shows the payment as completed.

    GCash confirmation can surface in any of three places depending on timing, so
    check all of them: the session's own ``payment_status``, the underlying
    ``payments`` list, and the ``payment_intent`` status.
    """
    if attributes.get("payment_status") == "paid":
        return True

    payments = attributes.get("payments") or []
    if any(p.get("attributes", {}).get("status") == "paid" for p in payments):
        return True

    intent = attributes.get("payment_intent") or {}
    if intent.get("attributes", {}).get("status") == "succeeded":
        return True

    return False


@app.get("/api/gcash/success")
def gcash_success(ref: str, attempt: int = 0):
    rows = query("SELECT * FROM gcash_checkouts WHERE ref = %s", (ref,))
    if not rows:
        return _gcash_result_page(False, "We couldn't find this checkout session.")
    checkout = rows[0]

    if checkout["status"] == "paid":
        return _gcash_result_page(
            True, f"Your GCash payment of PHP {float(checkout['final_amount_php']):,.2f} has been recorded."
        )

    checkout_id = checkout["checkout_id"]
    if not checkout_id:
        return _gcash_result_page(False, "This checkout session was never started with PayMongo.")

    import time
    import urllib.request
    import urllib.error

    paid = False
    attributes = {}
    max_attempts = 4
    for poll in range(max_attempts):
        try:
            req = urllib.request.Request(
                f"https://api.paymongo.com/v1/checkout_sessions/{checkout_id}",
                headers=_paymongo_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
        except Exception as e:
            print(f"[GCash] Could not verify checkout {checkout_id}: {e}")
            # Transient network/API hiccup — let the page retry rather than hard-fail.
            if attempt < _GCASH_MAX_PAGE_RETRIES:
                return _gcash_pending_page(ref, attempt + 1)
            return _gcash_result_page(False, "We couldn't verify your payment with PayMongo. Please contact support.")

        attributes = result.get("data", {}).get("attributes", {})
        if _checkout_is_paid(attributes):
            paid = True
            break

        if poll < max_attempts - 1:
            time.sleep(2)

    if not paid:
        # GCash confirmation often lags past the redirect. Auto-refresh a few times
        # before giving up so a slightly-late payment still gets recorded.
        if attempt < _GCASH_MAX_PAGE_RETRIES:
            return _gcash_pending_page(ref, attempt + 1)
        print(f"[GCash] checkout {checkout_id} still not paid after retries: {attributes}")
        return _gcash_result_page(False, "PayMongo has not confirmed this payment yet.")

    try:
        payment_payload = PaymentRecordPayload(
            vehicle_type=checkout["vehicle_type"] or "car",
            regular_price_php=float(checkout["regular_price_php"]),
            discount_type=checkout["discount_type"],
            discount_id_number=checkout["discount_id_number"],
            payment_method="gcash",
            notes="Paid via GCash (PayMongo)",
            user_id=checkout["user_id"],
        )
        _record_parking_payment(payment_payload)
        execute("UPDATE gcash_checkouts SET status='paid' WHERE ref=%s", (ref,))
    except Exception as e:
        print(f"[GCash] Could not auto-record payment: {e}")
        # PayMongo already charged the customer at this point — flag it so an admin
        # can find and manually reconcile it, instead of the money silently vanishing.
        execute(
            "UPDATE gcash_checkouts SET status='error', error_message=%s WHERE ref=%s",
            (str(e)[:500], ref),
        )
        return _gcash_result_page(False, "Payment was confirmed by PayMongo, but we couldn't save the record. Please contact support.")

    return _gcash_result_page(
        True, f"Your GCash payment of PHP {float(checkout['final_amount_php']):,.2f} has been recorded."
    )


@app.get("/api/gcash/status")
def gcash_status():
    return {"enabled": bool(PAYMONGO_SECRET_KEY)}


@app.get("/api/gcash/issues")
def gcash_issues(_admin=Depends(require_admin)):
    """
    Checkouts PayMongo confirmed as paid but that OccupAI failed to save as a
    parking_payments row — the customer was charged and nothing is recorded.
    Needs manual reconciliation.
    """
    rows = query("""
        SELECT ref, user_id, vehicle_type, final_amount_php, discount_type,
               error_message, created_at
        FROM gcash_checkouts
        WHERE status = 'error'
        ORDER BY created_at DESC
        LIMIT 50
    """)
    items = [{
        "ref": r.get("ref"),
        "user_id": r.get("user_id"),
        "vehicle_type": r.get("vehicle_type") or "car",
        "final_amount_php": _row_float(r, "final_amount_php"),
        "discount_type": r.get("discount_type") or "none",
        "error_message": r.get("error_message"),
        "created_at_ph": _coerce_datetime(r.get("created_at")).strftime("%Y-%m-%d %H:%M:%S"),
    } for r in rows]
    return {"items": items, "count": len(items)}


@app.get("/api/predictions/hourly-by-day")
def api_hourly_by_day():
    """
    Returns average vehicle count per hour per day-of-week
    from the last 30 days of parking_logs, in Philippine Time.
    Used by the Analytics page hourly-by-day chart.
    """
    try:
        active_capacity = max(1, int(_active_slot_capacity(LOT_CAPACITY)))
        rows = query("""
            SELECT
                EXTRACT(DOW  FROM logged_at AT TIME ZONE 'Asia/Manila') AS dow,
                EXTRACT(HOUR FROM logged_at AT TIME ZONE 'Asia/Manila') AS hour,
                AVG(occupied)      AS avg_vehicles,
                AVG(occupancy_pct) AS avg_occ_pct,
                COUNT(*)           AS sample_count
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '30 days'
            GROUP BY dow, hour
            ORDER BY dow, hour
        """)

        # Build dow->hour->value map
        # dow: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
        day_map = {i: {h: {"vehicles": 0.0, "occ_pct": 0.0, "samples": 0}
                       for h in range(24)} for i in range(7)}

        for r in rows:
            d = int(r["dow"])
            h = int(r["hour"])
            day_map[d][h] = {
                "vehicles": round(float(r["avg_vehicles"] or 0), 1),
                "occ_pct":  round(float(r["avg_occ_pct"]  or 0), 1),
                "samples":  int(r["sample_count"] or 0),
            }

        # If no DB data yet, fall back to historical pattern from training data
        # Mon=1 is busiest (144/day), Sat=6 quietest (85/day)
        DAY_AVG_HIST = {0:90, 1:144, 2:138, 3:135, 4:130, 5:125, 6:85}
        PROFILE = [0,0,0,0,0,2,6,14,18,13,9,8,11,8,7,6,8,6,4,3,2,1,0,0]
        prof_sum = sum(PROFILE)
        has_data = any(
            day_map[d][h]["samples"] > 0
            for d in range(7) for h in range(24)
        )
        if not has_data:
            for d in range(7):
                total = DAY_AVG_HIST[d]
                for h in range(24):
                    v = round(PROFILE[h] / prof_sum * total, 1)
                    day_map[d][h] = {
                        "vehicles": v,
                        "occ_pct":  round(min(v / active_capacity * 100, 100.0), 1),
                        "samples":  0,
                    }

        # Also compute overall daily totals and peak hours
        day_labels = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
        daily_summary = {}
        for d in range(7):
            hours_data = day_map[d]
            total_veh  = sum(hours_data[h]["vehicles"] for h in range(24))
            peak_h     = max(range(24), key=lambda h: hours_data[h]["vehicles"])
            daily_summary[day_labels[d]] = {
                "total_vehicles": round(total_veh, 1),
                "peak_hour":      peak_h,
                "peak_vehicles":  hours_data[peak_h]["vehicles"],
                "avg_occ_pct":    round(
                    sum(hours_data[h]["occ_pct"] for h in range(8, 20)) / 12, 1
                ),
            }

        return {
            "hourly_by_dow":  day_map,
            "daily_summary":  daily_summary,
            "day_labels":     day_labels,
            "lot_capacity":   active_capacity,
            "source":         "db" if has_data else "historical_fallback",
            "generated_at_ph": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"[hourly-by-day] {e}")
        raise HTTPException(500, str(e))

# ══════════════════════════════════════════════════════════════════
#  Insights
# ══════════════════════════════════════════════════════════════════
@app.get("/api/insights")
def api_insights():
    with _insight_lock:
        if not _insight_cache:
            result = {
                "live_status":        "⏳ Insights are being computed (ready in ~15s after startup).",
                "trend":              "—",
                "vehicle_forecast":   "—",
                "occupancy_forecast": "—",
                "pricing":            "—",
                "revenue_forecast":   "—",
                "peak_hours":         "—",
                "admin_action":       "Please wait — the system is initializing.",
                "admin_action_items": [],
                "generated_at":       datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "next_refresh":       "~15 seconds",
            }
            with state_lock:
                live_snapshot = dict(state)
            result.update(_live_status_from_state(live_snapshot))
            result["active_slot_capacity"] = _active_slot_capacity()
            return result
        result = dict(_insight_cache)

    try:
        last      = datetime.strptime(result["generated_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=PH_TZ)
        mins_ago  = int((datetime.now(PH_TZ) - last).total_seconds() / 60)
        mins_left = max(0, 60 - mins_ago)
        result["next_refresh"]   = f"Auto-refreshes in ~{mins_left} min" if mins_left > 1 else "Refreshing soon…"
        result["last_refreshed"] = "just now" if mins_ago < 1 else f"{mins_ago} min ago"
    except Exception:
        pass

    with state_lock:
        live_snapshot = dict(state)
    result.update(_live_status_from_state(live_snapshot))
    result["active_slot_capacity"] = _active_slot_capacity()

    return result


@app.post("/api/insights/refresh")
def api_insights_refresh(_admin=Depends(require_admin)):
    threading.Thread(target=_run_insights_now, daemon=True, name="insight-force").start()
    return {"ok": True, "message": "Recalculating — results ready in a few seconds."}

@app.get("/api/parking_logs_recent")
def parking_logs_recent(limit: int = 168):
    """
    Returns the most recent `limit` parking_logs rows (newest first).
    Used by the SlotAdjusterThread in detector.py to build lag features.
    Default 168 = 7 days of hourly rows.
    """
    safe_limit = max(1, min(int(limit), 1000))
    rows = query(
        """
        SELECT occupied, free, total, occupancy_pct, lot_full, logged_at
        FROM parking_logs
        ORDER BY logged_at DESC
        LIMIT %s
        """,
        (safe_limit,),
    )
    return [dict(r) for r in rows]

class SlotAdjustmentPayload(BaseModel):
    demand:         str
    forecast_veh:   float
    current_occ:    float
    n_slots:        int
    last_adjusted:  Optional[str] = None
    reason:         Optional[str] = None

class LayoutModePayload(BaseModel):
    mode: Optional[str] = None
    enabled: Optional[bool] = None

class DiscountSettingsPayload(BaseModel):
    pwd_enabled: Optional[bool] = None
    senior_enabled: Optional[bool] = None

class PricingSettingsPayload(BaseModel):
    price_php: Optional[float] = None
    price_php_car: Optional[float] = None
    price_php_motorcycle: Optional[float] = None
    enabled: Optional[bool] = None

# Keep last adjustment in memory so dashboard can read it
_last_slot_adjustment: dict = {}
_LAYOUT_MODES = {"NORMAL", "BUSY", "HIGH"}

# Admin-toggle settings (discounts, layout override, price override) are
# persisted in the admin_settings Postgres table, not a local .env file —
# Render wipes the local filesystem on every restart (including OOM
# crash-restarts), which used to silently reset these toggles back to their
# code defaults. Postgres survives restarts, so it's the durable store now.
# ON CONFLICT DO UPDATE makes concurrent saves (e.g. pricing and layout-mode
# at once) safe without an application-level lock.
def _read_setting(key: str, default: str = "") -> str:
    try:
        rows = query("SELECT value FROM admin_settings WHERE key = %s", (key,))
        if rows:
            return rows[0]["value"]
    except Exception as e:
        print(f"[settings] read failed for {key}: {e}")
    return os.getenv(key, default)

def _write_setting(key: str, value: str) -> None:
    execute(
        """
        INSERT INTO admin_settings (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """,
        (key, value),
    )

def _parse_price(value, default=None):
    try:
        price = round(float(value), 2)
    except (TypeError, ValueError):
        if default is None:
            raise HTTPException(400, "price_php must be a valid number")
        price = round(float(default), 2)
    if price < 1 or price > 10000:
        raise HTTPException(400, "price_php must be between 1 and 10000")
    return price

def _normalize_vehicle_type(value, default="car"):
    raw = str(value if value is not None else default).strip().lower()
    if raw in {"motorcycle", "motor", "moto", "bike", "motorbike"}:
        return "motorcycle"
    if raw in {"car", "auto", "automobile", "sedan"}:
        return "car"
    raise HTTPException(400, "vehicle_type must be 'car' or 'motorcycle'")

def _current_flat_rate(vehicle_type="car"):
    vt = _normalize_vehicle_type(vehicle_type)
    key = "FLAT_RATE_CAR" if vt == "car" else "FLAT_RATE_MOTORCYCLE"
    default = FLAT_RATE_CAR if vt == "car" else FLAT_RATE_MOTORCYCLE
    return _parse_price(_read_setting(key, os.getenv(key, str(default))), default)

def _env_bool(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

def _pricing_settings():
    flat_car = _current_flat_rate("car")
    flat_moto = _current_flat_rate("motorcycle")
    manual_car = _parse_price(_read_setting("PRICE_OVERRIDE_PHP_CAR", str(flat_car)), flat_car)
    manual_moto = _parse_price(_read_setting("PRICE_OVERRIDE_PHP_MOTORCYCLE", str(flat_moto)), flat_moto)
    enabled = _env_bool(_read_setting("PRICE_OVERRIDE_ENABLED", "false"))
    return {
        "enabled": enabled,
        "mode": "manual" if enabled else "dynamic",
        "price_php_car": manual_car,
        "price_php_motorcycle": manual_moto,
        "flat_rate_php_car": flat_car,
        "flat_rate_php_motorcycle": flat_moto,
        # Backward-compatible aliases (car rate) for older callers.
        "price_php": manual_car,
        "manual_price_php": manual_car,
        "flat_rate_php": flat_car,
        "pwd_senior_discount_rate": PWD_SENIOR_DISCOUNT_RATE,
        "pwd_senior_discount_pct": round(PWD_SENIOR_DISCOUNT_RATE * 100, 1),
        "currency": "PHP",
    }

def _discount_settings():
    return {
        "pwd_enabled": _env_bool(_read_setting("DISCOUNT_PWD_ENABLED", "true")),
        "senior_enabled": _env_bool(_read_setting("DISCOUNT_SENIOR_ENABLED", "true")),
        "pwd_senior_discount_rate": PWD_SENIOR_DISCOUNT_RATE,
        "pwd_senior_discount_pct": round(PWD_SENIOR_DISCOUNT_RATE * 100, 1),
    }

def _manual_price_override(vehicle_type="car"):
    settings = _pricing_settings()
    if not settings["enabled"]:
        return None
    vt = _normalize_vehicle_type(vehicle_type)
    return settings["price_php_car"] if vt == "car" else settings["price_php_motorcycle"]

def _clean_email(email: str) -> str:
    return (email or "").strip().lower()

def _is_reserved_admin_email(email: str) -> bool:
    return bool(ADMIN_EMAIL) and _clean_email(email) == ADMIN_EMAIL

def _is_reserved_admin_credentials(email: str, password: str) -> bool:
    return _is_reserved_admin_email(email) and (password or "") == ADMIN_PASSWORD

def _normalized_auth_role(role: str, email: str) -> str:
    if _is_reserved_admin_email(email):
        return "admin"
    return "admin" if str(role or "").strip().lower() == "admin" else "driver"


def _ensure_default_admin():
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        return

    pw_hash = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT user_id FROM users WHERE LOWER(email)=LOWER(%s)",
            (ADMIN_EMAIL,),
        )
        row = cur.fetchone()

        if row:
            admin_id = row["user_id"]
            cur.execute(
                """
                UPDATE users
                SET first_name=%s,
                    last_name=%s,
                    password_hash=%s,
                    role='admin',
                    is_active=TRUE
                WHERE user_id=%s
                """,
                ("John", "Cambiado", pw_hash, admin_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO users (first_name,last_name,email,password_hash,role)
                VALUES (%s,%s,%s,%s,'admin')
                RETURNING user_id
                """,
                ("John", "Cambiado", ADMIN_EMAIL, pw_hash),
            )
            admin_id = cur.fetchone()["user_id"]

        # The app now has only two login roles. Only the reserved account is admin.
        cur.execute(
            """
            UPDATE users
            SET role='driver'
            WHERE LOWER(email)<>LOWER(%s) AND COALESCE(role,'driver') <> 'driver'
            """,
            (ADMIN_EMAIL,),
        )

        try:
            cur.execute("DELETE FROM drivers WHERE user_id=%s", (admin_id,))
            cur.execute(
                """
                INSERT INTO drivers(user_id)
                SELECT u.user_id
                FROM users u
                WHERE LOWER(u.email)<>LOWER(%s)
                  AND NOT EXISTS (
                      SELECT 1 FROM drivers d WHERE d.user_id=u.user_id
                  )
                """,
                (ADMIN_EMAIL,),
            )
        except Exception as e:
            print(f"[auth] Driver table sync skipped: {e}")

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

@app.post("/yolo/slot_adjustment")
async def receive_slot_adjustment(
    payload: SlotAdjustmentPayload,
    x_cam_token: str = Header(None),
):
    _check_cam_token(x_cam_token)
    global _last_slot_adjustment
    _last_slot_adjustment = payload.dict()
    return {"status": "ok"}


@app.get("/api/slot_adjustment")
async def get_slot_adjustment():
    """Dashboard polls this to show current demand level + reason."""
    return _last_slot_adjustment or {
        "demand":       "NORMAL",
        "forecast_veh": 0,
        "current_occ":  0,
        "n_slots":      0,
        "reason":       "No adjustment yet",
    }
# ══════════════════════════════════════════════════════════════════
@app.get("/api/settings/layout-mode")
def get_layout_mode():
    raw_mode = _read_setting("FORCE_DEMAND_LEVEL", "").strip().upper()
    enabled = raw_mode in _LAYOUT_MODES
    mode = raw_mode if enabled else "NORMAL"
    return {"mode": mode, "enabled": enabled, "modes": ["NORMAL", "BUSY", "HIGH"]}

@app.post("/api/settings/layout-mode")
def set_layout_mode(payload: LayoutModePayload, _admin=Depends(require_admin)):
    enabled = True if payload.enabled is None else bool(payload.enabled)
    mode = (payload.mode or _read_setting("FORCE_DEMAND_LEVEL", "NORMAL") or "NORMAL").strip().upper()
    if mode not in _LAYOUT_MODES:
        raise HTTPException(400, "mode must be NORMAL, BUSY, or HIGH")
    if not enabled:
        _write_setting("FORCE_DEMAND_LEVEL", "")
        return {
            "ok": True,
            "mode": mode,
            "enabled": False,
            "message": "Manual layout override is off. Detector will choose the layout automatically.",
        }
    _write_setting("FORCE_DEMAND_LEVEL", mode)
    return {
        "ok": True,
        "mode": mode,
        "enabled": True,
        "message": "Layout mode saved. Detector applies it on the next adjustment cycle.",
    }

#  Auth
# ══════════════════════════════════════════════════════════════════
@app.get("/api/settings/pricing")
def get_pricing_settings():
    return _pricing_settings()

@app.post("/api/settings/pricing")
def set_pricing_settings(payload: PricingSettingsPayload, _admin=Depends(require_admin)):
    current = _pricing_settings()
    enabled = current["enabled"] if payload.enabled is None else bool(payload.enabled)
    car_input = payload.price_php_car if payload.price_php_car is not None else payload.price_php
    price_car = current["price_php_car"] if car_input is None else _parse_price(car_input)
    price_moto = (
        current["price_php_motorcycle"]
        if payload.price_php_motorcycle is None
        else _parse_price(payload.price_php_motorcycle)
    )

    _write_setting("PRICE_OVERRIDE_ENABLED", "true" if enabled else "false")
    _write_setting("PRICE_OVERRIDE_PHP_CAR", f"{price_car:.2f}")
    _write_setting("PRICE_OVERRIDE_PHP_MOTORCYCLE", f"{price_moto:.2f}")

    with _insight_lock:
        _insight_cache.clear()
    threading.Thread(target=_run_insights_now, daemon=True, name="insight-pricing-update").start()

    result = _pricing_settings()
    result.update({
        "ok": True,
        "message": (
            f"Manual rate saved — Car PHP {price_car:.2f}/hr, Motorcycle PHP {price_moto:.2f}/hr."
            if enabled else "Manual pricing is off. Dynamic pricing is active."
        ),
    })
    return result

@app.get("/api/settings/discounts")
def get_discount_settings():
    return _discount_settings()

@app.post("/api/settings/discounts")
def set_discount_settings(payload: DiscountSettingsPayload, _admin=Depends(require_admin)):
    current = _discount_settings()
    pwd_enabled = current["pwd_enabled"] if payload.pwd_enabled is None else bool(payload.pwd_enabled)
    senior_enabled = current["senior_enabled"] if payload.senior_enabled is None else bool(payload.senior_enabled)

    _write_setting("DISCOUNT_PWD_ENABLED", "true" if pwd_enabled else "false")
    _write_setting("DISCOUNT_SENIOR_ENABLED", "true" if senior_enabled else "false")

    result = _discount_settings()
    result.update({
        "ok": True,
        "message": f"PWD discount {'on' if pwd_enabled else 'off'}, Senior discount {'on' if senior_enabled else 'off'}.",
    })
    return result

def _validate_password_strength(password: str) -> None:
    pw = password or ""
    if len(pw) < 8:
        raise HTTPException(400, "Password must be at least 8 characters long")
    if not any(c.isalpha() for c in pw) or not any(c.isdigit() for c in pw):
        raise HTTPException(400, "Password must contain at least one letter and one number")


@app.post("/auth/register")
def register(data: UserRegister):
    email = _clean_email(data.email)
    if _is_reserved_admin_email(email) and not _is_reserved_admin_credentials(email, data.password):
        raise HTTPException(400, "This email is reserved for the admin account")

    _validate_password_strength(data.password)

    role = "admin" if _is_reserved_admin_credentials(email, data.password) else "driver"
    pw = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    conn = get_db()
    cur = conn.cursor()
    try:
        try:
            cur.execute(
                "INSERT INTO users (first_name,last_name,email,password_hash,role) "
                "VALUES (%s,%s,%s,%s,%s) RETURNING user_id",
                (data.first_name, data.last_name, email, pw, role),
            )
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            raise HTTPException(400, "Email already registered")
        new_id = cur.fetchone()["user_id"]
        if role == "driver":
            cur.execute("INSERT INTO drivers(user_id) VALUES(%s)", (new_id,))
        conn.commit()
        return {
            "ok": True,
            "user_id": new_id,
            "first_name": data.first_name,
            "last_name": data.last_name,
            "full_name": f"{data.first_name} {data.last_name}".strip(),
            "email": email,
            "role": role,
            "token": _sign_auth_token(new_id, role),
        }
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, str(e))
    finally:
        cur.close()
        conn.close()


@app.post("/auth/login")
def login(data: UserLogin, request: Request):
    _rate_limit(request, "login", max_calls=10, window_seconds=60)
    email = _clean_email(data.email)
    wait = _login_lockout_seconds_left(email)
    if wait > 0:
        raise HTTPException(429, f"Too many failed attempts. Try again in {max(1, wait // 60 + 1)} minute(s).")
    try:
        if _is_reserved_admin_credentials(email, data.password):
            _ensure_default_admin()

        rows = query(
            "SELECT user_id,first_name,last_name,full_name,email,password_hash,role,is_active "
            "FROM users WHERE LOWER(email)=LOWER(%s)", (email,)
        )
        u = rows[0] if rows else None
        # Check email-not-found and wrong-password together so the response can't be used
        # to enumerate registered accounts. "Account disabled" is only revealed once the
        # caller has already proven they know the correct password.
        if not u or not bcrypt.checkpw(data.password.encode(), u["password_hash"].encode()):
            _login_record_failure(email)
            raise HTTPException(401, "Invalid email or password")
        if not u["is_active"]: raise HTTPException(403, "Account disabled")
        _login_record_success(email)
        role = _normalized_auth_role(u["role"], u["email"])
        if role != str(u["role"] or "").strip().lower():
            execute("UPDATE users SET role=%s WHERE user_id=%s", (role, u["user_id"]))
        execute("UPDATE users SET last_login=%s WHERE user_id=%s",
                (datetime.now(PH_TZ), u["user_id"]))
        return {
            "ok": True, "user_id": u["user_id"],
            "first_name": u["first_name"], "last_name": u["last_name"],
            "full_name":  u["full_name"],  "email":     u["email"],
            "role":       role,
            "token": _sign_auth_token(u["user_id"], role),
        }
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/auth/logout")
def logout(authorization: str = Header(None)):
    if authorization and authorization.lower().startswith("bearer "):
        _revoke_token(authorization.split(" ", 1)[1].strip())
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n========================================")
    print("  OccupAI FastAPI Backend  v2.2")
    print("  http://localhost:8000")
    print("========================================\n")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)