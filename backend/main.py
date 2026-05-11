"""
backend/main.py — OccupAI FastAPI Backend  v2.2
Run: uvicorn backend.main:app --reload --port 8000

CHANGES in v2.2:
  - All timestamps now Philippine Time (Asia/Manila, UTC+8)
  - /api/predictions now includes weekday_revenue and today_revenue_forecast
  - Revenue forecast added to predictions endpoint
"""
import os, math, bcrypt, uvicorn, joblib, threading, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
from backend.db     import get_db, query, execute
from backend.models import UserRegister, UserLogin, YoloUpdate, PushFrame

warnings.filterwarnings("ignore")
load_dotenv()

PH_TZ        = ZoneInfo("Asia/Manila")

CAM_TOKEN    = os.getenv("CAM_TOKEN",   "occupai_cam_2027")
DEPLOY_MODE  = os.getenv("DEPLOY_MODE", "local")
STREAM_PORT  = int(os.getenv("STREAM_PORT", "8001"))
LOT_CAPACITY = int(os.getenv("LOT_CAPACITY", "44"))

BASE_DIR     = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "template"
MODEL_DIR    = Path(__file__).resolve().parent / "models"

INTERNAL_STREAM = f"http://127.0.0.1:{STREAM_PORT}/stream"

FLAT_RATE       = 25
OCC_LOW_THRESH  = 7.0
OCC_HIGH_THRESH = 20.0


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
        self._price       = None
        self._rev         = None
        self._scX         = None
        self._scY         = None
        self._occ_scX     = None
        self._occ_scY     = None
        self._occ_feats   = None
        self._price_feats = None
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
                    print(f"[ML] ✓ {fname}  input={self._nb1[name].input_shape}")
                except Exception as e:
                    print(f"[ML] ✗ {fname}: {e}")
            else:
                print(f"[ML] missing {fname}")

        for attr, fname in [
            ("_occ",         "occupancy_model.keras"),
            ("_price",       "pricing_model.pkl"),
            ("_rev",         "revenue_model.pkl"),
            ("_scX",         "scaler_nb1_X.pkl"),
            ("_scY",         "scaler_nb1_y.pkl"),
            ("_occ_scX",     "scaler_occ_X.pkl"),
            ("_occ_scY",     "scaler_occ_y.pkl"),
            ("_occ_feats",   "occ_features.pkl"),
            ("_price_feats", "pricing_features.pkl"),
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
                    print(f"[ML] ✓ {fname}{extra}")
                except Exception as e:
                    print(f"[ML] ✗ {fname}: {e}")

        self._ready = bool(self._nb1)
        if self._occ_feats is not None:
            print(f"[ML] occ_features.pkl has {len(self._occ_feats)} features: {self._occ_feats}")
        print(f"[ML] Ready={self._ready}  models={list(self._nb1)}")

    def predict_vehicles(self, history_df):
        if not self._nb1:
            raise RuntimeError("NB1 models not loaded")
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
        occ_pct = round(min(primary / LOT_CAPACITY * 100, 100.0), 1)
        next_ts = pd.to_datetime(df["datetime"].iloc[-1]) + timedelta(hours=1)
        status  = "LOW" if primary < OCC_LOW_THRESH else ("HIGH" if primary >= OCC_HIGH_THRESH else "MEDIUM")

        return {
            "predicted_vehicles": primary,
            "predicted_occ_pct":  occ_pct,
            "occupancy_status":   status,
            "prediction_for":     next_ts.strftime("%Y-%m-%d %H:%M"),
            "model_used":         "Spatio-Temporal",
            "confidence_pct":     94.79,
            "all_models":         preds,
        }

    def predict_occupancy(self, history_df):
        if self._occ is None:
            raise RuntimeError("Occupancy model not loaded")

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
        occupied = round(occ / 100 * LOT_CAPACITY)

        return {
            "predicted_occ_pct": occ,
            "occupancy_status":  "LOW" if occ < 30 else ("HIGH" if occ >= 70 else "MEDIUM"),
            "occupied_slots":    occupied,
            "free_slots":        LOT_CAPACITY - occupied,
            "model_used":        "Occupancy-BiLSTM",
            "confidence_pct":    84.0,
        }

    def predict_price(self, row):
        if self._price is None or self._price_feats is None:
            raise RuntimeError("Pricing model not loaded")
        X     = np.array([[row.get(f, 0.0) for f in self._price_feats]])
        price = round(max(15.0, min(50.0, float(self._price.predict(X)[0]))), 2)
        return {
            "recommended_price_php": price,
            "flat_rate_php":         FLAT_RATE,
            "price_change_pct":      round((price - FLAT_RATE) / FLAT_RATE * 100, 1),
            "model_used":            "Pricing-RF",
            "confidence_pct":        80.34,
        }

    def predict_revenue(self, row):
        if self._rev is None or self._rev_feats is None:
            raise RuntimeError("Revenue model not loaded")
        X   = np.array([[row.get(f, 0.0) for f in self._rev_feats]])
        rev = max(0.0, float(self._rev.predict(X)[0]))
        return {
            "predicted_daily_revenue_php": round(rev, 2),
            "model_used":                  "Revenue-GBR",
            "confidence_pct":              83.02,
        }


ml = _MLEngine()


# ══════════════════════════════════════════════════════════════════
#  Insight cache
# ══════════════════════════════════════════════════════════════════
_insight_cache: dict = {}
_insight_lock = threading.Lock()


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

    if s.get("lot_full"):
        out["live_status"] = (
            "🚫 The parking lot is completely full right now. "
            "No available spaces remain. Consider redirecting incoming vehicles."
        )
    elif total == 0:
        out["live_status"] = "⏳ Waiting for the camera detector to connect."
    else:
        out["live_status"] = (
            f"The parking lot is currently {_pct_word(occ)}. "
            f"{free} space{'s are' if free!=1 else ' is'} available out of {total} total. "
            f"{occupied} vehicle{'s are' if occupied!=1 else ' is'} parked right now."
        )

    out["trend"] = _trend(hist)

    if ml._ready and len(df) >= 24:
        try:
            r   = ml.predict_vehicles(df)
            n   = int(round(r["predicted_vehicles"]))
            pf  = r["prediction_for"]
            st  = r["occupancy_status"]
            cf  = r["confidence_pct"]
            desc = {
                "LOW":    "quiet — most spaces should be free",
                "MEDIUM": "moderately busy — about half the lot may be used",
                "HIGH":   "very busy — the lot could fill up",
            }
            out["vehicle_forecast"] = (
                f"Around {pf[-5:]}, the system expects roughly "
                f"{n} vehicle{'s' if n!=1 else ''} to be in the lot. "
                f"It will likely be {desc.get(st,'uncertain')}. "
                f"(Confidence: {cf:.0f}%)"
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
            r  = ml.predict_occupancy(df)
            pf = r["free_slots"]
            st = r["occupancy_status"]
            urgency = {
                "LOW":    "You should have plenty of space available.",
                "MEDIUM": "Expect moderate traffic — some spaces may run low.",
                "HIGH":   "The lot is predicted to fill up. Consider preparing overflow parking.",
            }
            out["occupancy_forecast"] = (
                f"In the next hour the lot will be {_pct_word(r['predicted_occ_pct'])}, "
                f"with roughly {pf} free space{'s' if pf!=1 else ''} remaining. "
                f"{urgency.get(st,'')}"
            )
        except Exception as e:
            out["occupancy_forecast"] = f"Occupancy forecast temporarily unavailable. ({e})"
    else:
        out["occupancy_forecast"] = (
            "Occupancy forecast will appear after 24+ hours of operation."
        )

    if ml._ready and ml._price is not None and ml._price_feats is not None and not df.empty:
        try:
            r     = ml.predict_price(_engineer_nb2_occ(df).iloc[-1].to_dict())
            price = r["recommended_price_php"]
            chg   = r["price_change_pct"]
            if abs(chg) < 5:
                out["pricing"] = f"Demand is normal. The standard flat rate of ₱{FLAT_RATE}/hr is appropriate."
            elif chg > 0:
                out["pricing"] = (
                    f"Demand is higher than usual. Raising the rate to ₱{price:.0f}/hr "
                    f"(+{chg:.0f}% above the ₱{FLAT_RATE} flat rate) is recommended."
                )
            else:
                out["pricing"] = (
                    f"Demand is lower than usual. Offering a discount of ₱{price:.0f}/hr "
                    f"({abs(chg):.0f}% below the ₱{FLAT_RATE} standard) could attract more drivers."
                )
        except Exception as e:
            out["pricing"] = f"Pricing recommendation temporarily unavailable. ({e})"
    else:
        out["pricing"] = "Dynamic pricing needs more historical data."

    if ml._ready and ml._rev is not None and ml._rev_feats is not None:
        try:
            rows = query("""
                SELECT DATE(logged_at AT TIME ZONE 'Asia/Manila') AS date,
                       SUM(occupied)      AS daily_vehicles,
                       AVG(occupancy_pct) AS avg_occ,
                       SUM(occupied * 25) AS daily_revenue
                FROM parking_logs
                WHERE logged_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(logged_at AT TIME ZONE 'Asia/Manila') ORDER BY date
            """)
            if len(rows) >= 7:
                d = pd.DataFrame(rows)
                d["date"]        = pd.to_datetime(d["date"])
                d["day_of_week"] = d["date"].dt.dayofweek
                d["month"]       = d["date"].dt.month
                d["is_weekend"]  = (d["day_of_week"] >= 5).astype(int)
                d["dow_sin"]     = np.sin(2*np.pi*d["day_of_week"]/7)
                d["dow_cos"]     = np.cos(2*np.pi*d["day_of_week"]/7)
                d["month_sin"]   = np.sin(2*np.pi*(d["month"]-1)/12)
                d["month_cos"]   = np.cos(2*np.pi*(d["month"]-1)/12)
                r = d["daily_revenue"].astype(float)
                v = d["daily_vehicles"].astype(float)
                d["rev_lag_1d"]   = r.shift(1).fillna(0)
                d["rev_lag_2d"]   = r.shift(2).fillna(0)
                d["rev_lag_7d"]   = r.shift(7).fillna(0)
                d["rev_roll_3d"]  = r.shift(1).rolling(3, min_periods=1).mean().fillna(0)
                d["rev_roll_7d"]  = r.shift(1).rolling(7, min_periods=1).mean().fillna(0)
                d["veh_lag_1d"]   = v.shift(1).fillna(0)
                d["veh_lag_7d"]   = v.shift(7).fillna(0)
                d["veh_roll_7d"]  = v.shift(1).rolling(7, min_periods=1).mean().fillna(0)
                d["avg_occ_lag"]  = d["avg_occ"].shift(1).fillna(0)
                d["same_dow_wk1"] = r.shift(7).fillna(0)
                d["same_dow_wk2"] = r.shift(14).fillna(0)
                d["same_dow_avg"] = (d["same_dow_wk1"] + d["same_dow_wk2"]) / 2
                rev_result        = ml.predict_revenue(d.iloc[-1].to_dict())
                rev_php = rev_result["predicted_daily_revenue_php"]
                out["revenue_forecast"] = (
                    f"Today's predicted revenue is approximately "
                    f"₱{rev_php:,.2f}. "
                    f"This estimate is based on current occupancy patterns and historical trends. "
                    f"(Model: {rev_result['model_used']} · Confidence: {rev_result['confidence_pct']:.0f}%)"
                )
            else:
                out["revenue_forecast"] = (
                    f"Revenue forecast needs 7+ days of data. "
                    f"Currently have {len(rows)} day(s) in the database."
                )
        except Exception as e:
            out["revenue_forecast"] = f"Revenue forecast temporarily unavailable. ({e})"
    else:
        out["revenue_forecast"] = "Revenue model not loaded. Check backend/models/revenue_model.pkl."

    try:
        rows = query("""
            SELECT EXTRACT(HOUR FROM logged_at AT TIME ZONE 'Asia/Manila') AS hour,
                   AVG(occupied) AS avg_veh
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '7 days'
            GROUP BY hour ORDER BY hour
        """)
        if rows and len(rows) >= 3:
            hourly  = {int(r["hour"]): float(r["avg_veh"]) for r in rows}
            peak_h  = max(hourly, key=lambda h: hourly[h])
            quiet_h = min(hourly, key=lambda h: hourly[h])
            now_desc = (
                "You are currently in a peak period — expect the lot to stay busy."
                if abs(hour - peak_h) <= 1
                else "Traffic should be relatively normal at this hour."
            )
            out["peak_hours"] = (
                f"Based on the last 7 days, the busiest time is around "
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
    if ml._ready and ml._price is not None and ml._price_feats is not None and not df.empty:
        try:
            r = ml.predict_price(_engineer_nb2_occ(df).iloc[-1].to_dict())
            if r["price_change_pct"] > 10:
                actions.append("Consider raising the parking rate — demand is high.")
            elif r["price_change_pct"] < -10:
                actions.append("Consider a promotional rate — the lot is underutilized.")
        except Exception:
            pass

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
@asynccontextmanager
async def lifespan(app: FastAPI):
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    import traceback
    return JSONResponse(500, {"error": str(exc), "trace": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════
#  Shared state
# ══════════════════════════════════════════════════════════════════
state = {
    "occupied": 0, "free": 0, "total": 0, "occupancy_pct": 0.0,
    "lot_full": False, "fps": 0.0, "timestamp": "",
    "yolo_count": 0, "yolo_boxes": [], "slots": [], "zones": {},
}
history    = deque(maxlen=200)
state_lock = threading.Lock()


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
@app.get("/analytics", response_class=FileResponse)
def analytics_page():
    return FileResponse(str(TEMPLATE_DIR / "analytics.html"))


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
    if x_cam_token != CAM_TOKEN:
        raise HTTPException(401, "Unauthorized")
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
            "yolo_boxes":    data.yolo_boxes,
            "slots":         data.slots,
            "zones":         data.zones,
        })
    history.append({
        "time": ts, "occupied": data.occupied,
        "total": data.total, "pct": round(data.occupancy_pct, 1),
    })
    try:
        execute(
            "INSERT INTO parking_logs (occupied,free,total,occupancy_pct,lot_full) "
            "VALUES (%s,%s,%s,%s,%s)",
            (data.occupied, data.free, data.total,
             round(data.occupancy_pct, 1), data.lot_full),
        )
    except Exception as e:
        print(f"[DB] {e}")
    return {"ok": True}


@app.post("/yolo/push-frame")
def push_frame(data: PushFrame, x_cam_token: str = Header(...)):
    if x_cam_token != CAM_TOKEN:
        raise HTTPException(401, "Unauthorized")
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
        rows = query(
            f"SELECT logged_at AS datetime, occupied AS vehicles_hour "
            f"FROM parking_logs ORDER BY logged_at DESC LIMIT {hours}"
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


# ══════════════════════════════════════════════════════════════════
#  ML endpoints
# ══════════════════════════════════════════════════════════════════
@app.get("/api/ml/predict/vehicles")
def ml_predict_vehicles():
    if not ml._ready: raise HTTPException(503, "ML not loaded")
    df = _db_history()
    if len(df) < NB1_SEQ_LEN:
        raise HTTPException(422, f"Need {NB1_SEQ_LEN} rows, have {len(df)}")
    try: return ml.predict_vehicles(df)
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/predict/occupancy")
def ml_predict_occupancy():
    if not ml._ready: raise HTTPException(503, "ML not loaded")
    df = _db_history()
    if len(df) < NB2_OCC_SEQ_LEN:
        raise HTTPException(422, f"Need {NB2_OCC_SEQ_LEN} rows, have {len(df)}")
    try: return ml.predict_occupancy(df)
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/predict/price")
def ml_predict_price():
    if not ml._ready: raise HTTPException(503, "ML not loaded")
    df = _db_history()
    if df.empty: raise HTTPException(422, "No history")
    try:
        return ml.predict_price(_engineer_nb2_occ(df).iloc[-1].to_dict())
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/predict/revenue")
def ml_predict_revenue():
    if not ml._ready: raise HTTPException(503, "ML not loaded")
    try:
        rows = query("""
            SELECT DATE(logged_at AT TIME ZONE 'Asia/Manila') AS date,
                   SUM(occupied)      AS daily_vehicles,
                   AVG(occupancy_pct) AS avg_occ,
                   SUM(occupied * 25) AS daily_revenue
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(logged_at AT TIME ZONE 'Asia/Manila') ORDER BY date
        """)
        if len(rows) < 7:
            raise HTTPException(422, f"Need 7+ days, have {len(rows)}")
        d = pd.DataFrame(rows)
        d["date"]        = pd.to_datetime(d["date"])
        d["day_of_week"] = d["date"].dt.dayofweek
        d["month"]       = d["date"].dt.month
        d["is_weekend"]  = (d["day_of_week"] >= 5).astype(int)
        d["dow_sin"]     = np.sin(2*np.pi*d["day_of_week"]/7)
        d["dow_cos"]     = np.cos(2*np.pi*d["day_of_week"]/7)
        d["month_sin"]   = np.sin(2*np.pi*(d["month"]-1)/12)
        d["month_cos"]   = np.cos(2*np.pi*(d["month"]-1)/12)
        r = d["daily_revenue"].astype(float)
        v = d["daily_vehicles"].astype(float)
        d["rev_lag_1d"]   = r.shift(1).fillna(0)
        d["rev_lag_2d"]   = r.shift(2).fillna(0)
        d["rev_lag_7d"]   = r.shift(7).fillna(0)
        d["rev_roll_3d"]  = r.shift(1).rolling(3, min_periods=1).mean().fillna(0)
        d["rev_roll_7d"]  = r.shift(1).rolling(7, min_periods=1).mean().fillna(0)
        d["veh_lag_1d"]   = v.shift(1).fillna(0)
        d["veh_lag_7d"]   = v.shift(7).fillna(0)
        d["veh_roll_7d"]  = v.shift(1).rolling(7, min_periods=1).mean().fillna(0)
        d["avg_occ_lag"]  = d["avg_occ"].shift(1).fillna(0)
        d["same_dow_wk1"] = r.shift(7).fillna(0)
        d["same_dow_wk2"] = r.shift(14).fillna(0)
        d["same_dow_avg"] = (d["same_dow_wk1"] + d["same_dow_wk2"]) / 2
        return ml.predict_revenue(d.iloc[-1].to_dict())
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/api/ml/dashboard")
def ml_dashboard():
    df  = _db_history()
    out = {"ml_ready": ml._ready, "timestamp": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S PHT")}
    if ml._ready and len(df) >= NB1_SEQ_LEN:
        for key, fn in [
            ("vehicles",  lambda: ml.predict_vehicles(df)),
            ("occupancy", lambda: ml.predict_occupancy(df)),
        ]:
            try:   out[key] = fn()
            except Exception as e: out[key] = {"error": str(e)}
        try:
            out["price"] = ml.predict_price(_engineer_nb2_occ(df).iloc[-1].to_dict())
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
        # Hourly avg occupancy in PH time (last 7 days)
        rows = query("""
            SELECT EXTRACT(HOUR FROM logged_at AT TIME ZONE 'Asia/Manila') AS hour,
                   AVG(occupancy_pct) AS avg_pct
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '7 days'
            GROUP BY hour ORDER BY hour
        """)
        hourly = {str(int(r["hour"])): round(float(r["avg_pct"]), 1) for r in rows}
        for h in range(24): hourly.setdefault(str(h), 0.0)
        peak = max(hourly, key=lambda h: hourly[h])

        # Daily revenue per weekday (last 30 days), in PH time
        rev_rows = query("""
            SELECT EXTRACT(DOW FROM logged_at AT TIME ZONE 'Asia/Manila') AS dow,
                   DATE(logged_at AT TIME ZONE 'Asia/Manila') AS date,
                   SUM(occupied * 25) AS daily_revenue
            FROM parking_logs
            WHERE logged_at >= NOW() - INTERVAL '30 days'
            GROUP BY dow, DATE(logged_at AT TIME ZONE 'Asia/Manila')
            ORDER BY date
        """)

        day_labels = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
        dow_totals = defaultdict(list)
        for r in rev_rows:
            dow_totals[int(r["dow"])].append(float(r["daily_revenue"] or 0))

        weekday_revenue = {
            day_labels[d]: round(sum(v) / len(v), 2) if v else 0.0
            for d, v in dow_totals.items()
        }
        for lbl in day_labels.values():
            weekday_revenue.setdefault(lbl, 0.0)

        # Today's ML revenue forecast
        today_forecast = None
        if ml._ready and ml._rev is not None and ml._rev_feats is not None:
            try:
                rows_30 = query("""
                    SELECT DATE(logged_at AT TIME ZONE 'Asia/Manila') AS date,
                           SUM(occupied)      AS daily_vehicles,
                           AVG(occupancy_pct) AS avg_occ,
                           SUM(occupied * 25) AS daily_revenue
                    FROM parking_logs
                    WHERE logged_at >= NOW() - INTERVAL '30 days'
                    GROUP BY DATE(logged_at AT TIME ZONE 'Asia/Manila')
                    ORDER BY date
                """)
                if len(rows_30) >= 7:
                    d = pd.DataFrame(rows_30)
                    d["date"]        = pd.to_datetime(d["date"])
                    d["day_of_week"] = d["date"].dt.dayofweek
                    d["month"]       = d["date"].dt.month
                    d["is_weekend"]  = (d["day_of_week"] >= 5).astype(int)
                    d["dow_sin"]     = np.sin(2*np.pi*d["day_of_week"]/7)
                    d["dow_cos"]     = np.cos(2*np.pi*d["day_of_week"]/7)
                    d["month_sin"]   = np.sin(2*np.pi*(d["month"]-1)/12)
                    d["month_cos"]   = np.cos(2*np.pi*(d["month"]-1)/12)
                    r2 = d["daily_revenue"].astype(float)
                    v2 = d["daily_vehicles"].astype(float)
                    d["rev_lag_1d"]   = r2.shift(1).fillna(0)
                    d["rev_lag_2d"]   = r2.shift(2).fillna(0)
                    d["rev_lag_7d"]   = r2.shift(7).fillna(0)
                    d["rev_roll_3d"]  = r2.shift(1).rolling(3, min_periods=1).mean().fillna(0)
                    d["rev_roll_7d"]  = r2.shift(1).rolling(7, min_periods=1).mean().fillna(0)
                    d["veh_lag_1d"]   = v2.shift(1).fillna(0)
                    d["veh_lag_7d"]   = v2.shift(7).fillna(0)
                    d["veh_roll_7d"]  = v2.shift(1).rolling(7, min_periods=1).mean().fillna(0)
                    d["avg_occ_lag"]  = d["avg_occ"].shift(1).fillna(0)
                    d["same_dow_wk1"] = r2.shift(7).fillna(0)
                    d["same_dow_wk2"] = r2.shift(14).fillna(0)
                    d["same_dow_avg"] = (d["same_dow_wk1"] + d["same_dow_wk2"]) / 2
                    rev_result = ml.predict_revenue(d.iloc[-1].to_dict())
                    today_forecast = rev_result["predicted_daily_revenue_php"]
            except Exception as e:
                print(f"[predictions revenue] {e}")

        return {
            "hourly_est":             hourly,
            "peak_hour":              int(peak),
            "peak_label":             f"{peak}:00 ({hourly[peak]:.0f}%)",
            "busy_days":              ["Mon", "Tue", "Wed"],
            "quiet_days":             ["Sat", "Sun"],
            "weekday_revenue":        weekday_revenue,
            "today_revenue_forecast": today_forecast,
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

@app.get("/api/predictions/hourly-by-day")
def api_hourly_by_day():
    """
    Returns average vehicle count per hour per day-of-week
    from the last 30 days of parking_logs, in Philippine Time.
    Used by the Analytics page hourly-by-day chart.
    """
    try:
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
                        "occ_pct":  round(v / LOT_CAPACITY * 100, 1),
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
            "lot_capacity":   LOT_CAPACITY,
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
            return {
                "live_status":        "⏳ Insights are being computed (ready in ~15s after startup).",
                "trend":              "—",
                "vehicle_forecast":   "—",
                "occupancy_forecast": "—",
                "pricing":            "—",
                "revenue_forecast":   "—",
                "peak_hours":         "—",
                "admin_action":       "Please wait — the system is initializing.",
                "generated_at":       datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "next_refresh":       "~15 seconds",
            }
        result = dict(_insight_cache)

    try:
        last      = datetime.strptime(result["generated_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=PH_TZ)
        mins_ago  = int((datetime.now(PH_TZ) - last).total_seconds() / 60)
        mins_left = max(0, 60 - mins_ago)
        result["next_refresh"]   = f"Auto-refreshes in ~{mins_left} min" if mins_left > 1 else "Refreshing soon…"
        result["last_refreshed"] = "just now" if mins_ago < 1 else f"{mins_ago} min ago"
    except Exception:
        pass

    return result


@app.post("/api/insights/refresh")
def api_insights_refresh():
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

# Keep last adjustment in memory so dashboard can read it
_last_slot_adjustment: dict = {}

@app.post("/yolo/slot_adjustment")
async def receive_slot_adjustment(
    payload: SlotAdjustmentPayload,
    x_cam_token: str = Header(None),
):
    if x_cam_token != CAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid cam token")
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
#  Auth
# ══════════════════════════════════════════════════════════════════
@app.post("/auth/register")
def register(data: UserRegister):
    pw = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    try:
        if query("SELECT user_id FROM users WHERE email=%s", (data.email,)):
            raise HTTPException(400, "Email already registered")
        r = query(
            "INSERT INTO users (first_name,last_name,email,password_hash,role) "
            "VALUES (%s,%s,%s,%s,'driver') RETURNING user_id",
            (data.first_name, data.last_name, data.email, pw),
        )
        new_id = r[0]["user_id"]
        execute("INSERT INTO drivers(user_id) VALUES(%s)", (new_id,))
        return {"ok": True, "user_id": new_id}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/auth/login")
def login(data: UserLogin):
    try:
        rows = query(
            "SELECT user_id,first_name,last_name,full_name,email,password_hash,role,is_active "
            "FROM users WHERE email=%s", (data.email,)
        )
        if not rows: raise HTTPException(404, "Email not found")
        u = rows[0]
        if not u["is_active"]: raise HTTPException(403, "Account disabled")
        if not bcrypt.checkpw(data.password.encode(), u["password_hash"].encode()):
            raise HTTPException(401, "Incorrect password")
        execute("UPDATE users SET last_login=%s WHERE user_id=%s",
                (datetime.now(PH_TZ), u["user_id"]))
        return {
            "ok": True, "user_id": u["user_id"],
            "first_name": u["first_name"], "last_name": u["last_name"],
            "full_name":  u["full_name"],  "email":     u["email"],
            "role":       u["role"],
        }
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/auth/logout")
def logout():
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║  OccupAI FastAPI Backend  v2.2       ║")
    print("║  http://localhost:8000               ║")
    print("╚══════════════════════════════════════╝\n")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
