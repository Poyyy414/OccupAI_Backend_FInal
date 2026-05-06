"""
seed_db.py — Seed parking_logs with parking_data_training.csv
=============================================================
KEY FIX: Seeds with CURRENT timestamps (today backwards) so
         NOW() - INTERVAL '7 days' queries actually return data.

Run from project root:
    python backend/seed_db.py

Or from inside backend/:
    cd backend && python seed_db.py
"""
import os
import sys
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Find .env ─────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
for _folder in [_here, os.path.dirname(_here)]:
    _env = os.path.join(_folder, ".env")
    if os.path.exists(_env):
        from dotenv import load_dotenv
        load_dotenv(_env)
        print(f"Loaded .env from: {_env}")
        break

# ── Find CSV ──────────────────────────────────────────────────────────────────
CSV_NAME = "parking_data_training.csv"
CSV_FILE = None
for _folder in [_here, os.path.dirname(_here)]:
    _c = os.path.join(_folder, CSV_NAME)
    if os.path.exists(_c):
        CSV_FILE = _c
        break

LOT_CAPACITY = 30


def seed():
    if not CSV_FILE:
        print(f"✗ Cannot find {CSV_NAME}")
        print(f"  Looked in: {_here} and {os.path.dirname(_here)}")
        sys.exit(1)

    print(f"Loading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Loaded {len(df)} rows")

    # ── KEY FIX: Restamp to current time ────────────────────────────────────
    # The CSV runs 2024-11-21 to 2026-02-18.
    # Today is May 2026. Gap = ~77 days → NOW()-7days returns nothing → 0%.
    # Solution: shift all timestamps so the LAST row = now,
    # keeping the same hour/minute pattern.
    now       = datetime.now().replace(second=0, microsecond=0)
    csv_end   = df["datetime"].iloc[-1]
    shift     = now - csv_end
    df["datetime"] = df["datetime"] + shift
    print(f"Restamped: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    print(f"(Shifted by {int(shift.total_seconds()/86400)} days to reach today)")

    # ── Compute fields ────────────────────────────────────────────────────────
    df["occupied"]      = df["vehicles_hour"].clip(0, LOT_CAPACITY).astype(int)
    df["free"]          = (LOT_CAPACITY - df["occupied"]).clip(lower=0).astype(int)
    df["total"]         = LOT_CAPACITY
    df["occupancy_pct"] = (df["occupied"] / LOT_CAPACITY * 100).round(1)
    df["lot_full"]      = df["free"] == 0

    # ── Connect ───────────────────────────────────────────────────────────────
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("✗ DATABASE_URL not set in .env")
        sys.exit(1)

    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── Check existing ────────────────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) AS cnt FROM parking_logs")
    existing = (cur.fetchone() or {}).get("cnt", 0)
    print(f"Existing rows: {existing}")

    if existing > 0:
        print(f"\n⚠  parking_logs has {existing} rows.")
        print("   Recommendation: DELETE FROM parking_logs; first to avoid duplicates.")
        ans = input("   Delete all existing rows and re-seed fresh? [Y/n]: ").strip().lower()
        if ans != "n":
            cur.execute("DELETE FROM parking_logs")
            conn.commit()
            print(f"   Deleted {existing} rows.")
        else:
            print("   Adding rows without deleting (may cause duplicate hours).")

    # ── Batch insert ──────────────────────────────────────────────────────────
    rows = [
        (int(r["occupied"]), int(r["free"]), int(r["total"]),
         float(r["occupancy_pct"]), bool(r["lot_full"]),
         r["datetime"].to_pydatetime())
        for _, r in df.iterrows()
    ]

    print(f"Inserting {len(rows)} rows...")
    try:
        psycopg2.extras.execute_batch(
            cur,
            """INSERT INTO parking_logs
               (occupied, free, total, occupancy_pct, lot_full, logged_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            rows, page_size=500,
        )
        conn.commit()
        inserted = len(rows)
        print(f"✓ Inserted {inserted} rows.")
    except Exception as e:
        conn.rollback()
        print(f"✗ Batch failed: {e} — trying row-by-row...")
        inserted = 0
        for row in rows:
            try:
                cur.execute(
                    "INSERT INTO parking_logs "
                    "(occupied,free,total,occupancy_pct,lot_full,logged_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s)", row)
                inserted += 1
                if inserted % 500 == 0:
                    conn.commit()
                    print(f"  {inserted}/{len(rows)}...")
            except Exception: pass
        conn.commit()

    cur.close(); conn.close()

    # ── Verify the 7-day window has data ──────────────────────────────────────
    print()
    recent = df[df["datetime"] >= datetime.now() - timedelta(days=7)]
    print(f"Rows in last 7 days: {len(recent)}  ← predictions will use these")
    print(f"Unique hours in last 7 days: {sorted(recent['datetime'].dt.hour.unique().tolist())}")

    print()
    print("=" * 55)
    print("  Seeding complete!")
    print("=" * 55)
    print(f"  Rows        : {inserted}")
    print(f"  Latest row  : {df['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  7-day rows  : {len(recent)}")
    print()
    print("  Restart uvicorn then refresh the dashboard.")
    print("  Predictions chart should now show real data.")
    print("=" * 55)


if __name__ == "__main__":
    seed()