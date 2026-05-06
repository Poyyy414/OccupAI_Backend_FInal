"""
seed_db.py  —  Seeds parking_logs with your REAL training dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run from project ROOT:
    cd C:\\Users\\arias\\Documents\\OccupAI_Backend_FInal
    python seed_db.py

Place parking_data_training.csv in the project root first.
"""
import os, sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from backend.db import execute, query
except ModuleNotFoundError:
    print("\n❌  Run from project ROOT:")
    print("    cd C:\\Users\\arias\\Documents\\OccupAI_Backend_FInal")
    print("    python seed_db.py\n")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("❌  pip install pandas"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────────
CSV_PATH     = Path("parking_data_training.csv")
LOT_CAPACITY = 45   # max vehicles_hour in dataset ≈ 43.67, round up to 45

def seed():
    print("=" * 60)
    print("  OccupAI — Real Dataset Seeder")
    print(f"  CSV: {CSV_PATH}")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"\n❌  {CSV_PATH} not found.")
        print("    Copy parking_data_training.csv to the project root.\n")
        sys.exit(1)

    # ── Load CSV ───────────────────────────────────────────────────
    print("\n  Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  {len(df)} rows  |  {df['datetime'].min()} → {df['datetime'].max()}")

    # ── Check existing rows ────────────────────────────────────────
    try:
        existing = query("SELECT COUNT(*) AS cnt FROM parking_logs")
        cnt = int(existing[0]["cnt"]) if existing else 0
    except Exception as e:
        print(f"\n❌  DB error: {e}")
        print("    Check your .env DATABASE_URL\n")
        sys.exit(1)

    if cnt > 100:
        print(f"\n  ⚠  parking_logs already has {cnt} rows.")
        ans = input("  Clear and re-seed with real data? (y/N): ").strip().lower()
        if ans == 'y':
            try:
                execute("DELETE FROM parking_logs")
                print("  ✓ Cleared old rows.")
            except Exception as e:
                print(f"  ✗ Could not clear: {e}")
                sys.exit(1)
        else:
            print("  Aborted.")
            return

    # ── Insert ─────────────────────────────────────────────────────
    print(f"\n  Inserting {len(df)} rows into parking_logs...")
    inserted = 0
    errors   = 0

    for _, row in df.iterrows():
        try:
            vehicles  = float(row["vehicles_hour"])
            occupied  = round(vehicles)
            free      = max(0, LOT_CAPACITY - occupied)
            occ_pct   = round(vehicles / LOT_CAPACITY * 100, 2)
            lot_full  = occupied >= LOT_CAPACITY
            ts        = row["datetime"].to_pydatetime()

            execute("""
                INSERT INTO parking_logs
                    (occupied, free, total, occupancy_pct, lot_full, logged_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (occupied, free, LOT_CAPACITY, occ_pct, lot_full, ts))
            inserted += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ✗ Row error ({row.get('datetime','?')}): {e}")

        if inserted % 200 == 0 and inserted > 0:
            pct_done = inserted / len(df) * 100
            bar = "█" * int(pct_done / 5) + "░" * (20 - int(pct_done / 5))
            print(f"  [{bar}] {inserted}/{len(df)} ({pct_done:.0f}%)", end='\r')

    print(f"\n\n  ✓ Inserted {inserted} rows  |  Errors: {errors}")

    # ── Verify ─────────────────────────────────────────────────────
    result = query("SELECT COUNT(*) AS cnt FROM parking_logs")
    total  = result[0]["cnt"] if result else "?"
    print(f"  Total rows in parking_logs: {total}")

    sample = query("""
        SELECT logged_at, occupied, free, total, occupancy_pct
        FROM parking_logs ORDER BY logged_at DESC LIMIT 5
    """)
    print("\n  Latest 5 rows:")
    print(f"  {'logged_at':<22} {'occ':>5} {'free':>5} {'pct':>7}")
    print("  " + "-" * 44)
    for r in (sample or []):
        print(f"  {str(r['logged_at']):<22} {r['occupied']:>5} {r['free']:>5} {r['occupancy_pct']:>6.1f}%")

    print()
    print("  ✅  Done! Now restart the backend:")
    print("      uvicorn backend.main:app --reload --port 8000")
    print("  Then open: http://127.0.0.1:8000/dashboard → AI Insights")
    print("=" * 60)

if __name__ == "__main__":
    seed()