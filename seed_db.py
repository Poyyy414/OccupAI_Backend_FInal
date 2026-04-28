"""
seed_parking_data.py
Inserts 30 days of realistic synthetic data into parking_logs.

Schema confirmed:
  log_id, occupied, free, total, occupancy_pct, lot_full, logged_at

Run from project ROOT:
    cd C:\\Users\\arias\\Documents\\OccupAI_Backend_FInal
    python seed_parking_data.py
"""
import os, random, sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Import your existing db helpers ──────────────────────────────────
try:
    from backend.db import execute, query
except ModuleNotFoundError:
    print("\n❌  Run this from the project ROOT, not from inside backend/")
    print("    cd C:\\Users\\arias\\Documents\\OccupAI_Backend_FInal")
    print("    python seed_parking_data.py\n")
    sys.exit(1)

LOT_CAPACITY = 8     # Z1-Z8  (change if different)
DAYS_BACK    = 30    # 30 days = 720 rows

# ── Realistic hourly occupancy curve ─────────────────────────────────
HOUR_BASE = [
    2,  2,  2,  2,  3,  5,    # 00-05  night
    10, 22, 58, 72, 66, 62,   # 06-11  morning ramp
    88, 92, 78, 68, 72, 82,   # 12-17  lunch + afternoon peak
    74, 58, 38, 22, 12,  6,   # 18-23  evening drop
]

def occ_pct(hour: int, dow: int) -> float:
    pct = HOUR_BASE[hour]
    if dow >= 5:   pct *= 0.68   # weekend quieter
    elif dow == 0: pct *= 1.12   # Monday busiest
    pct += pct * random.uniform(-0.12, 0.12)
    return round(max(0.0, min(100.0, pct)), 2)

def seed():
    print("=" * 55)
    print("  OccupAI — Synthetic Data Seeder")
    print(f"  Table: parking_logs  |  Capacity: {LOT_CAPACITY}")
    print("=" * 55)

    # Check existing rows
    try:
        existing = query("SELECT COUNT(*) AS cnt FROM parking_logs")
        cnt = existing[0]["cnt"] if existing else 0
    except Exception as e:
        print(f"\n❌  DB error: {e}")
        print("    Make sure DATABASE_URL is set in your .env")
        sys.exit(1)

    if cnt > 50:
        print(f"\n  ⚠  parking_logs already has {cnt} rows.")
        ans = input("  Add more anyway? (y/N): ").strip().lower()
        if ans != 'y':
            print("  Aborted.")
            return

    now   = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=DAYS_BACK)
    ts    = start

    total_hrs = DAYS_BACK * 24
    inserted  = 0
    errors    = 0

    print(f"\n  Inserting {total_hrs} rows...")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}  →  {now.strftime('%Y-%m-%d %H:%M')}\n")

    while ts <= now:
        pct      = occ_pct(ts.hour, ts.weekday())
        occupied = round(pct / 100 * LOT_CAPACITY)
        free     = LOT_CAPACITY - occupied
        lot_full = occupied >= LOT_CAPACITY

        try:
            execute("""
                INSERT INTO parking_logs
                    (occupied, free, total, occupancy_pct, lot_full, logged_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (occupied, free, LOT_CAPACITY, pct, lot_full, ts))
            inserted += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ✗ Row error at {ts}: {e}")

        ts += timedelta(hours=1)

        if inserted % 120 == 0 and inserted > 0:
            pct_done = inserted / total_hrs * 100
            bar = "█" * int(pct_done / 5) + "░" * (20 - int(pct_done / 5))
            print(f"  [{bar}] {inserted}/{total_hrs} ({pct_done:.0f}%)", end='\r')

    print(f"\n\n  ✓ Inserted {inserted} rows  |  Errors: {errors}")

    # Verify
    result = query("SELECT COUNT(*) AS cnt FROM parking_logs")
    total  = result[0]["cnt"] if result else "?"
    print(f"  Total rows in parking_logs: {total}")

    # Sample
    sample = query("""
        SELECT logged_at, occupied, free, total, occupancy_pct
        FROM parking_logs
        ORDER BY logged_at DESC
        LIMIT 6
    """)
    print()
    print("  Latest 6 rows:")
    print(f"  {'logged_at':<22} {'occupied':>8} {'free':>5} {'occ_pct':>8}")
    print("  " + "-" * 46)
    for r in (sample or []):
        print(f"  {str(r['logged_at']):<22} {r['occupied']:>8} {r['free']:>5} {r['occupancy_pct']:>7.1f}%")

    print()
    print("  ✅  Done! Now:")
    print("  1. Restart:  uvicorn backend.main:app --reload --port 8000")
    print("  2. Open:     http://127.0.0.1:8000/dashboard")
    print("  3. Click:    Predictions tab")
    print("=" * 55)


if __name__ == "__main__":
    seed()