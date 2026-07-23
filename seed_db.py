"""
seed_db.py — OccupAI local/dev database seeder

Populates a fresh database with demo data so the dashboard and driver
views have something to show without waiting for real camera/payment
traffic. Safe to run multiple times: everything is idempotent.

Run: python seed_db.py
"""
import random
import sys
from datetime import datetime, timedelta

import bcrypt

from backend.db import execute, query

DEMO_DRIVERS = [
    ("Juan", "Dela Cruz", "juan.delacruz@example.com", "Driver123"),
    ("Maria", "Santos", "maria.santos@example.com", "Driver123"),
]


def ensure_tables():
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
    execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower ON users (LOWER(email))")
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
    execute("""
        CREATE TABLE IF NOT EXISTS parking_payments (
            payment_id BIGSERIAL PRIMARY KEY,
            user_id INTEGER,
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


def seed_drivers():
    driver_ids = []
    for first, last, email, password in DEMO_DRIVERS:
        rows = query("SELECT user_id FROM users WHERE LOWER(email)=LOWER(%s)", (email,))
        if rows:
            driver_ids.append(rows[0]["user_id"])
            continue
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        rows = query(
            "INSERT INTO users (first_name,last_name,email,password_hash,role) "
            "VALUES (%s,%s,%s,%s,'driver') RETURNING user_id",
            (first, last, email, pw_hash),
        )
        user_id = rows[0]["user_id"]
        execute("INSERT INTO drivers(user_id) VALUES(%s) ON CONFLICT DO NOTHING", (user_id,))
        driver_ids.append(user_id)
        print(f"[seed] Created demo driver {email} (password: {password})")
    return driver_ids


def seed_parking_logs(hours=24, lot_capacity=44):
    existing = query("SELECT COUNT(*) AS n FROM parking_logs")
    if existing[0]["n"] > 0:
        print("[seed] parking_logs already has data, skipping")
        return
    now = datetime.utcnow()
    for i in range(hours, 0, -1):
        occupied = random.randint(0, lot_capacity)
        logged_at = now - timedelta(hours=i)
        execute(
            "INSERT INTO parking_logs (occupied,free,total,occupancy_pct,lot_full,logged_at) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (
                occupied,
                lot_capacity - occupied,
                lot_capacity,
                round(occupied / lot_capacity * 100, 2),
                occupied >= lot_capacity,
                logged_at,
            ),
        )
    print(f"[seed] Inserted {hours} sample parking_logs rows")


def seed_parking_payments(driver_ids, count=10):
    existing = query("SELECT COUNT(*) AS n FROM parking_payments")
    if existing[0]["n"] > 0:
        print("[seed] parking_payments already has data, skipping")
        return
    now = datetime.utcnow()
    for i in range(count):
        price = random.choice([25.0, 30.0, 35.0, 40.0])
        discount_type = random.choice(["none", "none", "pwd", "senior"])
        discount_rate = 0.20 if discount_type in ("pwd", "senior") else 0.0
        discount_amount = round(price * discount_rate, 2)
        final_amount = round(price - discount_amount, 2)
        paid_at = now - timedelta(hours=random.randint(0, 24 * 7))
        execute(
            """
            INSERT INTO parking_payments (
                user_id, regular_price_php, discount_type, discount_rate,
                discount_amount_php, final_amount_php, payment_method, notes, paid_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                random.choice(driver_ids) if driver_ids else None,
                price, discount_type, discount_rate,
                discount_amount, final_amount,
                random.choice(["cash", "gcash"]), "Seed data", paid_at,
            ),
        )
    print(f"[seed] Inserted {count} sample parking_payments rows")


def main():
    ensure_tables()
    driver_ids = seed_drivers()
    seed_parking_logs()
    seed_parking_payments(driver_ids)
    print("[seed] Done.")


if __name__ == "__main__":
    sys.exit(main())
