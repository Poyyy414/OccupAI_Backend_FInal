import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import HTTPException
from urllib.parse import urlparse

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

def _database_host():
    if not DATABASE_URL:
        return None
    try:
        return urlparse(DATABASE_URL).hostname
    except Exception:
        return None

def _database_error_detail(exc):
    host = _database_host()
    message = str(exc).strip()
    if "could not translate host name" in message:
        target = f" '{host}'" if host else ""
        return (
            f"Database host{target} could not be resolved. "
            "Check DATABASE_URL, DNS/network access, or use the database provider's public host."
        )
    if "timeout expired" in message.lower():
        target = f" '{host}'" if host else ""
        return (
            f"Timed out connecting to database host{target}. "
            "Check DATABASE_URL, firewall rules, and network access."
        )
    return "Database connection failed. Check DATABASE_URL and database network access."

def get_db():
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured.")
    try:
        return psycopg2.connect(
            DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor,
            connect_timeout=DB_CONNECT_TIMEOUT,
        )
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail=_database_error_detail(exc)) from exc

def query(sql, params=None, fetch="all"):
    """Run a SELECT and return all rows."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()

def execute(sql, params=None):
    """Run an INSERT/UPDATE/DELETE and commit."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        conn.commit()
        cur.close()
    finally:
        conn.close()
