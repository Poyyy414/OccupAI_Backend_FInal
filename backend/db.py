import os
import threading
import time
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from fastapi import HTTPException
from urllib.parse import urlparse

# Never let a local .env replace the deployment DATABASE_URL or credentials.
load_dotenv(override=False)


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[DB] Invalid integer env value {value!r}, falling back to {default}")
        return default


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"[DB] Invalid numeric env value {value!r}, falling back to {default}")
        return default


DB_CONNECT_TIMEOUT = _safe_int(os.getenv("DB_CONNECT_TIMEOUT", "5"), 5)
DB_POOL_MIN = _safe_int(os.getenv("DB_POOL_MIN", "1"), 1)
DB_POOL_MAX = _safe_int(os.getenv("DB_POOL_MAX", "10"), 10)
DB_POOL_WAIT_SECONDS = max(0.1, _safe_float(os.getenv("DB_POOL_WAIT_SECONDS", "3"), 3.0))


def _current_dsn():
    # Read fresh each time (not cached at import) so a runtime-rotated DATABASE_URL
    # is picked up instead of silently continuing to use a stale value forever.
    return os.getenv("DATABASE_URL")


def _database_host(dsn=None):
    dsn = dsn if dsn is not None else _current_dsn()
    if not dsn:
        return None
    try:
        return urlparse(dsn).hostname
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
    """Open a standalone connection. Caller owns its lifecycle (commit/rollback/close)."""
    dsn = _current_dsn()
    if not dsn:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured.")
    try:
        return psycopg2.connect(
            dsn,
            cursor_factory=psycopg2.extras.RealDictCursor,
            connect_timeout=DB_CONNECT_TIMEOUT,
        )
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail=_database_error_detail(exc)) from exc
    except psycopg2.Error as exc:
        raise HTTPException(status_code=503, detail="Database connection failed.") from exc


# ── Pooled path used internally by query()/execute() ──────────────────────
# Kept separate from get_db() so existing callers that manage their own
# connection (open it, run several statements, commit/rollback, close it
# themselves — e.g. payment/registration flows) are unaffected: mixing a
# pool with connections callers close directly would corrupt the pool's
# bookkeeping (a connection closed by the caller instead of returned via
# putconn() looks "still borrowed" forever).
_pool_lock = threading.Lock()
_pool = None
_pool_dsn = None


def _get_pool():
    global _pool, _pool_dsn
    dsn = _current_dsn()
    if not dsn:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured.")

    with _pool_lock:
        if _pool is None or dsn != _pool_dsn:
            if _pool is not None:
                try:
                    _pool.closeall()
                except Exception as e:
                    print(f"[DB] Error closing stale connection pool: {e}")
            try:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    DB_POOL_MIN, DB_POOL_MAX, dsn,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                    connect_timeout=DB_CONNECT_TIMEOUT,
                )
                _pool_dsn = dsn
            except psycopg2.OperationalError as exc:
                raise HTTPException(status_code=503, detail=_database_error_detail(exc)) from exc
        return _pool


def _borrow():
    """Borrow a connection, absorbing short request bursts before failing."""
    deadline = time.monotonic() + DB_POOL_WAIT_SECONDS
    while True:
        pool = _get_pool()
        try:
            return pool.getconn()
        except psycopg2.OperationalError as exc:
            raise HTTPException(status_code=503, detail=_database_error_detail(exc)) from exc
        except psycopg2.pool.PoolError as exc:
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=503,
                    detail="Database is busy. Please try again shortly.",
                ) from exc
            time.sleep(0.05)


def _release(conn):
    try:
        _get_pool().putconn(conn)
    except Exception as e:
        print(f"[DB] Error returning connection to pool: {e}")
        try:
            conn.close()
        except Exception:
            pass


def query(sql, params=None, fetch="all"):
    """Run a SELECT and return all rows."""
    conn = _borrow()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        # psycopg begins a transaction even for SELECT. End it before returning
        # the connection so Supavisor never sees pooled readers sitting "idle
        # in transaction" while the next request waits for a slot.
        conn.rollback()
        return rows
    except psycopg2.Error as exc:
        conn.rollback()
        print(f"[DB] query failed: {exc}")
        raise HTTPException(status_code=500, detail="Database query failed.") from exc
    finally:
        try:
            if cur is not None:
                cur.close()
        finally:
            _release(conn)


def execute(sql, params=None):
    """Run an INSERT/UPDATE/DELETE and commit."""
    conn = _borrow()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        print(f"[DB] execute failed: {exc}")
        raise HTTPException(status_code=500, detail="Database operation failed.") from exc
    finally:
        try:
            if cur is not None:
                cur.close()
        finally:
            _release(conn)
