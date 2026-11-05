import json
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
DB_POOL_MIN = max(1, _safe_int(os.getenv("DB_POOL_MIN", "1"), 1))
DB_POOL_MAX = max(DB_POOL_MIN, _safe_int(os.getenv("DB_POOL_MAX", "10"), 10))
DB_POOL_WAIT_SECONDS = max(0.1, _safe_float(os.getenv("DB_POOL_WAIT_SECONDS", "3"), 3.0))
DB_SLOW_QUERY_MS = max(1.0, _safe_float(os.getenv("DB_SLOW_QUERY_MS", "500"), 500.0))


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


class _PooledConnectionHandle:
    """Connection-compatible handle whose close() returns it to the pool."""

    def __init__(self, connection):
        self._connection = connection
        self._released = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        if not self._released:
            self._released = True
            _release(self._connection)


def get_db():
    """Borrow a pooled connection for a caller-managed short transaction."""
    return _PooledConnectionHandle(_borrow())


# ── Pooled path used internally by query()/execute() ──────────────────────
# query()/execute() and caller-managed payment/registration transactions share
# one bounded pool. get_db() wraps close() so existing transaction code returns
# rather than physically closes its borrowed connection.
_pool_lock = threading.Lock()
_pool = None
_pool_dsn = None
_pool_metrics_lock = threading.Lock()
_pool_metrics = {
    "borrow_count": 0,
    "borrow_wait_ms_total": 0.0,
    "borrow_wait_ms_max": 0.0,
    "borrow_timeouts": 0,
    "in_use": 0,
    "in_use_max": 0,
}


def _structured_db_log(event, **fields):
    """Emit operational metadata without SQL text, parameters, or credentials."""
    print(json.dumps({"component": "database", "event": event, **fields}, sort_keys=True))


def pool_metrics():
    """Return a safe snapshot for diagnostics and performance tests."""
    with _pool_metrics_lock:
        snapshot = dict(_pool_metrics)
    count = int(snapshot["borrow_count"])
    snapshot["borrow_wait_ms_avg"] = round(
        float(snapshot["borrow_wait_ms_total"]) / count, 3
    ) if count else 0.0
    snapshot["borrow_wait_ms_total"] = round(float(snapshot["borrow_wait_ms_total"]), 3)
    snapshot["borrow_wait_ms_max"] = round(float(snapshot["borrow_wait_ms_max"]), 3)
    snapshot["pool_min"] = DB_POOL_MIN
    snapshot["pool_max"] = DB_POOL_MAX
    return snapshot


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
    started = time.monotonic()
    deadline = started + DB_POOL_WAIT_SECONDS
    while True:
        pool = _get_pool()
        try:
            conn = pool.getconn()
            wait_ms = (time.monotonic() - started) * 1000
            with _pool_metrics_lock:
                _pool_metrics["borrow_count"] += 1
                _pool_metrics["borrow_wait_ms_total"] += wait_ms
                _pool_metrics["borrow_wait_ms_max"] = max(
                    _pool_metrics["borrow_wait_ms_max"], wait_ms
                )
                _pool_metrics["in_use"] += 1
                _pool_metrics["in_use_max"] = max(
                    _pool_metrics["in_use_max"], _pool_metrics["in_use"]
                )
            return conn
        except psycopg2.OperationalError as exc:
            raise HTTPException(status_code=503, detail=_database_error_detail(exc)) from exc
        except psycopg2.pool.PoolError as exc:
            if time.monotonic() >= deadline:
                with _pool_metrics_lock:
                    _pool_metrics["borrow_timeouts"] += 1
                _structured_db_log(
                    "pool_timeout",
                    wait_ms=round((time.monotonic() - started) * 1000, 3),
                    pool_max=DB_POOL_MAX,
                )
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
    finally:
        with _pool_metrics_lock:
            _pool_metrics["in_use"] = max(0, _pool_metrics["in_use"] - 1)


def query(sql, params=None, fetch="all"):
    """Run a SELECT and return all rows."""
    started = time.monotonic()
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
        duration_ms = (time.monotonic() - started) * 1000
        if duration_ms >= DB_SLOW_QUERY_MS:
            _structured_db_log(
                "slow_operation", operation="query", duration_ms=round(duration_ms, 3)
            )


def execute(sql, params=None):
    """Run an INSERT/UPDATE/DELETE and commit."""
    started = time.monotonic()
    conn = _borrow()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        affected = cur.rowcount
        conn.commit()
        return affected
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
        duration_ms = (time.monotonic() - started) * 1000
        if duration_ms >= DB_SLOW_QUERY_MS:
            _structured_db_log(
                "slow_operation", operation="execute", duration_ms=round(duration_ms, 3)
            )
