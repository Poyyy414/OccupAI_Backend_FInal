"""Read-only warm-path probe for OccupAI's main API service functions.

The probe never prints credentials, user IDs, tokens, or row data. It is
intended for local/Render-shell diagnostics when DATABASE_URL is configured.
"""

from __future__ import annotations

import json
import statistics
import time

import backend.main as app_module


def _active_user_id() -> int:
    rows = app_module.query(
        "SELECT user_id FROM users WHERE is_active=TRUE ORDER BY user_id LIMIT 1"
    )
    if not rows:
        raise RuntimeError("No active user is available for the probe")
    return int(rows[0]["user_id"])


def _measure(name, callback, repeats=5):
    original_query = app_module.query
    original_execute = app_module.execute
    samples = []
    counts = []
    payload_sizes = []

    for _ in range(repeats):
        counters = {"queries": 0, "writes": 0}

        def counted_query(*args, **kwargs):
            counters["queries"] += 1
            return original_query(*args, **kwargs)

        def counted_execute(*args, **kwargs):
            counters["writes"] += 1
            return original_execute(*args, **kwargs)

        app_module.query = counted_query
        app_module.execute = counted_execute
        started = time.perf_counter()
        try:
            payload = callback()
            payload_sizes.append(len(json.dumps(payload, default=str).encode("utf-8")))
        finally:
            samples.append((time.perf_counter() - started) * 1000)
            counts.append(dict(counters))
            app_module.query = original_query
            app_module.execute = original_execute

    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered)) - 1))
    return {
        "endpoint": name,
        "samples": repeats,
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "queries_per_call": max(item["queries"] for item in counts),
        "writes_per_call": max(item["writes"] for item in counts),
        "payload_bytes_max": max(payload_sizes),
    }


def main():
    user_id = _active_user_id()
    auth = {"user_id": user_id, "role": "driver"}

    app_module._invalidate_payment_revenue_dashboard_cache()
    probes = [
        ("/api/stats", lambda: app_module.api_stats(_auth=auth)),
        ("/api/driver/summary", lambda: app_module.api_driver_summary(_auth=auth)),
        ("/api/reservations/slots", lambda: app_module.reservation_slots(_auth=auth)),
        ("/api/revenue/dashboard", lambda: app_module.api_revenue_dashboard(_admin={})),
        ("/api/payments/recent", lambda: app_module.api_payments_recent(10, 0, _admin={})),
        ("/api/predictions", lambda: app_module.api_predictions(_auth=auth)),
    ]
    print(json.dumps([_measure(name, callback) for name, callback in probes], indent=2))


if __name__ == "__main__":
    main()
