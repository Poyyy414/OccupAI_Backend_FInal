"""Read-only, redacted PostgreSQL plan and capacity audit for OccupAI."""

from __future__ import annotations

import json

from backend.db import query


TABLES = ("parking_payments", "parking_reservations", "parking_logs")


def _nodes(plan):
    result = [plan.get("Node Type", "unknown")]
    for child in plan.get("Plans", []):
        result.extend(_nodes(child))
    return result


def _explain(label, sql, params=()):
    rows = query("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
    document = rows[0]["QUERY PLAN"][0]
    plan = document["Plan"]
    return {
        "label": label,
        "planning_ms": round(float(document.get("Planning Time", 0)), 3),
        "execution_ms": round(float(document.get("Execution Time", 0)), 3),
        "actual_rows": int(plan.get("Actual Rows", 0)),
        "node_types": _nodes(plan),
        "shared_hit_blocks": int(plan.get("Shared Hit Blocks", 0)),
        "shared_read_blocks": int(plan.get("Shared Read Blocks", 0)),
    }


def main():
    user_rows = query(
        "SELECT user_id FROM users WHERE is_active=TRUE ORDER BY user_id LIMIT 1"
    )
    if not user_rows:
        raise RuntimeError("No active user is available for the audit")
    user_id = int(user_rows[0]["user_id"])

    max_connections = query("SHOW max_connections")[0]["max_connections"]
    connections = query(
        "SELECT state, COUNT(*) AS count FROM pg_stat_activity GROUP BY state ORDER BY state"
    )
    tables = query(
        "SELECT relname AS table_name, n_live_tup AS estimated_rows, "
        "pg_total_relation_size(relid) AS total_bytes "
        "FROM pg_stat_user_tables WHERE relname = ANY(%s) ORDER BY relname",
        (list(TABLES),),
    )
    indexes = query(
        "SELECT tablename, indexname FROM pg_indexes "
        "WHERE schemaname='public' AND tablename = ANY(%s) "
        "ORDER BY tablename, indexname",
        (list(TABLES),),
    )
    plans = [
        _explain(
            "active reservation claims",
            "SELECT slot_id, status, user_id, duration_type, payment_status "
            "FROM parking_reservations "
            "WHERE (status='pending_payment' AND hold_expires_at>NOW()) OR "
            "(status IN ('reserved','occupied') AND expires_at>NOW())",
        ),
        _explain(
            "driver payment history",
            "SELECT payment_id, paid_at FROM parking_payments "
            "WHERE user_id=%s ORDER BY paid_at DESC LIMIT 50",
            (user_id,),
        ),
        _explain(
            "recent payments",
            "SELECT payment_id, paid_at FROM parking_payments "
            "ORDER BY paid_at DESC LIMIT 100",
        ),
    ]
    print(json.dumps({
        "max_connections": int(max_connections),
        "connections_by_state": connections,
        "tables": tables,
        "indexes": indexes,
        "plans": plans,
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
