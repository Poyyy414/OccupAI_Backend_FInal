# OccupAI Backend Performance Audit

Date: 2026-11-05 (Asia/Manila)

## Outcome

The backend now uses one bounded PostgreSQL connection pool for both single-statement helpers and caller-managed transactions. High-frequency reservation reads no longer perform expiry writes, repeated lot-wide reservation reads are coalesced, driver payment plans use a user-keyed cache, and three multi-query endpoints use fewer database round trips.

All 110 backend tests pass. The final read-only probe completed without pool timeouts or endpoint failures.

## Measurement scope

The measurements below were made from the local backend process against the configured Supabase PostgreSQL database. They measure service-function execution and real database network round trips; they do not include Render routing, TLS termination, browser rendering, or a Render cold start. Five samples were used per endpoint, so the figures are directional rather than a production load-test SLA.

| Endpoint | Baseline p50 / p95 | Final p50 / p95 | Baseline DB work | Final DB work |
|---|---:|---:|---:|---:|
| `/api/stats` | 565.18 / 576.91 ms | 0.44 / 307.27 ms | max 1 query + 1 write | max 1 cold query, 0 writes |
| `/api/driver/summary` | 275.85 / 277.19 ms | 0.45 / 317.43 ms | 1 query each call | 1 cold query; warm calls use the user cache |
| `/api/reservations/slots` | 825.28 / 840.63 ms | 304.94 / 613.08 ms | 2 queries + 1 write | 1 normal user query; up to 1 shared cold query; 0 writes |
| `/api/revenue/dashboard` | 0 / 2215.28 ms | 0.08 / 1532.58 ms | 8 cold queries | 5 cold queries; warm cache hit |
| `/api/payments/recent` | 549.31 / 553.75 ms | 307.87 / 315.94 ms | 2 queries | 1 query |
| `/api/predictions` | 0.09 / 238.23 ms | 0.06 / 172.52 ms | cached data path | cached data path |

The summary endpoint's cold p95 varied with the remote database connection, but its steady p50 fell below 1 ms because repeated three-second polls no longer reload the same user's payment plan.

Maximum measured JSON payloads were 6.9 KB (`stats`), 4.6 KB (`driver/summary`), 5.1 KB (`reservations/slots`), 6.4 KB (`revenue/dashboard`), 3.1 KB (`payments/recent`), and 0.8 KB (`predictions`). Global compression was intentionally not added: these payloads are already small, while a global middleware could waste CPU attempting to recompress JPEG camera responses.

## Database and pool findings

- Supabase reported `max_connections = 60` during the audit.
- The application pool defaults to minimum 1, maximum 10, a 3-second bounded wait, and a 5-second connect timeout. Values are configurable through `DB_POOL_MIN`, `DB_POOL_MAX`, `DB_POOL_WAIT_SECONDS`, and `DB_CONNECT_TIMEOUT`.
- All `get_db()` transaction flows now borrow from that same pool and return the connection when their existing `close()` call runs. This removes the previous unbounded standalone-connection path.
- Pool acquisition count, average/max wait, in-use/high-water count, and timeout count are recorded. Exhaustion fails with a controlled HTTP 503 instead of hanging indefinitely.
- Slow database operations emit structured JSON metadata without SQL text, parameters, credentials, email addresses, tokens, or uploaded IDs.
- A deployment should keep at most five application processes at the default pool maximum (`5 x 10 = 50`) to retain headroom beneath the provider's observed limit of 60. OccupAI currently expects one Render process, so a maximum of 10 is conservative.

Read-only `EXPLAIN (ANALYZE, BUFFERS)` results:

| Query | Execution | Plan observation |
|---|---:|---|
| Active reservation claims | 0.027 ms | Bitmap index + heap scan using the active reservation partial index |
| Driver payment history | 0.043 ms | Sequential scan/sort at the current tiny table size |
| Recent payments | 0.046 ms | Sequential scan/sort over 23 rows; appropriate at this size |

Current table sizes were approximately 1.29 MB for `parking_logs` (5,703 estimated rows), and 64 KB each for `parking_payments` and `parking_reservations`. A composite `(user_id, paid_at DESC)` payment index was added to schema setup for growth; PostgreSQL may continue preferring a sequential scan while the table remains tiny.

## Query and transaction changes

- Reservation expiry is handled immediately by a five-second scheduler and synchronously before a new hold. Read-only slot/status requests exclude expired claims by timestamp and no longer write to PostgreSQL.
- Camera vacancy updates only write on the first vacant frame and at the ten-second state transition. Stable frames do not repeatedly update the same row.
- Revenue vehicle, method, discount, and duration breakdowns are fetched with one `UNION ALL` request instead of four sequential requests.
- Recent payment rows and their total count are returned by one CTE query.
- Driver history uses `COUNT(*) OVER()` instead of a separate count query.
- Payment and reservation finalization remain short atomic transactions. External payment-provider calls are outside the database transaction.

## Cache inventory and invalidation

| Cache | TTL | Scope/key | Invalidation |
|---|---:|---|---|
| Active reservation claims | 1 second | Shared lot snapshot; never returned directly | Hold, cancellation, payment finalization, camera transition, or expiry |
| Driver payment plan | 10 seconds | Strictly keyed by numeric user ID; bounded to 512 entries | New parking payment or paid reservation finalization |
| Revenue dashboard | 15 seconds | Admin aggregate only | Any new payment/finalization |
| Prediction/model inputs | Existing bounded file/data caches | Lot aggregate only | Source-file modification or explicit refresh |
| Admin settings | Existing in-process settings cache | Setting key | Settings update |

The reservation cache returns copies, and user-specific `mine` flags are calculated after reading the shared snapshot. The payment-plan cache is keyed per user. Tests verify reuse, invalidation, and cross-user isolation.

## Polling behavior

| Client | Live polling | Supporting data | Protections |
|---|---|---|---|
| Admin web | Stats every 3 s; activity every 5 s only on relevant panels | Insights/predictions/revenue every 30 s when relevant; issues every 60 s | Hidden-tab guard, in-flight request deduplication, memory TTLs, one bounded retry for read-only GETs |
| Driver web | Summary every 3 s; slots every 5 s | Discount eligibility every 30 s | In-flight guards and short server caches |
| Flutter | Stats every 3 s (15 s in reduced-data mode) | Supporting data every 10 s (60 s reduced); forecast every 45 s | Progressive startup, in-flight guards, endpoint TTL cache, refresh cancellation on dispose |

Payment POST requests are not automatically retried, preventing accidental duplicate charges. Database uniqueness constraints on checkout/external references remain the final idempotency guard.

## Observability and HTTP behavior

- API and auth responses send `Cache-Control: no-store` so browsers/proxies cannot mix private or live responses.
- Every response includes an opaque `X-Request-ID` and an `app` duration in `Server-Timing`.
- Requests slower than `REQUEST_SLOW_MS` (default 1000 ms), and all 5xx responses, emit structured JSON containing only method, path, status, duration, request ID, and pool counters.
- Database operations slower than `DB_SLOW_QUERY_MS` (default 500 ms) emit redacted structured events.

## Verification

- `python -m py_compile backend\db.py backend\main.py scripts\performance_probe.py scripts\database_audit.py` — passed.
- `python -m pytest -q` — **110 passed**, 0 failed.
- `python -m scripts.performance_probe` — completed against Supabase; no endpoint failures or pool timeout logs.
- `python -m scripts.database_audit` — completed with read-only plans and capacity checks.

The warnings shown by the test run are existing TensorFlow/scikit-learn model serialization compatibility warnings; they did not fail any test and are separate from database/request performance.

## Remaining risks and deployment notes

1. In-process caches are per Render process. If the service is scaled to multiple workers, correctness remains in PostgreSQL, but cache hit rates fall and invalidation is not cross-process. Use Redis only if multi-process scaling becomes necessary.
2. The cold revenue dashboard still performs five aggregate queries. At substantially larger payment volume, replace those live aggregates with a scheduled summary table or materialized view after measuring production plans.
3. Keep the default pool maximum at 10 for one process. Recalculate `workers x DB_POOL_MAX` before increasing workers and leave provider/admin headroom.
4. The new payment composite index is created by startup schema setup after deployment. The audited database had not yet run this new code, which is why the audit index list did not include it.
5. For production SLA evidence, run a separate HTTP load test from a host near Render after deployment. Include Render cold-start and concurrent-user measurements; do not infer those from this local-to-Supabase probe.
