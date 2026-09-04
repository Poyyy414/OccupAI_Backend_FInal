# OccupAI Performance Audit

Date: 2026-09-05

## Result

The main loading problem was a request waterfall and repeated requests, not a
missing timeout. The web Overview requested its own data and then requested
many of the same endpoints again through the Insights panel. Hidden navigation
panels also kept polling. The Flutter dashboard awaited forecast and settings
requests one at a time, and `IndexedStack` constructed transaction/settings
children that immediately fetched data before the user opened those pages.

The changes keep live detector/camera state fresh and cache only short-lived,
user-scoped analytics and settings.

## Request-flow comparison

These are code-audit counts for one authenticated page load, not fabricated
browser timing measurements. Use the DevTools procedure below for exact
production numbers.

| Flow | Before | After |
|---|---:|---:|
| Web admin Overview unique API requests | Up to 17 attempts (12 top-level plus five nested Insights requests; several duplicated) | 7 critical unique requests; request broker deduplicates duplicates |
| Web navigation back to a loaded panel | Panel fetches repeated on every switch | Panel TTL/state prevents reload within the TTL; background polling is panel-aware |
| Flutter initial forecast/settings calls | 11 forecast calls serially, plus early settings and transaction-log calls | Independent forecast calls run in parallel; transaction logs/settings load on their tabs |
| Flutter navigation state | `IndexedStack` preserved widgets, but API calls could repeat after recreation | `IndexedStack` plus user-scoped in-memory TTL/dedup cache |

Live `/api/stats`, camera stream requests, and current occupancy are not cached.
History, predictions, insights, revenue summaries, ML metrics, settings, and
driver history use short TTLs. Successful writes clear the mobile cache; web
settings/payment writes clear the related dashboard cache; logout clears
user-specific cache state.

## Backend findings

- `/api/revenue/dashboard` performs several aggregate queries. It now shares a
  15-second in-process result cache and invalidates it after a recorded payment.
- Training CSV/model data already use bounded process caches, so repeated
  prediction calls do not repeatedly reload the same files.
- Existing indexes cover recent payment ordering and parking-log ordering.
- Local authenticated endpoint timing was not measured because no local server
  was running during this audit.

## Observed reachability checks

From this environment, the deployed Render service responded successfully:

- `GET /status`: HTTP 200 in about 0.371 seconds.
- `GET /dashboard`: HTTP 200 in about 0.363 seconds; response size about 203 KB.

These are public shell/health timings only. They do not represent an
authenticated Overview load or a cold Render instance.

## Reproducible browser measurement

1. Open the authenticated admin dashboard in Chromium.
2. Open DevTools → Network and enable “Disable cache” only for the baseline.
3. Hard reload Overview and record request count, duration, status, and size.
4. Visit Revenue, Reports, then Overview; record requests caused by each switch.
5. Repeat with normal cache enabled and compare the waterfall.
6. Repeat using Fast 3G and Slow 3G throttling.
7. Repeat with an empty database and a realistic staging dataset.

Do not use real payments in this test. Use a disposable/staging database and
mocked payment responses.

## Validation completed

- Backend: `python -m pytest -q` — 40 passed.
- Backend syntax: `python -m py_compile backend\\main.py tests\\test_predictions.py`.
- Flutter: `flutter analyze` — no issues found.
- Flutter: `flutter test` — all tests passed.

## Remaining manual checks

- Capture authenticated before/after Network-panel counts in local and
  production environments.
- Repeat with slow network, empty/minimal data, and realistic data volumes.
- Confirm live camera frames and occupancy continue changing while historical
  panels show cached data.
- Confirm logout/account switching cannot display another user’s cached data.
