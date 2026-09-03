# OccupAI QA Task List

Audit scope: FastAPI backend, YOLO/camera integration, admin/driver web dashboard, and the Flutter client at `C:\flutter\occupai_thesis_app_flutter`.

## Completed in this pass

- [x] Protect admin revenue data with backend authorization; the web admin request now sends its bearer token.
- [x] Prevent public registration from creating or taking over the reserved admin account.
- [x] Remove production fallbacks for `CAM_TOKEN`, `ADMIN_PASSWORD`, and `AUTH_SECRET_KEY`.
- [x] Require the camera token for `/api/parking_logs_recent` and keep detector/slot-adjuster callers sending it.
- [x] Bind GCash checkout ownership to the authenticated user instead of trusting a client-supplied `user_id`.
- [x] Add an atomic GCash checkout claim to prevent duplicate payment records from repeated success redirects.
- [x] Align web and Flutter password validation with the backend: at least 8 characters, including a letter and number.
- [x] Make discount availability/rates in Flutter follow the backend admin switches.
- [x] Reject oversized user, camera, snapshot, and YOLO payload fields; validate occupancy counts consistently.
- [x] Clamp driver-history limits and stop returning raw database/ML exception text to clients.
- [x] Revalidate the current database account status/role for every authenticated backend request.
- [x] Restrict non-local CORS defaults and separate Android debug cleartext networking from release builds.
- [x] Fix stale-camera presentation in Flutter: unavailable live values render as `--`, not misleading zeroes.
- [x] Fix Flutter startup role/session race; user data loads before admin-only requests and invalid sessions return to login.
- [x] Add a real Flutter widget test and configure `flutter_test`.
- [x] Add two-camera support: separate car/motorcycle detector workers, backend fan-in with stale-camera expiry, and a web dashboard POV selector. See [TWO_CAMERA_SETUP.md](TWO_CAMERA_SETUP.md).
- [x] Align Flutter admin navigation with the web dashboard: Overview, Revenue, Live Feed, AI Insights, Predictions, Reports, and Settings are available on desktop and in a horizontally scrollable mobile navigation.
- [x] Align Flutter admin Overview content with the web: operational summary, complete KPI set, GCash reconciliation alert, detection activity, revenue forecast, and parking-lot status.
- [x] Align Flutter Revenue workflows with the web: payment recording for vehicle/duration/discount/payment method, GCash issue review, paginated transaction logs, and revenue KPIs/trends.
- [x] Align Flutter Reports with the web's daily/weekly/monthly/yearly source data, KPIs, revenue/discount breakdown, AI predicted-vs-actual comparison, and mobile-safe CSV copying.
- [x] Align Flutter Settings with the backend controls used by the web: manual/dynamic car and motorcycle rates, duration rates, capacity, PWD/Senior switches, and Normal/Busy/High layout override.
- [x] Align Flutter Live Feed with the web two-camera workflow: car/motorcycle POV selector, camera-specific MJPEG endpoint, camera health, and detection log.
- [x] Match NB1 runtime prediction inputs to the 16-feature deployed model artifacts instead of truncating a 23-feature array after scaling.
- [x] Add the bearer token to the web Reports loader when calling the protected revenue dashboard endpoint.

## Priority 0: required before deployment

- [ ] **P0-01 PARTIAL / REQUIRES STAGING:** Set strong, unique values in the deployment environment for `ADMIN_PASSWORD`, `CAM_TOKEN`, and `AUTH_SECRET_KEY`. Restart the backend after setting them. Never reuse the development fallback. Code fail-closed checks are verified locally; the actual production environment still requires inspection.
- [ ] Set `ALLOWED_ORIGINS` to the exact HTTPS dashboard/client origins. The production default is same-origin only.
- [ ] Run a staging end-to-end test with a test PayMongo account: login, GCash checkout, redirect, repeated refresh, cancellation, and a payment where the browser is closed before redirect.
- [ ] Verify the camera and backend use the same `CAM_TOKEN` in the deployment environment. Confirm an invalid token receives 401 and a valid detector update still reaches the dashboard.
- [ ] Configure and calibrate both diorama POV workers using [TWO_CAMERA_SETUP.md](TWO_CAMERA_SETUP.md); verify combined capacity, separate vehicle counts, stream switching, and stale-camera expiry on staging hardware.
- [ ] Apply/verify database migrations on a staging copy, including the checkout `processing` status and payment/user indexes.

## P0-01 QA result — Production secrets

### Status

**PARTIAL / REQUIRES STAGING**

### What was inspected

- `backend/main.py`, `backend/db.py`, `backend/slot_adjuster.py`, and `yolo_service/detector_v7.py` environment loading and secret initialization.
- Root `.env` handling and `.gitignore`.
- Tracked source scan for secret literals and known development fallback use.
- Local environment metadata only; secret values were not printed.

### Tests performed and evidence

- Local `.env` metadata: `CAM_TOKEN` is configured with the known 16-character development fallback; `AUTH_SECRET_KEY` is 64 characters; `ADMIN_PASSWORD` is not configured; `.env` is ignored and not tracked.
- Synthetic precedence test before the fix: deployment values for `DEPLOY_MODE`, `CAM_TOKEN`, and `AUTH_SECRET_KEY` were all overridden by `.env` (`False` for each process-value assertion).
- Synthetic precedence test after the fix: all three deployment-value assertions returned `True`.
- Production startup negative test with missing secrets: `missing_production_secret_fails_closed=True`.
- Production startup negative test with the known camera fallback: `known_camera_fallback_rejected=True`.
- `python -m py_compile backend/main.py backend/db.py backend/slot_adjuster.py yolo_service/detector_v7.py` — passed.
- `python -m pytest -q` — **25 passed**.

### Bugs found and fixed

- **P0-01-B01 — High:** `load_dotenv(override=True)` allowed a local `.env` to replace deployment-provided secrets and `DEPLOY_MODE`. Fixed by using `override=False` in `backend/main.py`, `backend/db.py`, and `backend/slot_adjuster.py`.
- **P0-01-B02 — High:** Missing `DEPLOY_MODE` defaulted to `local`, which could activate development behavior in an undeclared deployment. Fixed by defaulting backend and detector mode to `production`; local mode now requires explicit `DEPLOY_MODE=local`.
- **P0-01-B03 — High:** Production accepted the known development camera token and weak secret lengths when explicitly supplied. Fixed with production validation for a unique 32+ character `CAM_TOKEN`, a 32+ character `AUTH_SECRET_KEY`, a 12+ character letter/number `ADMIN_PASSWORD`, and distinct secret values.

### Final task verdict

**PARTIAL / REQUIRES STAGING.** The application now fails closed and honors deployment environment precedence, but production secret values, uniqueness, and the required post-configuration restart cannot be verified from this local workspace.

## Production log follow-up

- The deployment is running successfully: dashboard, history, stats, predictions, insights, and settings requests returned `200 OK`.
- **LOG-B01 — Medium:** Vehicle forecasting logged `NB1 feature mismatch: have 23, scaler wants 16`. The three deployed vehicle models have input shape `(None, 24, 16)` and the fallback scaler also expects 16. Fixed by selecting the model's expected named features before scaling. A real local prediction returned all three model outputs without the warning.
- **LOG-B02 — Medium:** The Reports panel called protected `/api/revenue/dashboard` without the bearer token, producing intermittent `401 Unauthorized` responses. Fixed by passing `authHeaders()` in `fetchReports()`.
- The observed `401` responses are therefore an application request sequencing/auth-header issue, not a deployment crash. Re-test the Reports panel after the next deployment and confirm the endpoint remains `200 OK` after refresh and tab switching.

## Priority 1: security and reliability hardening

- [ ] Replace browser/local Flutter token storage with secure, HttpOnly, Secure, SameSite cookies (or a platform secure storage strategy for mobile). Current web `localStorage` tokens are exposed if an XSS bug is introduced.
- [ ] Add a PayMongo webhook with signature verification and an idempotency key. Redirect polling alone can miss a paid checkout when the user closes the browser; a stuck `processing` checkout also needs timeout/reconciliation handling.
- [ ] Protect `/api/stream` and review all currently public operational/ML endpoints. The MJPEG camera stream is currently reachable without application authentication.
- [ ] Move token revocation/session state out of the process memory when deploying multiple backend instances; use a shared store or short-lived access tokens with refresh-token rotation.
- [ ] Add a strict Content Security Policy, Subresource Integrity or self-hosted assets for dashboard CDN scripts, and a browser XSS test pass for all dynamic `innerHTML` rendering.
- [ ] Add database rate-limit/session audit logging shared across instances instead of process-local login and request buckets.

## Priority 2: product/data correctness

- [x] Fix the predictions data-source policy: recent live parking logs now drive hourly and revenue forecasts when available; the bundled CSV is an explicitly labeled fallback for a new system with no live logs.
- [ ] Return a server-side total count for driver history; the current `total_sessions` is the count of the limited page, not necessarily the user's total history.
- [ ] Decide whether the public pricing/occupancy/forecast endpoints are intentionally public. If not, add role-based authorization and update web/mobile callers.
- [ ] Replace the compile-time public Render URL with environment-specific `API_BASE_URL` values for development, staging, and production. Use the debug Android manifest only for local HTTP.
- [ ] Add integration tests using a disposable/staging database and mocked PayMongo responses. Do not use real charges in automated tests.

## Priority 3: UI/QA coverage

- [ ] Run the web dashboards in Chromium at 320px, 375px, 768px, and desktop widths with browser zoom/text scaling; verify no clipping or overflow in tables, charts, sidebars, and payment panels.
- [ ] Run the Flutter app on a small Android phone with large accessibility text and test onboarding, login, registration, offline camera state, history, profile, admin navigation, and GCash launch.
- [ ] On staging hardware, perform a cross-platform parity walkthrough at the same time: driver availability/rates/GCash and admin overview/revenue/live-feed/reports/settings, including both car and motorcycle camera POVs.
- [ ] Decide whether mobile needs a native PDF/share export for Reports and a light/gray theme preference equivalent to the web-only controls; mobile currently provides the report data through Copy CSV and uses its responsive native theme.
- [ ] Add widget tests for the offline `--` state, password validation, admin navigation, expired-session redirect, and rate/discount switch behavior.
- [ ] Resolve the remaining non-blocking Flutter `prefer_const` analyzer suggestions as a cleanup task.

## Verification performed

- Backend: `python -m pytest -q` — **25 passed**.
- Backend: `python -m py_compile backend/main.py backend/models.py backend/db.py yolo_service/detector_v7.py` — **passed**.
- Backend: route/model smoke check — protected routes loaded and bounded YOLO validation loaded.
- Flutter: `flutter test --no-pub` — **1 passed**.
- Flutter: `flutter analyze --no-pub` — **no errors/warnings; 33 optional lint/performance infos**.
- Android: `flutter build apk --debug` — **passed**; APK generated at `build/app/outputs/flutter-apk/app-debug.apk`.

Real database connectivity, live camera hardware, browser rendering, and real PayMongo transactions were not exercised in this environment and remain staging/manual test items above.
