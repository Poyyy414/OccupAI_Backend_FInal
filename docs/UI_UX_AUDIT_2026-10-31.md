# OccupAI UI/UX Audit and Design-System Update

**Date:** October 31, 2026  
**Primary implementation scope:** Admin web dashboard prediction and chart surfaces  
**Preserved:** APIs, calculations, authentication, payments, reservations, role permissions, and CV behavior

## Audit

| Page/Section | Current problem | User impact | Recommended improvement | Priority |
|---|---|---|---|---|
| Predictions / hourly forecast | A bar strip plus 24 large cards repeated the same metric | Slow scanning, excessive scrolling, weak hour-to-hour comparison | Use one line/area chart, highlight the peak, and provide an exact-values disclosure | Critical |
| Predictions / light theme | Pale mint and blue values had insufficient contrast on white | Values and trends became difficult to read | Use semantic light/dark chart tokens with a darker light-theme palette | Critical |
| Weekly heatmap | Opacity-based mint scale depended on the largest observed value and had no legend | Similar values were ambiguous; low values disappeared in light mode | Use fixed percentage bands, visible legend, and a separate no-data pattern | High |
| Weekly heatmap accessibility | Cells were non-focusable color blocks with mouse-only titles | Keyboard and assistive-technology users could not inspect exact values | Render cells as focusable buttons with exact accessible labels | High |
| Dashboard panels | Glass blur, large shadows, hover lifting, and gradients were repeated broadly | Operational information looked decorative and visually busy | Use neutral surfaces, restrained shadows, and border-only hover feedback | High |
| Chart axes and legends | Several charts used hard-coded gray and translucent colors | Theme switching produced inconsistent contrast | Read axes, grids, and series colors from semantic CSS tokens | High |
| Summary typography | Multiple supporting labels are 10–11px and heavily uppercase | Readability suffers on laptops, mobile, and browser zoom | Use 12px minimum for secondary labels and 14px for explanatory text | High |
| Empty data states | Some forecast paths converted missing values to zero | Users could interpret unavailable data as a real 0% prediction | Present “No data” or an em dash unless the source contains an actual value | High |
| Revenue reports | Dense tables and charts compete at the same hierarchy | Important totals are slower to find | Keep KPI summary first, chart second, horizontally scrollable detail table third | Medium |
| Settings | Long form sections use similar visual weight | Configuration groups are difficult to scan | Group by operational domain and use clear save feedback per group | Medium |
| Driver payment | Reservation and actual-parking payments previously looked related | Drivers could misunderstand whether payment held a slot | Retain the newly separated payment-purpose controls and direct wording | High |
| Flutter dashboard | Compact layouts pass automated tests but still contain dense payment content | Long mobile scrolling and cognitive load | Conduct a device review and consider a step-based payment flow | Medium |

## Design-system proposal

### Color tokens

All components should consume semantic tokens rather than fixed mint values.

| Purpose | Dark theme | Light theme |
|---|---:|---:|
| Page background | `#06100c` | `#d8dde4` |
| Panel surface | `#101b17` | `#eef1f5` |
| Primary text | `#f4f7fb` | `#1b2431` |
| Secondary text | `#97a5b8` | `#5f6d7f` |
| Interactive/available | `#32e6ad` | `#087a55` |
| Chart line | `#61adff` | `#1666b1` |
| Reserved/warning | `#f8d66d` | `#7a5b00` |
| Occupied/error | `#ff5c7a` | `#b4233d` |

The heatmap uses five fixed semantic levels. Missing data uses a striped neutral pattern and is not shown as 0%.

### Typography

- Page title: 24–30px, 700 weight
- Section title: 15–18px, 700 weight
- Body and explanations: 14–16px, 400–600 weight
- Secondary labels: minimum 12px
- Numeric KPI: 20–32px using the existing monospace family
- Uppercase and letter spacing: reserved for short eyebrow labels only

### Spacing and containers

- Base spacing unit: 4px
- Common gaps: 8, 12, 16, 24px
- Panel radius: 8px
- Compact control radius: 6–8px
- Panel shadow: one restrained elevation only
- No hover translation on operational panels
- Use one container per related information group, not one card per hourly number

### Controls and states

- Primary button: solid accent, explicit action label
- Secondary button: neutral surface and visible border
- Destructive button: semantic error color plus text label
- Minimum touch target: 44×44px
- Loading: skeleton or progress label that preserves layout
- Empty: explain why no data exists and what creates it
- Offline: show “Offline” or “Unknown,” never a fabricated zero
- Error: concise reason plus retry action where safe

## Implemented changes

### `template/dashboard.html`

- Replaced 24 hourly cards and the redundant bar strip with one responsive Chart.js forecast.
- Added a fixed 0–100% occupancy axis and a highlighted peak point.
- Added a visible peak summary and a collapsed exact-values grid.
- Added semantic chart, status, and heatmap variables for both themes.
- Darkened light-theme green, blue, orange, and red values for contrast.
- Converted heatmap cells into keyboard-focusable controls with exact labels.
- Added an occupancy legend and a striped missing-data state.
- Reduced panel glass effects, shadow weight, gradients, and hover movement.
- Updated major chart axes and legends to use theme-aware colors.

### `tests/test_dashboard_unknown_ui.py`

- Added regression checks proving the 24-card UI is removed.
- Added checks for the accessible forecast chart, exact-values fallback, semantic heatmap levels, legend, and cell labels.

## Before and after

- 24 hourly forecast cards plus one bar strip → one forecast chart plus an optional exact-values disclosure.
- Relative opacity heatmap with no scale → five fixed occupancy bands plus 0–100% legend.
- Missing heatmap values rendered like 0% → separate striped “No data” presentation.
- Mouse-only heatmap cells → keyboard-focusable, labeled cells.
- Pale light-theme green (`#32e6ad`) → darker interactive green (`#087a55`).
- Hard-coded chart axis colors → semantic theme-aware axes and grids.
- Blurred glass panels with hover lift → neutral panels with restrained elevation.

## Remaining recommendations

1. Perform a manual browser contrast audit with rendered Chart.js canvases; automated source checks cannot measure pixels inside canvas output.
2. Review every remaining 10–11px label and increase non-tertiary text to 12px or more.
3. Add Playwright or equivalent browser tests for 320, 390, 768, 1024, and 1440px widths.
4. Add automated keyboard tests for sidebar focus trapping, chart disclosure, heatmap cells, and dialogs.
5. Review the Flutter payment page on physical Android devices at 200% text scaling.
6. Replace remaining browser `alert()` error messages with the shared in-page status pattern.
7. Retrain or re-export the prediction model artifacts using the installed scikit-learn version before relying on forecast accuracy.

## Quality recommendation

The prediction page is suitable for supervised demonstration after manual light/dark browser review. A production accessibility sign-off still requires rendered contrast measurement, real keyboard testing, and responsive browser automation.

## Verification performed

- Backend and web regression suite: **101 passed**
- Flutter responsive/widget suite: **34 passed**
- Flutter static analysis: **No issues found**
- Dashboard inline JavaScript: **3 scripts compiled successfully**
- Source diff whitespace check: **Passed**
- Light panel contrast ratios:
  - Primary text: **13.80:1**
  - Secondary text: **4.66:1**
  - Green accent: **4.72:1**
  - Blue chart line: **5.19:1**
  - Orange: **5.30:1**
  - Red: **5.71:1**
  - Reserved yellow: **5.58:1**

All listed light-theme text/accent colors meet the WCAG AA 4.5:1 normal-text threshold against the light panel surface. Canvas rendering and focus behavior still require the manual browser review noted above.
