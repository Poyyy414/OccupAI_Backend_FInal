from pathlib import Path


def test_web_dashboard_keeps_capacity_and_hides_unavailable_occupancy():
    html = (Path(__file__).parents[1] / "template" / "dashboard.html").read_text(
        encoding="utf-8"
    )

    assert "Fixed at 10 car spaces + 20 motorcycle spaces = 30 total" in html
    assert "complete ? occupied : '--'" in html
    assert "complete ? free : '--'" in html
    assert "s.combined_complete === true" in html
    assert "setText('predFullCurrentOcc', '--')" in html
    assert "Neither camera is connected. The parking area still has 30 total spaces." in html
    assert "Parking totals will appear when both are ready." in html


def test_settings_use_one_duration_based_price_editor():
    root = Path(__file__).parents[1]
    dashboard = (root / "template" / "dashboard.html").read_text(encoding="utf-8")
    driver = (root / "template" / "driver.html").read_text(encoding="utf-8")

    assert "<strong>Parking Rates</strong>" in dashboard
    assert "<strong>Parking Price</strong>" not in dashboard
    assert "Demand Label" not in dashboard
    assert "per hour" not in driver
    assert driver.count("per day") == 2


def test_predictions_use_one_accessible_chart_instead_of_hour_cards():
    html = (Path(__file__).parents[1] / "template" / "dashboard.html").read_text(
        encoding="utf-8"
    )

    assert 'id="predFullForecastChart"' in html
    assert 'id="predFullForecastSummary"' in html
    assert 'id="predFullForecastValues"' in html
    assert 'id="predFullGrid"' not in html
    assert 'id="predFullHourBars"' not in html
    assert "renderFullOccupancyForecast" in html


def test_prediction_heatmap_has_theme_safe_levels_and_accessible_cells():
    html = (Path(__file__).parents[1] / "template" / "dashboard.html").read_text(
        encoding="utf-8"
    )

    assert "--heat-0:" in html
    assert "--chart-line:" in html
    assert 'class="heatmap-legend"' in html
    assert 'type="button" class="heatmap-cell ${level}"' in html
    assert 'aria-label="${escapeHtml(description)}"' in html
    assert "rgba(50,230,173,${alpha.toFixed(2)})" not in html
    assert "const dayHeader" in html
    assert 'class="heatmap-day-label"' in html
    assert "Array.from({ length: 24 }, (_, h)" in html


def test_reports_offer_real_office_downloads():
    html = (Path(__file__).parents[1] / "template" / "dashboard.html").read_text(
        encoding="utf-8"
    )

    assert "downloadOfficeReport('xlsx'" in html
    assert "downloadOfficeReport('docx'" in html
    assert "/api/revenue/export?period=" in html


def test_web_parking_status_uses_labeled_car_and_motorcycle_plan_views():
    root = Path(__file__).parents[1]
    dashboard = (root / "template" / "dashboard.html").read_text(encoding="utf-8")
    driver = (root / "template" / "driver.html").read_text(encoding="utf-8")

    assert "function parkingPlanMarkup" in dashboard
    assert "Car parking" in dashboard
    assert "Motorcycle parking" in dashboard
    assert "DRIVE LANE" in dashboard
    assert "lot-vehicle" in dashboard
    assert "space-plan-section" in driver
    assert "space-drive-lane" in driver
    assert "space-vehicle" in driver


def test_desktop_dashboard_uses_compact_labeled_navigation_rail():
    html = (Path(__file__).parents[1] / "template" / "dashboard.html").read_text(
        encoding="utf-8"
    )

    assert "Desktop navigation rail" in html
    assert 'class="logo-icon" aria-hidden="true">P</div>' in html
    assert 'title="Live Feed"' in html
    assert 'title="Settings"' in html
    assert 'id="i-logout"' in html
