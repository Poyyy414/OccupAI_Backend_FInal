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
    assert "Both camera zones are Unknown. Official capacity remains 30." in html
    assert "Partial data is not shown as complete lot occupancy." in html


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
