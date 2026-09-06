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
