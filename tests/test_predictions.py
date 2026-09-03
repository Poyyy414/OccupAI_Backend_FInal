import backend.main as m


def test_hourly_occupancy_percent_uses_active_capacity():
    result = m._hourly_occupancy_percent({8: 15}, 30)

    assert result["8"] == 50.0
    assert result["7"] == 0.0
    assert len(result) == 24


def test_api_predictions_prefers_live_data_over_training_data(monkeypatch):
    monkeypatch.setattr(m, "_active_slot_capacity", lambda default=None: 30)
    monkeypatch.setattr(m, "_logged_hourly_vehicle_avg", lambda: {8: 15.0})
    monkeypatch.setattr(
        m,
        "_weekday_revenue_forecast",
        lambda capacity: ({"Thu": 120.0}, 120.0, "deduplicated_live_logs"),
    )

    result = m.api_predictions()

    assert result["hourly_source"] == "last_7_days"
    assert result["revenue_source"] == "deduplicated_live_logs"
    assert result["hourly_est"]["8"] == 50.0
    assert result["peak_hour"] == 8
    assert result["today_revenue_forecast"] == 120.0


def test_weekday_revenue_falls_back_only_when_live_data_is_empty(monkeypatch):
    training = {"Sun": 10.0, "Mon": 20.0}
    monkeypatch.setattr(
        m,
        "_logged_weekday_revenue_forecast",
        lambda capacity: ({"Sun": 0.0, "Mon": 0.0}, None),
    )
    monkeypatch.setattr(
        m,
        "_training_weekday_revenue_forecast",
        lambda capacity: (training, 10.0),
    )

    values, today, source = m._weekday_revenue_forecast(30)

    assert values == training
    assert today == 10.0
    assert source == "training_data_fallback"
