import backend.main as m
import pandas as pd


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


def test_api_predictions_uses_training_pattern_when_live_history_is_rejected(monkeypatch):
    monkeypatch.setattr(m, "_active_slot_capacity", lambda default=None: 30)
    monkeypatch.setattr(m, "_logged_hourly_vehicle_avg", lambda: {})
    monkeypatch.setattr(
        m,
        "_logged_weekday_revenue_forecast",
        lambda capacity: ({day: 0.0 for day in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]}, None),
    )

    result = m.api_predictions()

    assert result["hourly_source"] == "training_data_fallback"
    assert result["revenue_source"] == "training_data_fallback"
    assert result["hourly_est"]["8"] == 74.7
    assert result["hourly_est"]["7"] == 49.2


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


def test_sparse_live_history_is_not_used_for_forecasts():
    rows = [
        {"hour": 8, "avg_vehicles": 1.0, "sample_count": 40},
        {"hour": 9, "avg_vehicles": 1.0, "sample_count": 40},
    ]

    assert not m._live_forecast_history_is_sufficient(rows)


def test_hourly_by_day_fallback_uses_training_csv(monkeypatch):
    monkeypatch.setattr(
        m,
        "_load_training_df",
        lambda: pd.DataFrame(
            {
                "day_of_week": [3, 3, 5],
                "hour": [8, 8, 8],
                "vehicles_hour": [10.0, 14.0, 4.0],
            }
        ),
    )

    result = m._training_hourly_by_day(30)

    # CSV day 3 is Thursday; the endpoint uses Sunday=0, so Thursday=4.
    assert result[4][8]["vehicles"] == 12.0
    assert result[4][8]["occ_pct"] == 40.0
    assert result[6][8]["vehicles"] == 4.0


def test_revenue_dashboard_cache_reuses_expensive_aggregate(monkeypatch):
    calls = 0

    def fake_dashboard():
        nonlocal calls
        calls += 1
        return {"today_revenue_php": 50.0}

    monkeypatch.setattr(m, "_payment_revenue_dashboard", fake_dashboard)
    m._invalidate_payment_revenue_dashboard_cache()

    first = m._cached_payment_revenue_dashboard()
    second = m._cached_payment_revenue_dashboard()

    assert first == second == {"today_revenue_php": 50.0}
    assert calls == 1
    m._invalidate_payment_revenue_dashboard_cache()
