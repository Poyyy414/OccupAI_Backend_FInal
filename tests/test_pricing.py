"""
Tests for the dynamic pricing formula in backend/main.py.

These pin down the occupancy-based multiplier tiers, the day-of-week
multiplier, the manual admin override, and the PWD/senior discount
calculation — the parts of the system most likely to get scrutinized,
and the easiest to silently break with an off-by-one during a refactor.
"""
from datetime import datetime

import backend.main as m

MONDAY = datetime(2024, 1, 1)     # weekday() == 0
TUESDAY = datetime(2024, 1, 2)    # a plain weekday, no multiplier
SATURDAY = datetime(2024, 1, 6)   # weekend
SUNDAY = datetime(2024, 1, 7)     # weekend

NO_OVERRIDE = {"PRICE_OVERRIDE_ENABLED": "false"}


def test_low_occupancy_uses_lowest_multiplier(env_settings):
    env_settings(NO_OVERRIDE)
    result = m._dynamic_price_formula(vehicles_hour=0, lot_capacity=44, when=TUESDAY)
    assert result["pricing_context"]["occupancy_multiplier"] == 0.80
    assert result["recommended_price_php"] == round(m.FLAT_RATE_CAR * 0.80, 2)
    assert result["pricing_reason"] != "manual_admin_override"


def test_near_full_occupancy_uses_highest_multiplier(env_settings):
    env_settings(NO_OVERRIDE)
    # 90%+ occupied should hit the top 1.80x tier.
    result = m._dynamic_price_formula(vehicles_hour=40, lot_capacity=44, when=TUESDAY)
    assert result["pricing_context"]["occupancy_pct"] >= 90
    assert result["pricing_context"]["occupancy_multiplier"] == 1.80


def test_occupancy_multiplier_tiers_are_monotonic():
    # Each higher occupancy band must charge at least as much as the one below it.
    tiers = [0, 15, 35, 55, 75, 95]
    multipliers = [m._occupancy_price_multiplier(pct) for pct in tiers]
    assert multipliers == sorted(multipliers)


def test_monday_multiplier_applied(env_settings):
    env_settings(NO_OVERRIDE)
    result = m._dynamic_price_formula(vehicles_hour=0, lot_capacity=44, when=MONDAY)
    assert result["pricing_context"]["day_multiplier"] == 1.10
    assert result["pricing_context"]["day_rule"] == "Monday"


def test_weekend_multiplier_applied(env_settings):
    env_settings(NO_OVERRIDE)
    for weekend_day in (SATURDAY, SUNDAY):
        result = m._dynamic_price_formula(vehicles_hour=0, lot_capacity=44, when=weekend_day)
        assert result["pricing_context"]["day_multiplier"] == 0.90
        assert result["pricing_context"]["day_rule"] == "Weekend"


def test_manual_override_takes_precedence_over_occupancy(env_settings):
    env_settings({
        "PRICE_OVERRIDE_ENABLED": "true",
        "PRICE_OVERRIDE_PHP_CAR": "77.00",
    })
    # Even at near-full occupancy, the manual admin price should win outright.
    result = m._dynamic_price_formula(vehicles_hour=40, lot_capacity=44, when=TUESDAY)
    assert result["pricing_reason"] == "manual_admin_override"
    assert result["recommended_price_php"] == 77.00


def test_pwd_senior_discount_matches_configured_rate(env_settings):
    env_settings(NO_OVERRIDE)
    result = m._dynamic_price_formula(vehicles_hour=0, lot_capacity=44, when=TUESDAY)
    price = result["recommended_price_php"]
    rate = m.PWD_SENIOR_DISCOUNT_RATE
    expected = round(price - round(price * rate, 2), 2)
    assert result["pwd_senior_price_php"] == expected
    assert result["pwd_senior_discount_rate"] == rate


def test_motorcycle_and_car_use_independent_flat_rates(env_settings):
    env_settings(NO_OVERRIDE)
    car = m._dynamic_price_formula(vehicles_hour=0, lot_capacity=44, when=TUESDAY, vehicle_type="car")
    moto = m._dynamic_price_formula(vehicles_hour=0, lot_capacity=44, when=TUESDAY, vehicle_type="motorcycle")
    assert car["flat_rate_php"] == m.FLAT_RATE_CAR
    assert moto["flat_rate_php"] == m.FLAT_RATE_MOTORCYCLE


def test_dynamic_duration_prices_use_normal_rates_at_standard_demand(env_settings):
    env_settings(NO_OVERRIDE)
    rates = m._effective_duration_pricing(
        vehicles_hour=10,
        lot_capacity=44,
        when=TUESDAY,
    )

    # 10/44 is in the standard 1.00x demand band. Daily prices therefore stay
    # at the approved normal rates: PHP 50 for cars and PHP 25 for motorcycles.
    assert rates["pricing_mode"] == "dynamic"
    assert rates["demand_pricing_enabled"] is True
    assert rates["daily_rate_php_car"] == 50.00
    assert rates["daily_rate_php_motorcycle"] == 25.00


def test_dynamic_duration_prices_increase_at_high_demand(env_settings):
    env_settings(NO_OVERRIDE)
    rates = m._effective_duration_pricing(
        vehicles_hour=40,
        lot_capacity=44,
        when=TUESDAY,
    )

    # 40/44 is above 90%, so the configured 1.80x demand multiplier applies.
    assert rates["pricing_context"]["occupancy_pct"] >= 90
    assert rates["daily_rate_php_car"] == 90.00
    assert rates["daily_rate_php_motorcycle"] == 45.00


def test_manual_mode_keeps_duration_prices_unchanged(env_settings):
    env_settings({"PRICE_OVERRIDE_ENABLED": "true"})
    rates = m._effective_duration_pricing(
        vehicles_hour=40,
        lot_capacity=44,
        when=TUESDAY,
    )

    assert rates["pricing_mode"] == "manual"
    assert rates["demand_pricing_enabled"] is False
    assert rates["daily_rate_php_car"] == 50.00
    assert rates["daily_rate_php_motorcycle"] == 25.00
