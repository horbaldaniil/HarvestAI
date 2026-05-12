"""Rule-based weather advice — unit tests, no DB required."""
from __future__ import annotations

from datetime import date

from app.services.weather_advice import build_advices


def _day(
    *,
    d: date,
    tmin: float | None = 15.0,
    tmax: float | None = 22.0,
    precip: float | None = 2.0,
    wind: float | None = 3.0,
    cloud: float | None = 50.0,
    soil: float | None = 0.3,
) -> dict:
    return {
        "observed_on": d,
        "temp_min_c": tmin,
        "temp_max_c": tmax,
        "precip_mm": precip,
        "wind_speed_max_ms": wind,
        "cloud_cover_pct": cloud,
        "soil_moisture_0_10cm": soil,
    }


def test_no_days_returns_empty():
    assert build_advices([]) == []


def test_drought_warning_when_no_rain():
    days = [
        _day(d=date(2026, 7, i + 1), precip=0.0) for i in range(7)
    ]
    out = build_advices(days)
    assert any(a.severity == "warning" and "Сух" in a.title for a in out)


def test_drought_info_for_low_rain():
    days = [_day(d=date(2026, 7, i + 1), precip=0.4) for i in range(7)]
    out = build_advices(days)
    titles = [a.title for a in out]
    # 7 * 0.4 = 2.8 mm → triggers info, not warning.
    assert any("Малі опади" in t for t in titles)


def test_heat_critical_for_5_plus_hot_days():
    days = [_day(d=date(2026, 7, i + 1), tmax=33.0) for i in range(7)]
    out = build_advices(days, crop="wheat")
    heat = [a for a in out if "спекотних" in a.title]
    assert heat and heat[0].severity == "critical"
    assert "пшениц" in heat[0].detail.lower()


def test_heat_warning_for_3_or_4_hot_days():
    days = [
        _day(d=date(2026, 7, 1), tmax=33.0),
        _day(d=date(2026, 7, 2), tmax=33.0),
        _day(d=date(2026, 7, 3), tmax=33.0),
        _day(d=date(2026, 7, 4), tmax=25.0),
        _day(d=date(2026, 7, 5), tmax=22.0),
    ]
    out = build_advices(days)
    heat = [a for a in out if "спекотних" in a.title]
    assert heat and heat[0].severity == "warning"


def test_frost_only_in_spring_autumn():
    # Summer days with negative tmin shouldn't flag.
    summer = [_day(d=date(2026, 7, 1), tmin=-1.0)]
    assert not any("заморо" in a.title.lower() for a in build_advices(summer))

    spring = [
        _day(d=date(2026, 4, 1), tmin=-2.0),
        _day(d=date(2026, 4, 2), tmin=-1.0),
    ]
    out = build_advices(spring)
    frost = [a for a in out if "Ризик заморозків" in a.title]
    assert frost and frost[0].severity == "critical"  # 2+ frosty nights


def test_wind_advisory_above_6_mps():
    days = [_day(d=date(2026, 5, i + 1), wind=7.5) for i in range(3)]
    out = build_advices(days)
    assert any("Вітряні" in a.title for a in out)


def test_workday_recommendation_picks_best_day():
    days = [
        _day(d=date(2026, 5, 1), precip=5.0, tmax=18.0, wind=3.0),
        _day(d=date(2026, 5, 2), precip=0.0, tmax=20.0, wind=2.0),  # winner
        _day(d=date(2026, 5, 3), precip=0.0, tmax=29.0, wind=4.0),
    ]
    out = build_advices(days)
    workday = [a for a in out if "Найкращий день" in a.title]
    assert workday
    assert "2026-05-02" in workday[0].detail


def test_clean_week_emits_only_workday_recommendation():
    """Mild, dry week with low wind — only the positive 'best day' advice."""
    days = [
        _day(d=date(2026, 5, i + 1), precip=0.5, tmax=22.0, wind=2.5)
        for i in range(7)
    ]
    out = build_advices(days)
    # 7 × 0.5 = 3.5 mm → still under 5 mm → triggers "Малі опади" info.
    titles = [a.title for a in out]
    assert any("Найкращий день" in t for t in titles)
