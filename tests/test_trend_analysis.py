"""Tests for Sub-Module 2.3 - plasma trend analysis and monitoring engine.

Monotonic-run and oscillation detection are tested against hand-crafted, exact
(noise-free) series so the algorithmic boundaries are checked precisely, rather
than relying on statistical tendencies of random data.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from digital_twin.chamber_config import ChamberParameters
from digital_twin.physics_engine import simulate
from digital_twin.session_manager import ExperimentDatabase
from ai_module.trend_analysis import (
    TRACKED_METRICS,
    TrendEventType,
    analyze_session_trends,
    detect_trend_events,
    ema_smooth,
    session_records_to_trend_frame,
    trend_chart,
    trend_summary_statistics,
)


# ---------------------------------------------------------------------------
# EMA smoothing [FE-2.3.2]
# ---------------------------------------------------------------------------
def test_ema_smooth_rejects_window_below_one() -> None:
    with pytest.raises(ValueError):
        ema_smooth(pd.Series([1.0, 2.0, 3.0]), window=0)


def test_ema_smooth_constant_series_stays_constant() -> None:
    series = pd.Series([5.0] * 10)
    smoothed = ema_smooth(series, window=4)
    assert (smoothed == 5.0).all()


def test_ema_smooth_reduces_variance_of_noisy_series() -> None:
    rng = np.random.default_rng(0)
    noisy = pd.Series(np.full(200, 1.0) + rng.normal(0, 0.1, 200))
    smoothed = ema_smooth(noisy, window=10)
    assert smoothed.std() < noisy.std()


def test_ema_smooth_larger_window_smooths_more() -> None:
    rng = np.random.default_rng(1)
    noisy = pd.Series(np.full(200, 1.0) + rng.normal(0, 0.1, 200))
    light = ema_smooth(noisy, window=3)
    heavy = ema_smooth(noisy, window=30)
    assert heavy.std() < light.std()


# ---------------------------------------------------------------------------
# Monotonic degradation / recovery detection [FE-2.3.3]
# ---------------------------------------------------------------------------
def test_detects_clean_degradation_and_recovery() -> None:
    series = pd.Series([10.0, 9.0, 8.0, 7.0, 9.0, 10.0, 11.0])
    events = detect_trend_events(series, "quality", min_run_length=3)
    types = [(e.event_type, e.start_index, e.end_index) for e in events]
    assert (TrendEventType.DEGRADATION, 0, 3) in types
    assert (TrendEventType.RECOVERY, 3, 6) in types


def test_short_monotonic_run_not_flagged() -> None:
    """Only 2 consecutive decreasing steps; default min_run_length=3 should not flag it."""
    series = pd.Series([10.0, 9.0, 8.0, 8.5, 9.0])
    events = detect_trend_events(series, "quality", min_run_length=3)
    degradations = [e for e in events if e.event_type == TrendEventType.DEGRADATION]
    assert degradations == []


def test_run_length_exactly_at_threshold_is_flagged() -> None:
    series = pd.Series([10.0, 9.0, 8.0, 7.0])  # exactly 3 decreasing steps
    events = detect_trend_events(series, "quality", min_run_length=3)
    assert any(e.event_type == TrendEventType.DEGRADATION for e in events)


def test_flat_series_produces_no_monotonic_events() -> None:
    series = pd.Series([5.0] * 10)
    events = detect_trend_events(series, "quality")
    assert events == []


def test_event_label_names_the_metric() -> None:
    series = pd.Series([10.0, 9.0, 8.0, 7.0])
    events = detect_trend_events(series, "reactivity_index", min_run_length=3)
    assert any("reactivity_index" in e.label for e in events)


# ---------------------------------------------------------------------------
# Oscillation detection [FE-2.3.3]
# ---------------------------------------------------------------------------
def test_detects_alternating_oscillation() -> None:
    series = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    events = detect_trend_events(series, "quality")
    assert any(e.event_type == TrendEventType.OSCILLATION for e in events)


def test_monotonic_series_has_no_oscillation_events() -> None:
    series = pd.Series(np.linspace(0.0, 1.0, 20))
    events = detect_trend_events(series, "quality")
    assert not any(e.event_type == TrendEventType.OSCILLATION for e in events)


def test_events_sorted_by_start_index() -> None:
    series = pd.Series([10.0, 9.0, 8.0, 7.0, 9.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    events = detect_trend_events(series, "quality", min_run_length=3)
    starts = [e.start_index for e in events]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Trend summary statistics [FE-2.3.4]
# ---------------------------------------------------------------------------
def test_trend_summary_statistics_basic_values() -> None:
    frame = pd.DataFrame({"quality": [1.0, 2.0, 3.0, 4.0, 5.0]})
    summary = trend_summary_statistics(frame, "quality")
    assert summary.mean == pytest.approx(3.0)
    assert summary.min == 1.0
    assert summary.max == 5.0


def test_trend_summary_slope_for_perfectly_linear_series() -> None:
    frame = pd.DataFrame({"quality": [1.0, 2.0, 3.0, 4.0, 5.0]})
    summary = trend_summary_statistics(frame, "quality")
    assert summary.trend_direction_coefficient == pytest.approx(1.0)
    assert summary.trend_r_squared == pytest.approx(1.0)


def test_trend_summary_negative_slope_for_declining_series() -> None:
    frame = pd.DataFrame({"quality": [5.0, 4.0, 3.0, 2.0, 1.0]})
    summary = trend_summary_statistics(frame, "quality")
    assert summary.trend_direction_coefficient < 0


def test_trend_summary_requires_at_least_two_points() -> None:
    frame = pd.DataFrame({"quality": [1.0]})
    with pytest.raises(ValueError):
        trend_summary_statistics(frame, "quality")


def test_trend_summary_flat_series_near_zero_slope_and_low_r_squared() -> None:
    frame = pd.DataFrame({"quality": [3.0, 3.0, 3.0, 3.0, 3.0]})
    summary = trend_summary_statistics(frame, "quality")
    assert summary.trend_direction_coefficient == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Session record conversion [FE-2.3.1]
# ---------------------------------------------------------------------------
def test_session_records_to_trend_frame_preserves_order(tmp_path) -> None:
    db = ExperimentDatabase(tmp_path / "sessions.db")
    session_ids = []
    for power in (80.0, 120.0, 160.0, 200.0):
        params = ChamberParameters(rf_power_w=power, pressure_mtorr=10.0)
        result = simulate(params.rf_power_w, params.pressure_mtorr)
        session_ids.append(db.create_session(params, result))

    records = [db.get_session(sid) for sid in session_ids]
    frame = session_records_to_trend_frame(records)

    assert list(frame["run_index"]) == [0, 1, 2, 3]
    for metric in TRACKED_METRICS:
        assert metric in frame.columns
    db.close()


# ---------------------------------------------------------------------------
# Charts and full orchestration [FE-2.3.1, full pipeline]
# ---------------------------------------------------------------------------
@pytest.fixture
def demo_frame() -> pd.DataFrame:
    n = 30
    base = np.concatenate([
        np.full(8, 0.6), np.linspace(0.6, 0.3, 8), np.linspace(0.3, 0.65, 8),
        0.5 + 0.1 * np.array([(-1) ** i for i in range(6)]),
    ])
    rng = np.random.default_rng(0)
    quality = base + rng.normal(0, 0.01, size=n)
    return pd.DataFrame({
        "run_index": np.arange(n),
        "process_quality": quality,
        "reactivity_index": quality,
        "uniformity_index": quality,
        "electron_temperature_ev": 3.0 + quality,
    })


def test_trend_chart_returns_figure(demo_frame) -> None:
    fig = trend_chart(demo_frame, "process_quality", smoothing_window=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # raw + smoothed traces


def test_trend_chart_adds_event_shapes(demo_frame) -> None:
    from ai_module.trend_analysis import TrendEvent

    event = TrendEvent("process_quality", TrendEventType.DEGRADATION, 8, 15, "test event")
    fig = trend_chart(demo_frame, "process_quality", smoothing_window=5, events=[event])
    assert len(fig.layout.shapes) >= 1


def test_analyze_session_trends_covers_all_tracked_metrics(demo_frame) -> None:
    reports = analyze_session_trends(demo_frame)
    assert set(reports.keys()) == set(TRACKED_METRICS)
    for metric, report in reports.items():
        assert report.metric == metric
        assert isinstance(report.figure, go.Figure)
        assert report.summary.mean == pytest.approx(demo_frame[metric].mean())


def test_analyze_session_trends_detects_the_built_in_degradation(demo_frame) -> None:
    """The synthetic fixture has a genuine decline in the middle (runs 8-15) -
    the pipeline should surface at least one degradation event for it."""
    reports = analyze_session_trends(demo_frame, metrics=["process_quality"])
    events = reports["process_quality"].events
    assert any(e.event_type == TrendEventType.DEGRADATION for e in events)
