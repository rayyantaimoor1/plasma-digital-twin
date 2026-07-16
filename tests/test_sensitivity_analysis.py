"""Tests for Sub-Module 1.5 - parameter sensitivity and sweep analysis."""
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import pytest

from digital_twin.chamber_config import PRESSURE_RANGE, RF_POWER_RANGE, default_parameters
from digital_twin.physics_engine import simulate
from digital_twin.sensitivity_analysis import (
    OATSensitivityResult,
    OUTPUT_METRICS,
    SweepPoint,
    parameter_effect_curve,
    paired_sweep_heatmap,
    run_full_oat_analysis,
    run_oat_sweep,
    run_paired_sweep,
    sensitivity_bar_chart,
    sensitivity_ranking,
)


# ---------------------------------------------------------------------------
# OAT sweep mechanics [FE-1.5.1]
# ---------------------------------------------------------------------------
def test_oat_sweep_produces_requested_number_of_points() -> None:
    baseline = default_parameters()
    result = run_oat_sweep("rf_power_w", baseline, n_points=12)
    assert len(result.sweep_points) == 12


def test_oat_sweep_spans_the_full_parameter_range() -> None:
    baseline = default_parameters()
    result = run_oat_sweep("rf_power_w", baseline, n_points=10)
    varied_values = [p.varied_value for p in result.sweep_points]
    assert min(varied_values) == pytest.approx(RF_POWER_RANGE.min_value)
    assert max(varied_values) == pytest.approx(RF_POWER_RANGE.max_value)


def test_oat_sweep_holds_the_other_parameter_fixed() -> None:
    baseline = default_parameters()
    result = run_oat_sweep("rf_power_w", baseline, n_points=8)
    pressures = [p.result.pressure_mtorr for p in result.sweep_points]
    assert all(p == pytest.approx(baseline.pressure_mtorr) for p in pressures)


def test_oat_sweep_unknown_parameter_raises() -> None:
    with pytest.raises(ValueError):
        run_oat_sweep("gas_flow_sccm", default_parameters())


def test_full_oat_analysis_covers_both_sweepable_parameters() -> None:
    results = run_full_oat_analysis(default_parameters(), n_points=6)
    assert set(results.keys()) == {"rf_power_w", "pressure_mtorr"}


# ---------------------------------------------------------------------------
# Physically meaningful sensitivity results - these double as regression tests
# for both physics fixes made alongside this sub-module.
# ---------------------------------------------------------------------------
def test_electron_temperature_has_zero_sensitivity_to_power() -> None:
    """Regression: Te is rigorously power-independent (particle balance only)."""
    oat = run_full_oat_analysis(default_parameters(), n_points=10)
    score = oat["rf_power_w"].sensitivity_score("electron_temperature_ev")
    assert score == pytest.approx(0.0, abs=1e-9)


def test_electron_temperature_has_high_sensitivity_to_pressure() -> None:
    oat = run_full_oat_analysis(default_parameters(), n_points=10)
    score = oat["pressure_mtorr"].sensitivity_score("electron_temperature_ev")
    assert score > 0.5


def test_density_has_nonzero_sensitivity_to_power() -> None:
    oat = run_full_oat_analysis(default_parameters(), n_points=10)
    score = oat["rf_power_w"].sensitivity_score("plasma_density_m3")
    assert score > 0.5


def test_ion_energy_has_nonzero_sensitivity_to_power() -> None:
    """Regression for the Sub-Module 1.2 sheath fix: before it, ion_energy_ev was
    completely power-independent, so this sensitivity score would have been 0."""
    oat = run_full_oat_analysis(default_parameters(), n_points=10)
    score = oat["rf_power_w"].sensitivity_score("ion_energy_ev")
    assert score > 0.05


@pytest.mark.parametrize("metric", OUTPUT_METRICS)
def test_sensitivity_score_is_non_negative_for_every_metric(metric) -> None:
    oat = run_full_oat_analysis(default_parameters(), n_points=8)
    for result in oat.values():
        assert result.sensitivity_score(metric) >= 0.0


def test_sensitivity_score_zero_division_guard_returns_zero() -> None:
    """No real metric in this model ever averages to ~0 (Phase 4 coverage pass
    found this guard was never exercised) - directly construct a synthetic sweep
    where a metric is identically zero, to verify the mean_abs < 1e-30 guard
    returns 0.0 rather than raising ZeroDivisionError or producing NaN/inf."""
    baseline = default_parameters()
    result = simulate(baseline.rf_power_w, baseline.pressure_mtorr)
    zero_result = replace(result, defect_probability=0.0)
    sweep = OATSensitivityResult(
        parameter_name="rf_power_w",
        baseline=baseline,
        sweep_points=[SweepPoint(varied_value=v, result=zero_result) for v in (50.0, 150.0, 300.0)],
    )
    score = sweep.sensitivity_score("defect_probability")
    assert score == 0.0


# ---------------------------------------------------------------------------
# Ranking [FE-1.5.4]
# ---------------------------------------------------------------------------
def test_sensitivity_ranking_is_sorted_descending() -> None:
    oat = run_full_oat_analysis(default_parameters(), n_points=10)
    ranking = sensitivity_ranking(oat, "process_quality")
    scores = [score for _name, score in ranking]
    assert scores == sorted(scores, reverse=True)


def test_sensitivity_ranking_pressure_dominates_electron_temperature() -> None:
    oat = run_full_oat_analysis(default_parameters(), n_points=10)
    ranking = dict(sensitivity_ranking(oat, "electron_temperature_ev"))
    assert ranking["pressure_mtorr"] > ranking["rf_power_w"]


# ---------------------------------------------------------------------------
# Visualisations [FE-1.5.2]
# ---------------------------------------------------------------------------
def test_sensitivity_bar_chart_returns_figure_with_bar_trace() -> None:
    oat = run_full_oat_analysis(default_parameters(), n_points=8)
    fig = sensitivity_bar_chart(oat, "process_quality")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)


def test_parameter_effect_curve_returns_figure_with_scatter_trace() -> None:
    baseline = default_parameters()
    result = run_oat_sweep("rf_power_w", baseline, n_points=8)
    fig = parameter_effect_curve(result, "plasma_density_m3")
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data[0].x) == 8


# ---------------------------------------------------------------------------
# Paired sweeps / response surfaces [FE-1.5.3]
# ---------------------------------------------------------------------------
def test_paired_sweep_grid_shape() -> None:
    sweep = run_paired_sweep("rf_power_w", "pressure_mtorr", n_x=6, n_y=4)
    assert sweep.x_values.shape == (6,)
    assert sweep.y_values.shape == (4,)
    for metric in OUTPUT_METRICS:
        assert sweep.metric_grid[metric].shape == (4, 6)


def test_paired_sweep_order_independent_axis_assignment() -> None:
    """Swapping x_name/y_name should swap which axis each parameter sits on, but
    produce the same underlying grid of values (just transposed roles)."""
    sweep_a = run_paired_sweep("rf_power_w", "pressure_mtorr", n_x=5, n_y=5)
    sweep_b = run_paired_sweep("pressure_mtorr", "rf_power_w", n_x=5, n_y=5)
    assert np.allclose(sweep_a.x_values, sweep_b.y_values)
    assert np.allclose(sweep_a.y_values, sweep_b.x_values)


def test_paired_sweep_same_parameter_twice_raises() -> None:
    with pytest.raises(ValueError):
        run_paired_sweep("rf_power_w", "rf_power_w")


def test_paired_sweep_invalid_parameter_raises() -> None:
    with pytest.raises(ValueError):
        run_paired_sweep("rf_power_w", "gas_flow_sccm")


def test_paired_sweep_matches_direct_simulation_at_grid_corner() -> None:
    """A grid corner must reproduce exactly what simulate() gives directly."""
    from digital_twin.physics_engine import simulate

    sweep = run_paired_sweep("rf_power_w", "pressure_mtorr", n_x=5, n_y=5)
    corner_power = float(sweep.x_values[0])
    corner_pressure = float(sweep.y_values[0])
    direct = simulate(corner_power, corner_pressure)
    assert sweep.metric_grid["process_quality"][0, 0] == pytest.approx(direct.process_quality)


def test_paired_sweep_heatmap_returns_figure_with_heatmap_trace() -> None:
    sweep = run_paired_sweep("rf_power_w", "pressure_mtorr", n_x=5, n_y=5)
    fig = paired_sweep_heatmap(sweep, "process_quality")
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Heatmap)
