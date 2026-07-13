"""Tests for Sub-Module 2.4 - multi-parameter correlation and relationship analysis."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ai_module.correlation_analysis import (
    CORRELATION_COLUMNS,
    KEY_RELATIONSHIPS,
    LOW_CORRELATION_THRESHOLD,
    correlation_heatmap,
    correlation_insights,
    correlation_matrix,
    key_relationship_plots,
    parallel_coordinates_plot,
    scatter_with_regression,
    simulation_sweep_dataframe,
)


@pytest.fixture(scope="module")
def sweep() -> pd.DataFrame:
    return simulation_sweep_dataframe(power_step_w=25.0, pressure_step_mtorr=2.0, noise_level=0.05)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def test_simulation_sweep_has_all_correlation_columns(sweep) -> None:
    assert set(CORRELATION_COLUMNS).issubset(sweep.columns)


def test_simulation_sweep_covers_the_operating_envelope(sweep) -> None:
    # fixture uses power_step_w=25.0 (50..300) and pressure_step_mtorr=2.0 (1..19,
    # since np.arange(1, 20+eps, 2) lands on 19 as its last step below 20).
    assert sweep["rf_power_w"].min() == pytest.approx(50.0)
    assert sweep["rf_power_w"].max() == pytest.approx(300.0)
    assert sweep["pressure_mtorr"].min() == pytest.approx(1.0)
    assert sweep["pressure_mtorr"].max() == pytest.approx(19.0)


def test_simulation_sweep_reproducible_with_seed() -> None:
    a = simulation_sweep_dataframe(power_step_w=50.0, pressure_step_mtorr=5.0, seed=42)
    b = simulation_sweep_dataframe(power_step_w=50.0, pressure_step_mtorr=5.0, seed=42)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Correlation matrix and heatmap [FE-2.4.1]
# ---------------------------------------------------------------------------
def test_correlation_matrix_is_symmetric_with_unit_diagonal(sweep) -> None:
    corr = correlation_matrix(sweep)
    assert corr.shape == (len(CORRELATION_COLUMNS), len(CORRELATION_COLUMNS))
    np.testing.assert_allclose(np.diag(corr), 1.0)
    pd.testing.assert_frame_equal(corr, corr.T, check_exact=False)


def test_correlation_matrix_values_in_valid_range(sweep) -> None:
    corr = correlation_matrix(sweep)
    assert (corr.to_numpy() >= -1.0 - 1e-9).all()
    assert (corr.to_numpy() <= 1.0 + 1e-9).all()


def test_power_and_reactivity_are_strongly_positively_correlated(sweep) -> None:
    """Known physics: reactivity scales with ion flux, which scales with power."""
    corr = correlation_matrix(sweep)
    assert corr.loc["rf_power_w", "reactivity_index"] > 0.9


def test_pressure_and_uniformity_are_strongly_negatively_correlated(sweep) -> None:
    """Known physics: uniformity worsens (falls) as pressure rises."""
    corr = correlation_matrix(sweep)
    assert corr.loc["pressure_mtorr", "uniformity_index"] < -0.7


def test_correlation_heatmap_returns_figure_matching_matrix_shape(sweep) -> None:
    corr = correlation_matrix(sweep)
    fig = correlation_heatmap(corr)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].z.shape == corr.shape
    assert list(fig.data[0].x) == list(corr.columns)


# ---------------------------------------------------------------------------
# Scatter + regression for key relationships [FE-2.4.2]
# ---------------------------------------------------------------------------
def test_scatter_with_regression_returns_figure_and_result(sweep) -> None:
    fig, result = scatter_with_regression(sweep, "rf_power_w", "reactivity_index")
    assert isinstance(fig, go.Figure)
    assert result.x == "rf_power_w"
    assert result.y == "reactivity_index"
    assert 0.0 <= result.r_squared <= 1.0


def test_scatter_regression_slope_sign_matches_known_physics(sweep) -> None:
    _fig, power_reactivity = scatter_with_regression(sweep, "rf_power_w", "reactivity_index")
    assert power_reactivity.slope > 0

    _fig, pressure_uniformity = scatter_with_regression(sweep, "pressure_mtorr", "uniformity_index")
    assert pressure_uniformity.slope < 0


def test_scatter_regression_r_squared_matches_pearson_r_squared(sweep) -> None:
    corr = correlation_matrix(sweep)
    _fig, result = scatter_with_regression(sweep, "rf_power_w", "reactivity_index")
    expected_r2 = corr.loc["rf_power_w", "reactivity_index"] ** 2
    assert result.r_squared == pytest.approx(expected_r2, abs=1e-6)


def test_key_relationship_plots_covers_all_three(sweep) -> None:
    plots = key_relationship_plots(sweep)
    assert set(plots.keys()) == set(KEY_RELATIONSHIPS)
    for (x, y), (fig, result) in plots.items():
        assert isinstance(fig, go.Figure)
        assert (result.x, result.y) == (x, y)


# ---------------------------------------------------------------------------
# Parallel coordinates [FE-2.4.3]
# ---------------------------------------------------------------------------
def test_parallel_coordinates_has_one_dimension_per_column(sweep) -> None:
    fig = parallel_coordinates_plot(sweep)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].dimensions) == len(CORRELATION_COLUMNS)


def test_parallel_coordinates_colored_by_specified_column(sweep) -> None:
    fig = parallel_coordinates_plot(sweep, color_by="process_quality")
    np.testing.assert_allclose(fig.data[0].line.color, sweep["process_quality"].to_numpy())


def test_parallel_coordinates_without_color(sweep) -> None:
    fig = parallel_coordinates_plot(sweep, color_by=None)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Automated insight narration [FE-2.4.4]
# ---------------------------------------------------------------------------
def test_insights_strongest_relationships_sorted_descending(sweep) -> None:
    insights = correlation_insights(sweep)
    magnitudes = [abs(r) for _x, _y, r in insights.strongest_relationships]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_insights_returns_requested_top_n(sweep) -> None:
    insights = correlation_insights(sweep, top_n=3)
    assert len(insights.strongest_relationships) == 3


def test_insights_flags_the_known_weak_relationship(sweep) -> None:
    """plasma_density_m3 vs process_quality is genuinely weak (r~0.2, verified) -
    this is exactly the "unexpectedly weak, physically expected" case FE-2.4.4
    exists to surface."""
    insights = correlation_insights(sweep)
    flagged_pairs = {(x, y) for x, y, _r in insights.weak_expected_relationships}
    assert ("plasma_density_m3", "process_quality") in flagged_pairs


def test_insights_does_not_flag_the_strong_key_relationships(sweep) -> None:
    insights = correlation_insights(sweep)
    flagged_pairs = {(x, y) for x, y, _r in insights.weak_expected_relationships}
    assert ("rf_power_w", "reactivity_index") not in flagged_pairs
    assert ("pressure_mtorr", "uniformity_index") not in flagged_pairs


def test_insights_narration_mentions_every_flagged_pair(sweep) -> None:
    insights = correlation_insights(sweep)
    for x, y, _r in insights.strongest_relationships + insights.weak_expected_relationships:
        assert any(x in line and y in line for line in insights.narration)


def test_insights_threshold_is_configurable(sweep) -> None:
    """A very strict (high) threshold should flag more pairs as 'weak'."""
    lenient = correlation_insights(sweep, low_threshold=0.01)
    strict = correlation_insights(sweep, low_threshold=0.99)
    assert len(strict.weak_expected_relationships) >= len(lenient.weak_expected_relationships)


def test_default_low_correlation_threshold_is_reasonable() -> None:
    assert 0.0 < LOW_CORRELATION_THRESHOLD < 1.0
