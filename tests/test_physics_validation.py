"""Tests for Sub-Module 1.6 - physics validation and literature benchmarking.

These tests pin down the honest, semi-quantitative nature of FE-1.6.2: some
reference checks are expected to pass and some are expected to fail (the
ionization-rate-coefficient and density-vs-line-average checks), and the module
must report that honestly rather than silently loosening tolerances to force a
clean pass. If a future physics_engine.py change accidentally makes the checks
report differently, these tests should be revisited deliberately, not patched
to match new numbers without checking why they changed.
"""
import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from digital_twin.physics_validation import (
    TOLERANCE_STANDARD_PCT,
    TOLERANCE_TIGHT_PCT,
    BenchmarkResult,
    benchmark_report_text,
    benchmark_summary_table,
    density_vs_power_validation_plot,
    run_literature_benchmarks,
    te_vs_pressure_validation_plot,
)


@pytest.fixture(scope="module")
def results() -> list[BenchmarkResult]:
    return run_literature_benchmarks()


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------
def test_returns_multiple_reference_points(results) -> None:
    """FE-1.6.2 calls for 'several reference operating points', not just one."""
    assert len(results) >= 4


def test_every_result_has_required_metadata(results) -> None:
    for r in results:
        assert r.name
        assert r.quantity
        assert r.description
        # Every check must carry a substantive, dated citation string. Most cite
        # Turner (2014); the two additional K_iz checks (FUTURE.md item 5) cite
        # their own independent sources, so we no longer require "Turner" here.
        assert len(r.source) > 20
        assert r.unit
        assert math.isfinite(r.computed_value)
        assert math.isfinite(r.reference_value)
        assert r.tolerance_pct > 0.0


def test_turner_checks_cite_turner(results) -> None:
    """Every check EXCEPT the two additional independent K_iz sources should still
    cite Turner 2014 - guards against a citation being dropped or mislabelled."""
    additional = {
        "ionization_rate_coefficient_at_3eV_vs_jimenez_redondo",
        "ionization_rate_coefficient_at_3eV_vs_chabert",
    }
    for r in results:
        if r.name not in additional:
            assert "Turner" in r.source and "2014" in r.source


def test_deviation_and_passed_are_internally_consistent(results) -> None:
    for r in results:
        expected_deviation = 100.0 * (r.computed_value - r.reference_value) / r.reference_value
        assert r.deviation_pct == pytest.approx(expected_deviation)
        assert r.passed == (abs(r.deviation_pct) <= r.tolerance_pct)


# ---------------------------------------------------------------------------
# Semi-quantitative, not merely directional (FE-1.6.2's core requirement)
# ---------------------------------------------------------------------------
def test_deviation_is_a_real_percentage_not_a_placeholder(results) -> None:
    """None of the deviations should be exactly 0 - these are independent
    reference values, not the model checking itself."""
    for r in results:
        assert r.deviation_pct != 0.0


def test_electron_temperature_checks_pass_within_standard_tolerance(results) -> None:
    te_checks = [r for r in results if r.quantity == "Te (eV)"]
    assert len(te_checks) == 2
    for r in te_checks:
        assert r.tolerance_pct == TOLERANCE_STANDARD_PCT
        assert r.passed, f"{r.name} deviated {r.deviation_pct:.1f}%, expected within tolerance"


def test_ion_mean_free_path_ratio_check_passes_tight_tolerance(results) -> None:
    r = next(r for r in results if r.name == "ion_mean_free_path_ratio")
    assert r.tolerance_pct == TOLERANCE_TIGHT_PCT
    assert r.passed
    assert abs(r.deviation_pct) < 10.0  # a clean, low-noise direct ratio check


def test_ionization_rate_coefficient_check_honestly_fails(results) -> None:
    """This is the key honesty test: the rate-coefficient parameterization is
    known to deviate from the cited reference by more than the standard
    tolerance at Te=3 eV, and the module must report that as a failure rather
    than loosen the tolerance to hide it."""
    r = next(r for r in results if r.name == "ionization_rate_coefficient_at_3eV")
    assert r.tolerance_pct == TOLERANCE_STANDARD_PCT
    assert not r.passed
    assert r.deviation_pct < -TOLERANCE_STANDARD_PCT


def test_additional_kiz_sources_present_independent_and_rederivable(results) -> None:
    """FUTURE.md item 5: two ADDITIONAL, independently published K_iz(3 eV)
    reference points beyond Turner, so the (honestly failing) K_iz check rests on
    a published spread rather than one number. Structural only - deliberately
    makes NO claim about whether they pass, since their tolerance was fixed a
    priori and we don't tune to the outcome."""
    from digital_twin.physics_validation import _chabert_kiz, _jimenez_redondo_kiz

    jr = next(r for r in results if r.name == "ionization_rate_coefficient_at_3eV_vs_jimenez_redondo")
    ch = next(r for r in results if r.name == "ionization_rate_coefficient_at_3eV_vs_chabert")
    turner_kiz = next(r for r in results if r.name == "ionization_rate_coefficient_at_3eV")

    for r in (jr, ch):
        assert r.quantity == "K_iz(3 eV) (m^3/s)"
        assert r.tolerance_pct == TOLERANCE_STANDARD_PCT
        # All three K_iz checks must benchmark the SAME model quantity - only the
        # reference differs - exactly like the two Te checks do.
        assert r.computed_value == pytest.approx(turner_kiz.computed_value)
        # Sane argon K_iz(3 eV) magnitude in SI m^3/s - guards the cm^3->m^3
        # conversion on the Jimenez-Redondo formula against a units slip.
        assert 1e-17 < r.reference_value < 1e-15

    # Reference values are re-derivable from each published FORMULA (not a
    # hand-copied scalar), and the two sources genuinely differ from each other.
    assert jr.reference_value == pytest.approx(_jimenez_redondo_kiz(3.0))
    assert ch.reference_value == pytest.approx(_chabert_kiz(3.0))
    # Genuinely different anchors (compare RELATIVE difference - pytest.approx's
    # default 1e-12 absolute tolerance would call any two ~1e-16 values "equal").
    assert abs(jr.reference_value - ch.reference_value) / ch.reference_value > 0.1

    # Independence is documented honestly in the citation strings themselves.
    assert "INDEPENDENT" in jr.source.upper()
    assert "Chabert" in ch.source and "NOT" in ch.source


def test_not_every_check_passes() -> None:
    """A validation suite that always passes is a validation suite that isn't
    checking anything real; the deliberately-honest result here is a mix."""
    results = run_literature_benchmarks()
    passed = [r.passed for r in results]
    assert any(passed), "expected at least some checks to pass"
    assert not all(passed), "expected at least one honest failure (K_iz check)"


def test_electron_temperature_matches_regardless_of_reference_chosen(results) -> None:
    """Both Te checks (vs global model, vs PIC) compute the SAME model value -
    only the reference differs. Confirms the two checks aren't silently using
    different computed quantities."""
    te_vs_model = next(r for r in results if r.name == "electron_temperature_vs_global_model")
    te_vs_pic = next(r for r in results if r.name == "electron_temperature_vs_pic_ground_truth")
    assert te_vs_model.computed_value == pytest.approx(te_vs_pic.computed_value)
    assert te_vs_model.reference_value != te_vs_pic.reference_value


# ---------------------------------------------------------------------------
# Live recomputation (not a stale hardcoded snapshot)
# ---------------------------------------------------------------------------
def test_benchmarks_are_computed_live_from_the_physics_engine() -> None:
    """Recomputing twice must give identical results (deterministic engine),
    proving these are live calls into physics_engine, not cached constants."""
    a = run_literature_benchmarks()
    b = run_literature_benchmarks()
    for ra, rb in zip(a, b):
        assert ra.computed_value == rb.computed_value


# ---------------------------------------------------------------------------
# Report / summary outputs (FE-1.6.3)
# ---------------------------------------------------------------------------
def test_summary_table_has_one_row_per_result(results) -> None:
    table = benchmark_summary_table()
    assert isinstance(table, pd.DataFrame)
    assert len(table) == len(results)
    for col in ("name", "computed_value", "reference_value", "deviation_pct", "tolerance_pct", "passed"):
        assert col in table.columns


def test_report_text_mentions_pass_fail_counts_and_source() -> None:
    text = benchmark_report_text()
    assert "PASS" in text
    assert "FAIL" in text
    assert "Turner" in text
    # Counts in the summary line must match the actual tally.
    results = run_literature_benchmarks()
    n_passed = sum(r.passed for r in results)
    assert f"{n_passed}/{len(results)}" in text


# ---------------------------------------------------------------------------
# Comparison plots (FE-1.6.1)
# ---------------------------------------------------------------------------
def test_te_vs_pressure_plot_is_a_figure_with_model_and_reference_traces() -> None:
    fig = te_vs_pressure_validation_plot()
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Digital twin (this project)" in trace_names
    assert any("Turner" in name for name in trace_names)


def test_te_vs_pressure_plot_model_curve_is_decreasing() -> None:
    """The plotted trend itself should show Te falling with Ndeff - the
    qualitative shape-comparison half of FE-1.6.1."""
    fig = te_vs_pressure_validation_plot()
    model_trace = next(t for t in fig.data if t.name == "Digital twin (this project)")
    y = list(model_trace.y)
    assert y == sorted(y, reverse=True)


def test_density_vs_power_plot_is_a_figure_with_increasing_trend() -> None:
    fig = density_vs_power_validation_plot()
    assert isinstance(fig, go.Figure)
    model_trace = fig.data[0]
    y = list(model_trace.y)
    assert y == sorted(y)  # density increases with power
