"""Tests for Sub-Module 2.5 - semiconductor process suitability analysis."""
import pandas as pd
import pytest

from digital_twin.chamber_config import ChamberParameters
from digital_twin.physics_engine import simulate
from digital_twin.session_manager import ExperimentDatabase
from ai_module.suitability_analysis import (
    SUITABILITY_WINDOWS,
    DefectProbabilityEstimate,
    SemiconductorApplication,
    _ceiling_compliance,
    _window_compliance,
    all_application_defect_estimates,
    application_defect_estimate,
    classify_suitability,
    compare_sessions_for_application,
)


# ---------------------------------------------------------------------------
# Window definitions [FE-2.5.1]
# ---------------------------------------------------------------------------
def test_all_four_applications_have_windows() -> None:
    assert set(SUITABILITY_WINDOWS.keys()) == set(SemiconductorApplication)


def test_windows_are_internally_consistent() -> None:
    for window in SUITABILITY_WINDOWS.values():
        assert window.ion_energy_min_ev < window.ion_energy_max_ev
        assert 0.0 < window.max_defect_probability <= 1.0
        assert len(window.rationale) > 20  # a real justification, not a stub string


def test_etching_requires_higher_ion_energy_than_cleaning() -> None:
    """The cited ordering: etching needs much higher ion energy than cleaning."""
    etching = SUITABILITY_WINDOWS[SemiconductorApplication.PLASMA_ETCHING]
    cleaning = SUITABILITY_WINDOWS[SemiconductorApplication.WAFER_CLEANING]
    assert etching.ion_energy_min_ev > cleaning.ion_energy_max_ev


def test_etching_has_the_most_permissive_defect_ceiling() -> None:
    """Etching inherently tolerates more aggressive bombardment than the
    damage-sensitive cleaning/deposition processes."""
    etching = SUITABILITY_WINDOWS[SemiconductorApplication.PLASMA_ETCHING].max_defect_probability
    for app in (SemiconductorApplication.WAFER_CLEANING, SemiconductorApplication.THIN_FILM_DEPOSITION):
        assert etching > SUITABILITY_WINDOWS[app].max_defect_probability


# ---------------------------------------------------------------------------
# Compliance scoring primitives
# ---------------------------------------------------------------------------
def test_window_compliance_inside_window_is_full() -> None:
    assert _window_compliance(30.0, 5.0, 50.0) == 1.0
    assert _window_compliance(5.0, 5.0, 50.0) == 1.0   # boundary inclusive
    assert _window_compliance(50.0, 5.0, 50.0) == 1.0  # boundary inclusive


def test_window_compliance_degrades_outside_window() -> None:
    partial = _window_compliance(60.0, 5.0, 50.0)  # 10 past a 45-wide window
    assert 0.0 < partial < 1.0
    assert _window_compliance(50.0 + 45.0, 5.0, 50.0) == pytest.approx(0.0, abs=1e-9)
    assert _window_compliance(1000.0, 5.0, 50.0) == 0.0


def test_ceiling_compliance_at_and_below_ceiling_is_full() -> None:
    assert _ceiling_compliance(0.3, 0.5) == 1.0
    assert _ceiling_compliance(0.5, 0.5) == 1.0


def test_ceiling_compliance_degrades_above_ceiling() -> None:
    partial = _ceiling_compliance(0.7, 0.5)
    assert 0.0 < partial < 1.0
    assert _ceiling_compliance(1.0, 0.5) == 0.0


# ---------------------------------------------------------------------------
# Suitability scorecard [FE-2.5.1, FE-2.5.2]
# ---------------------------------------------------------------------------
def test_scorecard_covers_all_applications() -> None:
    scorecard = classify_suitability(150.0, 10.0)
    assert set(scorecard.ratings.keys()) == set(SemiconductorApplication)


def test_gentle_default_condition_suits_cleaning_not_etching() -> None:
    """Verified: at 150W/10mTorr with no RF voltage, ion_energy sits well inside
    the cleaning/treatment/deposition windows but below the etching window."""
    scorecard = classify_suitability(150.0, 10.0)
    assert scorecard.ratings[SemiconductorApplication.WAFER_CLEANING].is_suitable
    assert scorecard.ratings[SemiconductorApplication.SURFACE_TREATMENT].is_suitable
    assert scorecard.ratings[SemiconductorApplication.THIN_FILM_DEPOSITION].is_suitable
    assert not scorecard.ratings[SemiconductorApplication.PLASMA_ETCHING].is_suitable


def test_rf_voltage_driven_condition_suits_etching_not_cleaning() -> None:
    """The core physical narrative of this sub-module: etching only becomes
    reachable once a driven RF voltage pushes ion energy into its window
    (verified: 200W/5mTorr, rf_voltage_v=300V)."""
    scorecard = classify_suitability(200.0, 5.0, rf_voltage_v=300.0)
    assert scorecard.ratings[SemiconductorApplication.PLASMA_ETCHING].is_suitable
    assert not scorecard.ratings[SemiconductorApplication.WAFER_CLEANING].is_suitable


def test_best_application_matches_max_overall_compliance() -> None:
    scorecard = classify_suitability(200.0, 5.0, rf_voltage_v=300.0)
    best = scorecard.best_application()
    best_value = scorecard.ratings[best].overall_compliance_pct
    assert all(r.overall_compliance_pct <= best_value for r in scorecard.ratings.values())
    assert best == SemiconductorApplication.PLASMA_ETCHING  # unique winner in this scenario


def test_scorecard_to_dataframe_has_one_row_per_application() -> None:
    scorecard = classify_suitability(150.0, 10.0)
    df = scorecard.to_dataframe()
    assert len(df) == len(SemiconductorApplication)
    for col in ("application", "is_suitable", "ion_energy_compliance_pct",
                "defect_compliance_pct", "overall_compliance_pct"):
        assert col in df.columns


def test_compliance_percentages_are_bounded() -> None:
    scorecard = classify_suitability(150.0, 10.0)
    for rating in scorecard.ratings.values():
        assert 0.0 <= rating.ion_energy_compliance_pct <= 100.0
        assert 0.0 <= rating.defect_compliance_pct <= 100.0
        assert 0.0 <= rating.overall_compliance_pct <= 100.0


def test_is_suitable_implies_full_compliance() -> None:
    scorecard = classify_suitability(150.0, 10.0)
    for rating in scorecard.ratings.values():
        if rating.is_suitable:
            assert rating.ion_energy_compliance_pct == 100.0
            assert rating.defect_compliance_pct == 100.0


# ---------------------------------------------------------------------------
# Application-specific defect confidence interval [FE-2.5.3]
# ---------------------------------------------------------------------------
def test_defect_estimate_ci_bounds_are_ordered() -> None:
    estimate = application_defect_estimate(
        150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, n_bootstrap=100, seed=1
    )
    assert estimate.ci_lower <= estimate.point_estimate <= estimate.ci_upper


def test_defect_estimate_is_a_genuine_interval_not_a_point() -> None:
    """FE-2.5.3 explicitly requires an interval, not a single point estimate."""
    estimate = application_defect_estimate(
        150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, n_bootstrap=100, seed=1
    )
    assert estimate.ci_upper > estimate.ci_lower


def test_defect_estimate_reproducible_with_seed() -> None:
    a = application_defect_estimate(150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, n_bootstrap=50, seed=42)
    b = application_defect_estimate(150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, n_bootstrap=50, seed=42)
    assert a == b


def test_out_of_window_application_has_higher_risk_than_in_window() -> None:
    """Verified: at 150W/10mTorr, etching (ion energy below its window) shows
    materially higher defect risk than cleaning (ion energy inside its window),
    at the SAME physical operating point - the deviation penalty is doing real
    work, not just echoing the raw physics defect_probability unchanged."""
    in_window = application_defect_estimate(
        150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, n_bootstrap=150, seed=5
    )
    out_of_window = application_defect_estimate(
        150.0, 10.0, SemiconductorApplication.PLASMA_ETCHING, n_bootstrap=150, seed=5
    )
    assert out_of_window.point_estimate > in_window.point_estimate


def test_defect_estimate_rejects_invalid_confidence_level() -> None:
    with pytest.raises(ValueError):
        application_defect_estimate(
            150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, confidence_level=1.5
        )
    with pytest.raises(ValueError):
        application_defect_estimate(
            150.0, 10.0, SemiconductorApplication.WAFER_CLEANING, confidence_level=0.0
        )


def test_all_application_defect_estimates_covers_every_application() -> None:
    estimates = all_application_defect_estimates(150.0, 10.0, n_bootstrap=30, seed=1)
    assert set(estimates.keys()) == set(SemiconductorApplication)
    for est in estimates.values():
        assert isinstance(est, DefectProbabilityEstimate)


# ---------------------------------------------------------------------------
# Comparative suitability across stored sessions [FE-2.5.4]
# ---------------------------------------------------------------------------
def test_compare_sessions_ranks_by_overall_compliance(tmp_path) -> None:
    db = ExperimentDatabase(tmp_path / "sessions.db")
    for power, pressure in [(150.0, 10.0), (280.0, 1.5), (100.0, 15.0)]:
        params = ChamberParameters(rf_power_w=power, pressure_mtorr=pressure)
        result = simulate(params.rf_power_w, params.pressure_mtorr)
        db.create_session(params, result)
    sessions = db.list_sessions()

    ranked = compare_sessions_for_application(sessions, SemiconductorApplication.WAFER_CLEANING)
    compliance = ranked["overall_compliance_pct"].tolist()
    assert compliance == sorted(compliance, reverse=True)
    db.close()


def test_compare_sessions_is_suitable_matches_classify_suitability(tmp_path) -> None:
    """Cross-consistency: the session-comparison path and the direct
    classify_suitability path must agree on the same operating point, since both
    share the underlying _rate_application logic."""
    db = ExperimentDatabase(tmp_path / "sessions.db")
    params = ChamberParameters(rf_power_w=150.0, pressure_mtorr=10.0)
    result = simulate(params.rf_power_w, params.pressure_mtorr)
    session_id = db.create_session(params, result)
    session = db.get_session(session_id)

    ranked = compare_sessions_for_application([session], SemiconductorApplication.WAFER_CLEANING)
    direct = classify_suitability(150.0, 10.0)

    assert bool(ranked.iloc[0]["is_suitable"]) == direct.ratings[SemiconductorApplication.WAFER_CLEANING].is_suitable
    db.close()


def test_compare_sessions_empty_list_returns_empty_dataframe() -> None:
    result = compare_sessions_for_application([], SemiconductorApplication.WAFER_CLEANING)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
