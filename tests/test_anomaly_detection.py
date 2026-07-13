"""Tests for Sub-Module 2.2 - anomaly detection.

The most important tests enforce non-negotiable principle #4: the injected
anomalies must be IN-RANGE (so a plain range check misses them) yet detected by
the Isolation Forest (because it operates on physics residuals). If those ever
break, the sub-module would be reduced to a range check, defeating its purpose.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from digital_twin.dataset_generation import FEATURE_COLUMNS
from ai_module.anomaly_detection import (
    RESIDUAL_COLUMNS,
    AnomalyFault,
    AnomalySeverity,
    PlasmaAnomalyDetector,
    SPCMonitor,
    anomaly_timeline_plot,
    evaluate_detector,
    generate_anomalous_data,
    generate_normal_operating_data,
    inject_anomaly,
    load_anomaly_events,
    normal_feature_ranges,
    range_check_is_anomaly,
    store_anomaly_event,
)


@pytest.fixture(scope="module")
def normal_data() -> pd.DataFrame:
    return generate_normal_operating_data()


@pytest.fixture(scope="module")
def detector(normal_data) -> PlasmaAnomalyDetector:
    return PlasmaAnomalyDetector().fit(normal_data)


@pytest.fixture(scope="module")
def anomalous_data() -> pd.DataFrame:
    return generate_anomalous_data(n_samples=150)


# ---------------------------------------------------------------------------
# The core principle #4 guarantees
# ---------------------------------------------------------------------------
def test_injected_anomalies_are_all_in_range(normal_data, anomalous_data) -> None:
    """Every injected anomaly must be within the normal per-feature range, so a
    naive range check cannot catch it - this is what makes them genuine
    RELATIONSHIP violations rather than out-of-range values."""
    ranges = normal_feature_ranges(normal_data)
    flagged = range_check_is_anomaly(anomalous_data[FEATURE_COLUMNS], ranges)
    assert flagged.sum() == 0, "some injected anomalies were out-of-range (not relationship violations)"


def test_isolation_forest_catches_what_range_check_misses(detector, normal_data, anomalous_data) -> None:
    """The headline demonstration of principle #4: Isolation Forest recall is
    high while the naive range check's recall is ~zero on the same anomalies."""
    normal_test = generate_normal_operating_data(seed=99, replicates=1)
    report = evaluate_detector(detector, normal_test, anomalous_data)
    assert report.range_check_recall < 0.05      # range check essentially blind
    assert report.isolation_forest_recall > 0.80  # Isolation Forest catches them
    assert report.isolation_forest_recall > report.range_check_recall + 0.5


def test_multi_channel_faults_detected_near_perfectly(detector) -> None:
    """Pressure-gauge and electrode-coupling faults corrupt several channels and
    should be caught almost every time."""
    for fault in (AnomalyFault.PRESSURE_GAUGE_FAULT, AnomalyFault.ELECTRODE_COUPLING_FAULT):
        anom = generate_anomalous_data(n_samples=60, seed=5, faults=(fault,))
        recall = detector.is_anomaly(anom[FEATURE_COLUMNS]).mean()
        assert recall > 0.95


def test_low_false_positive_rate_on_normal_data(detector) -> None:
    normal_test = generate_normal_operating_data(seed=123, replicates=1)
    fpr = detector.is_anomaly(normal_test).mean()
    assert fpr < 0.05


# ---------------------------------------------------------------------------
# Residual representation
# ---------------------------------------------------------------------------
def test_normal_residuals_are_small(detector, normal_data) -> None:
    """A normal run's residual is just measurement noise - small in every channel."""
    residuals = detector.relative_residuals(normal_data.iloc[:50]).abs()
    # 5% noise: essentially all residuals well under 25%.
    assert (residuals < 0.25).mean().mean() > 0.98


def test_anomaly_residual_is_large_in_the_faulted_channel(detector) -> None:
    """A Te-sensor drift must produce a large residual specifically in Te."""
    rng = np.random.default_rng(0)
    row = inject_anomaly(150.0, 10.0, AnomalyFault.TE_SENSOR_DRIFT, rng)
    residuals = detector.relative_residuals(pd.DataFrame([row])).abs().iloc[0]
    assert residuals["electron_temperature_ev"] > 0.3
    # other channels stay near noise level
    for c in RESIDUAL_COLUMNS:
        if c != "electron_temperature_ev":
            assert residuals[c] < 0.1


def test_residual_columns_exclude_the_input_setpoints() -> None:
    assert "rf_power_w" not in RESIDUAL_COLUMNS
    assert "pressure_mtorr" not in RESIDUAL_COLUMNS
    assert set(RESIDUAL_COLUMNS) == set(FEATURE_COLUMNS) - {"rf_power_w", "pressure_mtorr"}


# ---------------------------------------------------------------------------
# Severity levels (FE-2.2.2)
# ---------------------------------------------------------------------------
def test_severity_normal_for_normal_data(detector) -> None:
    normal_test = generate_normal_operating_data(seed=7, replicates=1)
    severities = detector.severity(normal_test)
    normal_fraction = sum(s == AnomalySeverity.NORMAL for s in severities) / len(severities)
    assert normal_fraction > 0.95


def test_severity_elevated_for_anomalies(detector, anomalous_data) -> None:
    severities = detector.severity(anomalous_data[FEATURE_COLUMNS])
    elevated = sum(s in (AnomalySeverity.WARNING, AnomalySeverity.CRITICAL) for s in severities)
    assert elevated / len(severities) > 0.80


def test_severity_thresholds_are_ordered(detector) -> None:
    assert detector.critical_threshold < detector.warning_threshold


# ---------------------------------------------------------------------------
# Root-cause indication (BO-5)
# ---------------------------------------------------------------------------
def test_root_cause_points_at_temperature_for_te_drift(detector) -> None:
    rng = np.random.default_rng(1)
    row = inject_anomaly(150.0, 10.0, AnomalyFault.TE_SENSOR_DRIFT, rng)
    cause = detector.root_cause(pd.DataFrame([row]))[0]
    assert "temperature" in cause.lower()


def test_root_cause_points_at_density_for_electrode_fault(detector) -> None:
    rng = np.random.default_rng(2)
    row = inject_anomaly(100.0, 10.0, AnomalyFault.ELECTRODE_COUPLING_FAULT, rng)
    cause = detector.root_cause(pd.DataFrame([row]))[0]
    assert "density" in cause.lower() or "reactive" in cause.lower() or "etch" in cause.lower()


# ---------------------------------------------------------------------------
# Reproducibility & validation
# ---------------------------------------------------------------------------
def test_detector_is_reproducible_with_seed(normal_data, anomalous_data) -> None:
    d1 = PlasmaAnomalyDetector(seed=42).fit(normal_data)
    d2 = PlasmaAnomalyDetector(seed=42).fit(normal_data)
    np.testing.assert_allclose(
        d1.anomaly_score(anomalous_data[FEATURE_COLUMNS]),
        d2.anomaly_score(anomalous_data[FEATURE_COLUMNS]),
    )


def test_normal_data_requires_noise() -> None:
    with pytest.raises(ValueError):
        generate_normal_operating_data(noise_level=0.0)


# ---------------------------------------------------------------------------
# Statistical process control (FE-2.2.4)
# ---------------------------------------------------------------------------
def test_spc_control_limits_are_three_sigma() -> None:
    baseline = np.array([0.30, 0.32, 0.31, 0.29, 0.33, 0.30, 0.31])
    monitor = SPCMonitor.from_baseline(baseline)
    assert monitor.upper_control_limit == pytest.approx(monitor.center_line + 3 * monitor.sigma)
    assert monitor.lower_control_limit == pytest.approx(monitor.center_line - 3 * monitor.sigma)


def test_spc_flags_a_gross_quality_excursion() -> None:
    baseline = np.array([0.30, 0.32, 0.31, 0.29, 0.33, 0.30, 0.31])
    monitor = SPCMonitor.from_baseline(baseline)
    values = np.array([0.31, 0.30, 0.05, 0.31])  # third point is a gross drop
    flags = monitor.out_of_control(values)
    assert flags.tolist() == [False, False, True, False]


def test_spc_does_not_flag_in_control_runs() -> None:
    baseline = np.array([0.30, 0.32, 0.31, 0.29, 0.33, 0.30, 0.31])
    monitor = SPCMonitor.from_baseline(baseline)
    assert not monitor.out_of_control(np.array([0.30, 0.31, 0.32])).any()


def test_spc_requires_baseline_of_at_least_two() -> None:
    with pytest.raises(ValueError):
        SPCMonitor.from_baseline(np.array([0.3]))


# ---------------------------------------------------------------------------
# Anomaly event logging (FE-2.2.3)
# ---------------------------------------------------------------------------
def test_store_and_load_anomaly_event(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    event_id = store_anomaly_event(
        "session-1", 150.0, 10.0, -0.2, AnomalySeverity.CRITICAL,
        root_cause="density channel off", db_path=db_path,
    )
    events = load_anomaly_events("session-1", db_path=db_path)
    assert len(events) == 1
    assert events.iloc[0]["event_id"] == event_id
    assert events.iloc[0]["severity"] == "Critical"
    assert events.iloc[0]["rf_power_w"] == 150.0


def test_load_anomaly_events_filters_by_session(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    store_anomaly_event("session-A", 100.0, 5.0, -0.1, AnomalySeverity.WARNING, db_path=db_path)
    store_anomaly_event("session-B", 200.0, 15.0, -0.3, AnomalySeverity.CRITICAL, db_path=db_path)
    assert len(load_anomaly_events("session-A", db_path=db_path)) == 1
    assert len(load_anomaly_events(db_path=db_path)) == 2  # all sessions


# ---------------------------------------------------------------------------
# Timeline visualisation (FE-2.2.5)
# ---------------------------------------------------------------------------
def test_anomaly_timeline_plot_builds_a_figure(detector) -> None:
    scores = [0.05, 0.04, -0.2, 0.03]
    severities = [
        AnomalySeverity.NORMAL, AnomalySeverity.NORMAL,
        AnomalySeverity.CRITICAL, AnomalySeverity.NORMAL,
    ]
    fig = anomaly_timeline_plot(
        [0, 1, 2, 3], scores, severities,
        detector.warning_threshold, detector.critical_threshold,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert list(fig.data[0].y) == scores
