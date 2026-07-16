"""Tests for FE-2.2.6 - the optional PyTorch autoencoder anomaly detector.

Marked `torch` and gated with importorskip so the suite still runs (and CI stays
lightweight) when the optional torch dependency is not installed. Install it with
`pip install -r requirements-optional.txt` to exercise these locally.

Like the Isolation Forest tests, the central assertions enforce non-negotiable
principle #4: the injected anomalies are in-range (a range check misses them) yet
the deep-learning detector catches them by learning the normal manifold.
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.torch
pytest.importorskip("torch", reason="optional deep-learning dependency (requirements-optional.txt)")

from ai_module.anomaly_detection import (
    generate_anomalous_data,
    generate_normal_operating_data,
    normal_feature_ranges,
    range_check_is_anomaly,
)
from ai_module.autoencoder_detector import (
    PlasmaAutoencoderDetector,
    compare_with_isolation_forest,
)
from digital_twin.dataset_generation import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def normal_data() -> pd.DataFrame:
    return generate_normal_operating_data()


@pytest.fixture(scope="module")
def detector(normal_data) -> PlasmaAutoencoderDetector:
    return PlasmaAutoencoderDetector(seed=0).fit(normal_data)


@pytest.fixture(scope="module")
def anomalous_data() -> pd.DataFrame:
    return generate_anomalous_data(n_samples=150)


# ---------------------------------------------------------------------------
# Reconstruction-error behaviour
# ---------------------------------------------------------------------------
def test_normal_reconstruction_error_is_low(detector, normal_data) -> None:
    """Normal points lie on the learned manifold, so they reconstruct well."""
    errors = detector.reconstruction_error(normal_data)
    assert np.median(errors) < detector.threshold


def test_anomalies_reconstruct_worse_than_normal(detector, normal_data, anomalous_data) -> None:
    normal_median = float(np.median(detector.reconstruction_error(normal_data)))
    anomaly_median = float(np.median(detector.reconstruction_error(anomalous_data[FEATURE_COLUMNS])))
    assert anomaly_median > normal_median


def test_anomaly_score_higher_means_more_anomalous(detector, normal_data, anomalous_data) -> None:
    """Sanity on the sign convention (higher = more anomalous, unlike IF)."""
    normal_mean = detector.anomaly_score(normal_data).mean()
    anomaly_mean = detector.anomaly_score(anomalous_data[FEATURE_COLUMNS]).mean()
    assert anomaly_mean > normal_mean


def test_scoring_before_fit_raises() -> None:
    unfitted = PlasmaAutoencoderDetector()
    normal = generate_normal_operating_data(replicates=1).iloc[:3]
    with pytest.raises(RuntimeError):
        unfitted.reconstruction_error(normal)


# ---------------------------------------------------------------------------
# Detection performance + principle #4 contrast
# ---------------------------------------------------------------------------
def test_low_false_positive_rate_on_normal_data(detector) -> None:
    normal_test = generate_normal_operating_data(seed=123, replicates=1)
    fpr = detector.is_anomaly(normal_test).mean()
    assert fpr < 0.05


def test_catches_relationship_violations_range_check_misses(detector, anomalous_data) -> None:
    """Principle #4: the deep-learning detector must catch, from raw features
    alone, the in-range relationship violations that a naive range check cannot."""
    normal_test = generate_normal_operating_data(seed=7, replicates=1)
    ranges = normal_feature_ranges(normal_test)
    range_recall = range_check_is_anomaly(anomalous_data[FEATURE_COLUMNS], ranges).mean()
    ae_recall = detector.is_anomaly(anomalous_data[FEATURE_COLUMNS]).mean()
    assert range_recall < 0.05
    assert ae_recall > 0.7
    assert ae_recall > range_recall + 0.5


def test_multi_channel_faults_well_detected(detector) -> None:
    from ai_module.anomaly_detection import AnomalyFault

    for fault in (AnomalyFault.PRESSURE_GAUGE_FAULT, AnomalyFault.ELECTRODE_COUPLING_FAULT):
        anom = generate_anomalous_data(n_samples=60, seed=5, faults=(fault,))
        recall = detector.is_anomaly(anom[FEATURE_COLUMNS]).mean()
        assert recall > 0.85


# ---------------------------------------------------------------------------
# Reproducibility (a defended result must be reproducible)
# ---------------------------------------------------------------------------
def test_detector_is_reproducible_with_seed(normal_data, anomalous_data) -> None:
    d1 = PlasmaAutoencoderDetector(seed=42, epochs=150).fit(normal_data)
    d2 = PlasmaAutoencoderDetector(seed=42, epochs=150).fit(normal_data)
    np.testing.assert_allclose(
        d1.reconstruction_error(anomalous_data[FEATURE_COLUMNS]),
        d2.reconstruction_error(anomalous_data[FEATURE_COLUMNS]),
        rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Classical vs deep-learning comparison [FE-2.2.6]
# ---------------------------------------------------------------------------
def test_comparison_reports_both_detectors_and_baseline(normal_data, anomalous_data) -> None:
    normal_test = generate_normal_operating_data(seed=99, replicates=1)
    comparison = compare_with_isolation_forest(normal_data, normal_test, anomalous_data)

    # both learned detectors clear the naive baseline by a wide margin
    assert comparison.range_check_recall < 0.05
    assert comparison.isolation_forest_recall > 0.7
    assert comparison.autoencoder_recall > 0.7
    # both keep a low false-positive rate
    assert comparison.isolation_forest_fpr < 0.1
    assert comparison.autoencoder_fpr < 0.1
    # per-fault breakdown covers the injected fault types
    assert set(comparison.autoencoder_per_fault_recall).issubset(
        {"pressure_gauge_fault", "electrode_coupling_fault", "te_sensor_drift"}
    )
