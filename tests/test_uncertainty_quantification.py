"""Tests for Sub-Module 2.8 - uncertainty quantification and trust layer (MAPIE).

The centrepiece test is test_empirical_coverage_is_close_to_target: it checks
that "calibrated" is a measured, verified property (on a genuinely held-out test
set) rather than an assumed consequence of using MAPIE.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from digital_twin.dataset_generation import (
    DEFAULT_SEED,
    SUITABILITY_CLASSES,
    generate_dataset,
    random_split,
)
from ai_module.classification import ClassifierKind, train_classifiers
from ai_module.uncertainty_quantification import (
    DEFECT_TARGET_COLUMN,
    assess_configuration,
    build_uncertainty_layer,
    conformalize_classifier,
    conformalize_defect_regressor,
    evaluate_coverage,
    train_defect_probability_regressor,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(power_step_w=25.0, pressure_step_mtorr=2.0, replicates_per_recipe=10)


@pytest.fixture(scope="module")
def layer(dataset):
    return build_uncertainty_layer(dataset)


# ---------------------------------------------------------------------------
# Conformal classification [FE-2.8.1]
# ---------------------------------------------------------------------------
def test_conformalize_classifier_produces_valid_class_sets(dataset) -> None:
    train_df, calib_df = random_split(dataset, test_fraction=0.3, seed=1)
    base = train_classifiers(train_df)[ClassifierKind.RANDOM_FOREST]
    conformal = conformalize_classifier(base, calib_df, confidence_level=0.9)

    from digital_twin.dataset_generation import features_and_labels
    X_test, _y_test = features_and_labels(calib_df.iloc[:20])
    pred_sets = conformal.predict_set(X_test)

    assert len(pred_sets) == 20
    for s in pred_sets:
        assert len(s) >= 1
        assert set(s).issubset(set(SUITABILITY_CLASSES))


def test_conformal_classifier_confidence_level_is_recorded(dataset) -> None:
    train_df, calib_df = random_split(dataset, test_fraction=0.3, seed=1)
    base = train_classifiers(train_df)[ClassifierKind.RANDOM_FOREST]
    conformal = conformalize_classifier(base, calib_df, confidence_level=0.8)
    assert conformal.confidence_level == 0.8


# ---------------------------------------------------------------------------
# Conformal defect-probability regression [FE-2.8.1]
# ---------------------------------------------------------------------------
def test_defect_regressor_predicts_within_plausible_range(dataset) -> None:
    train_df, calib_df = random_split(dataset, test_fraction=0.3, seed=1)
    model = train_defect_probability_regressor(train_df)
    preds = model.predict(calib_df[["rf_power_w", "pressure_mtorr", "electron_temperature_ev",
                                     "plasma_density_m3", "reactivity_index", "uniformity_index",
                                     "etch_rate_nm_min"]].to_numpy())
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_conformal_regressor_intervals_are_ordered(dataset) -> None:
    """lower <= point_estimate <= upper must hold for every row."""
    train_df, rest = random_split(dataset, test_fraction=0.4, seed=1)
    calib_df, test_df = random_split(rest, test_fraction=0.5, seed=2)
    model = train_defect_probability_regressor(train_df)
    conformal = conformalize_defect_regressor(model, calib_df, confidence_level=0.9)

    result = conformal.predict_interval(test_df)
    assert (result["lower"] <= result["point_estimate"]).all()
    assert (result["point_estimate"] <= result["upper"]).all()


def test_conformal_regressor_intervals_are_genuine_intervals(dataset) -> None:
    """FE-2.8.1 requires an interval, not a single point - width must be positive."""
    train_df, rest = random_split(dataset, test_fraction=0.4, seed=1)
    calib_df, test_df = random_split(rest, test_fraction=0.5, seed=2)
    model = train_defect_probability_regressor(train_df)
    conformal = conformalize_defect_regressor(model, calib_df, confidence_level=0.9)

    result = conformal.predict_interval(test_df.iloc[:20])
    assert ((result["upper"] - result["lower"]) > 0.0).all()


# ---------------------------------------------------------------------------
# Empirical coverage validation - the core "calibrated, not assumed" guarantee
# ---------------------------------------------------------------------------
def test_empirical_coverage_is_close_to_target(layer) -> None:
    """Both conformal predictors must achieve coverage reasonably close to their
    90% target on a genuinely held-out test set - verifying "calibrated" is a
    measured property, not an assumption. A generous band (>=75%) absorbs
    finite-calibration-set statistical noise while still catching a badly broken
    conformalization (e.g. wrong feature space, mismatched label encoding)."""
    coverage = layer.coverage
    assert coverage.target_confidence_level == pytest.approx(0.9)
    assert coverage.empirical_classification_coverage >= 0.75
    assert coverage.empirical_regression_coverage >= 0.75


def test_coverage_report_fields_in_valid_ranges(layer) -> None:
    coverage = layer.coverage
    assert 0.0 <= coverage.empirical_classification_coverage <= 1.0
    assert 0.0 <= coverage.empirical_regression_coverage <= 1.0
    assert coverage.mean_prediction_set_size >= 1.0
    assert coverage.mean_interval_width > 0.0


def test_evaluate_coverage_reproducible_via_direct_call(dataset) -> None:
    train_df, rest = random_split(dataset, test_fraction=0.4, seed=DEFAULT_SEED)
    calib_df, test_df = random_split(rest, test_fraction=0.5, seed=DEFAULT_SEED + 1)
    base = train_classifiers(train_df, seed=DEFAULT_SEED)[ClassifierKind.RANDOM_FOREST]
    conformal_clf = conformalize_classifier(base, calib_df, seed=DEFAULT_SEED)
    model = train_defect_probability_regressor(train_df, seed=DEFAULT_SEED)
    conformal_reg = conformalize_defect_regressor(model, calib_df)

    coverage = evaluate_coverage(conformal_clf, conformal_reg, test_df)
    assert coverage.empirical_classification_coverage >= 0.75


# ---------------------------------------------------------------------------
# Per-configuration trust assessment [FE-2.8.2]
# ---------------------------------------------------------------------------
def test_assess_configuration_returns_valid_prediction_set(layer) -> None:
    report = assess_configuration(150.0, 10.0, layer.classifier, layer.regressor)
    assert len(report.predicted_classes) >= 1
    assert set(report.predicted_classes).issubset(set(SUITABILITY_CLASSES))


def test_ambiguous_flag_matches_set_size(layer) -> None:
    for power, pressure in [(150.0, 10.0), (280.0, 1.5), (60.0, 19.0), (100.0, 5.0)]:
        report = assess_configuration(power, pressure, layer.classifier, layer.regressor)
        assert report.classification_is_ambiguous == (len(report.predicted_classes) > 1)


def test_low_confidence_flag_matches_ambiguity_or_wide_interval(layer) -> None:
    for power, pressure in [(150.0, 10.0), (280.0, 1.5), (60.0, 19.0), (100.0, 5.0)]:
        report = assess_configuration(power, pressure, layer.classifier, layer.regressor)
        expected = report.classification_is_ambiguous or (
            report.defect_interval_width > 0.3
        )
        assert report.is_low_confidence == expected


def test_clear_cut_high_power_low_pressure_is_unambiguous(layer) -> None:
    """Verified: 280W/1.5mTorr sits clearly in the Optimal regime, away from any
    class boundary, and should yield a single-class prediction set."""
    report = assess_configuration(280.0, 1.5, layer.classifier, layer.regressor)
    assert len(report.predicted_classes) == 1


def test_defect_interval_bounds_are_ordered_in_report(layer) -> None:
    report = assess_configuration(150.0, 10.0, layer.classifier, layer.regressor)
    assert report.defect_interval_lower <= report.defect_point_estimate <= report.defect_interval_upper
    assert report.defect_interval_width == pytest.approx(
        report.defect_interval_upper - report.defect_interval_lower
    )


def test_assess_configuration_deterministic_without_noise(layer) -> None:
    a = assess_configuration(150.0, 10.0, layer.classifier, layer.regressor)
    b = assess_configuration(150.0, 10.0, layer.classifier, layer.regressor)
    assert a.predicted_classes == b.predicted_classes
    assert a.defect_point_estimate == pytest.approx(b.defect_point_estimate)


def test_wide_interval_threshold_is_configurable(layer) -> None:
    permissive = assess_configuration(
        150.0, 10.0, layer.classifier, layer.regressor, wide_interval_threshold=100.0
    )
    strict = assess_configuration(
        150.0, 10.0, layer.classifier, layer.regressor, wide_interval_threshold=0.0
    )
    # a threshold of 0.0 forces the width check to trigger low-confidence.
    assert strict.is_low_confidence
    # a threshold of 100.0 means low_confidence can only come from ambiguity.
    assert permissive.is_low_confidence == permissive.classification_is_ambiguous


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------
def test_build_uncertainty_layer_returns_all_components(layer) -> None:
    assert layer.classifier is not None
    assert layer.regressor is not None
    assert layer.coverage is not None


def test_dataset_carries_the_defect_target_column_used_by_this_module(dataset) -> None:
    assert DEFECT_TARGET_COLUMN in dataset.columns
    assert dataset[DEFECT_TARGET_COLUMN].between(0.0, 1.0).all()
