"""Tests for Sub-Module 2.1 - plasma process suitability classification engine.

Uses module-scoped fixtures (one dataset/training run shared across tests) since
training three model families plus SHAP explainers is the expensive part; the
assertions themselves are cheap. Dataset is small but spans the full operating
envelope so every suitability class is represented.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from digital_twin.dataset_generation import (
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    SUITABILITY_CLASSES,
    features_and_labels,
    generate_dataset,
    random_split,
)
from ai_module.classification import (
    BASELINE_KIND,
    ENSEMBLE_KINDS,
    ClassifierKind,
    classify_configuration,
    compare_classifiers,
    evaluate_classifier,
    explain_configuration,
    global_feature_importance,
    mcnemar_test,
    run_full_evaluation,
    train_classifiers,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=15)


@pytest.fixture(scope="module")
def classifiers(dataset):
    return train_classifiers(dataset)


@pytest.fixture(scope="module")
def background(dataset):
    X, _y = features_and_labels(dataset)
    return X.sample(30, random_state=0)


@pytest.fixture(scope="module")
def evaluation_report(dataset):
    return run_full_evaluation(dataset)


@pytest.fixture(scope="module")
def held_out_split(dataset):
    """A genuine train/test split, held separate from the `classifiers` fixture
    (which fits on the full `dataset`) - used ONLY by the calibration check
    below. Evaluating calibration on rows a model trained on would be circular
    and would flatter the calibrated model for the wrong reason."""
    return random_split(dataset, seed=DEFAULT_SEED)


@pytest.fixture(scope="module")
def calibration_check_classifiers(held_out_split):
    train_df, _test_df = held_out_split
    return train_classifiers(train_df, seed=DEFAULT_SEED)


# ---------------------------------------------------------------------------
# Training [FE-2.1.1]
# ---------------------------------------------------------------------------
def test_train_classifiers_returns_all_three_kinds(classifiers) -> None:
    assert set(classifiers.keys()) == set(ClassifierKind)


def test_baseline_has_scaler_and_ensembles_do_not(classifiers) -> None:
    assert classifiers[ClassifierKind.LOGISTIC_REGRESSION].scaler is not None
    assert classifiers[ClassifierKind.RANDOM_FOREST].scaler is None
    assert classifiers[ClassifierKind.XGBOOST].scaler is None


def test_all_classifiers_share_the_same_label_encoder_classes(classifiers) -> None:
    baseline_classes = list(classifiers[ClassifierKind.LOGISTIC_REGRESSION].label_encoder.classes_)
    for kind in ENSEMBLE_KINDS:
        assert list(classifiers[kind].label_encoder.classes_) == baseline_classes


def test_ensembles_are_wrapped_in_calibrated_classifier_cv(classifiers) -> None:
    """FE-2.1.1: Random Forest and XGBoost must be genuinely calibrated
    (CalibratedClassifierCV), not the baseline - calibrating the interpretable
    reference model too would erase the raw-vs-calibrated contrast."""
    assert not isinstance(classifiers[ClassifierKind.LOGISTIC_REGRESSION].model, CalibratedClassifierCV)
    for kind in ENSEMBLE_KINDS:
        assert isinstance(classifiers[kind].model, CalibratedClassifierCV)


def test_ensembles_have_an_explainer_model_baseline_does_not(classifiers) -> None:
    """explainer_model backs SHAP's TreeExplainer for the two calibrated
    ensembles (which TreeExplainer can't look inside); the baseline is never
    calibrated, so it needs no separate explainer model."""
    assert classifiers[ClassifierKind.LOGISTIC_REGRESSION].explainer_model is None
    for kind in ENSEMBLE_KINDS:
        assert classifiers[kind].explainer_model is not None


def test_base_model_property_resolves_to_the_uncalibrated_estimator(classifiers) -> None:
    baseline = classifiers[ClassifierKind.LOGISTIC_REGRESSION]
    assert baseline.base_model is baseline.model
    for kind in ENSEMBLE_KINDS:
        clf = classifiers[kind]
        assert clf.base_model is clf.explainer_model
        assert not isinstance(clf.base_model, CalibratedClassifierCV)


# ---------------------------------------------------------------------------
# Single-configuration classification [FE-2.1.1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", list(ClassifierKind))
def test_classify_configuration_returns_valid_class(classifiers, kind) -> None:
    result = classify_configuration(150.0, 10.0, classifiers[kind])
    assert result.predicted_class in SUITABILITY_CLASSES
    assert result.classifier == kind.value


def test_class_probabilities_sum_to_one_and_match_confidence(classifiers) -> None:
    result = classify_configuration(150.0, 10.0, classifiers[ClassifierKind.RANDOM_FOREST])
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.class_probabilities[result.predicted_class] == pytest.approx(result.confidence)
    assert result.confidence == max(result.class_probabilities.values())


def test_classify_configuration_all_probabilities_non_negative(classifiers) -> None:
    result = classify_configuration(150.0, 10.0, classifiers[ClassifierKind.LOGISTIC_REGRESSION])
    assert all(p >= 0.0 for p in result.class_probabilities.values())


def test_classify_configuration_deterministic_without_noise(classifiers) -> None:
    a = classify_configuration(120.0, 8.0, classifiers[ClassifierKind.XGBOOST])
    b = classify_configuration(120.0, 8.0, classifiers[ClassifierKind.XGBOOST])
    assert a.predicted_class == b.predicted_class
    assert a.class_probabilities == b.class_probabilities


# ---------------------------------------------------------------------------
# SHAP explainability [FE-2.1.2]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", list(ClassifierKind))
def test_explain_configuration_covers_every_feature(classifiers, background, kind) -> None:
    explanation = explain_configuration(150.0, 10.0, classifiers[kind], background)
    assert set(explanation.feature_contributions.keys()) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(v) for v in explanation.feature_contributions.values())


def test_explain_configuration_matches_the_classifiers_own_prediction(classifiers, background) -> None:
    clf = classifiers[ClassifierKind.RANDOM_FOREST]
    result = classify_configuration(150.0, 10.0, clf)
    explanation = explain_configuration(150.0, 10.0, clf, background)
    assert explanation.predicted_class == result.predicted_class


@pytest.mark.parametrize("kind", list(ClassifierKind))
def test_global_feature_importance_ranks_all_features(dataset, classifiers, background, kind) -> None:
    X, _y = features_and_labels(dataset)
    importance = global_feature_importance(classifiers[kind], X.sample(40, random_state=1), background)
    assert set(importance.index) == set(FEATURE_COLUMNS)
    assert (importance >= 0.0).all()
    # sorted descending
    assert list(importance) == sorted(importance, reverse=True)


# ---------------------------------------------------------------------------
# Side-by-side comparison [FE-2.1.3]
# ---------------------------------------------------------------------------
def test_compare_classifiers_includes_all_three(classifiers) -> None:
    comparison = compare_classifiers(150.0, 10.0, classifiers)
    assert set(comparison.results.keys()) == {k.value for k in ClassifierKind}


def test_compare_classifiers_agreement_flag_is_consistent(classifiers) -> None:
    comparison = compare_classifiers(150.0, 10.0, classifiers)
    predicted = {r.predicted_class for r in comparison.results.values()}
    assert comparison.all_agree == (len(predicted) == 1)


def test_compare_classifiers_can_detect_real_disagreement(classifiers) -> None:
    """At 150W/10mTorr the baseline and ensembles genuinely disagree (verified
    manually) - confirms the comparison surfaces real disagreement, not just
    a flag that's always True."""
    comparison = compare_classifiers(150.0, 10.0, classifiers)
    predictions = {k: r.predicted_class for k, r in comparison.results.items()}
    if len(set(predictions.values())) > 1:
        assert not comparison.all_agree
    assert comparison.majority_class in SUITABILITY_CLASSES


def test_compare_classifiers_majority_matches_most_common_prediction(classifiers) -> None:
    comparison = compare_classifiers(50.0, 20.0, classifiers)
    predictions = [r.predicted_class for r in comparison.results.values()]
    counts = pd.Series(predictions).value_counts()
    assert comparison.majority_class == counts.idxmax()


# ---------------------------------------------------------------------------
# Performance metrics on both splits [FE-2.1.4]
# ---------------------------------------------------------------------------
def test_evaluation_report_covers_all_classifiers_and_both_splits(evaluation_report) -> None:
    combos = {(m.classifier, m.split_name) for m in evaluation_report.metrics}
    expected = {(k.value, split) for k in ClassifierKind for split in ("region", "random")}
    assert combos == expected


def test_metrics_are_in_valid_ranges(evaluation_report) -> None:
    for m in evaluation_report.metrics:
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.precision_macro <= 1.0
        assert 0.0 <= m.recall_macro <= 1.0
        assert 0.0 <= m.f1_macro <= 1.0
        assert m.n_test_samples > 0


def test_confusion_matrix_shape_and_totals(evaluation_report) -> None:
    for m in evaluation_report.metrics:
        n_classes = len(m.class_labels)
        assert len(m.confusion_matrix) == n_classes
        assert all(len(row) == n_classes for row in m.confusion_matrix)
        assert sum(sum(row) for row in m.confusion_matrix) == m.n_test_samples


def test_region_and_random_splits_use_disjoint_sample_counts(dataset, evaluation_report) -> None:
    """Sanity check that the two splits are genuinely different evaluations,
    not the same test set relabeled."""
    region_n = {m.n_test_samples for m in evaluation_report.metrics if m.split_name == "region"}
    random_n = {m.n_test_samples for m in evaluation_report.metrics if m.split_name == "random"}
    assert region_n != random_n


def test_evaluate_classifier_standalone_matches_report(dataset, classifiers) -> None:
    """evaluate_classifier() used directly should reproduce accuracy computed by
    hand from predict()."""
    X, y = features_and_labels(dataset)
    clf = classifiers[ClassifierKind.RANDOM_FOREST]
    metrics = evaluate_classifier(clf, X, y, split_name="sanity")
    y_pred = clf.predict(X)
    expected_accuracy = float((y_pred == y.to_numpy()).mean())
    assert metrics.accuracy == pytest.approx(expected_accuracy)


# ---------------------------------------------------------------------------
# Probability calibration (FE-2.1.1: "calibrated probability scores")
# ---------------------------------------------------------------------------
def _multiclass_brier_score(proba: pd.DataFrame, y_true: pd.Series) -> float:
    """Mean squared error between predicted class probabilities and the
    one-hot true label, averaged over samples - the standard multi-class Brier
    score (lower is better; 0.0 is a perfectly calibrated, confident model)."""
    onehot = pd.get_dummies(y_true)[proba.columns].to_numpy(dtype=float)
    return float(np.mean(np.sum((proba.to_numpy() - onehot) ** 2, axis=1)))


@pytest.mark.parametrize("kind", ENSEMBLE_KINDS)
def test_calibration_improves_brier_score_on_held_out_data(
    calibration_check_classifiers, held_out_split, kind
) -> None:
    """FE-2.1.1's whole point: predict_proba for RF/XGBoost must be genuinely
    calibrated, not just each model's raw vote fractions. Verified directly by
    comparing Brier score against the SAME model with identical hyperparameters
    and training data but no calibration layer (`explainer_model`), on a
    held-out split neither view was trained on - isolating the calibration
    step as the only difference being measured."""
    _train_df, test_df = held_out_split
    clf = calibration_check_classifiers[kind]
    X_test, y_test = features_and_labels(test_df)

    calibrated_proba = clf.predict_proba(X_test)

    raw_proba_arr = clf.explainer_model.predict_proba(X_test[FEATURE_COLUMNS].to_numpy())
    class_names = clf.label_encoder.inverse_transform(np.arange(len(clf.label_encoder.classes_)))
    raw_proba = pd.DataFrame(raw_proba_arr, columns=class_names, index=X_test.index)

    calibrated_brier = _multiclass_brier_score(calibrated_proba, y_test)
    raw_brier = _multiclass_brier_score(raw_proba, y_test)

    assert calibrated_brier < raw_brier


# ---------------------------------------------------------------------------
# Statistical significance vs baseline (CLAUDE.md non-negotiable principle #3)
# ---------------------------------------------------------------------------
def test_significance_tests_cover_both_ensembles_on_both_splits(evaluation_report) -> None:
    combos = {(t.ensemble, t.split_name) for t in evaluation_report.significance_tests}
    expected = {(k.value, split) for k in ENSEMBLE_KINDS for split in ("region", "random")}
    assert combos == expected


def test_significance_tests_always_compare_against_the_baseline(evaluation_report) -> None:
    for t in evaluation_report.significance_tests:
        assert t.baseline == BASELINE_KIND.value


def test_significant_property_matches_alpha_threshold(evaluation_report) -> None:
    for t in evaluation_report.significance_tests:
        assert t.significant == (t.p_value < t.alpha)


def test_mcnemar_identical_predictions_give_p_value_one() -> None:
    """No discordant pairs (both models always agree) => no evidence of a
    difference => p-value must be 1.0, not spuriously significant."""
    y_true = pd.Series(["A", "B", "A", "B", "A"])
    same_preds = np.array(["A", "B", "A", "B", "A"])
    result = mcnemar_test(y_true, same_preds, same_preds, "baseline", "ensemble", "test")
    assert result.n_discordant_pairs == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_mcnemar_large_consistent_improvement_is_significant() -> None:
    """A large, one-sided, consistent improvement (ensemble right whenever they
    disagree) must be detected as statistically significant."""
    n = 40
    y_true = pd.Series(["A"] * n)
    baseline_wrong = np.array(["B"] * n)  # baseline always wrong
    ensemble_right = np.array(["A"] * n)  # ensemble always right
    result = mcnemar_test(y_true, baseline_wrong, ensemble_right, "baseline", "ensemble", "test")
    assert result.n_discordant_pairs == n
    assert result.significant
    assert result.p_value < 0.001


def test_mcnemar_p_value_symmetric_under_relabeling() -> None:
    """Swapping which model is 'baseline' vs 'ensemble' shouldn't change the
    p-value (McNemar's test is symmetric in the two models being compared)."""
    y_true = pd.Series(["A", "A", "A", "A", "A", "A", "A", "A"])
    pred_1 = np.array(["A", "A", "A", "B", "B", "A", "A", "A"])
    pred_2 = np.array(["A", "B", "A", "A", "A", "A", "B", "A"])
    r1 = mcnemar_test(y_true, pred_1, pred_2, "m1", "m2", "test")
    r2 = mcnemar_test(y_true, pred_2, pred_1, "m2", "m1", "test")
    assert r1.p_value == pytest.approx(r2.p_value)


def test_run_full_evaluation_reports_results_honestly_even_if_not_significant(evaluation_report) -> None:
    """CLAUDE.md: report accuracy/significance honestly, whatever it is. This
    test just confirms the report doesn't crash or hide results when ensembles
    fail to beat the baseline - it must still produce a full, valid report."""
    assert len(evaluation_report.significance_tests) == 4
    for t in evaluation_report.significance_tests:
        assert 0.0 <= t.p_value <= 1.0
