"""Sub-Module 2.1 - Plasma Process Suitability Classification Engine.

Trains Random Forest and XGBoost classifiers, benchmarked against an interpretable
logistic-regression BASELINE, to predict plasma process suitability (Optimal /
Acceptable / Marginal / Unsuitable) from the observable features produced by
Sub-Module 1.3's synthetic dataset generator.

Per CLAUDE.md non-negotiable principle #3: classification success is measured as
a STATISTICALLY SIGNIFICANT improvement over the logistic-regression baseline
(McNemar's exact test via scipy.stats), not an absolute accuracy figure.
Whatever accuracy results, it is reported honestly - see `run_full_evaluation`.

Design notes:
  * All three models are trained on IDENTICAL data through a uniform interface
    (`PlasmaClassifier`), so the significance test compares model quality, not an
    accident of preprocessing. Labels are integer-encoded for all three (verified
    empirically that XGBoost 3.x rejects raw string labels - see git history for
    the verification script), and decoded back to class names on prediction.
  * Only the logistic-regression baseline is feature-scaled (StandardScaler);
    Random Forest and XGBoost are tree-based and scale-invariant, so scaling them
    would be a no-op that only adds a moot preprocessing step.
  * "Calibrated probability scores" (FE-2.1.1): Random Forest and XGBoost are
    each wrapped in CalibratedClassifierCV (isotonic, 5-fold) so predict_proba
    returns genuinely calibrated probabilities rather than raw tree-vote
    fractions - empirically verified to reduce Brier score vs the uncalibrated
    model (tests/test_classification.py). The logistic-regression BASELINE is
    left uncalibrated on purpose: as the interpretable reference model it should
    reflect what an off-the-shelf LogisticRegression actually outputs, and
    calibrating it too would erase the "raw vs. calibrated ensemble" contrast
    this exists to show. This is calibration of probability VALUES (reliability);
    Sub-Module 2.8's conformal prediction is a separate, complementary form of
    uncertainty quantification with a formal coverage guarantee on prediction
    SETS/intervals, not duplicated here.
  * SHAP explainer choice is model-specific (verified empirically per model type):
    TreeExplainer(feature_perturbation="tree_path_dependent") for Random Forest
    and XGBoost (the plain default explainer fails on XGBoost 3.x with a
    "categorical split not yet supported" error), and LinearExplainer on the
    SCALED features for the logistic-regression baseline (far faster than the
    generic permutation explainer, and exact for a linear model).
  * Evaluation follows FE-1.3.5's region/random split distinction throughout:
    a classifier trained and evaluated on the RANDOM split measures interpolation
    within seen operating regions; one trained and evaluated on the REGION split
    measures extrapolation to the held-out high-power/high-pressure corner. These
    are reported separately, never conflated into one figure (FE-2.1.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import shap
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

from digital_twin.dataset_generation import (
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    features_and_labels,
    random_split,
    region_based_split,
)
from digital_twin.physics_engine import ChamberGeometry, DEFAULT_GEOMETRY, simulate

SIGNIFICANCE_ALPHA = 0.05  # standard threshold for the McNemar significance test

# FE-2.1.1 calibration settings for the Random Forest / XGBoost predict_proba.
# Isotonic (a free-form monotonic fit) rather than sigmoid/Platt scaling, since
# each class has well over a hundred calibration samples in this project's
# datasets - isotonic needs more data than sigmoid but is not restricted to a
# parametric (logistic) miscalibration shape, so it is the better default once
# there is enough data to fit it, which there reliably is here.
CALIBRATION_METHOD = "isotonic"
# 5-fold: for each fold, CalibratedClassifierCV fits a fresh copy of the base
# estimator on the other four folds and calibrates it on the held-out fold, so
# no row is ever used to calibrate a probability the model saw during its own
# training - the leakage-free property the "genuinely calibrated" claim relies on.
CALIBRATION_CV_FOLDS = 5


class ClassifierKind(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


BASELINE_KIND = ClassifierKind.LOGISTIC_REGRESSION
ENSEMBLE_KINDS = (ClassifierKind.RANDOM_FOREST, ClassifierKind.XGBOOST)


@dataclass
class PlasmaClassifier:
    """Uniform wrapper around one trained model: label encoding, optional
    feature scaling (baseline only), and a model-appropriate SHAP explainer,
    all hidden behind the same interface regardless of which of the three
    underlying libraries is used [FE-2.1.1].

    `model` is what predict()/predict_proba() actually call. For Random Forest
    and XGBoost this is a CalibratedClassifierCV wrapper (isotonic, 5-fold), so
    predict_proba returns genuinely calibrated probabilities rather than raw
    tree-vote fractions. `explainer_model` - only set for those two - holds a
    plain (uncalibrated) estimator fit with the SAME hyperparameters and data,
    used exclusively by shap_values(): SHAP's TreeExplainer needs direct access
    to real tree structure, which it cannot get from CalibratedClassifierCV's
    internal fold-wise copies.
    """
    kind: ClassifierKind
    model: object  # sklearn/xgboost estimator, or CalibratedClassifierCV wrapper, already fit
    label_encoder: LabelEncoder
    scaler: Optional[StandardScaler]  # only set for the logistic-regression baseline
    explainer_model: Optional[object] = None  # RF/XGBoost only - raw model backing SHAP, see above

    @property
    def base_model(self) -> object:
        """The plain (uncalibrated) fitted estimator describing this
        classifier's actual hyperparameters - `explainer_model` for RF/XGBoost
        (whose `model` is a calibration wrapper), `model` itself for the
        never-calibrated baseline."""
        return self.explainer_model if self.explainer_model is not None else self.model

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[FEATURE_COLUMNS].to_numpy()
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return arr

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.model.predict(self._prepare(X))
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(self._prepare(X))
        class_names = self.label_encoder.inverse_transform(np.arange(len(self.label_encoder.classes_)))
        return pd.DataFrame(proba, columns=class_names, index=X.index)

    def shap_values(self, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
        """SHAP values for X, shape (n_samples, n_features, n_classes), in
        FEATURE_COLUMNS / label_encoder.classes_ order (verified empirically)."""
        if self.kind == ClassifierKind.LOGISTIC_REGRESSION:
            explainer = shap.LinearExplainer(self.model, self._prepare(background))
            return np.asarray(explainer(self._prepare(X)).values)
        explainer = shap.TreeExplainer(self.base_model, feature_perturbation="tree_path_dependent")
        return np.asarray(explainer(self._prepare(X)).values)


def _calibrated(estimator: object) -> CalibratedClassifierCV:
    """Wrap an unfit RF/XGBoost estimator for isotonic probability calibration
    [FE-2.1.1]. `cv=CALIBRATION_CV_FOLDS` makes CalibratedClassifierCV fit its
    own internal copies of `estimator` on each fold and calibrate each on that
    fold's held-out rows - scikit-learn's standard leakage-free pattern, so no
    separate calibration split needs to be threaded through this module."""
    return CalibratedClassifierCV(estimator, method=CALIBRATION_METHOD, cv=CALIBRATION_CV_FOLDS)


def train_classifiers(
    train_df: pd.DataFrame, seed: int = DEFAULT_SEED
) -> dict[ClassifierKind, PlasmaClassifier]:
    """Train the baseline and both ensemble classifiers on identical data [FE-2.1.1].

    Hyperparameters are modest, fixed defaults appropriate for a dataset of a few
    hundred to a few thousand rows (this project's scale) - not tuned per dataset,
    keeping the comparison about model FAMILY, not a hyperparameter search.

    Random Forest and XGBoost are each fit TWICE with IDENTICAL hyperparameters:
    once as a plain estimator on all of `train_df` (kept as `explainer_model`,
    used only by SHAP - TreeExplainer needs real tree structure, which it can't
    reach through a CalibratedClassifierCV wrapper), and once wrapped in
    CalibratedClassifierCV for genuinely calibrated predict_proba output. Both
    fits see the same data and hyperparameters, so this is one predictive model
    exposed through two views, not two models that could disagree.
    """
    X, y = features_and_labels(train_df)
    X_arr = X[FEATURE_COLUMNS].to_numpy()
    label_encoder = LabelEncoder().fit(y)
    y_enc = label_encoder.transform(y)

    # Fit on plain arrays (matching _prepare()'s numpy conversion at predict time)
    # so scikit-learn doesn't warn about a fit-time/predict-time feature-name mismatch.
    scaler = StandardScaler().fit(X_arr)
    baseline = LogisticRegression(max_iter=2000, random_state=seed)
    baseline.fit(scaler.transform(X_arr), y_enc)

    rf_kwargs = dict(n_estimators=200, random_state=seed)
    rf_explainer = RandomForestClassifier(**rf_kwargs).fit(X_arr, y_enc)
    rf_calibrated = _calibrated(RandomForestClassifier(**rf_kwargs)).fit(X_arr, y_enc)

    xgb_kwargs = dict(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed,
        eval_metric="mlogloss",
    )
    xgb_explainer = xgb.XGBClassifier(**xgb_kwargs).fit(X_arr, y_enc)
    xgb_calibrated = _calibrated(xgb.XGBClassifier(**xgb_kwargs)).fit(X_arr, y_enc)

    return {
        ClassifierKind.LOGISTIC_REGRESSION: PlasmaClassifier(
            BASELINE_KIND, baseline, label_encoder, scaler
        ),
        ClassifierKind.RANDOM_FOREST: PlasmaClassifier(
            ClassifierKind.RANDOM_FOREST, rf_calibrated, label_encoder, None,
            explainer_model=rf_explainer,
        ),
        ClassifierKind.XGBOOST: PlasmaClassifier(
            ClassifierKind.XGBOOST, xgb_calibrated, label_encoder, None,
            explainer_model=xgb_explainer,
        ),
    }


# ---------------------------------------------------------------------------
# Single-configuration classification [FE-2.1.1]
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    classifier: str
    predicted_class: str
    class_probabilities: dict[str, float]
    confidence: float  # probability assigned to the predicted class


def _features_for_configuration(
    rf_power_w: float,
    pressure_mtorr: float,
    noise_level: float,
    seed: Optional[int],
    geometry: ChamberGeometry,
) -> pd.DataFrame:
    result = simulate(rf_power_w, pressure_mtorr, noise_level=noise_level, seed=seed, geometry=geometry)
    row = {col: getattr(result, col) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def classify_configuration(
    rf_power_w: float,
    pressure_mtorr: float,
    classifier: PlasmaClassifier,
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> ClassificationResult:
    """Run the physics engine for one (power, pressure) configuration and classify
    the resulting observable features [FE-2.1.1]. Mirrors the real pipeline: a
    submitted configuration is simulated, and the classifier sees only the
    resulting observables, never the confounders (Sub-Module 1.3's contract)."""
    X = _features_for_configuration(rf_power_w, pressure_mtorr, noise_level, seed, geometry)
    proba = classifier.predict_proba(X).iloc[0]
    predicted = proba.idxmax()
    return ClassificationResult(
        classifier=classifier.kind.value,
        predicted_class=predicted,
        class_probabilities=proba.to_dict(),
        confidence=float(proba[predicted]),
    )


# ---------------------------------------------------------------------------
# SHAP explainability [FE-2.1.2]
# ---------------------------------------------------------------------------
@dataclass
class ShapExplanation:
    classifier: str
    predicted_class: str
    feature_contributions: dict[str, float]  # SHAP value per feature, for the predicted class


def explain_configuration(
    rf_power_w: float,
    pressure_mtorr: float,
    classifier: PlasmaClassifier,
    background: pd.DataFrame,
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> ShapExplanation:
    """Per-prediction SHAP breakdown: each feature's contribution to THIS
    configuration's predicted class [FE-2.1.2]. `background` should be a sample
    of training features (e.g. a few dozen rows) used as the SHAP reference
    distribution."""
    X = _features_for_configuration(rf_power_w, pressure_mtorr, noise_level, seed, geometry)
    result = classify_configuration(rf_power_w, pressure_mtorr, classifier, noise_level, seed, geometry)

    # LabelEncoder.transform gives the encoded index directly (classes_ is sorted,
    # and both predict_proba's columns and SHAP's class axis follow that same order).
    class_idx = int(classifier.label_encoder.transform([result.predicted_class])[0])
    values = classifier.shap_values(X, background)[0, :, class_idx]
    return ShapExplanation(
        classifier=classifier.kind.value,
        predicted_class=result.predicted_class,
        feature_contributions=dict(zip(FEATURE_COLUMNS, values.tolist())),
    )


def global_feature_importance(classifier: PlasmaClassifier, X: pd.DataFrame, background: pd.DataFrame) -> pd.Series:
    """Global feature importance ranking: mean |SHAP value| across all samples
    and all classes, sorted descending [FE-2.1.2]."""
    values = classifier.shap_values(X, background)  # (n_samples, n_features, n_classes)
    importance = np.abs(values).mean(axis=(0, 2))
    return pd.Series(importance, index=FEATURE_COLUMNS).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Side-by-side model comparison [FE-2.1.3]
# ---------------------------------------------------------------------------
@dataclass
class ClassifierComparison:
    rf_power_w: float
    pressure_mtorr: float
    results: dict[str, ClassificationResult]
    all_agree: bool
    majority_class: str


def compare_classifiers(
    rf_power_w: float,
    pressure_mtorr: float,
    classifiers: dict[ClassifierKind, PlasmaClassifier],
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> ClassifierComparison:
    """Classify the same configuration with every model, flagging agreement
    or disagreement between them [FE-2.1.3]."""
    results = {
        kind.value: classify_configuration(rf_power_w, pressure_mtorr, clf, noise_level, seed, geometry)
        for kind, clf in classifiers.items()
    }
    predictions = [r.predicted_class for r in results.values()]
    majority_class = pd.Series(predictions).mode().iloc[0]
    return ClassifierComparison(
        rf_power_w=rf_power_w,
        pressure_mtorr=pressure_mtorr,
        results=results,
        all_agree=len(set(predictions)) == 1,
        majority_class=majority_class,
    )


# ---------------------------------------------------------------------------
# Performance metrics [FE-2.1.4]
# ---------------------------------------------------------------------------
@dataclass
class EvaluationMetrics:
    classifier: str
    split_name: str  # "region" (extrapolation) or "random" (interpolation)
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    n_test_samples: int


def evaluate_classifier(
    classifier: PlasmaClassifier, X_test: pd.DataFrame, y_test: pd.Series, split_name: str
) -> EvaluationMetrics:
    """Accuracy, macro precision/recall/F1, and confusion matrix for one
    classifier on one test set [FE-2.1.4]. Macro-averaged because the four
    suitability classes are treated as equally important, not weighted by size."""
    y_pred = classifier.predict(X_test)
    class_labels = list(classifier.label_encoder.classes_)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    accuracy = float((y_pred == y_test.to_numpy()).mean())
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    return EvaluationMetrics(
        classifier=classifier.kind.value,
        split_name=split_name,
        accuracy=accuracy,
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        confusion_matrix=cm.tolist(),
        class_labels=class_labels,
        n_test_samples=len(y_test),
    )


# ---------------------------------------------------------------------------
# Statistical significance vs baseline (CLAUDE.md non-negotiable principle #3)
# ---------------------------------------------------------------------------
@dataclass
class SignificanceTestResult:
    """McNemar's exact test result comparing one ensemble model against the
    logistic-regression baseline on the SAME test set [BO-2]."""
    baseline: str
    ensemble: str
    split_name: str
    baseline_accuracy: float
    ensemble_accuracy: float
    n_discordant_pairs: int
    p_value: float
    alpha: float = SIGNIFICANCE_ALPHA

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def mcnemar_test(
    y_true: pd.Series,
    y_pred_baseline: np.ndarray,
    y_pred_ensemble: np.ndarray,
    baseline_name: str,
    ensemble_name: str,
    split_name: str,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> SignificanceTestResult:
    """Exact McNemar's test via scipy.stats.binomtest on the discordant pairs
    (CLAUDE.md principle #3). Correct/incorrect per sample is a valid, standard
    binarization of a multi-class prediction for this test.

    b = baseline correct, ensemble wrong; c = baseline wrong, ensemble correct.
    Under the null (no true difference), each discordant pair is equally likely
    to favour either model, so binomtest(c, b+c, 0.5) is the exact McNemar p-value
    - no chi-square/continuity-correction approximation needed.
    """
    y_true_arr = y_true.to_numpy()
    correct_baseline = y_pred_baseline == y_true_arr
    correct_ensemble = y_pred_ensemble == y_true_arr

    b = int((correct_baseline & ~correct_ensemble).sum())
    c = int((~correct_baseline & correct_ensemble).sum())
    n = b + c
    p_value = 1.0 if n == 0 else binomtest(c, n, 0.5, alternative="two-sided").pvalue

    return SignificanceTestResult(
        baseline=baseline_name,
        ensemble=ensemble_name,
        split_name=split_name,
        baseline_accuracy=float(correct_baseline.mean()),
        ensemble_accuracy=float(correct_ensemble.mean()),
        n_discordant_pairs=n,
        p_value=float(p_value),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Full dual-split evaluation, orchestrating FE-2.1.4 + principle #3
# ---------------------------------------------------------------------------
@dataclass
class FullEvaluationReport:
    metrics: list[EvaluationMetrics]
    significance_tests: list[SignificanceTestResult]
    classifiers_region: dict[ClassifierKind, PlasmaClassifier]
    classifiers_random: dict[ClassifierKind, PlasmaClassifier]


def run_full_evaluation(dataset: pd.DataFrame, seed: int = DEFAULT_SEED) -> FullEvaluationReport:
    """Train and evaluate all three classifiers on BOTH the region-based
    (extrapolation) and random (interpolation) splits, and run the McNemar
    significance test for each ensemble model against the baseline on each
    split [FE-2.1.4, principle #3].

    A classifier trained on the region-split TRAIN set is evaluated only on the
    region-split TEST set (never mixed with the random split), so "interpolation
    vs extrapolation" measures a genuinely different generalisation question
    rather than one model evaluated on two arbitrarily different test sets.
    """
    train_region, test_region = region_based_split(dataset)
    train_random, test_random = random_split(dataset, seed=seed)

    classifiers_region = train_classifiers(train_region, seed=seed)
    classifiers_random = train_classifiers(train_random, seed=seed)

    X_test_region, y_test_region = features_and_labels(test_region)
    X_test_random, y_test_random = features_and_labels(test_random)

    metrics: list[EvaluationMetrics] = []
    predictions_region: dict[ClassifierKind, np.ndarray] = {}
    predictions_random: dict[ClassifierKind, np.ndarray] = {}

    for kind, clf in classifiers_region.items():
        y_pred = clf.predict(X_test_region)
        predictions_region[kind] = y_pred
        metrics.append(evaluate_classifier(clf, X_test_region, y_test_region, split_name="region"))

    for kind, clf in classifiers_random.items():
        y_pred = clf.predict(X_test_random)
        predictions_random[kind] = y_pred
        metrics.append(evaluate_classifier(clf, X_test_random, y_test_random, split_name="random"))

    significance_tests: list[SignificanceTestResult] = []
    for ensemble_kind in ENSEMBLE_KINDS:
        significance_tests.append(mcnemar_test(
            y_test_region, predictions_region[BASELINE_KIND], predictions_region[ensemble_kind],
            BASELINE_KIND.value, ensemble_kind.value, "region",
        ))
        significance_tests.append(mcnemar_test(
            y_test_random, predictions_random[BASELINE_KIND], predictions_random[ensemble_kind],
            BASELINE_KIND.value, ensemble_kind.value, "random",
        ))

    return FullEvaluationReport(
        metrics=metrics,
        significance_tests=significance_tests,
        classifiers_region=classifiers_region,
        classifiers_random=classifiers_random,
    )


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.classification
    from digital_twin.dataset_generation import generate_dataset

    print("Generating dataset...")
    df = generate_dataset()
    print(f"{len(df)} samples\n")

    report = run_full_evaluation(df)
    print("Performance metrics (accuracy, macro F1) by classifier and split:")
    for m in report.metrics:
        print(
            f"  {m.classifier:20s} [{m.split_name:6s}]  "
            f"acc={m.accuracy:.3f}  f1={m.f1_macro:.3f}  n={m.n_test_samples}"
        )

    print("\nSignificance vs logistic-regression baseline (McNemar's exact test):")
    for t in report.significance_tests:
        verdict = "SIGNIFICANT" if t.significant else "not significant"
        print(
            f"  {t.ensemble:15s} vs {t.baseline:20s} [{t.split_name:6s}]  "
            f"acc {t.ensemble_accuracy:.3f} vs {t.baseline_accuracy:.3f}  "
            f"p={t.p_value:.4f}  ({verdict})"
        )
