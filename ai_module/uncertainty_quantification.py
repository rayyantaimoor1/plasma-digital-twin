"""Sub-Module 2.8 - Uncertainty Quantification and Trust Layer.

Presenting classification confidence and defect probability as single point
estimates overstates the certainty the underlying model actually has, especially
once Sub-Module 1.3's genuine hidden confounders are introduced. This sub-module
replaces point estimates with MAPIE-based CONFORMAL PREDICTION, which gives a
distribution-free, statistically guaranteed coverage property: at a chosen
confidence level (e.g. 90%), the predicted set/interval contains the true outcome
at least that often, verified empirically on held-out data (see `evaluate_coverage`
and the "verified, not assumed" numbers in this module's git history).

Two conformal predictors, both built via the standard split-conformal workflow
(train / calibrate / test - three genuinely disjoint splits, required for the
coverage guarantee to hold):

  1. Classification (FE-2.8.1): wraps one of Sub-Module 2.1's already-trained
     classifiers (Random Forest by default, per CLAUDE.md's preference for tree
     ensembles) with `mapie.classification.SplitConformalClassifier`. Instead of
     one predicted label, `predict_set` returns the SET of classes consistent
     with the target coverage - a set of size 1 is an unambiguous, high-confidence
     call; a set of size 2+ means the model genuinely cannot distinguish between
     those classes at this confidence level. That set size IS the confidence
     signal (FE-2.8.2).

  2. Defect-probability regression (FE-2.8.1): a RandomForestRegressor trained to
     predict Sub-Module 1.3's CONFOUNDED `true_defect_probability` from the
     visible features - a genuine regression task with real irreducible
     uncertainty from the hidden confounders (wall-temperature drift, electrode
     aging, gas purity), not a deterministic function of its inputs. Wrapped with
     `mapie.regression.SplitConformalRegressor` for a calibrated interval instead
     of a point estimate - directly extending Sub-Module 2.5's bootstrap-CI
     precursor with a genuinely calibrated method.

MAPIE's classic top-level `MapieClassifier`/`MapieRegressor` API (matching the
0.8-era floor originally listed for this project) was superseded by MAPIE 1.x's
`SplitConformalClassifier`/`SplitConformalRegressor` classes, verified directly
against the installed mapie==1.4.1 before writing this module (see git history) -
consistent with CLAUDE.md's own instruction to pin actual installed versions
rather than hand-copy stale floors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor

from ai_module.classification import ClassifierKind, PlasmaClassifier, train_classifiers
from digital_twin.dataset_generation import DEFAULT_SEED, FEATURE_COLUMNS, features_and_labels, random_split
from digital_twin.physics_engine import ChamberGeometry, DEFAULT_GEOMETRY, simulate

DEFAULT_CONFIDENCE_LEVEL = 0.9
# A defect-probability interval wider than this (on the 0-1 scale) is flagged as
# low-confidence for FE-2.8.2's explicit signal - a round, documented choice, not
# a literature figure (this scale is specific to this platform's own model).
DEFAULT_WIDE_INTERVAL_THRESHOLD = 0.3


def _prepared_features(classifier: PlasmaClassifier, X: pd.DataFrame) -> np.ndarray:
    """Feature array in the same space `classifier.model` was originally fit on -
    scaled for the logistic-regression baseline, raw for the tree ensembles -
    required so MAPIE's prefit conformalization sees consistent inputs."""
    arr = X[FEATURE_COLUMNS].to_numpy()
    return arr if classifier.scaler is None else classifier.scaler.transform(arr)


# ---------------------------------------------------------------------------
# Conformal classification [FE-2.8.1]
# ---------------------------------------------------------------------------
@dataclass
class ConformalClassifier:
    base: PlasmaClassifier
    mapie: SplitConformalClassifier
    confidence_level: float

    def predict_set(self, X: pd.DataFrame) -> list[list[str]]:
        """Calibrated prediction SET per row: the class names whose membership is
        guaranteed at >= confidence_level marginal coverage (empirically verified
        via `evaluate_coverage`, not merely assumed)."""
        _point, mask = self.mapie.predict_set(_prepared_features(self.base, X))
        classes = self.base.label_encoder.classes_
        return [
            [str(c) for c, included in zip(classes, mask[i, :, 0]) if included]
            for i in range(mask.shape[0])
        ]


def conformalize_classifier(
    classifier: PlasmaClassifier,
    calibration_df: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> ConformalClassifier:
    """Calibrate an already-trained Sub-Module 2.1 classifier for conformal
    prediction sets [FE-2.8.1]. `calibration_df` MUST be disjoint from the data
    the classifier was trained on - conformal prediction's coverage guarantee
    requires a genuinely held-out, exchangeable calibration set.
    """
    X_calib, y_calib = features_and_labels(calibration_df)
    y_calib_enc = classifier.label_encoder.transform(y_calib)
    mapie = SplitConformalClassifier(
        estimator=classifier.model, confidence_level=confidence_level,
        prefit=True, random_state=seed,
    )
    mapie.conformalize(_prepared_features(classifier, X_calib), y_calib_enc)
    return ConformalClassifier(base=classifier, mapie=mapie, confidence_level=confidence_level)


# ---------------------------------------------------------------------------
# Conformal defect-probability regression [FE-2.8.1]
# ---------------------------------------------------------------------------
DEFECT_TARGET_COLUMN = "true_defect_probability"


def train_defect_probability_regressor(
    train_df: pd.DataFrame, seed: int = DEFAULT_SEED
) -> RandomForestRegressor:
    """Train a regressor predicting the CONFOUNDED defect probability
    (Sub-Module 1.3's `true_defect_probability`) from the visible features - a
    genuine regression problem with real irreducible uncertainty from hidden
    confounders, exactly what conformal prediction is meant to quantify."""
    X = train_df[FEATURE_COLUMNS].to_numpy()
    y = train_df[DEFECT_TARGET_COLUMN].to_numpy()
    model = RandomForestRegressor(n_estimators=200, random_state=seed)
    model.fit(X, y)
    return model


@dataclass
class ConformalDefectRegressor:
    model: RandomForestRegressor
    mapie: SplitConformalRegressor
    confidence_level: float

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """Point estimate + calibrated [lower, upper] interval per row."""
        point, intervals = self.mapie.predict_interval(X[FEATURE_COLUMNS].to_numpy())
        return pd.DataFrame(
            {"point_estimate": point, "lower": intervals[:, 0, 0], "upper": intervals[:, 1, 0]},
            index=X.index,
        )


def conformalize_defect_regressor(
    model: RandomForestRegressor,
    calibration_df: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> ConformalDefectRegressor:
    """Calibrate a trained defect-probability regressor for conformal intervals
    [FE-2.8.1]. `calibration_df` must be disjoint from the regressor's training data."""
    X_calib = calibration_df[FEATURE_COLUMNS].to_numpy()
    y_calib = calibration_df[DEFECT_TARGET_COLUMN].to_numpy()
    mapie = SplitConformalRegressor(estimator=model, confidence_level=confidence_level, prefit=True)
    mapie.conformalize(X_calib, y_calib)
    return ConformalDefectRegressor(model=model, mapie=mapie, confidence_level=confidence_level)


# ---------------------------------------------------------------------------
# Empirical coverage validation - "calibrated" checked, not assumed [FE-2.8.1]
# ---------------------------------------------------------------------------
@dataclass
class CoverageReport:
    target_confidence_level: float
    empirical_classification_coverage: float
    empirical_regression_coverage: float
    mean_prediction_set_size: float
    mean_interval_width: float


def evaluate_coverage(
    conformal_classifier: ConformalClassifier,
    conformal_regressor: ConformalDefectRegressor,
    test_df: pd.DataFrame,
) -> CoverageReport:
    """Measure empirical coverage on a genuinely held-out test set (disjoint from
    both training and calibration data) - the honest check that "calibrated"
    means something measured, not an assumed property of using MAPIE [FE-2.8.1].
    """
    X_test, y_test = features_and_labels(test_df)
    pred_sets = conformal_classifier.predict_set(X_test)
    class_covered = [true_label in pred_set for true_label, pred_set in zip(y_test, pred_sets)]

    interval = conformal_regressor.predict_interval(X_test)
    y_true_defect = test_df[DEFECT_TARGET_COLUMN].to_numpy()
    reg_covered = (
        (y_true_defect >= interval["lower"].to_numpy())
        & (y_true_defect <= interval["upper"].to_numpy())
    )

    return CoverageReport(
        target_confidence_level=conformal_classifier.confidence_level,
        empirical_classification_coverage=float(np.mean(class_covered)),
        empirical_regression_coverage=float(reg_covered.mean()),
        mean_prediction_set_size=float(np.mean([len(s) for s in pred_sets])),
        mean_interval_width=float((interval["upper"] - interval["lower"]).mean()),
    )


# ---------------------------------------------------------------------------
# Per-configuration trust-layer assessment [FE-2.8.2]
# ---------------------------------------------------------------------------
@dataclass
class UncertaintyReport:
    """The explicit "confidence band" alongside a suitability prediction and a
    defect-probability estimate, so low-confidence configurations are visibly
    distinguished from high-confidence ones [FE-2.8.2]."""
    rf_power_w: float
    pressure_mtorr: float
    predicted_classes: list[str]        # conformal prediction set (>=1 class names)
    classification_is_ambiguous: bool   # True iff the set has more than one class
    defect_point_estimate: float
    defect_interval_lower: float
    defect_interval_upper: float
    defect_interval_width: float
    is_low_confidence: bool
    confidence_level: float


def assess_configuration(
    rf_power_w: float,
    pressure_mtorr: float,
    conformal_classifier: ConformalClassifier,
    conformal_regressor: ConformalDefectRegressor,
    wide_interval_threshold: float = DEFAULT_WIDE_INTERVAL_THRESHOLD,
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> UncertaintyReport:
    """Full trust-layer assessment for one submitted configuration: run the
    physics engine, then report a calibrated classification set and a calibrated
    defect-probability interval with an explicit low-confidence flag [FE-2.8.2]."""
    result = simulate(rf_power_w, pressure_mtorr, noise_level=noise_level, seed=seed, geometry=geometry)
    X = pd.DataFrame([{c: getattr(result, c) for c in FEATURE_COLUMNS}])

    predicted_classes = conformal_classifier.predict_set(X)[0]
    interval_row = conformal_regressor.predict_interval(X).iloc[0]
    width = float(interval_row["upper"] - interval_row["lower"])
    ambiguous = len(predicted_classes) > 1

    return UncertaintyReport(
        rf_power_w=rf_power_w, pressure_mtorr=pressure_mtorr,
        predicted_classes=predicted_classes, classification_is_ambiguous=ambiguous,
        defect_point_estimate=float(interval_row["point_estimate"]),
        defect_interval_lower=float(interval_row["lower"]),
        defect_interval_upper=float(interval_row["upper"]),
        defect_interval_width=width,
        is_low_confidence=(ambiguous or width > wide_interval_threshold),
        confidence_level=conformal_classifier.confidence_level,
    )


# ---------------------------------------------------------------------------
# End-to-end orchestration: train + calibrate + validate from one dataset
# ---------------------------------------------------------------------------
@dataclass
class UncertaintyLayer:
    classifier: ConformalClassifier
    regressor: ConformalDefectRegressor
    coverage: CoverageReport


def build_uncertainty_layer(
    dataset: pd.DataFrame,
    base_classifier_kind: ClassifierKind = ClassifierKind.RANDOM_FOREST,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> UncertaintyLayer:
    """Train, calibrate, and validate the full uncertainty-quantification trust
    layer from one dataset, via a genuine 3-way train/calibrate/test split
    (60/20/20) - the split conformal prediction methodology requires all three
    to be disjoint for the coverage guarantee to hold [FE-2.8.1]."""
    train_df, rest = random_split(dataset, test_fraction=0.4, seed=seed)
    calib_df, test_df = random_split(rest, test_fraction=0.5, seed=seed + 1)

    base = train_classifiers(train_df, seed=seed)[base_classifier_kind]
    conformal_clf = conformalize_classifier(base, calib_df, confidence_level, seed=seed)

    regressor = train_defect_probability_regressor(train_df, seed=seed)
    conformal_reg = conformalize_defect_regressor(regressor, calib_df, confidence_level)

    coverage = evaluate_coverage(conformal_clf, conformal_reg, test_df)
    return UncertaintyLayer(classifier=conformal_clf, regressor=conformal_reg, coverage=coverage)


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.uncertainty_quantification
    from digital_twin.dataset_generation import generate_dataset

    print("Generating dataset and building the conformal uncertainty layer...")
    df = generate_dataset()
    layer = build_uncertainty_layer(df)

    c = layer.coverage
    print(f"\nTarget confidence level: {c.target_confidence_level:.0%}")
    print(f"Empirical classification coverage: {c.empirical_classification_coverage:.1%}")
    print(f"Empirical regression coverage:     {c.empirical_regression_coverage:.1%}")
    print(f"Mean prediction set size: {c.mean_prediction_set_size:.2f}")
    print(f"Mean defect interval width: {c.mean_interval_width:.3f}")

    print("\nPer-configuration trust assessment:")
    for power, pressure in [(150.0, 10.0), (280.0, 1.5), (60.0, 19.0)]:
        report = assess_configuration(power, pressure, layer.classifier, layer.regressor)
        flag = "LOW CONFIDENCE" if report.is_low_confidence else "high confidence"
        print(
            f"  {power:5.0f} W / {pressure:4.1f} mTorr -> classes={report.predicted_classes} "
            f"defect={report.defect_point_estimate:.2f} "
            f"[{report.defect_interval_lower:.2f}, {report.defect_interval_upper:.2f}]  ({flag})"
        )
