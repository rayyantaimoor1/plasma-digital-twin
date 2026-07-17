"""Sub-Module 2.7 - AI Model Training, Evaluation, and Retraining Interface.

Wraps Sub-Module 2.1's classifiers with a model registry (FE-2.7.1), retraining
from accumulated stored datasets (FE-2.7.2), stratified k-fold cross-validation
(FE-2.7.3), a cross-model feature-importance comparison (FE-2.7.4), and MLflow
local-tracking of every training run (FE-2.7.5).

Design note on FE-2.7.2 ("retraining using accumulated session data"): this
platform has two different SQLite-backed record types, and only one of them
carries a genuine label. Sub-Module 1.4's `sessions` table logs individual
simulate() runs from ad hoc chamber-configuration experimentation - it has no
confounder-based suitability label unless this very classifier retroactively
assigns one, which would be circular (training on your own past predictions).
Sub-Module 1.3's `dataset_rows` table stores independently-generated, seeded
synthetic datasets that DO carry genuine confounded labels. "Accumulated session
data" is therefore implemented here as accumulated DATASET GENERATIONS (more
seeded runs of Sub-Module 1.3, each stored via `store_dataset_to_db`) concatenated
together - the honest way to let performance improve "as more experimental
records are generated" without reintroducing the label-circularity problem
Sub-Module 1.3 was built to eliminate in the first place.
"""
from __future__ import annotations

import os

# MLflow 3.x put the raw file-store backend into "maintenance mode" and refuses
# a plain file:// tracking URI unless explicitly opted back in (verified
# empirically - mlflow.set_experiment() otherwise raises MlflowException
# pointing at a SQLite-backed alternative). CLAUDE.md specifically calls for the
# classic local mlruns/ FILE store, so this must be set before any mlflow
# tracking call. Must happen at import time, before mlflow's tracking client
# is first constructed.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold

from ai_module.classification import (
    ClassifierKind,
    EvaluationMetrics,
    FullEvaluationReport,
    PlasmaClassifier,
    evaluate_classifier,
    global_feature_importance,
    run_full_evaluation,
    train_classifiers,
)
from digital_twin.dataset_generation import (
    DEFAULT_SEED,
    features_and_labels,
    load_dataset_from_db,
)
from digital_twin.session_manager import DEFAULT_DB_PATH

MLRUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "mlruns"
MLFLOW_EXPERIMENT_NAME = "plasma_suitability_classification"

# Params logged as-is must be simple, MLflow-serialisable scalars; anything else
# (e.g. an estimator or array-valued hyperparameter) is skipped rather than
# stringified unpredictably.
_LOGGABLE_PARAM_TYPES = (int, float, str, bool, type(None))


def _configure_mlflow(tracking_dir: Optional[Path] = None) -> None:
    """Point MLflow's classic tracking API at a local mlruns/ file store [FE-2.7.5].

    Looks up MLRUNS_DIR from the module namespace at CALL time rather than as a
    default parameter value (which would bind at function-definition time and
    silently ignore any later override of the module-level constant, e.g. in tests).
    """
    if tracking_dir is None:
        tracking_dir = MLRUNS_DIR
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# Model registry [FE-2.7.1]
# ---------------------------------------------------------------------------
@dataclass
class ModelRegistryEntry:
    """One deployed classifier's training parameters, dataset size, timestamp,
    and latest performance metrics."""
    classifier: str
    hyperparameters: dict
    n_train_samples: int
    trained_at: str
    dataset_seed: Optional[int]
    metrics: Optional[EvaluationMetrics]
    mlflow_run_id: Optional[str]


class ModelRegistry:
    """In-application model management: what's currently deployed, how it was
    trained, and how well it performs [FE-2.7.1]."""

    def __init__(self) -> None:
        self._classifiers: dict[ClassifierKind, PlasmaClassifier] = {}
        self._entries: dict[ClassifierKind, ModelRegistryEntry] = {}

    def register(
        self,
        kind: ClassifierKind,
        classifier: PlasmaClassifier,
        n_train_samples: int,
        dataset_seed: Optional[int] = None,
        metrics: Optional[EvaluationMetrics] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        self._classifiers[kind] = classifier
        self._entries[kind] = ModelRegistryEntry(
            classifier=kind.value,
            hyperparameters=classifier.base_model.get_params(),
            n_train_samples=n_train_samples,
            trained_at=datetime.now(timezone.utc).isoformat(),
            dataset_seed=dataset_seed,
            metrics=metrics,
            mlflow_run_id=mlflow_run_id,
        )

    def get_classifier(self, kind: ClassifierKind) -> PlasmaClassifier:
        return self._classifiers[kind]

    def get_entry(self, kind: ClassifierKind) -> ModelRegistryEntry:
        return self._entries[kind]

    def deployed_kinds(self) -> list[ClassifierKind]:
        return list(self._entries.keys())

    def summary_table(self) -> pd.DataFrame:
        """Tabular view for the model management interface [FE-2.7.1]."""
        rows = []
        for entry in self._entries.values():
            row = {
                "classifier": entry.classifier,
                "n_train_samples": entry.n_train_samples,
                "trained_at": entry.trained_at,
                "dataset_seed": entry.dataset_seed,
                "mlflow_run_id": entry.mlflow_run_id,
            }
            if entry.metrics is not None:
                row.update({
                    "eval_split": entry.metrics.split_name,
                    "accuracy": entry.metrics.accuracy,
                    "precision_macro": entry.metrics.precision_macro,
                    "recall_macro": entry.metrics.recall_macro,
                    "f1_macro": entry.metrics.f1_macro,
                })
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training orchestration with MLflow logging [FE-2.7.5]
# ---------------------------------------------------------------------------
def _log_params(params: dict) -> None:
    for name, value in params.items():
        if isinstance(value, _LOGGABLE_PARAM_TYPES):
            mlflow.log_param(name, value)


def train_and_log(
    dataset: pd.DataFrame,
    registry: ModelRegistry,
    seed: int = DEFAULT_SEED,
    dataset_seed: Optional[int] = None,
    log_to_mlflow: bool = True,
) -> FullEvaluationReport:
    """Train all three classifiers, evaluate on both the region and random
    splits (reusing Sub-Module 2.1's `run_full_evaluation`), log every run's
    hyperparameters and metrics to MLflow (FE-2.7.5), and register each
    classifier in the model registry (FE-2.7.1).

    The RANDOM-split-trained classifier is what gets registered as "deployed"
    per kind: the region-split-trained classifier exists specifically to measure
    extrapolation to the held-out hardest corner and was deliberately not
    trained on it, so it is evaluated but not deployed.
    """
    report = run_full_evaluation(dataset, seed=seed)

    def random_metrics(kind: ClassifierKind) -> EvaluationMetrics:
        return next(m for m in report.metrics if m.classifier == kind.value and m.split_name == "random")

    run_ids: dict[ClassifierKind, Optional[str]] = {kind: None for kind in ClassifierKind}

    if log_to_mlflow:
        _configure_mlflow()
        run_name = f"training_{datetime.now(timezone.utc).isoformat()}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("dataset_seed", dataset_seed)
            mlflow.log_param("n_samples", len(dataset))
            for kind in ClassifierKind:
                clf = report.classifiers_random[kind]
                metrics = random_metrics(kind)
                with mlflow.start_run(run_name=kind.value, nested=True) as child_run:
                    _log_params(clf.base_model.get_params())
                    mlflow.log_metric("accuracy", metrics.accuracy)
                    mlflow.log_metric("precision_macro", metrics.precision_macro)
                    mlflow.log_metric("recall_macro", metrics.recall_macro)
                    mlflow.log_metric("f1_macro", metrics.f1_macro)
                    run_ids[kind] = child_run.info.run_id

    for kind in ClassifierKind:
        clf = report.classifiers_random[kind]
        metrics = random_metrics(kind)
        n_train = len(dataset) - metrics.n_test_samples
        registry.register(
            kind, clf, n_train_samples=n_train, dataset_seed=dataset_seed,
            metrics=metrics, mlflow_run_id=run_ids[kind],
        )

    return report


def training_history(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    """Query MLflow for the full history of past training runs - the queryable
    record FE-2.7.5 asks for, replacing ad hoc console logging."""
    _configure_mlflow()
    try:
        return mlflow.search_runs(experiment_names=[experiment_name])
    except mlflow.exceptions.MlflowException:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Retraining from accumulated stored datasets [FE-2.7.2]
# ---------------------------------------------------------------------------
def accumulate_stored_datasets(
    dataset_ids: list[str], db_path: Path | str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Load and concatenate multiple previously-generated, stored datasets into
    one larger accumulated training set (see module docstring for why this is
    the honest interpretation of "accumulated session data")."""
    if not dataset_ids:
        raise ValueError("dataset_ids must be non-empty.")
    frames = [load_dataset_from_db(did, db_path=db_path) for did in dataset_ids]
    return pd.concat(frames, ignore_index=True)


def retrain_from_accumulated_datasets(
    dataset_ids: list[str],
    registry: ModelRegistry,
    db_path: Path | str = DEFAULT_DB_PATH,
    seed: int = DEFAULT_SEED,
    log_to_mlflow: bool = True,
) -> FullEvaluationReport:
    """Retrain on the union of several stored dataset generations [FE-2.7.2]."""
    accumulated = accumulate_stored_datasets(dataset_ids, db_path=db_path)
    return train_and_log(
        accumulated, registry, seed=seed, dataset_seed=None, log_to_mlflow=log_to_mlflow
    )


# ---------------------------------------------------------------------------
# k-fold cross-validation [FE-2.7.3]
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    classifier: str
    fold_index: int
    accuracy: float
    f1_macro: float
    n_test_samples: int


@dataclass
class CrossValidationReport:
    k: int
    fold_results: list[FoldResult]

    def aggregate_table(self) -> pd.DataFrame:
        """Mean/std accuracy and F1 per classifier across folds, alongside the
        fold-by-fold detail in `fold_results` [FE-2.7.3]."""
        rows = [
            {"classifier": f.classifier, "accuracy": f.accuracy, "f1_macro": f.f1_macro}
            for f in self.fold_results
        ]
        df = pd.DataFrame(rows)
        return (
            df.groupby("classifier")
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_f1_macro=("f1_macro", "mean"),
                std_f1_macro=("f1_macro", "std"),
            )
            .reset_index()
        )


def run_cross_validation(
    dataset: pd.DataFrame, k: int = 5, seed: int = DEFAULT_SEED
) -> CrossValidationReport:
    """Stratified k-fold cross-validation for all three classifiers [FE-2.7.3].

    Stratified (not plain k-fold) so every fold keeps roughly the overall class
    balance despite the 4-class, not-perfectly-even label distribution.
    """
    if k < 2:
        raise ValueError("k must be at least 2.")
    X, y = features_and_labels(dataset)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_results: list[FoldResult] = []
    for fold_index, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_fold = dataset.iloc[train_idx].reset_index(drop=True)
        test_fold = dataset.iloc[test_idx].reset_index(drop=True)
        classifiers = train_classifiers(train_fold, seed=seed)
        X_test, y_test = features_and_labels(test_fold)
        for kind, clf in classifiers.items():
            metrics = evaluate_classifier(clf, X_test, y_test, split_name=f"fold_{fold_index}")
            fold_results.append(FoldResult(
                classifier=kind.value,
                fold_index=fold_index,
                accuracy=metrics.accuracy,
                f1_macro=metrics.f1_macro,
                n_test_samples=metrics.n_test_samples,
            ))
    return CrossValidationReport(k=k, fold_results=fold_results)


# ---------------------------------------------------------------------------
# Cross-model feature importance comparison [FE-2.7.4]
# ---------------------------------------------------------------------------
def feature_importance_comparison_plot(
    classifiers: dict[ClassifierKind, PlasmaClassifier],
    X: pd.DataFrame,
    background: pd.DataFrame,
) -> go.Figure:
    """Grouped bar chart comparing feature importance across Random Forest,
    XGBoost, and the logistic-regression baseline [FE-2.7.4].

    Uses the SAME SHAP-based importance measure (Sub-Module 2.1's
    `global_feature_importance`) for all three models rather than mixing each
    library's own native importance metric (Gini / gain / raw coefficient
    magnitude, which live on different, non-comparable scales) - this is what
    makes the three bars actually comparable to each other.
    """
    fig = go.Figure()
    for kind, clf in classifiers.items():
        importance = global_feature_importance(clf, X, background)
        fig.add_trace(go.Bar(x=importance.index.tolist(), y=importance.values.tolist(), name=kind.value))
    fig.update_layout(
        barmode="group",
        title="Sub-Module 2.7: feature importance comparison (mean |SHAP value|)",
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP value|",
    )
    return fig


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.training
    from digital_twin.dataset_generation import generate_dataset

    print("Generating dataset and training all classifiers...")
    df = generate_dataset()
    registry = ModelRegistry()
    train_and_log(df, registry, dataset_seed=None)

    print("\nModel registry:")
    print(registry.summary_table().to_string(index=False))

    print("\n5-fold cross-validation:")
    cv_report = run_cross_validation(df, k=5)
    print(cv_report.aggregate_table().to_string(index=False))

    print(f"\nTraining history logged to {MLRUNS_DIR}:")
    print(training_history()[["run_id", "status", "params.n_samples"]].to_string(index=False))
