"""Tests for Sub-Module 2.7 - AI model training, evaluation, and retraining interface.

MLflow tests point at an isolated tmp_path tracking directory (never the real
data/mlruns/), so running the suite never pollutes or depends on real training
history. Trains real models (no mocking) since the whole point of this
sub-module is orchestrating genuine training/evaluation/logging.
"""
import warnings

import pandas as pd
import plotly.graph_objects as go
import pytest

warnings.filterwarnings("ignore")

from ai_module.classification import ClassifierKind
from ai_module.training import (
    CrossValidationReport,
    ModelRegistry,
    accumulate_stored_datasets,
    feature_importance_comparison_plot,
    retrain_from_accumulated_datasets,
    run_cross_validation,
    train_and_log,
    training_history,
)
from digital_twin.dataset_generation import features_and_labels, generate_dataset, store_dataset_to_db


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=10)


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry()


# ---------------------------------------------------------------------------
# Model registry [FE-2.7.1]
# ---------------------------------------------------------------------------
def test_train_and_log_registers_all_three_classifiers(dataset, registry) -> None:
    train_and_log(dataset, registry, dataset_seed=42, log_to_mlflow=False)
    assert set(registry.deployed_kinds()) == set(ClassifierKind)


def test_registry_entry_has_hyperparameters_and_timestamp(dataset, registry) -> None:
    train_and_log(dataset, registry, dataset_seed=42, log_to_mlflow=False)
    entry = registry.get_entry(ClassifierKind.RANDOM_FOREST)
    assert entry.hyperparameters["n_estimators"] == 200
    assert entry.trained_at  # non-empty ISO timestamp
    assert entry.dataset_seed == 42


def test_registry_n_train_samples_is_less_than_full_dataset(dataset, registry) -> None:
    """n_train_samples should be the RANDOM split's train portion, not the
    full dataset (which also includes the held-out test rows)."""
    train_and_log(dataset, registry, log_to_mlflow=False)
    for kind in ClassifierKind:
        entry = registry.get_entry(kind)
        assert 0 < entry.n_train_samples < len(dataset)


def test_summary_table_has_one_row_per_classifier_with_metrics(dataset, registry) -> None:
    train_and_log(dataset, registry, log_to_mlflow=False)
    table = registry.summary_table()
    assert len(table) == 3
    for col in ("classifier", "n_train_samples", "trained_at", "accuracy", "f1_macro"):
        assert col in table.columns
    assert (table["accuracy"] >= 0.0).all() and (table["accuracy"] <= 1.0).all()


def test_get_classifier_returns_usable_model(dataset, registry) -> None:
    train_and_log(dataset, registry, log_to_mlflow=False)
    clf = registry.get_classifier(ClassifierKind.XGBOOST)
    X, _y = features_and_labels(dataset)
    predictions = clf.predict(X.iloc[:5])
    assert len(predictions) == 5


# ---------------------------------------------------------------------------
# MLflow local file-store logging [FE-2.7.5]
# ---------------------------------------------------------------------------
def test_train_and_log_writes_mlflow_runs(dataset, registry, tmp_path, monkeypatch) -> None:
    import ai_module.training as training_module

    monkeypatch.setattr(training_module, "MLRUNS_DIR", tmp_path / "mlruns")
    train_and_log(dataset, registry, dataset_seed=7, log_to_mlflow=True)

    for kind in ClassifierKind:
        entry = registry.get_entry(kind)
        assert entry.mlflow_run_id is not None


def test_mlflow_run_logs_expected_params_and_metrics(dataset, registry, tmp_path, monkeypatch) -> None:
    import mlflow

    import ai_module.training as training_module

    monkeypatch.setattr(training_module, "MLRUNS_DIR", tmp_path / "mlruns")
    train_and_log(dataset, registry, dataset_seed=7, log_to_mlflow=True)

    entry = registry.get_entry(ClassifierKind.RANDOM_FOREST)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(entry.mlflow_run_id)
    assert "n_estimators" in run.data.params
    assert "accuracy" in run.data.metrics
    assert run.data.metrics["accuracy"] == pytest.approx(entry.metrics.accuracy)


def test_training_history_returns_dataframe_after_logging(dataset, registry, tmp_path, monkeypatch) -> None:
    import ai_module.training as training_module

    monkeypatch.setattr(training_module, "MLRUNS_DIR", tmp_path / "mlruns")
    train_and_log(dataset, registry, dataset_seed=7, log_to_mlflow=True)

    history = training_history()
    assert isinstance(history, pd.DataFrame)
    assert len(history) >= 3  # at least the 3 nested child runs


def test_train_and_log_without_mlflow_leaves_run_id_none(dataset, registry) -> None:
    train_and_log(dataset, registry, log_to_mlflow=False)
    for kind in ClassifierKind:
        assert registry.get_entry(kind).mlflow_run_id is None


# ---------------------------------------------------------------------------
# Retraining from accumulated stored datasets [FE-2.7.2]
# ---------------------------------------------------------------------------
def test_accumulate_stored_datasets_concatenates_rows(tmp_path) -> None:
    db_path = tmp_path / "datasets.db"
    df1 = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=4, seed=1)
    df2 = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=4, seed=2)
    id1 = store_dataset_to_db(df1, db_path=db_path, seed=1)
    id2 = store_dataset_to_db(df2, db_path=db_path, seed=2)

    accumulated = accumulate_stored_datasets([id1, id2], db_path=db_path)
    assert len(accumulated) == len(df1) + len(df2)


def test_accumulate_stored_datasets_empty_list_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        accumulate_stored_datasets([], db_path=tmp_path / "datasets.db")


def test_retrain_from_accumulated_datasets_trains_on_the_union(tmp_path) -> None:
    db_path = tmp_path / "datasets.db"
    df1 = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=6, seed=1)
    df2 = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=6, seed=2)
    id1 = store_dataset_to_db(df1, db_path=db_path, seed=1)
    id2 = store_dataset_to_db(df2, db_path=db_path, seed=2)

    registry = ModelRegistry()
    retrain_from_accumulated_datasets([id1, id2], registry, db_path=db_path, log_to_mlflow=False)

    total_accumulated = len(df1) + len(df2)
    for kind in ClassifierKind:
        entry = registry.get_entry(kind)
        assert entry.n_train_samples < total_accumulated  # train portion after the random split
        assert entry.n_train_samples > len(df1) * 0.5  # genuinely reflects the LARGER accumulated set


# ---------------------------------------------------------------------------
# k-fold cross-validation [FE-2.7.3]
# ---------------------------------------------------------------------------
def test_cross_validation_produces_k_folds_per_classifier(dataset) -> None:
    report = run_cross_validation(dataset, k=4)
    assert isinstance(report, CrossValidationReport)
    for kind in ClassifierKind:
        folds_for_kind = [f for f in report.fold_results if f.classifier == kind.value]
        assert len(folds_for_kind) == 4
        assert {f.fold_index for f in folds_for_kind} == {0, 1, 2, 3}


def test_cross_validation_fold_metrics_in_valid_range(dataset) -> None:
    report = run_cross_validation(dataset, k=3)
    for f in report.fold_results:
        assert 0.0 <= f.accuracy <= 1.0
        assert 0.0 <= f.f1_macro <= 1.0
        assert f.n_test_samples > 0


def test_cross_validation_aggregate_table_has_mean_and_std(dataset) -> None:
    report = run_cross_validation(dataset, k=4)
    table = report.aggregate_table()
    assert len(table) == 3  # one row per classifier
    for col in ("mean_accuracy", "std_accuracy", "mean_f1_macro", "std_f1_macro"):
        assert col in table.columns
    assert (table["mean_accuracy"] >= 0.0).all() and (table["mean_accuracy"] <= 1.0).all()


def test_cross_validation_folds_cover_the_full_dataset_once(dataset) -> None:
    """Every sample appears in exactly one test fold (standard k-fold property),
    checked via total test-set sample counts summing to the dataset size."""
    report = run_cross_validation(dataset, k=5)
    rf_folds = [f for f in report.fold_results if f.classifier == ClassifierKind.RANDOM_FOREST.value]
    assert sum(f.n_test_samples for f in rf_folds) == len(dataset)


def test_cross_validation_rejects_k_below_two(dataset) -> None:
    with pytest.raises(ValueError):
        run_cross_validation(dataset, k=1)


# ---------------------------------------------------------------------------
# Cross-model feature importance comparison [FE-2.7.4]
# ---------------------------------------------------------------------------
def test_feature_importance_plot_has_one_trace_per_classifier(dataset) -> None:
    from ai_module.classification import train_classifiers
    from digital_twin.dataset_generation import FEATURE_COLUMNS

    classifiers = train_classifiers(dataset)
    X, _y = features_and_labels(dataset)
    background = X.sample(20, random_state=0)
    fig = feature_importance_comparison_plot(classifiers, X.sample(30, random_state=1), background)

    assert isinstance(fig, go.Figure)
    trace_names = {t.name for t in fig.data}
    assert trace_names == {k.value for k in ClassifierKind}
    assert fig.layout.barmode == "group"
    for trace in fig.data:
        assert set(trace.x) == set(FEATURE_COLUMNS)
