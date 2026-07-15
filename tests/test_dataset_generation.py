"""Tests for Sub-Module 1.3 - synthetic dataset generation with hidden confounders.

The most important tests here enforce non-negotiable principle #2: the confounders
must genuinely influence the label while being absent from the features, so that
the visible parameters do not determine the outcome. If those guarantees ever
break, the classification accuracy in Phase 2 would become meaningless.
"""
import numpy as np
import pandas as pd
import pytest

from digital_twin.dataset_generation import (
    ANALYSIS_COLUMNS,
    CONFOUNDER_COLUMNS,
    DEFAULT_CONFOUNDER_CONFIG,
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    POWER_EXTRAPOLATION_THRESHOLD_W,
    PRESSURE_EXTRAPOLATION_THRESHOLD_MTORR,
    SUITABILITY_CLASSES,
    ConfounderConfig,
    export_csv,
    features_and_labels,
    generate_dataset,
    load_dataset_from_db,
    random_split,
    region_based_split,
    store_dataset_to_db,
)


# A small, fast dataset reused across tests.
@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=6)


# ---------------------------------------------------------------------------
# Column contract & structure (FE-1.3.1)
# ---------------------------------------------------------------------------
def test_dataset_has_features_label_and_analysis_columns(dataset) -> None:
    expected = FEATURE_COLUMNS + [LABEL_COLUMN] + ANALYSIS_COLUMNS
    assert list(dataset.columns) == expected


def test_features_exclude_confounders_and_quality(dataset) -> None:
    """Principle #2: confounders and the label-basis quality must NOT be features."""
    for hidden in CONFOUNDER_COLUMNS:
        assert hidden not in FEATURE_COLUMNS
    assert "process_quality" not in FEATURE_COLUMNS
    assert "true_process_quality" not in FEATURE_COLUMNS
    assert "defect_probability" not in FEATURE_COLUMNS


def test_features_and_labels_returns_only_feature_columns(dataset) -> None:
    X, y = features_and_labels(dataset)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == LABEL_COLUMN
    assert len(X) == len(y) == len(dataset)


def test_recipe_columns_are_exact_not_noisy(dataset) -> None:
    """Operator setpoints (power, pressure) are exact; only measured observables are noisy."""
    for power in dataset["rf_power_w"].unique():
        assert power in np.arange(50.0, 300.0 + 1e-9, 50.0)


# ---------------------------------------------------------------------------
# Hidden confounders genuinely influence the label (the core guarantee)
# ---------------------------------------------------------------------------
def test_identical_recipes_can_carry_different_labels(dataset) -> None:
    """The signature of hidden confounding: same visible features, different labels."""
    distinct_labels = dataset.groupby(["rf_power_w", "pressure_mtorr"])[LABEL_COLUMN].nunique()
    assert (distinct_labels > 1).any(), "no recipe had label variation - confounders inert"


def test_confounders_change_the_true_quality(dataset) -> None:
    """The confounded true quality must differ from the nominal quality per recipe."""
    # For at least many samples, confounders shift true quality off the nominal value.
    diff = (dataset["true_process_quality"] - dataset["nominal_process_quality"]).abs()
    assert (diff > 1e-6).mean() > 0.9


def test_removing_confounders_makes_recipe_determine_label() -> None:
    """Control experiment: with all confounder strengths zeroed and no measurement or
    label noise, a recipe maps to a single label - proving the confounders (not some
    other randomness) are what break determinism in the real dataset."""
    inert = ConfounderConfig(
        wall_temp_drift_mean_k=0.0, wall_temp_drift_sd_k=0.0, wall_temp_drift_max_k=0.0,
        electrode_aging_max_coupling_loss=0.0,
        gas_impurity_sd=0.0, gas_impurity_max=0.0, gas_impurity_quality_sensitivity=0.0,
        feature_measurement_noise=0.0, label_jitter_sd=0.0,
    )
    df = generate_dataset(
        power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=6,
        confounder_config=inert,
    )
    distinct = df.groupby(["rf_power_w", "pressure_mtorr"])[LABEL_COLUMN].nunique()
    assert (distinct == 1).all(), "with confounders off, each recipe must map to one label"


# ---------------------------------------------------------------------------
# Labelling (FE-1.3.3)
# ---------------------------------------------------------------------------
def test_labels_are_all_valid_classes(dataset) -> None:
    assert set(dataset[LABEL_COLUMN].unique()).issubset(set(SUITABILITY_CLASSES))


def test_all_four_classes_present(dataset) -> None:
    assert set(dataset[LABEL_COLUMN].unique()) == set(SUITABILITY_CLASSES)


def test_soft_boundaries_cause_class_overlap_in_quality(dataset) -> None:
    """Soft labelling means adjacent classes overlap in true-quality range, rather
    than being separated by a razor-sharp threshold (FE-1.3.3)."""
    marginal_max = dataset.loc[dataset[LABEL_COLUMN] == "Marginal", "true_process_quality"].max()
    acceptable_min = dataset.loc[dataset[LABEL_COLUMN] == "Acceptable", "true_process_quality"].min()
    # With hard thresholds acceptable_min would exceed marginal_max; jitter overlaps them.
    assert marginal_max > acceptable_min


# ---------------------------------------------------------------------------
# Confounder value ranges match the documented a-priori bounds (FE-1.3.7)
# ---------------------------------------------------------------------------
def test_confounder_values_within_documented_bounds(dataset) -> None:
    cfg = DEFAULT_CONFOUNDER_CONFIG
    assert dataset["wall_temp_drift_k"].between(0.0, cfg.wall_temp_drift_max_k).all()
    assert dataset["electrode_age"].between(0.0, 1.0).all()
    assert dataset["gas_purity"].between(1.0 - cfg.gas_impurity_max, 1.0).all()


def test_true_defect_probability_is_valid_and_confounded(dataset) -> None:
    """true_defect_probability (added for Sub-Module 2.8's regression target)
    must be a valid probability, and - like true_process_quality - must genuinely
    vary across replicates of the SAME recipe, since it is read off the same
    confounder-perturbed simulate() call."""
    assert dataset["true_defect_probability"].between(0.0, 1.0).all()
    per_recipe_spread = dataset.groupby(["rf_power_w", "pressure_mtorr"])["true_defect_probability"].std()
    assert (per_recipe_spread > 0).any()


# ---------------------------------------------------------------------------
# Reproducibility (FE-1.3.7)
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_identical_dataset() -> None:
    a = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=4, seed=123)
    b = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=4, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_confounder_draws() -> None:
    a = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=4, seed=1)
    b = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=4, seed=2)
    assert not a["wall_temp_drift_k"].equals(b["wall_temp_drift_k"])


# ---------------------------------------------------------------------------
# Splits (FE-1.3.5)
# ---------------------------------------------------------------------------
def test_region_split_holds_out_the_extrapolation_corner(dataset) -> None:
    train, test = region_based_split(dataset)
    # Training set must contain NO samples from the held-out high-power/high-pressure corner.
    assert (train["rf_power_w"] <= POWER_EXTRAPOLATION_THRESHOLD_W).all()
    assert (train["pressure_mtorr"] <= PRESSURE_EXTRAPOLATION_THRESHOLD_MTORR).all()
    # Every test sample is in that corner.
    in_corner = (
        (test["rf_power_w"] > POWER_EXTRAPOLATION_THRESHOLD_W)
        | (test["pressure_mtorr"] > PRESSURE_EXTRAPOLATION_THRESHOLD_MTORR)
    )
    assert in_corner.all()
    assert len(train) + len(test) == len(dataset)


def test_random_split_sizes_and_reproducibility(dataset) -> None:
    train_a, test_a = random_split(dataset, test_fraction=0.2, seed=7)
    train_b, test_b = random_split(dataset, test_fraction=0.2, seed=7)
    assert len(test_a) == round(len(dataset) * 0.2)
    assert len(train_a) + len(test_a) == len(dataset)
    pd.testing.assert_frame_equal(test_a, test_b)  # reproducible


def test_random_split_covers_full_power_range_unlike_region_split(dataset) -> None:
    """Interpolation split should span the envelope, including the high-power corner
    that the extrapolation split holds out entirely."""
    _train, test = random_split(dataset, seed=7)
    assert test["rf_power_w"].max() > POWER_EXTRAPOLATION_THRESHOLD_W


# ---------------------------------------------------------------------------
# Export & persistence (FE-1.3.4, FE-1.3.6)
# ---------------------------------------------------------------------------
def test_csv_export_roundtrip(dataset, tmp_path) -> None:
    path = export_csv(dataset, tmp_path / "ds.csv")
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == list(dataset.columns)
    assert len(reloaded) == len(dataset)


def test_sqlite_store_and_load_roundtrip(dataset, tmp_path) -> None:
    db_path = tmp_path / "datasets.db"
    dataset_id = store_dataset_to_db(dataset, db_path=db_path)
    reloaded = load_dataset_from_db(dataset_id, db_path=db_path)
    assert len(reloaded) == len(dataset)
    assert set(FEATURE_COLUMNS + [LABEL_COLUMN]).issubset(reloaded.columns)


def test_sqlite_metadata_records_seed_and_config(dataset, tmp_path) -> None:
    import json
    import sqlite3

    db_path = tmp_path / "datasets.db"
    dataset_id = store_dataset_to_db(dataset, seed=DEFAULT_SEED, db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT seed, n_samples, config_json FROM datasets WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    finally:
        conn.close()
    seed, n_samples, config_json = row
    assert seed == DEFAULT_SEED
    assert n_samples == len(dataset)
    # The full confounder config is recorded for auditability (FE-1.3.7).
    assert "wall_temp_drift_mean_k" in json.loads(config_json)


def test_load_missing_dataset_raises(tmp_path) -> None:
    db_path = tmp_path / "datasets.db"
    store_dataset_to_db(
        generate_dataset(power_step_w=100.0, pressure_step_mtorr=10.0, replicates_per_recipe=2),
        db_path=db_path,
    )
    with pytest.raises(KeyError):
        load_dataset_from_db("no-such-id", db_path=db_path)
