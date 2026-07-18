"""Shared backend for the Streamlit dashboard: cached model/data builders and the
common chamber-configuration sidebar.

The Digital Twin and AI sub-modules were all deliberately built UI-agnostic (pure
functions returning dataclasses and Plotly figures), so this dashboard layer is
thin: it caches the expensive one-time work (dataset generation, model training,
conformal calibration) behind Streamlit's cache so it runs once per process, and
renders the already-built figures.

Nothing here trains or simulates anything new - it only orchestrates and caches
calls into ai_module / digital_twin.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_module.anomaly_detection import PlasmaAnomalyDetector, generate_normal_operating_data
from ai_module.classification import train_classifiers
from ai_module.correlation_analysis import simulation_sweep_dataframe
from ai_module.recommendation_engine import Configuration
from ai_module.uncertainty_quantification import build_uncertainty_layer
from digital_twin.chamber_config import (
    EXPERIMENT_MODE_PRESETS,
    PRESSURE_RANGE,
    RF_POWER_RANGE,
    ChamberParameters,
    ExperimentMode,
    validate_parameters,
)
from digital_twin.dataset_generation import features_and_labels, generate_dataset

# Modest dataset so first-load model training stays responsive for a live demo;
# the fixed default seed keeps the whole dashboard reproducible run-to-run.
_DATASET_POWER_STEP = 25.0
_DATASET_PRESSURE_STEP = 2.0
_DATASET_REPLICATES = 8


# ---------------------------------------------------------------------------
# Cached heavy resources (built once per process)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating synthetic dataset (Sub-Module 1.3)...")
def load_dataset() -> pd.DataFrame:
    return generate_dataset(
        power_step_w=_DATASET_POWER_STEP,
        pressure_step_mtorr=_DATASET_PRESSURE_STEP,
        replicates_per_recipe=_DATASET_REPLICATES,
    )


@st.cache_resource(show_spinner="Training classifiers (Sub-Module 2.1)...")
def get_classifiers():
    return train_classifiers(load_dataset())


@st.cache_data
def get_shap_background() -> pd.DataFrame:
    features, _labels = features_and_labels(load_dataset())
    return features.sample(40, random_state=0)


@st.cache_resource(show_spinner="Fitting anomaly detector (Sub-Module 2.2)...")
def get_anomaly_detector() -> PlasmaAnomalyDetector:
    return PlasmaAnomalyDetector().fit(generate_normal_operating_data())


@st.cache_resource(show_spinner="Building uncertainty layer (Sub-Module 2.8)...")
def get_uncertainty_layer():
    return build_uncertainty_layer(load_dataset())


@st.cache_data(show_spinner="Computing correlation sweep (Sub-Module 2.4)...")
def get_correlation_sweep() -> pd.DataFrame:
    return simulation_sweep_dataframe(power_step_w=_DATASET_POWER_STEP, pressure_step_mtorr=_DATASET_PRESSURE_STEP)


@st.cache_data(show_spinner="Running sensitivity sweep (Sub-Module 1.5)...")
def get_oat_analysis(rf_power_w: float, pressure_mtorr: float, n_points: int = 15):
    """Cached one-at-a-time sensitivity sweep (Sub-Module 1.5) for the Digital Twin
    page. It depends only on the operating point, NOT on which output metric the user
    is viewing, so caching it lets the metric selectbox (now a fragment) re-draw
    instantly instead of re-running ~30 physics solves on every switch
    (DASHBOARD_UX_REVIEW.md U3). Pure memoisation: the cached result is byte-identical
    to a fresh call, so no numerical output changes."""
    from digital_twin.chamber_config import ChamberParameters
    from digital_twin.sensitivity_analysis import run_full_oat_analysis
    return run_full_oat_analysis(ChamberParameters(rf_power_w, pressure_mtorr), n_points=n_points)


@st.cache_data(show_spinner="Computing paired RF-power x pressure sweep (Sub-Module 1.5)...")
def get_paired_sweep(n_x: int = 12, n_y: int = 12):
    """Cached RF-power x pressure paired sweep (Sub-Module 1.5). It spans the full
    operating envelope, so it is independent of BOTH the current operating point and
    the selected metric and only ever needs computing once per process
    (DASHBOARD_UX_REVIEW.md U3). Pure memoisation — identical grid to a fresh call."""
    from digital_twin.sensitivity_analysis import run_paired_sweep
    return run_paired_sweep("rf_power_w", "pressure_mtorr", n_x=n_x, n_y=n_y)


@st.cache_data(show_spinner="Bootstrapping application defect-risk intervals (Sub-Module 2.5)...")
def get_defect_estimates(rf_power_w: float, pressure_mtorr: float, n_bootstrap: int = 80):
    """Cached application-specific defect-risk bootstrap (Sub-Module 2.5, FE-2.5.3).
    It is deterministic (fixed default seed) and independent of the recommendation
    widgets below it on the Suitability page, so caching stops its bootstrap from
    re-running on every unrelated rerun (DASHBOARD_UX_REVIEW.md U3). Same seed and
    inputs -> identical estimates; pure memoisation."""
    from ai_module.suitability_analysis import all_application_defect_estimates
    return all_application_defect_estimates(rf_power_w, pressure_mtorr, n_bootstrap=n_bootstrap)


@st.cache_resource(show_spinner="Evaluating models on both splits (Sub-Module 2.1)...")
def get_evaluation_report():
    from ai_module.classification import run_full_evaluation
    return run_full_evaluation(load_dataset())


def open_db():
    """Open a fresh ExperimentDatabase connection. Deliberately NOT cached: a
    sqlite3 connection is not safe to share across Streamlit's reruns/threads, so
    each caller opens and closes its own (cheap, and SQLite handles the file-level
    concurrency for a single-user dashboard)."""
    from digital_twin.session_manager import ExperimentDatabase
    return ExperimentDatabase()


@st.cache_resource(show_spinner="Training the optional deep-learning autoencoder (FE-2.2.6)...")
def get_autoencoder_comparison():
    """Classical (Isolation Forest) vs deep-learning (autoencoder) detector
    comparison. torch is imported LAZILY here so the whole dashboard still runs
    when the optional dependency is absent - the caller guards against ImportError
    and shows an install hint instead."""
    from ai_module.anomaly_detection import generate_anomalous_data, generate_normal_operating_data
    from ai_module.autoencoder_detector import compare_with_isolation_forest

    normal_train = generate_normal_operating_data()
    normal_test = generate_normal_operating_data(seed=99, replicates=1)
    anomalous = generate_anomalous_data(n_samples=150)
    return compare_with_isolation_forest(normal_train, normal_test, anomalous)


# ---------------------------------------------------------------------------
# Shared chamber-configuration sidebar (Sub-Module 1.1)
# ---------------------------------------------------------------------------
def _apply_experiment_mode_preset() -> None:
    """on_change callback: set the sliders to the selected preset's operating point.
    Runs before the widgets are re-instantiated, so the sliders pick up the values."""
    preset = EXPERIMENT_MODE_PRESETS[ExperimentMode(st.session_state["exp_mode"])]
    st.session_state["rf_power"] = preset.rf_power_w
    st.session_state["pressure"] = preset.pressure_mtorr


def render_sidebar() -> Configuration:
    """Render the chamber-configuration controls (shared across all pages) and
    return the current Configuration. Values persist across pages via session
    state, so the whole dashboard reflects one consistent operating point."""
    st.session_state.setdefault("rf_power", RF_POWER_RANGE.default)
    st.session_state.setdefault("pressure", PRESSURE_RANGE.default)

    st.sidebar.header("Chamber configuration")
    st.sidebar.caption("Sub-Module 1.1 — virtual chamber & parameter config")

    st.sidebar.selectbox(
        "Experiment mode preset",
        [m.value for m in ExperimentMode],
        key="exp_mode",
        on_change=_apply_experiment_mode_preset,
        help="Selecting a preset sets the sliders to that mode's operating point.",
    )
    st.sidebar.slider(
        f"RF power ({RF_POWER_RANGE.unit})",
        RF_POWER_RANGE.min_value, RF_POWER_RANGE.max_value,
        step=RF_POWER_RANGE.step, key="rf_power",
    )
    st.sidebar.slider(
        f"Chamber pressure ({PRESSURE_RANGE.unit})",
        PRESSURE_RANGE.min_value, PRESSURE_RANGE.max_value,
        step=PRESSURE_RANGE.step, key="pressure",
    )

    apply_bias = st.sidebar.checkbox(
        "Apply RF bias voltage (drives the Child–Langmuir sheath, Sub-Module 1.2)",
        value=False, key="apply_bias",
    )
    rf_voltage = None
    if apply_bias:
        rf_voltage = st.sidebar.slider("RF bias voltage (V)", 0.0, 600.0, 300.0, step=10.0, key="rf_voltage")

    config = Configuration(
        rf_power_w=float(st.session_state["rf_power"]),
        pressure_mtorr=float(st.session_state["pressure"]),
        rf_voltage_v=rf_voltage,
    )

    # Physics-grounded validation (Sub-Module 1.1) surfaced live in the sidebar.
    validation = validate_parameters(
        ChamberParameters(rf_power_w=config.rf_power_w, pressure_mtorr=config.pressure_mtorr)
    )
    for warning in validation.warnings:
        st.sidebar.warning(warning)
    for error in validation.errors:
        st.sidebar.error(error)

    return config
