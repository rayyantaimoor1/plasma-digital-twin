"""Reactor Control Room - FastAPI backend (FUTURE.md item 1).

A DELIBERATELY THIN JSON wrapper over the project's existing pure functions. It
performs NO physics or AI computation of its own: every endpoint calls one
already-tested function from `digital_twin/` or `ai_module/` and returns its
result serialized to JSON. The serialization helpers below only convert types
(dataclass -> dict, numpy scalar -> Python scalar); they never recompute a value.

Why this matters (the correctness guarantee from FUTURE.md item 1): because the
Streamlit dashboard imports the SAME functions, the two UIs are structurally
guaranteed to show identical numbers for identical inputs - there is only ever
one place the math executes. `tests/test_reactor_backend.py` pins this down by
asserting each endpoint returns exactly what calling the underlying function
directly returns. If real logic ever leaks into this file, that test breaks.

This is a SEPARATE companion app; it is NOT part of the graded core stack locked
in CLAUDE.md. Its extra dependencies (fastapi, uvicorn, psutil) live in
`reactor_control_room/requirements.txt`, not the project's `requirements.txt`.

Run (from the project root):
    uvicorn reactor_control_room.backend.app:app --reload
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from functools import lru_cache
from typing import Any, Optional

# Put the project root on sys.path so `import digital_twin` / `import ai_module`
# resolve no matter what directory uvicorn is launched from (same pattern the
# Streamlit pages use).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- existing project functions: the ONLY place any computation happens ---
from ai_module.anomaly_detection import (
    AnomalyFault,
    PlasmaAnomalyDetector,
    generate_normal_operating_data,
    inject_anomaly,
)
from ai_module.classification import (
    ClassifierKind,
    classify_configuration,
    explain_configuration,
    train_classifiers,
)
from ai_module.suitability_analysis import all_application_defect_estimates, classify_suitability
from digital_twin.dataset_generation import FEATURE_COLUMNS, features_and_labels, generate_dataset
from digital_twin.physics_engine import simulate
from digital_twin.physics_validation import run_literature_benchmarks
from digital_twin.session_manager import ExperimentDatabase

# Valid fault names for the anomaly endpoint: "none" (healthy) plus the three
# physically-named relationship-violating faults the detector was built around.
_FAULT_NAMES = frozenset({"none", *(f.value for f in AnomalyFault)})
# Fixed RNG seed for fault injection so /api/anomaly is deterministic (identity-testable).
_ANOMALY_SEED = 0

# Mirror the Streamlit dashboard's dataset parameters (see dashboard/backend.py)
# so the classifier this backend trains is the SAME one the dashboard trains -
# then both UIs classify identically, per the correctness guarantee above.
_DATASET_POWER_STEP = 25.0
_DATASET_PRESSURE_STEP = 2.0
_DATASET_REPLICATES = 8

app = FastAPI(title="Reactor Control Room API", version="0.1.0")

# The companion frontend is a separate app (a different origin), so it needs CORS
# to read these endpoints from the browser. Safe to open up here: every route is
# a read-only GET over a local, non-authenticated demo API - no state is mutated.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# One-time resources (built lazily, cached) - thin calls into existing code
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_classifiers():
    """Train the classifiers ONCE (cached) via the existing `train_classifiers`,
    on the same dataset parameters the dashboard uses. No model logic here - a
    cached call into ai_module. `with_explainer=False` skips the SHAP-only second
    fit of each ensemble (roughly halving training time); the explanation endpoint
    builds its own explainer-enabled set lazily - see `get_explainer_classifiers`."""
    return train_classifiers(get_dataset(), with_explainer=False)


@lru_cache(maxsize=1)
def get_dataset():
    """The synthetic training dataset, built once (cached). Same parameters the
    dashboard uses, so both UIs train on identical data."""
    return generate_dataset(
        power_step_w=_DATASET_POWER_STEP,
        pressure_step_mtorr=_DATASET_PRESSURE_STEP,
        replicates_per_recipe=_DATASET_REPLICATES,
    )


@lru_cache(maxsize=1)
def get_explainer_classifiers():
    """Classifiers trained WITH the SHAP explainer fit, built LAZILY.

    `get_classifiers()` above deliberately passes `with_explainer=False`, which
    skips a second (uncalibrated) fit of each ensemble and roughly halves training
    time - but leaves `shap_values()` unavailable. SHAP therefore needs its own
    build. It is a separate lru_cache rather than simply flipping that flag so the
    cost is paid ONLY if someone opens the explanation panel: the Reactor View and
    the rest of AI Verdict stay as fast as before.

    Predictions from these classifiers are identical to `get_classifiers()`' -
    same data, same hyperparameters, same seed - so the explanation always matches
    the verdict shown alongside it."""
    return train_classifiers(get_dataset(), with_explainer=True)


@lru_cache(maxsize=1)
def get_shap_background():
    """SHAP reference distribution: 40 sampled training rows, matching
    `dashboard/backend.py:get_shap_background` exactly (same size, same seed) so
    the companion app and the dashboard report identical SHAP values."""
    features, _labels = features_and_labels(get_dataset())
    return features.sample(40, random_state=0)


@lru_cache(maxsize=1)
def get_anomaly_detector() -> PlasmaAnomalyDetector:
    """Fit the Isolation-Forest anomaly detector ONCE (cached) on normal operating
    data, exactly as the dashboard does. No detector logic here - a cached call
    into ai_module."""
    return PlasmaAnomalyDetector().fit(generate_normal_operating_data())


def _anomaly_feature_row(rf_power_w: float, pressure_mtorr: float, fault: str) -> pd.DataFrame:
    """Build the single observed-feature row the detector scores. For a healthy
    run it is simulate()'s own output; for an injected fault it is the existing
    `inject_anomaly` construction (seeded, so deterministic). No physics is redone
    here beyond calling those existing functions."""
    if fault == "none":
        result = simulate(rf_power_w, pressure_mtorr)
        return pd.DataFrame([{c: getattr(result, c) for c in FEATURE_COLUMNS}])
    rng = np.random.default_rng(_ANOMALY_SEED)
    row = inject_anomaly(rf_power_w, pressure_mtorr, AnomalyFault(fault), rng)
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def open_session_db() -> ExperimentDatabase:
    """Open the session store. Factored into its own function so tests can point
    it at a temporary database. Deliberately not cached: a sqlite connection is
    not safe to share across threads/requests (same reasoning as the dashboard's
    `open_db`)."""
    return ExperimentDatabase()


# ---------------------------------------------------------------------------
# Serialization helpers - PURE format conversion, never a recomputation
# ---------------------------------------------------------------------------
def _records_from_dataframe(df) -> list[dict[str, Any]]:
    """A pandas DataFrame -> a list of plain-typed dict rows. numpy scalars
    (float64/bool_) are converted to native Python via `.item()` so FastAPI can
    serialize them and no precision is lost; string cells pass through untouched.
    No cell value is altered - this only changes Python types, not numbers."""
    return [
        {col: (val.item() if hasattr(val, "item") else val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _serialize_defect_estimates(estimates) -> dict[str, dict[str, Any]]:
    """dict[SemiconductorApplication, DefectProbabilityEstimate] -> plain JSON:
    enum keys become their string `.value`, dataclass values become plain dicts
    (with the enum `application` field also flattened to its string value). No
    numbers are recomputed - just reshaped for JSON."""
    return {
        app.value: dataclasses.asdict(est) | {"application": est.application.value}
        for app, est in estimates.items()
    }


def _serialize_benchmarks(results) -> list[dict[str, Any]]:
    """list[BenchmarkResult] -> plain JSON, one row per check. Reads the dataclass
    fields plus its `deviation_pct`/`passed` PROPERTIES (which the summary-table
    DataFrame also exposes) and, additionally, the human `description` and literature
    `source` citation the Physics Validation page needs. Pure field access - nothing
    is recomputed."""
    return [
        {
            "name": r.name,
            "quantity": r.quantity,
            "description": r.description,
            "source": r.source,
            "computed_value": r.computed_value,
            "reference_value": r.reference_value,
            "unit": r.unit,
            "deviation_pct": r.deviation_pct,
            "tolerance_pct": r.tolerance_pct,
            "passed": r.passed,
        }
        for r in results
    ]


def _serialize_scorecard(scorecard) -> dict[str, Any]:
    """SuitabilityScorecard -> plain JSON: the operating point, the best-fit
    application, its compliance %, and every application's SuitabilityRating.
    Enum fields flattened to their string values. No numbers recomputed."""
    best = scorecard.best_application()
    return {
        "rf_power_w": scorecard.rf_power_w,
        "pressure_mtorr": scorecard.pressure_mtorr,
        "ion_energy_ev": scorecard.ion_energy_ev,
        "defect_probability": scorecard.defect_probability,
        "best_application": best.value,
        "best_compliance_pct": scorecard.ratings[best].overall_compliance_pct,
        "ratings": [
            dataclasses.asdict(r) | {"application": r.application.value}
            for r in scorecard.ratings.values()
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints - each wraps exactly one existing function
# ---------------------------------------------------------------------------
@app.get("/api")
def root() -> dict[str, Any]:
    """Service banner + endpoint index (the frontend itself is served at `/`)."""
    return {
        "service": "Reactor Control Room API",
        "status": "ok",
        "note": "Thin JSON wrapper over the digital twin / AI functions; no math runs here.",
        "endpoints": [
            "/api/simulate", "/api/classify", "/api/explain", "/api/suitability",
            "/api/suitability-scorecard", "/api/anomaly", "/api/physics-validation",
            "/api/sessions", "/api/sessions/{session_id}", "/api/system/stats",
        ],
    }


@app.get("/api/simulate")
def api_simulate(
    rf_power_w: float, pressure_mtorr: float, rf_voltage_v: Optional[float] = None
) -> dict[str, float]:
    """Reactor View: the full 0D global-model output vector for one operating
    point. Wraps `digital_twin.physics_engine.simulate`."""
    return simulate(rf_power_w, pressure_mtorr, rf_voltage_v=rf_voltage_v).to_dict()


def _classifier_kind(name: str) -> ClassifierKind:
    """Validate a classifier query param, 422 on anything unknown."""
    try:
        return ClassifierKind(name)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown classifier {name!r}; must be one of "
                   f"{sorted(k.value for k in ClassifierKind)}",
        )


@app.get("/api/classify")
def api_classify(
    rf_power_w: float, pressure_mtorr: float, classifier: str = "random_forest"
) -> dict[str, Any]:
    """AI Verdict: the suitability-class prediction + confidence for one operating
    point. Defaults to the Random Forest (the dashboard's primary classifier);
    pass `classifier=xgboost` to get the second ensemble's independent verdict, so
    the UI can show whether the two models agree. Wraps
    `ai_module.classification.classify_configuration`."""
    model = get_classifiers()[_classifier_kind(classifier)]
    return dataclasses.asdict(classify_configuration(rf_power_w, pressure_mtorr, model))


@app.get("/api/explain")
def api_explain(
    rf_power_w: float, pressure_mtorr: float, classifier: str = "random_forest"
) -> dict[str, Any]:
    """AI Verdict: per-prediction SHAP breakdown - how much each observable feature
    pushed THIS operating point toward its predicted class. Wraps
    `ai_module.classification.explain_configuration`.

    Uses the lazily-built explainer-enabled classifiers, so the first call is slow
    (it fits the SHAP-only estimators) and every later call is cached."""
    kind = _classifier_kind(classifier)
    if kind is ClassifierKind.LOGISTIC_REGRESSION:
        # The baseline has no TreeExplainer; explaining it is out of scope here.
        raise HTTPException(
            status_code=422,
            detail="SHAP explanation is available for the ensemble models "
                   "(random_forest, xgboost), not the logistic-regression baseline.",
        )
    model = get_explainer_classifiers()[kind]
    explanation = explain_configuration(
        rf_power_w, pressure_mtorr, model, get_shap_background()
    )
    return dataclasses.asdict(explanation)


@app.get("/api/suitability")
def api_suitability(
    rf_power_w: float,
    pressure_mtorr: float,
    n_bootstrap: int = 200,
    rf_voltage_v: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """AI Verdict: the bootstrapped defect-probability interval for every
    semiconductor application. Wraps
    `ai_module.suitability_analysis.all_application_defect_estimates`."""
    estimates = all_application_defect_estimates(
        rf_power_w, pressure_mtorr, n_bootstrap=n_bootstrap, rf_voltage_v=rf_voltage_v
    )
    return _serialize_defect_estimates(estimates)


@app.get("/api/suitability-scorecard")
def api_suitability_scorecard(
    rf_power_w: float, pressure_mtorr: float, rf_voltage_v: Optional[float] = None
) -> dict[str, Any]:
    """AI Verdict: window-compliance scorecard - best-fit application + per-
    application compliance %. Wraps
    `ai_module.suitability_analysis.classify_suitability` (the same function the
    dashboard's best-fit uses, so the two UIs agree)."""
    scorecard = classify_suitability(rf_power_w, pressure_mtorr, rf_voltage_v=rf_voltage_v)
    return _serialize_scorecard(scorecard)


@app.get("/api/anomaly")
def api_anomaly(
    rf_power_w: float, pressure_mtorr: float, fault: str = "none"
) -> dict[str, Any]:
    """AI Verdict: relationship-anomaly severity + Isolation-Forest score +
    root-cause for the current operating point, optionally with an injected fault.
    Wraps `PlasmaAnomalyDetector` (+ the existing `inject_anomaly`); the detector
    does all the work."""
    if fault not in _FAULT_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown fault {fault!r}; must be one of {sorted(_FAULT_NAMES)}",
        )
    detector = get_anomaly_detector()
    df = _anomaly_feature_row(rf_power_w, pressure_mtorr, fault)
    return {
        "fault": fault,
        "severity": detector.severity(df)[0].value,
        "score": float(detector.anomaly_score(df)[0]),
        "is_anomaly": bool(detector.is_anomaly(df)[0]),
        "root_cause": detector.root_cause(df)[0],
    }


@app.get("/api/physics-validation")
def api_physics_validation() -> list[dict[str, Any]]:
    """Physics Validation: the literature benchmark checks, one row per check,
    including each check's tolerance verdict and literature source. Wraps
    `digital_twin.physics_validation.run_literature_benchmarks` (the same call the
    summary-table builds on, serialized with its source/description fields too)."""
    return _serialize_benchmarks(run_literature_benchmarks())


@app.get("/api/sessions")
def api_sessions() -> list[dict[str, Any]]:
    """Session Replay: every stored experiment, newest first. Wraps
    `ExperimentDatabase.list_sessions`."""
    db = open_session_db()
    try:
        return [dataclasses.asdict(s) for s in db.list_sessions()]
    finally:
        db.close()


@app.get("/api/sessions/{session_id}")
def api_session(session_id: str) -> dict[str, Any]:
    """Session Replay: one stored experiment by id. Wraps
    `ExperimentDatabase.get_session` (404 if the id is unknown)."""
    db = open_session_db()
    try:
        return dataclasses.asdict(db.get_session(session_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No session with id {session_id!r}")
    finally:
        db.close()


@app.get("/api/system/stats")
def api_system_stats() -> dict[str, Any]:
    """Persistent strip: live host CPU% / RAM% via psutil. GPU is reported
    honestly as idle - the core models are CPU-based tree ensembles (CLAUDE.md
    principle #7), so nothing touches the GPU by design; only the optional
    PyTorch autoencoder (Sub-Module 2.2) ever would. This endpoint reports host
    metrics only - no physics or AI is involved."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
        "gpu": {
            "status": "idle",
            "note": (
                "GPU idle by design - core models are CPU-based tree ensembles "
                "(CLAUDE.md principle #7); only the optional PyTorch autoencoder "
                "(Sub-Module 2.2) would ever use it."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Serve the companion frontend (the Reactor Control Room UI) at `/` so it is
# same-origin with the API above - fetches to /api/* need no CORS and one command
# (uvicorn) serves the whole app. Mounted LAST so every /api/... route is matched
# first; StaticFiles only handles what's left (index.html, JS, CSS). Guarded on the
# directory existing so the API still boots if the frontend hasn't been built yet.
# ---------------------------------------------------------------------------
_FRONTEND_DIR = pathlib.Path(__file__).resolve().parents[1] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
