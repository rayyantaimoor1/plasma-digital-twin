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

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- existing project functions: the ONLY place any computation happens ---
from ai_module.classification import ClassifierKind, classify_configuration, train_classifiers
from ai_module.suitability_analysis import all_application_defect_estimates
from digital_twin.dataset_generation import generate_dataset
from digital_twin.physics_engine import simulate
from digital_twin.physics_validation import benchmark_summary_table
from digital_twin.session_manager import ExperimentDatabase

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
    cached call into ai_module. `with_explainer=False` because this backend only
    ever predicts; SHAP is a dashboard-only concern (halves ensemble training)."""
    dataset = generate_dataset(
        power_step_w=_DATASET_POWER_STEP,
        pressure_step_mtorr=_DATASET_PRESSURE_STEP,
        replicates_per_recipe=_DATASET_REPLICATES,
    )
    return train_classifiers(dataset, with_explainer=False)


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


# ---------------------------------------------------------------------------
# Endpoints - each wraps exactly one existing function
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict[str, Any]:
    """Service banner + endpoint index."""
    return {
        "service": "Reactor Control Room API",
        "status": "ok",
        "note": "Thin JSON wrapper over the digital twin / AI functions; no math runs here.",
        "endpoints": [
            "/api/simulate", "/api/classify", "/api/suitability",
            "/api/physics-validation", "/api/sessions",
            "/api/sessions/{session_id}", "/api/system/stats",
        ],
    }


@app.get("/api/simulate")
def api_simulate(
    rf_power_w: float, pressure_mtorr: float, rf_voltage_v: Optional[float] = None
) -> dict[str, float]:
    """Reactor View: the full 0D global-model output vector for one operating
    point. Wraps `digital_twin.physics_engine.simulate`."""
    return simulate(rf_power_w, pressure_mtorr, rf_voltage_v=rf_voltage_v).to_dict()


@app.get("/api/classify")
def api_classify(rf_power_w: float, pressure_mtorr: float) -> dict[str, Any]:
    """AI Verdict: the suitability-class prediction + confidence for one operating
    point, using the Random Forest (the dashboard's primary classifier). Wraps
    `ai_module.classification.classify_configuration`."""
    classifier = get_classifiers()[ClassifierKind.RANDOM_FOREST]
    return dataclasses.asdict(classify_configuration(rf_power_w, pressure_mtorr, classifier))


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


@app.get("/api/physics-validation")
def api_physics_validation() -> list[dict[str, Any]]:
    """Physics Validation: the literature-benchmark table, one row per check.
    Wraps `digital_twin.physics_validation.benchmark_summary_table`."""
    return _records_from_dataframe(benchmark_summary_table())


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
