"""Identity tests for the Reactor Control Room FastAPI backend (FUTURE.md item 1).

The backend's entire contract is that it NEVER computes anything itself - every
endpoint just serializes the output of an existing project function. These tests
prove exactly that: for each endpoint, calling it returns the same values as
calling the underlying function directly. If someone ever slips real physics or
AI logic into the backend, one of these breaks.

The route handlers are plain functions, so they are called DIRECTLY here rather
than over HTTP. That keeps the backend's scoped dependencies to exactly the three
approved packages (fastapi, uvicorn, psutil) - no test-only HTTP client (httpx)
is pulled in. Because every handler already returns JSON-native types, its direct
return value is identical to the JSON body FastAPI would send; only FastAPI's own
(well-tested) query-string coercion is skipped, which is not our logic to verify.
"""
import dataclasses

import pytest

from ai_module.classification import ClassifierKind, classify_configuration, explain_configuration
from ai_module.suitability_analysis import (
    SemiconductorApplication,
    all_application_defect_estimates,
    classify_suitability,
)
from digital_twin.chamber_config import ChamberParameters
from digital_twin.dataset_generation import FEATURE_COLUMNS
from digital_twin.physics_engine import simulate
from digital_twin.physics_validation import run_literature_benchmarks
from digital_twin.session_manager import ExperimentDatabase
from reactor_control_room.backend import app as backend

_RF_POWER = 150.0
_PRESSURE = 10.0


def test_simulate_endpoint_matches_direct_call() -> None:
    """/api/simulate == simulate(...).to_dict() (the project's own serializer)."""
    endpoint = backend.api_simulate(_RF_POWER, _PRESSURE)
    direct = simulate(_RF_POWER, _PRESSURE).to_dict()
    assert endpoint == direct


def test_simulate_endpoint_passes_rf_voltage_through() -> None:
    """The optional rf_voltage_v is genuinely forwarded to simulate (it changes
    the Child-Langmuir sheath), and the endpoint still matches the direct call."""
    endpoint = backend.api_simulate(_RF_POWER, _PRESSURE, rf_voltage_v=300.0)
    direct = simulate(_RF_POWER, _PRESSURE, rf_voltage_v=300.0).to_dict()
    assert endpoint == direct
    # sanity: supplying the bias really does change the output (arg isn't ignored)
    assert endpoint != simulate(_RF_POWER, _PRESSURE).to_dict()


def test_classify_endpoint_matches_direct_call() -> None:
    """/api/classify == asdict(classify_configuration(..., same_classifier))."""
    classifier = backend.get_classifiers()[ClassifierKind.RANDOM_FOREST]
    endpoint = backend.api_classify(_RF_POWER, _PRESSURE)
    direct = dataclasses.asdict(classify_configuration(_RF_POWER, _PRESSURE, classifier))
    assert endpoint == direct


def test_classify_endpoint_supports_xgboost_and_matches_direct_call() -> None:
    """/api/classify?classifier=xgboost returns XGBoost's own verdict (option B:
    the model-agreement signal), identical to calling it directly."""
    model = backend.get_classifiers()[ClassifierKind.XGBOOST]
    endpoint = backend.api_classify(_RF_POWER, _PRESSURE, classifier="xgboost")
    direct = dataclasses.asdict(classify_configuration(_RF_POWER, _PRESSURE, model))
    assert endpoint == direct
    assert endpoint["classifier"] == "xgboost"
    # and it is genuinely a different model, not the default silently reused
    assert endpoint != backend.api_classify(_RF_POWER, _PRESSURE, classifier="random_forest")


def test_classify_endpoint_rejects_unknown_classifier() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        backend.api_classify(_RF_POWER, _PRESSURE, classifier="not_a_model")
    assert exc_info.value.status_code == 422


def test_explain_endpoint_matches_direct_call() -> None:
    """/api/explain == asdict(explain_configuration(...)) using the SAME
    explainer-enabled classifier and the SAME background sample."""
    model = backend.get_explainer_classifiers()[ClassifierKind.RANDOM_FOREST]
    background = backend.get_shap_background()
    endpoint = backend.api_explain(_RF_POWER, _PRESSURE)
    direct = dataclasses.asdict(
        explain_configuration(_RF_POWER, _PRESSURE, model, background)
    )
    assert endpoint == direct


def test_explain_background_matches_the_dashboards() -> None:
    """The SHAP reference distribution must be identical to the dashboard's, or the
    two UIs would report different SHAP values for the same operating point."""
    from dashboard.backend import get_shap_background as dashboard_background

    ours = backend.get_shap_background()
    theirs = dashboard_background()
    assert list(ours.index) == list(theirs.index)
    assert ours.equals(theirs)


def test_explain_agrees_with_the_verdict_it_explains() -> None:
    """The explanation must be for the class the UI actually displays - otherwise
    the panel would justify a verdict the user never saw."""
    verdict = backend.api_classify(_RF_POWER, _PRESSURE)
    explanation = backend.api_explain(_RF_POWER, _PRESSURE)
    assert explanation["predicted_class"] == verdict["predicted_class"]
    assert set(explanation["feature_contributions"]) == set(FEATURE_COLUMNS)


def test_explain_endpoint_rejects_the_baseline_model() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        backend.api_explain(_RF_POWER, _PRESSURE, classifier="logistic_regression")
    assert exc_info.value.status_code == 422


def test_suitability_endpoint_matches_direct_call() -> None:
    """/api/suitability values tie back field-by-field to the DefectProbability-
    Estimate dataclasses from all_application_defect_estimates (same seed -> same
    bootstrap, so this is exact, not approximate)."""
    n_bootstrap = 50
    endpoint = backend.api_suitability(_RF_POWER, _PRESSURE, n_bootstrap=n_bootstrap)
    direct = all_application_defect_estimates(_RF_POWER, _PRESSURE, n_bootstrap=n_bootstrap)

    assert set(endpoint.keys()) == {app.value for app in direct}
    for app, est in direct.items():
        row = endpoint[app.value]
        assert row["application"] == app.value
        assert row["point_estimate"] == est.point_estimate
        assert row["ci_lower"] == est.ci_lower
        assert row["ci_upper"] == est.ci_upper
        assert row["confidence_level"] == est.confidence_level


def test_physics_validation_endpoint_matches_direct_call() -> None:
    """/api/physics-validation rows tie back field-by-field to the BenchmarkResult
    objects from run_literature_benchmarks(), including the source citation and the
    deviation_pct/passed properties, with full float precision (no rounding)."""
    endpoint = backend.api_physics_validation()
    results = run_literature_benchmarks()

    assert len(endpoint) == len(results)
    for row, r in zip(endpoint, results):
        assert row["name"] == r.name
        assert row["quantity"] == r.quantity
        assert row["description"] == r.description
        assert row["source"] == r.source
        assert row["computed_value"] == r.computed_value
        assert row["reference_value"] == r.reference_value
        assert row["unit"] == r.unit
        assert row["deviation_pct"] == r.deviation_pct
        assert row["tolerance_pct"] == r.tolerance_pct
        assert row["passed"] == r.passed


def test_suitability_scorecard_endpoint_matches_direct_call() -> None:
    """/api/suitability-scorecard ties back to classify_suitability(): best-fit,
    compliance %, and every application's rating, exactly (deterministic engine)."""
    endpoint = backend.api_suitability_scorecard(_RF_POWER, _PRESSURE)
    scorecard = classify_suitability(_RF_POWER, _PRESSURE)

    best = scorecard.best_application()
    assert endpoint["best_application"] == best.value
    assert endpoint["best_compliance_pct"] == scorecard.ratings[best].overall_compliance_pct
    assert endpoint["ion_energy_ev"] == scorecard.ion_energy_ev
    assert endpoint["defect_probability"] == scorecard.defect_probability
    assert {r["application"] for r in endpoint["ratings"]} == {a.value for a in scorecard.ratings}
    for row in endpoint["ratings"]:
        rating = scorecard.ratings[SemiconductorApplication(row["application"])]
        assert row["overall_compliance_pct"] == rating.overall_compliance_pct
        assert row["ion_energy_compliance_pct"] == rating.ion_energy_compliance_pct
        assert row["defect_compliance_pct"] == rating.defect_compliance_pct
        assert row["is_suitable"] == rating.is_suitable


def test_anomaly_endpoint_matches_direct_call() -> None:
    """/api/anomaly ties back to the detector's own severity/score/root-cause for
    every fault mode, healthy and injected. The fault injection is seeded, so the
    endpoint is deterministic and the comparison is exact."""
    detector = backend.get_anomaly_detector()
    for fault in ["none", "pressure_gauge_fault", "electrode_coupling_fault", "te_sensor_drift"]:
        endpoint = backend.api_anomaly(_RF_POWER, _PRESSURE, fault=fault)
        df = backend._anomaly_feature_row(_RF_POWER, _PRESSURE, fault)
        assert endpoint["fault"] == fault
        assert endpoint["severity"] == detector.severity(df)[0].value
        assert endpoint["score"] == float(detector.anomaly_score(df)[0])
        assert endpoint["is_anomaly"] == bool(detector.is_anomaly(df)[0])
        assert endpoint["root_cause"] == detector.root_cause(df)[0]


def test_anomaly_endpoint_rejects_unknown_fault() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        backend.api_anomaly(_RF_POWER, _PRESSURE, fault="not_a_real_fault")
    assert exc_info.value.status_code == 422


def test_sessions_endpoints_match_direct_calls(tmp_path, monkeypatch) -> None:
    """/api/sessions and /api/sessions/{id} return exactly what list_sessions /
    get_session return, against a temp DB seeded with real simulate() outputs."""
    db_path = tmp_path / "reactor_sessions.db"
    seed_db = ExperimentDatabase(db_path)
    session_ids = []
    for power, pressure in [(120.0, 8.0), (200.0, 12.0)]:
        result = simulate(power, pressure)
        session_ids.append(
            seed_db.create_session(ChamberParameters(power, pressure), result)
        )
    seed_db.close()

    # Point the backend's DB accessor at the temp database (same seam the
    # dashboard uses to keep sqlite connections per-caller).
    monkeypatch.setattr(backend, "open_session_db", lambda: ExperimentDatabase(db_path))

    ref_db = ExperimentDatabase(db_path)
    try:
        endpoint_list = backend.api_sessions()
        direct_list = [dataclasses.asdict(s) for s in ref_db.list_sessions()]
        assert endpoint_list == direct_list
        assert len(endpoint_list) == 2

        endpoint_one = backend.api_session(session_ids[0])
        direct_one = dataclasses.asdict(ref_db.get_session(session_ids[0]))
        assert endpoint_one == direct_one
    finally:
        ref_db.close()


def test_session_not_found_returns_404(tmp_path, monkeypatch) -> None:
    from fastapi import HTTPException

    db_path = tmp_path / "empty.db"
    monkeypatch.setattr(backend, "open_session_db", lambda: ExperimentDatabase(db_path))
    with pytest.raises(HTTPException) as exc_info:
        backend.api_session("does-not-exist")
    assert exc_info.value.status_code == 404


def test_system_stats_endpoint_wraps_psutil_faithfully(monkeypatch) -> None:
    """Live host metrics can't be value-identical across two reads, so instead
    patch psutil to known values and assert the endpoint echoes exactly them -
    proving it wraps psutil without recomputing - plus the honest idle-GPU note."""
    monkeypatch.setattr(backend.psutil, "cpu_percent", lambda interval=None: 42.5)

    class _FakeVM:
        percent = 37.0

    monkeypatch.setattr(backend.psutil, "virtual_memory", lambda: _FakeVM())

    stats = backend.api_system_stats()
    assert stats["cpu_percent"] == 42.5
    assert stats["ram_percent"] == 37.0
    assert stats["gpu"]["status"] == "idle"
    assert "CPU-based tree ensembles" in stats["gpu"]["note"]
