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

from ai_module.classification import ClassifierKind, classify_configuration
from ai_module.suitability_analysis import all_application_defect_estimates
from digital_twin.chamber_config import ChamberParameters
from digital_twin.physics_engine import simulate
from digital_twin.physics_validation import benchmark_summary_table
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
    """/api/physics-validation rows tie back to benchmark_summary_table() cell by
    cell, with full float precision preserved (no rounding in serialization)."""
    endpoint = backend.api_physics_validation()
    df = benchmark_summary_table()

    assert len(endpoint) == len(df)
    assert [r["name"] for r in endpoint] == df["name"].tolist()
    for i, row in enumerate(endpoint):
        assert row["computed_value"] == df.iloc[i]["computed_value"]
        assert row["reference_value"] == df.iloc[i]["reference_value"]
        assert row["deviation_pct"] == df.iloc[i]["deviation_pct"]
        assert row["tolerance_pct"] == df.iloc[i]["tolerance_pct"]
        assert row["passed"] == bool(df.iloc[i]["passed"])


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
