"""Tests for Sub-Module 1.4 - multi-session experiment management and comparison.

All tests use an isolated tmp_path database, never the real data/experiments.db,
so running the suite never pollutes (or depends on) real experiment history.
"""
import sqlite3

import pandas as pd
import pytest

from digital_twin.chamber_config import ChamberParameters, ExperimentMode
from digital_twin.physics_engine import simulate
from digital_twin.session_manager import ExperimentDatabase, MAX_COMPARISON_SESSIONS


@pytest.fixture
def db(tmp_path):
    database = ExperimentDatabase(tmp_path / "test_experiments.db")
    yield database
    database.close()


def _make_session(database, power=100.0, pressure=10.0, mode=None):
    params = ChamberParameters(rf_power_w=power, pressure_mtorr=pressure)
    result = simulate(params.rf_power_w, params.pressure_mtorr)
    session_id = database.create_session(params, result, mode=mode)
    return session_id, params, result


# ---------------------------------------------------------------------------
# Create / persist [FE-1.4.1]
# ---------------------------------------------------------------------------
def test_create_session_assigns_unique_ids(db) -> None:
    id1, _, _ = _make_session(db)
    id2, _, _ = _make_session(db)
    assert id1 != id2
    assert len(id1) == 36  # UUID4 string length


def test_create_session_persists_full_configuration_and_outputs(db) -> None:
    session_id, params, result = _make_session(
        db, power=150.0, pressure=8.0, mode=ExperimentMode.STABLE_PLASMA
    )
    record = db.get_session(session_id)
    assert record.rf_power_w == pytest.approx(params.rf_power_w)
    assert record.pressure_mtorr == pytest.approx(params.pressure_mtorr)
    assert record.gas_flow_sccm == pytest.approx(params.gas_flow_sccm)
    assert record.electron_temperature_ev == pytest.approx(result.electron_temperature_ev)
    assert record.plasma_density_m3 == pytest.approx(result.plasma_density_m3)
    assert record.process_quality == pytest.approx(result.process_quality)
    assert record.mode == "Stable Plasma"
    assert record.suitability_classification is None
    assert record.anomaly_count is None


def test_create_session_without_mode_stores_null_mode(db) -> None:
    session_id, _, _ = _make_session(db, mode=None)
    record = db.get_session(session_id)
    assert record.mode is None


# ---------------------------------------------------------------------------
# Retrieval / replay [FE-1.4.2]
# ---------------------------------------------------------------------------
def test_get_session_missing_id_raises(db) -> None:
    with pytest.raises(KeyError):
        db.get_session("does-not-exist")


def test_list_sessions_orders_most_recent_first(db) -> None:
    _make_session(db, power=60.0)
    _make_session(db, power=120.0)
    _make_session(db, power=200.0)
    sessions = db.list_sessions()
    timestamps = [s.created_at for s in sessions]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_sessions_respects_limit(db) -> None:
    for _ in range(4):
        _make_session(db)
    sessions = db.list_sessions(limit=2)
    assert len(sessions) == 2


def test_data_persists_across_reconnects(tmp_path) -> None:
    db_path = tmp_path / "persist.db"
    database = ExperimentDatabase(db_path)
    session_id, _params, _result = _make_session(database, power=120.0, pressure=6.0)
    database.close()

    reopened = ExperimentDatabase(db_path)
    record = reopened.get_session(session_id)
    assert record.rf_power_w == pytest.approx(120.0)
    reopened.close()


def test_context_manager_closes_connection(tmp_path) -> None:
    db_path = tmp_path / "ctx.db"
    with ExperimentDatabase(db_path) as database:
        _make_session(database)
        internal_conn = database._conn
    with pytest.raises(sqlite3.ProgrammingError):
        internal_conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Multi-experiment overlay comparison [FE-1.4.3]
# ---------------------------------------------------------------------------
def test_comparison_fetches_requested_sessions(db) -> None:
    id1, _, _ = _make_session(db, power=60.0)
    id2, _, _ = _make_session(db, power=200.0)
    records = db.get_sessions_for_comparison([id1, id2])
    assert {r.session_id for r in records} == {id1, id2}


def test_comparison_allows_exactly_five(db) -> None:
    ids = [_make_session(db, power=p)[0] for p in (50, 100, 150, 200, 250)]
    assert len(ids) == MAX_COMPARISON_SESSIONS
    records = db.get_sessions_for_comparison(ids)
    assert len(records) == MAX_COMPARISON_SESSIONS


def test_comparison_rejects_more_than_five(db) -> None:
    ids = [_make_session(db, power=p)[0] for p in (50, 100, 150, 200, 250, 300)]
    assert len(ids) == MAX_COMPARISON_SESSIONS + 1
    with pytest.raises(ValueError):
        db.get_sessions_for_comparison(ids)


def test_comparison_missing_session_id_raises(db) -> None:
    id1, _, _ = _make_session(db)
    with pytest.raises(KeyError):
        db.get_sessions_for_comparison([id1, "not-a-real-id"])


# ---------------------------------------------------------------------------
# AI results attachment (schema readiness for Module 2.x)
# ---------------------------------------------------------------------------
def test_update_ai_results_attaches_classification_and_anomaly_count(db) -> None:
    session_id, _, _ = _make_session(db)
    db.update_ai_results(session_id, suitability_classification="Optimal", anomaly_count=0)
    record = db.get_session(session_id)
    assert record.suitability_classification == "Optimal"
    assert record.anomaly_count == 0


def test_update_ai_results_partial_update_preserves_other_field(db) -> None:
    session_id, _, _ = _make_session(db)
    db.update_ai_results(session_id, suitability_classification="Marginal")
    db.update_ai_results(session_id, anomaly_count=2)
    record = db.get_session(session_id)
    assert record.suitability_classification == "Marginal"
    assert record.anomaly_count == 2


def test_update_ai_results_missing_session_raises(db) -> None:
    with pytest.raises(KeyError):
        db.update_ai_results("does-not-exist", suitability_classification="Optimal")


# ---------------------------------------------------------------------------
# Session summary reports [FE-1.4.4]
# ---------------------------------------------------------------------------
def test_summary_report_is_dataframe_with_one_row_per_session(db) -> None:
    _make_session(db, power=60.0)
    _make_session(db, power=200.0)
    report = db.summary_report()
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 2
    assert "process_quality" in report.columns
    assert "rf_power_w" in report.columns
    assert "anomaly_count" in report.columns


def test_summary_report_empty_database_returns_empty_dataframe(tmp_path) -> None:
    database = ExperimentDatabase(tmp_path / "empty.db")
    report = database.summary_report()
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 0
    database.close()
