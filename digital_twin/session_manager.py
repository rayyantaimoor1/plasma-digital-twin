"""Sub-Module 1.4 - Multi-Session Experiment Management and Historical Comparison.

Persists every simulation run to a local SQLite database so students can review,
replay, and compare historical experiments. This module owns the `sessions` table
schema; AI Module 2.x sub-modules (suitability classification, anomaly detection)
will write into the `suitability_classification` / `anomaly_count` columns once
they exist - those columns are defined now (nullable) so the schema does not need
to change later, per FE-1.4.1's requirement to persist "AI analysis results"
alongside parameters and simulation outputs.

Plain sqlite3 (stdlib) is used deliberately rather than an ORM: the schema is
small and fixed, and CLAUDE.md's tech stack lists SQLite directly - an ORM would
be an abstraction this project does not need.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from digital_twin.chamber_config import ChamberParameters, ExperimentMode
from digital_twin.physics_engine import SimulationResult

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "experiments.db"

# FE-1.4.3: the overlay comparison chart becomes unreadable beyond this many
# simultaneous traces, so the retrieval API enforces the same cap the dashboard
# will use.
MAX_COMPARISON_SESSIONS = 5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT,
    rf_power_w REAL NOT NULL,
    pressure_mtorr REAL NOT NULL,
    gas_flow_sccm REAL,
    exposure_time_s REAL,
    electron_temperature_ev REAL NOT NULL,
    plasma_density_m3 REAL NOT NULL,
    ion_flux_m2s REAL NOT NULL,
    sheath_voltage_v REAL NOT NULL,
    ion_energy_ev REAL NOT NULL,
    reactivity_index REAL NOT NULL,
    uniformity_index REAL NOT NULL,
    etch_rate_nm_min REAL NOT NULL,
    process_quality REAL NOT NULL,
    defect_probability REAL NOT NULL,
    suitability_classification TEXT,
    anomaly_count INTEGER
);
"""


@dataclass
class SessionRecord:
    """One stored experiment: full parameter configuration + simulation outputs,
    plus (once Module 2 exists) AI analysis results [FE-1.4.1].
    """
    session_id: str
    created_at: str
    mode: Optional[str]
    rf_power_w: float
    pressure_mtorr: float
    gas_flow_sccm: Optional[float]
    exposure_time_s: Optional[float]
    electron_temperature_ev: float
    plasma_density_m3: float
    ion_flux_m2s: float
    sheath_voltage_v: float
    ion_energy_ev: float
    reactivity_index: float
    uniformity_index: float
    etch_rate_nm_min: float
    process_quality: float
    defect_probability: float
    suitability_classification: Optional[str]
    anomaly_count: Optional[int]

    @classmethod
    def column_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SessionRecord":
        return cls(**{name: row[name] for name in cls.column_names()})


class ExperimentDatabase:
    """SQLite-backed store for session persistence, retrieval, and comparison."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentDatabase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def create_session(
        self,
        parameters: ChamberParameters,
        result: SimulationResult,
        mode: Optional[ExperimentMode] = None,
    ) -> str:
        """Persist one simulation run under a new session id [FE-1.4.1]."""
        session_id = str(uuid.uuid4())
        row = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode.value if mode is not None else None,
            "rf_power_w": parameters.rf_power_w,
            "pressure_mtorr": parameters.pressure_mtorr,
            "gas_flow_sccm": parameters.gas_flow_sccm,
            "exposure_time_s": parameters.exposure_time_s,
            "electron_temperature_ev": result.electron_temperature_ev,
            "plasma_density_m3": result.plasma_density_m3,
            "ion_flux_m2s": result.ion_flux_m2s,
            "sheath_voltage_v": result.sheath_voltage_v,
            "ion_energy_ev": result.ion_energy_ev,
            "reactivity_index": result.reactivity_index,
            "uniformity_index": result.uniformity_index,
            "etch_rate_nm_min": result.etch_rate_nm_min,
            "process_quality": result.process_quality,
            "defect_probability": result.defect_probability,
            "suitability_classification": None,
            "anomaly_count": None,
        }
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self._conn.execute(f"INSERT INTO sessions ({columns}) VALUES ({placeholders})", row)
        self._conn.commit()
        return session_id

    def get_session(self, session_id: str) -> SessionRecord:
        """Reload one historical experiment by id, for replay/re-visualisation [FE-1.4.2]."""
        cursor = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"No session found with id {session_id!r}")
        return SessionRecord.from_row(row)

    def list_sessions(self, limit: Optional[int] = None) -> list[SessionRecord]:
        """List stored sessions, most recent first."""
        query = "SELECT * FROM sessions ORDER BY created_at DESC"
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        cursor = self._conn.execute(query, params)
        return [SessionRecord.from_row(row) for row in cursor.fetchall()]

    def get_sessions_for_comparison(self, session_ids: list[str]) -> list[SessionRecord]:
        """Fetch specific sessions to overlay-compare, capped at 5 [FE-1.4.3]."""
        if len(session_ids) > MAX_COMPARISON_SESSIONS:
            raise ValueError(
                f"Cannot compare more than {MAX_COMPARISON_SESSIONS} sessions at once "
                f"(got {len(session_ids)})."
            )
        return [self.get_session(sid) for sid in session_ids]

    def update_ai_results(
        self,
        session_id: str,
        suitability_classification: Optional[str] = None,
        anomaly_count: Optional[int] = None,
    ) -> None:
        """Attach AI Module results to an existing session (Sub-Modules 2.1/2.2)."""
        self.get_session(session_id)  # raises KeyError if missing
        self._conn.execute(
            "UPDATE sessions SET "
            "suitability_classification = COALESCE(?, suitability_classification), "
            "anomaly_count = COALESCE(?, anomaly_count) "
            "WHERE session_id = ?",
            (suitability_classification, anomaly_count, session_id),
        )
        self._conn.commit()

    def summary_report(self) -> pd.DataFrame:
        """Tabular summary of every stored session: configs, outcomes, AI results,
        and anomaly counts across all experiments [FE-1.4.4].
        """
        sessions = self.list_sessions()
        if not sessions:
            return pd.DataFrame(columns=SessionRecord.column_names())
        return pd.DataFrame([asdict(s) for s in sessions])
