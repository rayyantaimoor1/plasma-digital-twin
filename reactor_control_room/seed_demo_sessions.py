"""Populate a few demo experiment sessions for the Reactor Control Room's Session
Replay page (and, since it is the same store, the Streamlit dashboard's Session
History).

These are REAL `simulate()` outputs persisted through the real `ExperimentDatabase`
- genuine physics-engine runs tagged by experiment mode, never fabricated numbers.
The Session Replay page reads them back through /api/sessions.

Idempotent: if the store already has sessions, it leaves them alone.

Run from the project root:
    python reactor_control_room/seed_demo_sessions.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from digital_twin.chamber_config import ChamberParameters, ExperimentMode
from digital_twin.physics_engine import simulate
from digital_twin.session_manager import ExperimentDatabase

# A spread across the three experiment modes so the trend sparklines have shape.
_DEMO_RUNS = [
    (100.0, 10.0, ExperimentMode.STABLE_PLASMA),
    (120.0, 9.0, ExperimentMode.STABLE_PLASMA),
    (150.0, 10.0, ExperimentMode.EXPLORATORY_SWEEP),
    (80.0, 15.0, ExperimentMode.EXPLORATORY_SWEEP),
    (210.0, 5.0, ExperimentMode.EXPLORATORY_SWEEP),
    (280.0, 1.5, ExperimentMode.STRESS_TEST),
    (300.0, 2.0, ExperimentMode.STRESS_TEST),
    (100.0, 12.0, ExperimentMode.STABLE_PLASMA),
]


def main() -> None:
    db = ExperimentDatabase()
    try:
        existing = len(db.summary_report())
        if existing:
            print(f"{existing} session(s) already present - leaving the store untouched.")
            print("(To reseed from scratch, delete data/experiments.db first.)")
            return
        for rf_power_w, pressure_mtorr, mode in _DEMO_RUNS:
            result = simulate(rf_power_w, pressure_mtorr)
            session_id = db.create_session(
                ChamberParameters(rf_power_w=rf_power_w, pressure_mtorr=pressure_mtorr),
                result,
                mode=mode,
            )
            print(f"  seeded {session_id[:8]} - {mode.value:<17} {rf_power_w:>5.0f} W / {pressure_mtorr:>4.1f} mTorr")
        print(f"Seeded {len(_DEMO_RUNS)} demo sessions into data/experiments.db.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
