"""Sub-Module 2.6 - Intelligent, Counterfactual Recommendation Engine.

Per CLAUDE.md non-negotiable principle #5: every recommendation is COUNTERFACTUAL.
A rule-based best-practice layer proposes candidate parameter adjustments, but no
suggestion is ever presented as bare advisory text - each candidate is re-run
through the actual digital twin (Sub-Module 1.2's simulate()) and the REAL
predicted numeric outcome is reported. The quantified before/after is the whole
point: "reduce RF power from 200 W to 180 W -> process quality 0.271 -> 0.288".

Layering (FE-2.6.2 -> FE-2.6.1 -> FE-2.6.3):
  * FE-2.6.2 (rule layer): plasma-processing best practices propose candidate
    adjustments (reduce pressure when uniformity is low, reduce power/bias when
    defect risk is high, raise power when reactivity is low), plus an exploratory
    palette of single-parameter steps so there are always enough candidates to
    rank and at least three to return.
  * FE-2.6.1 (counterfactual quantification): every candidate is re-simulated;
    the reported effect is the real re-simulated delta, not the rule's guess.
  * FE-2.6.3 (ML-derived scoring): candidates are RANKED by the re-simulated
    process-quality-index improvement - the quantified result, not the heuristic
    that proposed them - optionally enhanced by an ML classifier's predicted
    suitability-class transition (Sub-Module 2.1), so a move that also lifts the
    class (e.g. Acceptable -> Optimal) is preferred.
  * FE-2.6.4 (history + verification): recommendations are logged with their
    quantified predicted effect, and a later "apply" can be verified against the
    prediction.

Single-parameter adjustments are used deliberately so each recommendation is
interpretable ("reduce RF power by 20 W"), rather than an opaque multi-parameter
move.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from digital_twin.chamber_config import (
    PRESSURE_MAX_MTORR,
    PRESSURE_MIN_MTORR,
    RF_POWER_MAX_W,
    RF_POWER_MIN_W,
)
from digital_twin.dataset_generation import SUITABILITY_CLASSES
from digital_twin.physics_engine import ChamberGeometry, DEFAULT_GEOMETRY, SimulationResult, simulate
from digital_twin.session_manager import DEFAULT_DB_PATH
from ai_module.classification import PlasmaClassifier, classify_configuration

# rf_voltage has no chamber-config range (it is a physics option, not a core
# chamber input); bound candidate bias voltages to a physically sane span.
RF_VOLTAGE_MIN_V = 0.0
RF_VOLTAGE_MAX_V = 600.0

# Best-practice rule trigger thresholds (when a candidate is worth PROPOSING;
# whether it actually helps is decided by re-simulation, not these).
UNIFORMITY_CONCERN = 0.5
DEFECT_CONCERN = 0.5
REACTIVITY_CONCERN = 0.5

# A one-class suitability improvement is worth this much process-quality-equivalent
# when ranking (quality spans ~0-0.5 in this model, so 0.1 makes a class jump
# meaningful but not dominant over a large quality gain).
CLASS_IMPROVEMENT_WEIGHT = 0.1
DEFAULT_MAX_RECOMMENDATIONS = 3

_CLASS_RANK = {name: i for i, name in enumerate(SUITABILITY_CLASSES)}


@dataclass(frozen=True)
class Configuration:
    """A parameter operating point the engine can recommend moving to."""
    rf_power_w: float
    pressure_mtorr: float
    rf_voltage_v: Optional[float] = None


@dataclass
class Recommendation:
    """One counterfactual recommendation: a proposed change plus its REAL
    re-simulated predicted outcome (principle #5)."""
    action_text: str
    rationale: str
    baseline: Configuration
    recommended: Configuration
    quality_before: float
    quality_after: float
    quality_delta: float
    uniformity_before: float
    uniformity_after: float
    defect_before: float
    defect_after: float
    ion_energy_before: float
    ion_energy_after: float
    class_before: Optional[str]
    class_after: Optional[str]
    score: float
    prediction_text: str

    @property
    def improves(self) -> bool:
        return self.quality_delta > 0.0


# ---------------------------------------------------------------------------
# Envelope clamping and candidate identity
# ---------------------------------------------------------------------------
def _clamp(cfg: Configuration) -> Configuration:
    power = min(max(cfg.rf_power_w, RF_POWER_MIN_W), RF_POWER_MAX_W)
    pressure = min(max(cfg.pressure_mtorr, PRESSURE_MIN_MTORR), PRESSURE_MAX_MTORR)
    voltage = (
        None if cfg.rf_voltage_v is None
        else min(max(cfg.rf_voltage_v, RF_VOLTAGE_MIN_V), RF_VOLTAGE_MAX_V)
    )
    return Configuration(power, pressure, voltage)


def _key(cfg: Configuration) -> tuple:
    return (
        round(cfg.rf_power_w, 4),
        round(cfg.pressure_mtorr, 4),
        None if cfg.rf_voltage_v is None else round(cfg.rf_voltage_v, 4),
    )


# ---------------------------------------------------------------------------
# FE-2.6.2 - rule-based candidate proposal (best practices + exploration)
# ---------------------------------------------------------------------------
def _candidate_moves(cfg: Configuration, result: SimulationResult) -> list[tuple[Configuration, str]]:
    """Propose (candidate configuration, best-practice rationale) pairs. Ordered
    best-practice-first so that when an exploratory step lands on the same config,
    de-duplication keeps the richer best-practice rationale."""
    moves: list[tuple[Configuration, str]] = []

    # --- conditional best-practice rules ---
    if result.uniformity_index < UNIFORMITY_CONCERN:
        moves.append((replace(cfg, pressure_mtorr=cfg.pressure_mtorr * 0.7),
                      f"Uniformity is low ({result.uniformity_index:.2f}); reducing chamber "
                      f"pressure flattens the radial density profile (plasma-processing best practice)."))
    if result.defect_probability > DEFECT_CONCERN:
        moves.append((replace(cfg, rf_power_w=cfg.rf_power_w * 0.85),
                      f"Defect probability is high ({result.defect_probability:.2f}); reducing "
                      f"RF power lowers ion bombardment energy (best practice)."))
        if cfg.rf_voltage_v is not None:
            moves.append((replace(cfg, rf_voltage_v=cfg.rf_voltage_v * 0.7),
                          f"Defect probability is high ({result.defect_probability:.2f}); reducing "
                          f"RF bias voltage lowers ion bombardment energy (best practice)."))
    if result.reactivity_index < REACTIVITY_CONCERN:
        moves.append((replace(cfg, rf_power_w=cfg.rf_power_w * 1.15),
                      f"Reactivity is low ({result.reactivity_index:.2f}); increasing RF power "
                      f"raises the reactive ion flux (best practice)."))

    # --- exploratory single-parameter palette (guarantees enough candidates) ---
    for frac in (-0.2, -0.1, 0.1, 0.2):
        moves.append((replace(cfg, rf_power_w=cfg.rf_power_w * (1 + frac)),
                      f"Exploratory RF power adjustment ({frac:+.0%})."))
    for frac in (-0.3, -0.15, 0.15, 0.3):
        moves.append((replace(cfg, pressure_mtorr=cfg.pressure_mtorr * (1 + frac)),
                      f"Exploratory chamber pressure adjustment ({frac:+.0%})."))
    if cfg.rf_voltage_v is not None:
        for frac in (-0.3, 0.3):
            moves.append((replace(cfg, rf_voltage_v=cfg.rf_voltage_v * (1 + frac)),
                          f"Exploratory RF bias voltage adjustment ({frac:+.0%})."))
    return moves


# ---------------------------------------------------------------------------
# FE-2.6.1 - counterfactual quantification
# ---------------------------------------------------------------------------
def _action_text(a: Configuration, b: Configuration) -> str:
    if b.rf_power_w != a.rf_power_w:
        verb = "Reduce" if b.rf_power_w < a.rf_power_w else "Increase"
        return f"{verb} RF power from {a.rf_power_w:.0f} W to {b.rf_power_w:.0f} W"
    if b.pressure_mtorr != a.pressure_mtorr:
        verb = "Reduce" if b.pressure_mtorr < a.pressure_mtorr else "Increase"
        return f"{verb} chamber pressure from {a.pressure_mtorr:.1f} mTorr to {b.pressure_mtorr:.1f} mTorr"
    if (b.rf_voltage_v or 0.0) != (a.rf_voltage_v or 0.0):
        verb = "Reduce" if (b.rf_voltage_v or 0.0) < (a.rf_voltage_v or 0.0) else "Increase"
        return f"{verb} RF bias voltage from {a.rf_voltage_v:.0f} V to {b.rf_voltage_v:.0f} V"
    return "No change"


def _evaluate_candidate(
    baseline: Configuration,
    baseline_result: SimulationResult,
    candidate: Configuration,
    rationale: str,
    classifier: Optional[PlasmaClassifier],
    geometry: ChamberGeometry,
    class_before: Optional[str],
) -> Recommendation:
    """Re-run the candidate through the digital twin and build the quantified
    recommendation from the REAL predicted outcome (principle #5).

    `class_before` is the baseline's predicted class, computed ONCE by the caller
    and passed in [EFFICIENCY_REVIEW.md F5]: it is identical for every candidate
    (same baseline), so classifying the baseline per candidate re-ran an identical
    simulate()+predict for no benefit. It is None exactly when `classifier` is None.
    """
    candidate_result = simulate(
        candidate.rf_power_w, candidate.pressure_mtorr,
        rf_voltage_v=candidate.rf_voltage_v, geometry=geometry,
    )
    quality_delta = candidate_result.process_quality - baseline_result.process_quality

    class_after = None
    class_delta = 0
    if classifier is not None:
        # Classify on (power, pressure) only - the classifier was trained on
        # data generated without an applied RF voltage, so classifying is done
        # in that same distribution while the physics outcome above uses the full
        # (voltage-aware) simulation.
        class_after = classify_configuration(
            candidate.rf_power_w, candidate.pressure_mtorr, classifier, geometry=geometry
        ).predicted_class
        class_delta = _CLASS_RANK[class_after] - _CLASS_RANK[class_before]

    score = quality_delta + CLASS_IMPROVEMENT_WEIGHT * class_delta
    action = _action_text(baseline, candidate)

    prediction = (
        f"{action}: process quality {baseline_result.process_quality:.3f} -> "
        f"{candidate_result.process_quality:.3f} ({quality_delta:+.3f}), "
        f"uniformity {baseline_result.uniformity_index:.2f} -> {candidate_result.uniformity_index:.2f}, "
        f"defect probability {baseline_result.defect_probability:.2f} -> "
        f"{candidate_result.defect_probability:.2f}, "
        f"ion energy {baseline_result.ion_energy_ev:.0f} -> {candidate_result.ion_energy_ev:.0f} eV."
    )
    if classifier is not None:
        prediction += f" Classification: {class_before} -> {class_after}."

    return Recommendation(
        action_text=action, rationale=rationale, baseline=baseline, recommended=candidate,
        quality_before=baseline_result.process_quality, quality_after=candidate_result.process_quality,
        quality_delta=quality_delta,
        uniformity_before=baseline_result.uniformity_index, uniformity_after=candidate_result.uniformity_index,
        defect_before=baseline_result.defect_probability, defect_after=candidate_result.defect_probability,
        ion_energy_before=baseline_result.ion_energy_ev, ion_energy_after=candidate_result.ion_energy_ev,
        class_before=class_before, class_after=class_after, score=score, prediction_text=prediction,
    )


# ---------------------------------------------------------------------------
# FE-2.6.1 + FE-2.6.3 - generate, quantify, and ML-rank
# ---------------------------------------------------------------------------
def recommend(
    baseline: Configuration,
    classifier: Optional[PlasmaClassifier] = None,
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> list[Recommendation]:
    """Produce up to `max_recommendations` counterfactual recommendations, ranked
    by re-simulated process-quality improvement (FE-2.6.1, 2.6.3). At least three
    distinct candidates are evaluated whenever the envelope allows.

    Every returned recommendation carries its real re-simulated numeric outcome;
    re-running simulate() on `recommended` reproduces the reported after-values
    exactly (principle #5) - the tests assert this.
    """
    baseline = _clamp(baseline)
    baseline_result = simulate(
        baseline.rf_power_w, baseline.pressure_mtorr,
        rf_voltage_v=baseline.rf_voltage_v, geometry=geometry,
    )

    # Classify the baseline ONCE (it is the same for every candidate) rather than
    # re-classifying it inside each _evaluate_candidate call [EFFICIENCY_REVIEW.md F5].
    class_before = None
    if classifier is not None:
        class_before = classify_configuration(
            baseline.rf_power_w, baseline.pressure_mtorr, classifier, geometry=geometry
        ).predicted_class

    seen = {_key(baseline)}
    recommendations: list[Recommendation] = []
    for candidate, rationale in _candidate_moves(baseline, baseline_result):
        candidate = _clamp(candidate)
        key = _key(candidate)
        if key in seen:  # duplicate or a no-op after clamping
            continue
        seen.add(key)
        recommendations.append(
            _evaluate_candidate(baseline, baseline_result, candidate, rationale, classifier, geometry, class_before)
        )

    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations[:max_recommendations]


# ---------------------------------------------------------------------------
# FE-2.6.4 - recommendation history log + apply/verify
# ---------------------------------------------------------------------------
_RECOMMENDATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendation_events (
    recommendation_id TEXT PRIMARY KEY,
    session_id TEXT,
    created_at TEXT NOT NULL,
    baseline_power_w REAL NOT NULL,
    baseline_pressure_mtorr REAL NOT NULL,
    baseline_voltage_v REAL,
    recommended_power_w REAL NOT NULL,
    recommended_pressure_mtorr REAL NOT NULL,
    recommended_voltage_v REAL,
    quality_before REAL NOT NULL,
    quality_after REAL NOT NULL,
    quality_delta REAL NOT NULL,
    action_text TEXT NOT NULL,
    rationale TEXT NOT NULL,
    prediction_text TEXT NOT NULL,
    applied INTEGER,
    observed_quality_after REAL,
    prediction_verified INTEGER
);
"""


def store_recommendation(
    recommendation: Recommendation,
    session_id: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Log a recommendation with its quantified predicted effect (FE-2.6.4)."""
    recommendation_id = str(uuid.uuid4())
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_RECOMMENDATIONS_SCHEMA)
        conn.execute(
            "INSERT INTO recommendation_events ("
            "recommendation_id, session_id, created_at, "
            "baseline_power_w, baseline_pressure_mtorr, baseline_voltage_v, "
            "recommended_power_w, recommended_pressure_mtorr, recommended_voltage_v, "
            "quality_before, quality_after, quality_delta, "
            "action_text, rationale, prediction_text, "
            "applied, observed_quality_after, prediction_verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                recommendation_id, session_id, datetime.now(timezone.utc).isoformat(),
                recommendation.baseline.rf_power_w, recommendation.baseline.pressure_mtorr,
                recommendation.baseline.rf_voltage_v,
                recommendation.recommended.rf_power_w, recommendation.recommended.pressure_mtorr,
                recommendation.recommended.rf_voltage_v,
                recommendation.quality_before, recommendation.quality_after, recommendation.quality_delta,
                recommendation.action_text, recommendation.rationale, recommendation.prediction_text,
                None, None, None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return recommendation_id


def load_recommendations(
    session_id: Optional[str] = None, db_path: Path | str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Load the recommendation history, optionally filtered to one session (FE-2.6.4)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_RECOMMENDATIONS_SCHEMA)
        if session_id is None:
            frame = pd.read_sql_query("SELECT * FROM recommendation_events ORDER BY created_at", conn)
        else:
            frame = pd.read_sql_query(
                "SELECT * FROM recommendation_events WHERE session_id = ? ORDER BY created_at",
                conn, params=(session_id,),
            )
    finally:
        conn.close()
    return frame


def verify_recommendation(
    recommendation_id: str,
    applied: Configuration,
    db_path: Path | str = DEFAULT_DB_PATH,
    tolerance: float = 1e-6,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> bool:
    """Record whether an actually-applied adjustment produced the predicted
    improvement (FE-2.6.4).

    Re-simulates the applied configuration, compares the observed quality
    improvement (relative to the recorded baseline) against the predicted delta,
    and marks the recommendation verified if the applied change achieved at least
    the predicted gain (within tolerance). Because the twin is deterministic,
    applying exactly the recommended configuration verifies True; applying a
    weaker change verifies False.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_RECOMMENDATIONS_SCHEMA)
        row = conn.execute(
            "SELECT quality_before, quality_delta FROM recommendation_events WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No recommendation with id {recommendation_id!r}")
        quality_before, predicted_delta = row

        observed = simulate(
            applied.rf_power_w, applied.pressure_mtorr,
            rf_voltage_v=applied.rf_voltage_v, geometry=geometry,
        ).process_quality
        observed_delta = observed - quality_before
        verified = observed_delta >= predicted_delta - tolerance

        conn.execute(
            "UPDATE recommendation_events SET applied = 1, observed_quality_after = ?, "
            "prediction_verified = ? WHERE recommendation_id = ?",
            (observed, int(verified), recommendation_id),
        )
        conn.commit()
    finally:
        conn.close()
    return verified


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.recommendation_engine
    baseline = Configuration(rf_power_w=250.0, pressure_mtorr=18.0)
    print(f"Baseline: {baseline.rf_power_w:.0f} W, {baseline.pressure_mtorr:.0f} mTorr\n")

    recommendations = recommend(baseline)
    print(f"Top {len(recommendations)} counterfactual recommendations (ranked by "
          f"re-simulated quality gain):\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.prediction_text}")
        print(f"   rationale: {rec.rationale}\n")

    # Verify the top recommendation by "applying" exactly it.
    top = recommendations[0]
    import tempfile
    db = Path(tempfile.mkdtemp()) / "rec.db"
    rec_id = store_recommendation(top, session_id="demo", db_path=db)
    verified = verify_recommendation(rec_id, top.recommended, db_path=db)
    print(f"Applied the top recommendation exactly -> prediction verified: {verified}")
