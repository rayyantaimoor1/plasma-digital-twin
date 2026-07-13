"""Sub-Module 2.2 - Real-Time Anomaly Detection and Process Instability Monitor.

Per CLAUDE.md non-negotiable principle #4: anomalies are PHYSICS-RELATIONSHIP
VIOLATIONS, not out-of-range parameter values. A plain range check would already
catch out-of-range inputs; the point of Isolation Forest here is to catch faults
where every individual value is perfectly in-range but the input->output
RELATIONSHIP the digital twin defines has been broken - e.g. a pressure gauge
reading that makes the observed electron temperature inconsistent with the logged
pressure, or an electrode coupling fault that makes the density inconsistent with
the logged power.

================================================================================
WHY RAW FEATURES DON'T WORK, AND WHAT WE DO INSTEAD (verified empirically)
================================================================================
A relationship violation sits INSIDE the marginal hull of every individual
feature (each value occurs in normal data; only the COMBINATION is impossible),
so it is not isolated by axis-aligned cuts - an Isolation Forest on the raw 7
features achieves ~1% recall (verified). The physically-principled representation
is the RESIDUAL between the observed outputs and what the digital twin PREDICTS
for the logged inputs:

    residual_c = (observed_c - physics_predicted_c(logged_power, logged_pressure))
                 / |physics_predicted_c|          for each output channel c

For a normal run the residual is just measurement noise (small in every channel);
for a relationship violation it is large in the channel(s) the fault corrupts.
Computing it requires the physics model, so it is fundamentally NOT a raw range
check.

Isolation Forest is then trained on residual SUMMARY statistics
(max-abs, L2-norm, count-of-channels-over-3-sigma) rather than the raw per-channel
residual vector. This is deliberate: Isolation Forest averages path length over
all feature dimensions, so an anomaly confined to a MINORITY of dimensions - e.g.
a single-sensor drift, 1 of 5 residuals large - is diluted by the normal
dimensions and missed (verified: 0% recall on the raw residual vector for
single-channel faults). Every fault, single- or multi-channel, elevates all three
summary statistics, so there is no normal dimension to dilute against. The
per-channel residuals are retained separately for ROOT-CAUSE indication (which
channel is off -> which subsystem is faulty), supporting BO-5.

Measured performance on injected faults (see `evaluate_detector`): ~100% recall on
multi-channel faults (pressure-gauge, electrode-coupling), ~80% on single-channel
Te-sensor drift (the misses are high-pressure points where Te is physically flat,
so even a full-range drift is a small residual - an honest limitation, reported
rather than hidden), at ~1% false-positive rate. A naive range check gets ~0%.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

from digital_twin.physics_engine import ChamberGeometry, DEFAULT_GEOMETRY, simulate
from digital_twin.dataset_generation import FEATURE_COLUMNS
from digital_twin.session_manager import DEFAULT_DB_PATH

# The output channels whose relationship to the inputs a fault can break. The two
# inputs (rf_power_w, pressure_mtorr) are the logged setpoints and have no residual.
RESIDUAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("rf_power_w", "pressure_mtorr")]

NORMAL_MEASUREMENT_NOISE = 0.05  # 5% - the noise floor normal residuals sit at
N_OVER_SIGMA = 3.0               # a channel counts as "violated" above 3 residual-sigma

# Severity thresholds as quantiles of the NORMAL training-score distribution, so
# false-positive rates are calibrated by construction: ~1% of normal runs reach
# Warning and ~0.2% reach Critical.
WARNING_QUANTILE = 0.01
CRITICAL_QUANTILE = 0.002

# Minimum shift used when injecting a relationship-violating fault, large enough
# that the resulting residual clears the measurement-noise floor but constructed
# from valid operating points so every feature stays in-range.
MIN_PRESSURE_SHIFT_MTORR = 6.0
MIN_POWER_SHIFT_W = 100.0


class AnomalyFault(str, Enum):
    """Physically-named relationship-violating faults (support root-cause)."""
    PRESSURE_GAUGE_FAULT = "pressure_gauge_fault"      # outputs match a different pressure
    ELECTRODE_COUPLING_FAULT = "electrode_coupling_fault"  # density matches a different power
    TE_SENSOR_DRIFT = "te_sensor_drift"                # only the Te channel reads wrong


class AnomalySeverity(str, Enum):
    NORMAL = "Normal"
    WARNING = "Warning"
    CRITICAL = "Critical"


# ---------------------------------------------------------------------------
# Normal + anomalous data generation
# ---------------------------------------------------------------------------
def _feature_row(
    rf_power_w: float, pressure_mtorr: float, noise: float, seed: Optional[int],
    geometry: ChamberGeometry,
) -> dict[str, float]:
    result = simulate(rf_power_w, pressure_mtorr, noise_level=noise, seed=seed, geometry=geometry)
    return {c: getattr(result, c) for c in FEATURE_COLUMNS}


def generate_normal_operating_data(
    power_step_w: float = 12.5,
    pressure_step_mtorr: float = 1.0,
    replicates: int = 3,
    noise_level: float = NORMAL_MEASUREMENT_NOISE,
    seed: int = 0,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> pd.DataFrame:
    """Normal operating-condition data for training the detector (FE-2.2.1).

    A sweep of the operating envelope with measurement noise; every row is
    consistent with the physics (outputs match inputs up to noise). `noise_level`
    must be > 0 so the residual noise floor is well-defined.
    """
    if noise_level <= 0.0:
        raise ValueError("normal data must include measurement noise (noise_level > 0).")
    rng = np.random.default_rng(seed)
    powers = np.arange(50.0, 300.0 + 1e-9, power_step_w)
    pressures = np.arange(1.0, 20.0 + 1e-9, pressure_step_mtorr)
    rows = [
        _feature_row(float(p), float(q), noise_level, int(rng.integers(0, 2**32 - 1)), geometry)
        for p in powers for q in pressures for _ in range(replicates)
    ]
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


_ENVELOPE_POWERS = np.arange(50.0, 300.0 + 1e-9, 12.5)
_ENVELOPE_PRESSURES = np.arange(1.0, 20.0 + 1e-9, 1.0)


def inject_anomaly(
    rf_power_w: float,
    pressure_mtorr: float,
    fault: AnomalyFault,
    rng: np.random.Generator,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> dict[str, float]:
    """Construct one in-range, relationship-violating feature row (FE-2.2.1).

    Each fault makes the observed outputs correspond to a DIFFERENT valid
    operating point than the logged inputs, so every value is a genuine
    physics-engine output (hence in-range) but the joint tuple is impossible.
    """
    if fault == AnomalyFault.PRESSURE_GAUGE_FAULT:
        # Outputs computed at a different (valid) pressure; logged pressure unchanged.
        candidates = [p for p in _ENVELOPE_PRESSURES if abs(p - pressure_mtorr) >= MIN_PRESSURE_SHIFT_MTORR]
        p_actual = float(rng.choice(candidates))
        row = _feature_row(rf_power_w, p_actual, 0.0, None, geometry)
        row["rf_power_w"] = rf_power_w
        row["pressure_mtorr"] = pressure_mtorr
        return row

    if fault == AnomalyFault.ELECTRODE_COUPLING_FAULT:
        # Density/reactivity/etch computed at a different (valid) power; logged
        # power unchanged. (Te is pressure-set, so it stays correct - physically
        # right for a coupling fault, which changes delivered power not Te.)
        candidates = [p for p in _ENVELOPE_POWERS if abs(p - rf_power_w) >= MIN_POWER_SHIFT_W]
        p_actual = float(rng.choice(candidates))
        row = _feature_row(p_actual, pressure_mtorr, 0.0, None, geometry)
        row["rf_power_w"] = rf_power_w
        row["pressure_mtorr"] = pressure_mtorr
        return row

    if fault == AnomalyFault.TE_SENSOR_DRIFT:
        # Only the Te channel reads wrong: drifted toward the opposite pressure
        # extreme (where Te differs most), still a valid in-range Te value.
        row = _feature_row(rf_power_w, pressure_mtorr, 0.0, None, geometry)
        p_other = 1.0 if pressure_mtorr > 8.0 else 20.0
        row["electron_temperature_ev"] = _feature_row(rf_power_w, p_other, 0.0, None, geometry)[
            "electron_temperature_ev"
        ]
        return row

    raise ValueError(f"Unknown fault: {fault}")


def generate_anomalous_data(
    n_samples: int = 150,
    seed: int = 1,
    faults: Optional[tuple[AnomalyFault, ...]] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> pd.DataFrame:
    """Labelled anomalous dataset (FEATURE_COLUMNS + 'fault_type') for evaluation."""
    faults = faults or tuple(AnomalyFault)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        power = float(rng.choice(_ENVELOPE_POWERS))
        pressure = float(rng.choice(_ENVELOPE_PRESSURES))
        fault = faults[int(rng.integers(0, len(faults)))]
        row = inject_anomaly(power, pressure, fault, rng, geometry)
        row["fault_type"] = fault.value
        rows.append(row)
    return pd.DataFrame(rows)[FEATURE_COLUMNS + ["fault_type"]]


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------
@dataclass
class DetectionReport:
    """Detection performance, including the range-check contrast that
    demonstrates principle #4."""
    isolation_forest_recall: float
    isolation_forest_fpr: float
    range_check_recall: float
    per_fault_recall: dict[str, float]
    n_normal: int
    n_anomalous: int


class PlasmaAnomalyDetector:
    """Isolation Forest anomaly detector operating on physics residuals."""

    def __init__(
        self, n_estimators: int = 300, seed: int = 0, geometry: ChamberGeometry = DEFAULT_GEOMETRY
    ) -> None:
        self.geometry = geometry
        self._forest = IsolationForest(n_estimators=n_estimators, random_state=seed)
        self._residual_std: Optional[pd.Series] = None
        self.warning_threshold: Optional[float] = None
        self.critical_threshold: Optional[float] = None

    # -- residual computation --
    def _predicted_outputs(self, powers: np.ndarray, pressures: np.ndarray) -> pd.DataFrame:
        preds = [
            _feature_row(float(p), float(q), 0.0, None, self.geometry)
            for p, q in zip(powers, pressures)
        ]
        return pd.DataFrame(preds)[RESIDUAL_COLUMNS]

    def relative_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-channel relative residual (observed - predicted)/|predicted|."""
        predicted = self._predicted_outputs(df["rf_power_w"].to_numpy(), df["pressure_mtorr"].to_numpy())
        observed = df[RESIDUAL_COLUMNS].reset_index(drop=True)
        return (observed - predicted) / (predicted.abs() + 1e-30)

    def _standardized_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._residual_std is None:
            raise RuntimeError("Detector must be fit before computing standardized residuals.")
        return self.relative_residuals(df) / self._residual_std

    def _residual_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """The three residual summary features Isolation Forest trains on - each
        elevated by any fault, so single-channel anomalies are not diluted."""
        abs_residuals = self._standardized_residuals(df).abs()
        return pd.DataFrame({
            "max_abs_residual": abs_residuals.max(axis=1),
            "l2_norm_residual": np.sqrt((abs_residuals**2).sum(axis=1)),
            "n_channels_over_sigma": (abs_residuals > N_OVER_SIGMA).sum(axis=1),
        })

    # -- fit / score --
    def fit(self, normal_df: pd.DataFrame) -> "PlasmaAnomalyDetector":
        """Train on normal operating data; calibrate residual scale and severity
        thresholds from the normal residual/score distributions."""
        residuals = self.relative_residuals(normal_df)
        # Clamp to avoid divide-by-zero if a channel had no variation (shouldn't
        # happen with noise > 0, but keeps standardisation safe).
        self._residual_std = residuals.std().replace(0.0, 1e-9)

        summary = self._residual_summary(normal_df)
        self._forest.fit(summary)
        normal_scores = self._forest.decision_function(summary)
        self.warning_threshold = float(np.quantile(normal_scores, WARNING_QUANTILE))
        self.critical_threshold = float(np.quantile(normal_scores, CRITICAL_QUANTILE))
        return self

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """Isolation Forest decision-function score; lower = more anomalous."""
        return self._forest.decision_function(self._residual_summary(df))

    def is_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        return self.anomaly_score(df) < self.warning_threshold

    def severity(self, df: pd.DataFrame) -> list[AnomalySeverity]:
        """Normal / Warning / Critical per row (FE-2.2.2)."""
        scores = self.anomaly_score(df)
        out = []
        for s in scores:
            if s < self.critical_threshold:
                out.append(AnomalySeverity.CRITICAL)
            elif s < self.warning_threshold:
                out.append(AnomalySeverity.WARNING)
            else:
                out.append(AnomalySeverity.NORMAL)
        return out

    def root_cause(self, df: pd.DataFrame) -> list[str]:
        """Indicate the likely faulty subsystem from the largest-residual channel
        (BO-5 root-cause indication)."""
        residuals = self._standardized_residuals(df).abs()
        worst = residuals.idxmax(axis=1)
        hints = {
            "electron_temperature_ev": "electron-temperature channel off (pressure gauge / Te sensor)",
            "plasma_density_m3": "density channel off (electrode coupling / power delivery)",
            "reactivity_index": "reactive-flux channel off (electrode coupling / power delivery)",
            "uniformity_index": "uniformity channel off (pressure/geometry)",
            "etch_rate_nm_min": "etch-rate channel off (electrode coupling / power delivery)",
        }
        return [hints.get(channel, channel) for channel in worst]


# ---------------------------------------------------------------------------
# Range-check baseline + evaluation (demonstrates principle #4)
# ---------------------------------------------------------------------------
def normal_feature_ranges(normal_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {c: (float(normal_df[c].min()), float(normal_df[c].max())) for c in FEATURE_COLUMNS}


def range_check_is_anomaly(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    """The NAIVE baseline: flag a row only if some feature is out of the normal
    per-feature range. This is what principle #4 says is NOT enough - it catches
    ~0% of relationship violations, because they are all in-range."""
    out = np.zeros(len(df), dtype=bool)
    for c, (lo, hi) in ranges.items():
        out |= (df[c].to_numpy() < lo) | (df[c].to_numpy() > hi)
    return out


def evaluate_detector(
    detector: PlasmaAnomalyDetector,
    normal_test: pd.DataFrame,
    anomalous_test: pd.DataFrame,
) -> DetectionReport:
    """Compare Isolation Forest detection against the naive range check on the
    same injected physics-relationship violations."""
    ranges = normal_feature_ranges(normal_test)
    anomalous_features = anomalous_test[FEATURE_COLUMNS]

    if_anomaly = detector.is_anomaly(anomalous_features)
    if_normal = detector.is_anomaly(normal_test)
    range_anomaly = range_check_is_anomaly(anomalous_features, ranges)

    per_fault = {
        fault: float(if_anomaly[(anomalous_test["fault_type"] == fault).to_numpy()].mean())
        for fault in anomalous_test["fault_type"].unique()
    }
    return DetectionReport(
        isolation_forest_recall=float(if_anomaly.mean()),
        isolation_forest_fpr=float(if_normal.mean()),
        range_check_recall=float(range_anomaly.mean()),
        per_fault_recall=per_fault,
        n_normal=len(normal_test),
        n_anomalous=len(anomalous_test),
    )


# ---------------------------------------------------------------------------
# Statistical process control (FE-2.2.4)
# ---------------------------------------------------------------------------
@dataclass
class SPCMonitor:
    """Shewhart control chart on the process quality index across sequential runs
    (FE-2.2.4). Control limits are the classic mean +- 3-sigma of a baseline;
    runs outside them are flagged out-of-control. This is a univariate SPC check
    complementary to the multivariate Isolation Forest relationship detector."""
    center_line: float
    sigma: float

    @classmethod
    def from_baseline(cls, baseline_quality: np.ndarray) -> "SPCMonitor":
        arr = np.asarray(baseline_quality, dtype=float)
        if len(arr) < 2:
            raise ValueError("SPC baseline needs at least 2 runs.")
        return cls(center_line=float(arr.mean()), sigma=float(arr.std(ddof=1)))

    @property
    def upper_control_limit(self) -> float:
        return self.center_line + 3.0 * self.sigma

    @property
    def lower_control_limit(self) -> float:
        return self.center_line - 3.0 * self.sigma

    def out_of_control(self, quality_values: np.ndarray) -> np.ndarray:
        arr = np.asarray(quality_values, dtype=float)
        return (arr > self.upper_control_limit) | (arr < self.lower_control_limit)


# ---------------------------------------------------------------------------
# Anomaly event logging (FE-2.2.3)
# ---------------------------------------------------------------------------
_ANOMALY_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT,
    created_at TEXT NOT NULL,
    rf_power_w REAL NOT NULL,
    pressure_mtorr REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    severity TEXT NOT NULL,
    root_cause TEXT
);
"""


def store_anomaly_event(
    session_id: Optional[str],
    rf_power_w: float,
    pressure_mtorr: float,
    anomaly_score: float,
    severity: AnomalySeverity,
    root_cause: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Record a detected anomaly with parameter state, score, severity, timestamp
    and session id for post-session review (FE-2.2.3)."""
    event_id = str(uuid.uuid4())
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_ANOMALY_EVENTS_SCHEMA)
        conn.execute(
            "INSERT INTO anomaly_events (event_id, session_id, created_at, rf_power_w, "
            "pressure_mtorr, anomaly_score, severity, root_cause) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id, session_id, datetime.now(timezone.utc).isoformat(),
                rf_power_w, pressure_mtorr, anomaly_score, severity.value, root_cause,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return event_id


def load_anomaly_events(
    session_id: Optional[str] = None, db_path: Path | str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Load logged anomaly events, optionally filtered to one session."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_ANOMALY_EVENTS_SCHEMA)
        if session_id is None:
            frame = pd.read_sql_query("SELECT * FROM anomaly_events ORDER BY created_at", conn)
        else:
            frame = pd.read_sql_query(
                "SELECT * FROM anomaly_events WHERE session_id = ? ORDER BY created_at",
                conn, params=(session_id,),
            )
    finally:
        conn.close()
    return frame


# ---------------------------------------------------------------------------
# Anomaly timeline visualisation (FE-2.2.5)
# ---------------------------------------------------------------------------
_SEVERITY_COLOR = {
    AnomalySeverity.NORMAL.value: "#2ca02c",
    AnomalySeverity.WARNING.value: "#ff7f0e",
    AnomalySeverity.CRITICAL.value: "#d62728",
}


def anomaly_timeline_plot(
    run_index: list[int],
    anomaly_scores: list[float],
    severities: list[AnomalySeverity],
    warning_threshold: float,
    critical_threshold: float,
) -> go.Figure:
    """Timeline of anomaly score across sequential runs in a session, coloured by
    severity, with the Warning/Critical threshold lines (FE-2.2.5)."""
    severity_values = [s.value if isinstance(s, AnomalySeverity) else s for s in severities]
    colors = [_SEVERITY_COLOR.get(s, "#7f7f7f") for s in severity_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run_index, y=anomaly_scores, mode="lines+markers",
        marker=dict(color=colors, size=9), line=dict(color="#888", width=1),
        text=severity_values, name="anomaly score",
    ))
    fig.add_hline(y=warning_threshold, line_dash="dash", line_color="#ff7f0e",
                  annotation_text="Warning")
    fig.add_hline(y=critical_threshold, line_dash="dash", line_color="#d62728",
                  annotation_text="Critical")
    fig.update_layout(
        title="Sub-Module 2.2: anomaly timeline (lower score = more anomalous)",
        xaxis_title="Run index in session",
        yaxis_title="Isolation Forest anomaly score",
    )
    return fig


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.anomaly_detection
    print("Training anomaly detector on normal operating data...")
    normal = generate_normal_operating_data()
    detector = PlasmaAnomalyDetector().fit(normal)

    normal_test = generate_normal_operating_data(seed=99, replicates=1)
    anomalous = generate_anomalous_data(n_samples=150)
    report = evaluate_detector(detector, normal_test, anomalous)

    print(f"\nNormal runs: {report.n_normal}   Anomalous runs: {report.n_anomalous}")
    print(f"Isolation Forest recall:  {report.isolation_forest_recall:.1%}")
    print(f"Isolation Forest FPR:     {report.isolation_forest_fpr:.1%}")
    print(f"Naive range-check recall: {report.range_check_recall:.1%}  "
          f"<- principle #4: range check misses relationship violations")
    print("Per-fault recall:")
    for fault, recall in report.per_fault_recall.items():
        print(f"    {fault:26s} {recall:.1%}")
