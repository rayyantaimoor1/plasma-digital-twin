"""Sub-Module 2.3 - Plasma Trend Analysis and Monitoring Engine.

Tracks how the digital twin's key outputs evolve across SEQUENTIAL runs within a
session - i.e. a time series ordered by when each run happened, not necessarily a
systematic parameter sweep (that is Sub-Module 1.5's job). This is the natural
counterpart to Sub-Module 1.4's session history: feed it a chronologically-ordered
list of session results and it tracks whether the process is drifting, degrading,
recovering, or oscillating over the course of a session.

Pipeline (FE-2.3.1 -> FE-2.3.2 -> FE-2.3.3 -> FE-2.3.4):
  1. Track the four monitored metrics per run (raw trend chart).
  2. EMA-smooth each metric to separate systematic trend from simulation noise.
  3. Detect trend-change events on the SMOOTHED series (monotonic degradation /
     recovery runs, and oscillation), since detecting them on the raw noisy
     series would just re-detect noise - defeating the point of smoothing.
  4. Summary statistics (mean/std/min/max/slope) over the session window.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress

from digital_twin.session_manager import SessionRecord

# The four metrics FE-2.3.1 names explicitly.
TRACKED_METRICS = ["reactivity_index", "process_quality", "uniformity_index", "electron_temperature_ev"]

# A monotonic run shorter than this many consecutive steps is treated as noise,
# not a genuine degradation/recovery sequence.
DEFAULT_MIN_RUN_LENGTH = 3
# Oscillation is checked in a sliding window of this many consecutive steps.
DEFAULT_OSCILLATION_WINDOW = 4
# Minimum sign flips among adjacent steps within that window to call it oscillating
# (out of at most window-1 possible flips) - "most, not necessarily all, steps
# reverse direction," more realistic for real instability than requiring perfect
# alternation.
DEFAULT_MIN_SIGN_CHANGES = 2


def session_records_to_trend_frame(records: list[SessionRecord]) -> pd.DataFrame:
    """Convert a chronologically-ordered list of SessionRecords (e.g. from
    ExperimentDatabase.list_sessions()) into a trend DataFrame indexed by run
    order within the session [FE-2.3.1]."""
    rows = [
        {"run_index": i, "created_at": r.created_at, **{m: getattr(r, m) for m in TRACKED_METRICS}}
        for i, r in enumerate(records)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EMA smoothing [FE-2.3.2]
# ---------------------------------------------------------------------------
def ema_smooth(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with an adjustable window (EMA span).

    Uses pandas' built-in `.ewm(span=window)` - the standard, well-tested EMA
    implementation - rather than hand-rolling the recursion.
    """
    if window < 1:
        raise ValueError("window must be >= 1.")
    return series.ewm(span=window, adjust=False).mean()


# ---------------------------------------------------------------------------
# Trend-change event detection [FE-2.3.3]
# ---------------------------------------------------------------------------
class TrendEventType(str, Enum):
    DEGRADATION = "degradation"
    RECOVERY = "recovery"
    OSCILLATION = "oscillation"


@dataclass
class TrendEvent:
    metric: str
    event_type: TrendEventType
    start_index: int
    end_index: int  # inclusive point index range [start_index, end_index]
    label: str       # human-readable annotation text for the chart


def _monotonic_runs(values: np.ndarray, min_run_length: int) -> list[tuple[int, int, TrendEventType]]:
    """Maximal runs of >= min_run_length consecutive same-sign steps.

    Returns (start_point_index, end_point_index, DEGRADATION|RECOVERY) triples.
    A zero-diff step (plateau) neither extends nor breaks a run; it is skipped.
    """
    diffs = np.diff(values)
    signs = np.sign(diffs)
    n = len(signs)
    runs: list[tuple[int, int, TrendEventType]] = []
    i = 0
    while i < n:
        if signs[i] == 0:
            i += 1
            continue
        j = i
        while j < n and signs[j] == signs[i]:
            j += 1
        steps = j - i
        if steps >= min_run_length:
            event_type = TrendEventType.DEGRADATION if signs[i] < 0 else TrendEventType.RECOVERY
            runs.append((i, j, event_type))
        i = j
    return runs


def _oscillation_events(values: np.ndarray, window: int, min_sign_changes: int) -> list[tuple[int, int]]:
    """Sliding-window scan for stretches where the step direction flips
    repeatedly rather than trending consistently. Non-overlapping: once a window
    is flagged, the scan jumps past it rather than re-flagging near-duplicates."""
    diffs = np.diff(values)
    signs = np.sign(diffs)
    n = len(signs)
    events: list[tuple[int, int]] = []
    i = 0
    while i + window <= n:
        window_signs = signs[i : i + window]
        nonzero = window_signs[window_signs != 0]
        changes = int(np.sum(np.diff(nonzero) != 0)) if len(nonzero) > 1 else 0
        if changes >= min_sign_changes:
            events.append((i, i + window))
            i += window
        else:
            i += 1
    return events


def detect_trend_events(
    smoothed: pd.Series,
    metric: str,
    min_run_length: int = DEFAULT_MIN_RUN_LENGTH,
    oscillation_window: int = DEFAULT_OSCILLATION_WINDOW,
    min_sign_changes: int = DEFAULT_MIN_SIGN_CHANGES,
) -> list[TrendEvent]:
    """Detect degradation, recovery, and oscillation events on a SMOOTHED metric
    series [FE-2.3.3]. Must be called on the EMA-smoothed series (FE-2.3.2), not
    the raw series - detecting on raw noise would just re-detect the noise.
    """
    values = smoothed.to_numpy()
    events: list[TrendEvent] = []

    for start, end, event_type in _monotonic_runs(values, min_run_length):
        verb = "declining" if event_type == TrendEventType.DEGRADATION else "improving"
        label = f"{event_type.value.capitalize()}: {metric} {verb} over runs {start}-{end}"
        events.append(TrendEvent(metric, event_type, start, end, label))

    for start, end in _oscillation_events(values, oscillation_window, min_sign_changes):
        label = f"Instability: {metric} oscillating over runs {start}-{end}"
        events.append(TrendEvent(metric, TrendEventType.OSCILLATION, start, end, label))

    return sorted(events, key=lambda e: e.start_index)


# ---------------------------------------------------------------------------
# Trend summary statistics [FE-2.3.4]
# ---------------------------------------------------------------------------
@dataclass
class TrendSummary:
    metric: str
    mean: float
    std: float
    min: float
    max: float
    trend_direction_coefficient: float  # slope of metric vs. run index (linear regression)
    trend_r_squared: float


def trend_summary_statistics(frame: pd.DataFrame, metric: str) -> TrendSummary:
    """Mean, std, min, max, and a linear trend-direction coefficient (the slope
    of metric vs. run index) for one metric across the session window [FE-2.3.4].
    """
    values = frame[metric].to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("Need at least 2 runs to compute trend statistics.")
    run_index = np.arange(len(values), dtype=float)
    slope, _intercept, r, _p, _se = linregress(run_index, values)
    return TrendSummary(
        metric=metric,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        min=float(values.min()),
        max=float(values.max()),
        trend_direction_coefficient=float(slope),
        trend_r_squared=float(r**2),
    )


# ---------------------------------------------------------------------------
# Trend charts [FE-2.3.1, FE-2.3.2, FE-2.3.3]
# ---------------------------------------------------------------------------
_EVENT_COLOR = {
    TrendEventType.DEGRADATION: "rgba(214, 39, 40, 0.15)",
    TrendEventType.RECOVERY: "rgba(44, 160, 44, 0.15)",
    TrendEventType.OSCILLATION: "rgba(255, 127, 14, 0.15)",
}


def trend_chart(
    frame: pd.DataFrame,
    metric: str,
    smoothing_window: Optional[int] = None,
    events: Optional[list[TrendEvent]] = None,
) -> go.Figure:
    """Scrollable trend chart for one metric: raw series, optional EMA overlay,
    and optional shaded/annotated trend-event regions [FE-2.3.1, 2.3.2, 2.3.3]."""
    run_index = frame["run_index"] if "run_index" in frame.columns else np.arange(len(frame))
    raw = frame[metric]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=run_index, y=raw, mode="lines+markers", name=f"{metric} (raw)",
                              line=dict(color="#7f7f7f", width=1)))
    if smoothing_window is not None:
        smoothed = ema_smooth(raw, smoothing_window)
        fig.add_trace(go.Scatter(x=run_index, y=smoothed, mode="lines",
                                  name=f"{metric} (EMA-{smoothing_window})",
                                  line=dict(color="#1f77b4", width=2)))

    for event in events or []:
        fig.add_vrect(
            x0=event.start_index, x1=event.end_index,
            fillcolor=_EVENT_COLOR[event.event_type], line_width=0,
            annotation_text=event.event_type.value, annotation_position="top left",
        )

    fig.update_layout(title=f"Sub-Module 2.3: {metric} trend", xaxis_title="Run index", yaxis_title=metric)
    return fig


# ---------------------------------------------------------------------------
# Full-session orchestration
# ---------------------------------------------------------------------------
@dataclass
class MetricTrendReport:
    metric: str
    summary: TrendSummary
    events: list[TrendEvent]
    figure: go.Figure


def analyze_session_trends(
    frame: pd.DataFrame,
    metrics: list[str] = TRACKED_METRICS,
    smoothing_window: int = 5,
) -> dict[str, MetricTrendReport]:
    """Run the full FE-2.3.1-2.3.4 pipeline for every tracked metric in one call."""
    reports: dict[str, MetricTrendReport] = {}
    for metric in metrics:
        smoothed = ema_smooth(frame[metric], smoothing_window)
        events = detect_trend_events(smoothed, metric)
        summary = trend_summary_statistics(frame, metric)
        figure = trend_chart(frame, metric, smoothing_window=smoothing_window, events=events)
        reports[metric] = MetricTrendReport(metric=metric, summary=summary, events=events, figure=figure)
    return reports


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.trend_analysis
    # Synthetic session: a stable start, a degradation, a recovery, then instability.
    rng = np.random.default_rng(0)
    n = 30
    base = np.concatenate([
        np.full(8, 0.6), np.linspace(0.6, 0.3, 8), np.linspace(0.3, 0.65, 8),
        0.5 + 0.15 * np.array([(-1) ** i for i in range(6)]),
    ])
    quality = base + rng.normal(0, 0.02, size=n)
    demo = pd.DataFrame({"run_index": np.arange(n), "process_quality": quality,
                          "reactivity_index": quality, "uniformity_index": quality,
                          "electron_temperature_ev": 3.0 + quality})

    reports = analyze_session_trends(demo, metrics=["process_quality"])
    report = reports["process_quality"]
    print(f"Summary: mean={report.summary.mean:.3f} slope={report.summary.trend_direction_coefficient:+.4f} "
          f"R^2={report.summary.trend_r_squared:.3f}")
    print("Detected events:")
    for e in report.events:
        print(f"  {e.label}")
