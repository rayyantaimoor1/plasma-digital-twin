"""Sub-Module 2.4 - Multi-Parameter Correlation and Relationship Analysis.

Analyses the relationships between plasma input parameters and the full digital
twin output vector (Sub-Module 1.2's SimulationResult fields) across a collection
of experiment results - a parameter sweep, a session history, or any DataFrame
with the same columns. Distinct from Sub-Module 2.3 (which tracks ONE metric over
sequential time) and Sub-Module 1.5 (which does controlled OAT/paired sweeps):
this sub-module looks at correlation structure across an arbitrary collection of
results, exactly as an operator's accumulated experiment history would look.

"AI-generated text summaries" (FE-2.4.4) here means template-generated narration
from the computed statistics, not an LLM call - consistent with CLAUDE.md's
scoping of this project to classical ML (the platform's own tech stack has no
generative-language component).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress

from digital_twin.physics_engine import ChamberGeometry, DEFAULT_GEOMETRY, simulate

# All numeric fields of SimulationResult: the two input setpoints plus every
# derived output, rigorous and illustrative alike (FE-2.4.1 asks for correlation
# across "all plasma input parameters and process output metrics").
CORRELATION_COLUMNS = [
    "rf_power_w",
    "pressure_mtorr",
    "electron_temperature_ev",
    "plasma_density_m3",
    "ion_flux_m2s",
    "sheath_voltage_v",
    "ion_energy_ev",
    "reactivity_index",
    "uniformity_index",
    "etch_rate_nm_min",
    "process_quality",
    "defect_probability",
]

# The three parameter-output relationships FE-2.4.2 names explicitly.
KEY_RELATIONSHIPS = [
    ("rf_power_w", "reactivity_index"),
    ("pressure_mtorr", "uniformity_index"),
    ("plasma_density_m3", "process_quality"),
]

# A |correlation| below this among a KEY_RELATIONSHIPS pair (physically expected
# to be related) is flagged as unexpectedly weak (FE-2.4.4).
LOW_CORRELATION_THRESHOLD = 0.3
INSIGHT_TOP_N = 5


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def simulation_sweep_dataframe(
    power_step_w: float = 12.5,
    pressure_step_mtorr: float = 1.0,
    noise_level: float = 0.05,
    seed: int = 0,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> pd.DataFrame:
    """A parameter sweep over the full operating envelope, with all
    SimulationResult fields - a convenient, reproducible data source for
    correlation analysis (an operator's real session history, via Sub-Module
    1.4's SessionRecord, has the same columns and works identically)."""
    rng = np.random.default_rng(seed)
    powers = np.arange(50.0, 300.0 + 1e-9, power_step_w)
    pressures = np.arange(1.0, 20.0 + 1e-9, pressure_step_mtorr)
    rows = []
    for p in powers:
        for q in pressures:
            result = simulate(
                float(p), float(q), noise_level=noise_level,
                seed=int(rng.integers(0, 2**32 - 1)), geometry=geometry,
            )
            rows.append({c: getattr(result, c) for c in CORRELATION_COLUMNS})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation matrix and heatmap [FE-2.4.1]
# ---------------------------------------------------------------------------
def correlation_matrix(frame: pd.DataFrame, columns: list[str] = CORRELATION_COLUMNS) -> pd.DataFrame:
    """Pearson correlation coefficients between every pair of parameters/outputs."""
    return frame[columns].corr(method="pearson")


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Interactive correlation heatmap (FE-2.4.1)."""
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0.0, zmin=-1.0, zmax=1.0,
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(title="Sub-Module 2.4: parameter-output correlation heatmap")
    return fig


# ---------------------------------------------------------------------------
# Scatter plots with regression trend lines [FE-2.4.2]
# ---------------------------------------------------------------------------
@dataclass
class RegressionResult:
    x: str
    y: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float


def scatter_with_regression(frame: pd.DataFrame, x: str, y: str) -> tuple[go.Figure, RegressionResult]:
    """Scatter plot of y vs x with a linear regression trend line and R^2 (FE-2.4.2)."""
    xs = frame[x].to_numpy(dtype=float)
    ys = frame[y].to_numpy(dtype=float)
    slope, intercept, r, p_value, _se = linregress(xs, ys)

    line_x = np.linspace(xs.min(), xs.max(), 50)
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="observations",
                              marker=dict(size=5, opacity=0.5)))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                              name=f"fit (R^2={r**2:.3f})", line=dict(color="#d62728")))
    fig.update_layout(title=f"Sub-Module 2.4: {y} vs {x}", xaxis_title=x, yaxis_title=y)

    result = RegressionResult(x=x, y=y, slope=float(slope), intercept=float(intercept),
                               r_squared=float(r**2), p_value=float(p_value))
    return fig, result


def key_relationship_plots(
    frame: pd.DataFrame, relationships: list[tuple[str, str]] = KEY_RELATIONSHIPS
) -> dict[tuple[str, str], tuple[go.Figure, RegressionResult]]:
    """Scatter+regression for every FE-2.4.2 key relationship in one call."""
    return {(x, y): scatter_with_regression(frame, x, y) for x, y in relationships}


# ---------------------------------------------------------------------------
# Parallel coordinates [FE-2.4.3]
# ---------------------------------------------------------------------------
def parallel_coordinates_plot(
    frame: pd.DataFrame,
    columns: list[str] = CORRELATION_COLUMNS,
    color_by: Optional[str] = "process_quality",
) -> go.Figure:
    """Parallel-coordinates view of all parameter/output dimensions across
    multiple experimental data points at once (FE-2.4.3)."""
    dimensions = [dict(label=c, values=frame[c].to_numpy()) for c in columns]
    line = dict(color=frame[color_by].to_numpy(), colorscale="Viridis",
                colorbar=dict(title=color_by)) if color_by else None
    fig = go.Figure(go.Parcoords(line=line, dimensions=dimensions))
    fig.update_layout(title="Sub-Module 2.4: parallel coordinates")
    return fig


# ---------------------------------------------------------------------------
# Automated correlation insight narration [FE-2.4.4]
# ---------------------------------------------------------------------------
@dataclass
class CorrelationInsights:
    strongest_relationships: list[tuple[str, str, float]]  # (x, y, r) sorted by |r| desc
    weak_expected_relationships: list[tuple[str, str, float]]  # KEY_RELATIONSHIPS below threshold
    narration: list[str]


def _all_pairs_by_strength(corr: pd.DataFrame, columns: list[str]) -> list[tuple[str, str, float]]:
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            pairs.append((a, b, float(corr.loc[a, b])))
    return sorted(pairs, key=lambda t: abs(t[2]), reverse=True)


def correlation_insights(
    frame: pd.DataFrame,
    columns: list[str] = CORRELATION_COLUMNS,
    key_relationships: list[tuple[str, str]] = KEY_RELATIONSHIPS,
    top_n: int = INSIGHT_TOP_N,
    low_threshold: float = LOW_CORRELATION_THRESHOLD,
) -> CorrelationInsights:
    """Identify the strongest observed relationships and flag any physically
    expected relationship that turns out surprisingly weak (FE-2.4.4)."""
    corr = correlation_matrix(frame, columns)
    strongest = _all_pairs_by_strength(corr, columns)[:top_n]

    weak_expected = [
        (x, y, float(corr.loc[x, y]))
        for x, y in key_relationships
        if abs(float(corr.loc[x, y])) < low_threshold
    ]

    narration: list[str] = []
    for x, y, r in strongest:
        direction = "positive" if r > 0 else "negative"
        narration.append(f"Strong {direction} relationship between {x} and {y} (r={r:.2f}).")
    for x, y, r in weak_expected:
        narration.append(
            f"Unexpectedly weak relationship between {x} and {y} (r={r:.2f}) despite a "
            f"physically expected link."
        )

    return CorrelationInsights(
        strongest_relationships=strongest, weak_expected_relationships=weak_expected, narration=narration
    )


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.correlation_analysis
    print("Generating sweep data and computing correlations...")
    df = simulation_sweep_dataframe()
    insights = correlation_insights(df)

    print(f"\n{len(df)} samples\n")
    print("Strongest relationships:")
    for x, y, r in insights.strongest_relationships:
        print(f"  {x:26s} {y:26s} r={r:+.3f}")
    print("\nUnexpectedly weak physically-expected relationships:")
    for x, y, r in insights.weak_expected_relationships:
        print(f"  {x:26s} {y:26s} r={r:+.3f}")
    print("\nNarration:")
    for line in insights.narration:
        print(f"  - {line}")
