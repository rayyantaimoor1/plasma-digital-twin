"""Sub-Module 1.5 - Parameter Sensitivity and Sweep Analysis.

Systematically sweeps the digital twin's inputs to quantify how each one drives
every output metric. Because the physics engine (Sub-Module 1.2) currently accepts
exactly two simulation inputs - RF power and pressure - "one-at-a-time" (OAT)
sensitivity here means: vary one of those two, hold the other fixed. (gas_flow_sccm
and exposure_time_s are recorded in ChamberParameters but do not feed the physics
engine; see chamber_config.py's module docstring, so sweeping them would trivially
show zero sensitivity and is not included.)

Chart-building functions return plotly.graph_objects.Figure objects directly
(Plotly does not require a running dashboard to construct a figure) so Phase 3
can embed them with st.plotly_chart() without this module needing to know
anything about Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from digital_twin.chamber_config import ChamberParameters, PRESSURE_RANGE, RF_POWER_RANGE
from digital_twin.physics_engine import SimulationResult, simulate

# Output metrics reported for every sweep - the numeric fields of SimulationResult
# excluding the two inputs it echoes back (rf_power_w, pressure_mtorr).
OUTPUT_METRICS = [
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

_SWEEPABLE_RANGES = {
    "rf_power_w": RF_POWER_RANGE,
    "pressure_mtorr": PRESSURE_RANGE,
}


@dataclass
class SweepPoint:
    """One evaluated point of a sweep: the varied input value and the full result."""
    varied_value: float
    result: SimulationResult


@dataclass
class OATSensitivityResult:
    """One parameter's one-at-a-time sweep: every other input held at `baseline` [FE-1.5.1]."""
    parameter_name: str
    baseline: ChamberParameters
    sweep_points: list[SweepPoint]

    def metric_series(self, metric: str) -> tuple[list[float], list[float]]:
        """Return (varied_values, metric_values) for plotting/analysis."""
        xs = [p.varied_value for p in self.sweep_points]
        ys = [getattr(p.result, metric) for p in self.sweep_points]
        return xs, ys

    def sensitivity_score(self, metric: str) -> float:
        """Normalized marginal-effect magnitude: (max - min) / mean(|values|).

        A simple, interpretable range-based measure appropriate for OAT analysis,
        where the question is "how much can this parameter alone move the output
        across its full range," not a local gradient at one point.
        """
        _xs, ys = self.metric_series(metric)
        y_arr = np.asarray(ys)
        mean_abs = float(np.mean(np.abs(y_arr)))
        if mean_abs < 1e-30:
            return 0.0
        return float((y_arr.max() - y_arr.min()) / mean_abs)


def run_oat_sweep(
    parameter_name: str, baseline: ChamberParameters, n_points: int = 15, noise_level: float = 0.0
) -> OATSensitivityResult:
    """Vary one input parameter across its full range, holding the other fixed [FE-1.5.1]."""
    if parameter_name not in _SWEEPABLE_RANGES:
        raise ValueError(
            f"Unknown sweep parameter: {parameter_name!r}. "
            f"Must be one of {sorted(_SWEEPABLE_RANGES)}."
        )
    param_range = _SWEEPABLE_RANGES[parameter_name]

    values = np.linspace(param_range.min_value, param_range.max_value, n_points)
    points: list[SweepPoint] = []
    for v in values:
        power = float(v) if parameter_name == "rf_power_w" else baseline.rf_power_w
        pressure = float(v) if parameter_name == "pressure_mtorr" else baseline.pressure_mtorr
        result = simulate(power, pressure, noise_level=noise_level)
        points.append(SweepPoint(varied_value=float(v), result=result))
    return OATSensitivityResult(parameter_name=parameter_name, baseline=baseline, sweep_points=points)


def run_full_oat_analysis(
    baseline: ChamberParameters, n_points: int = 15
) -> dict[str, OATSensitivityResult]:
    """Run an OAT sweep for every sweepable input parameter [FE-1.5.1]."""
    return {
        name: run_oat_sweep(name, baseline, n_points=n_points) for name in _SWEEPABLE_RANGES
    }


def sensitivity_ranking(
    oat_results: dict[str, OATSensitivityResult], metric: str
) -> list[tuple[str, float]]:
    """Rank parameters by sensitivity score for one output metric, descending [FE-1.5.4]."""
    scores = [(name, res.sensitivity_score(metric)) for name, res in oat_results.items()]
    return sorted(scores, key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Visualisations [FE-1.5.2] - return Figure objects; Phase 3 embeds them with
# st.plotly_chart(). Never called with side effects (no .show()) so this module
# has no dependency on a display backend and is safe to import/test headlessly.
# ---------------------------------------------------------------------------
def sensitivity_bar_chart(oat_results: dict[str, OATSensitivityResult], metric: str) -> go.Figure:
    """Bar chart ranking each parameter's sensitivity score for one metric [FE-1.5.2/1.5.4]."""
    ranking = sensitivity_ranking(oat_results, metric)
    names = [name for name, _score in ranking]
    scores = [score for _name, score in ranking]
    fig = go.Figure(go.Bar(x=names, y=scores))
    fig.update_layout(
        title=f"Parameter sensitivity - {metric}",
        xaxis_title="Parameter",
        yaxis_title="Sensitivity score (range / mean)",
    )
    return fig


def parameter_effect_curve(oat_result: OATSensitivityResult, metric: str) -> go.Figure:
    """Line chart of one output metric versus the swept input parameter [FE-1.5.2]."""
    xs, ys = oat_result.metric_series(metric)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        title=f"{metric} vs. {oat_result.parameter_name}",
        xaxis_title=oat_result.parameter_name,
        yaxis_title=metric,
    )
    return fig


# ---------------------------------------------------------------------------
# Paired sweeps [FE-1.5.3]
# ---------------------------------------------------------------------------
@dataclass
class PairedSweepResult:
    """A 2D co-variation sweep over the two available inputs.

    `metric_grid[metric]` is a (len(y_values), len(x_values)) array, indexed
    [row=y, col=x] to match Plotly's Heatmap(x=..., y=..., z=2D array) convention.
    """
    x_name: str
    y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    metric_grid: dict[str, np.ndarray]


def run_paired_sweep(
    x_name: str, y_name: str, n_x: int = 10, n_y: int = 10, noise_level: float = 0.0
) -> PairedSweepResult:
    """Co-vary two parameters over a grid, e.g. RF power vs. pressure [FE-1.5.3].

    With only two sweepable inputs currently available, x_name/y_name must be
    "rf_power_w" and "pressure_mtorr" in either order - every grid point is then
    fully determined without needing a third "held fixed" parameter.
    """
    if x_name not in _SWEEPABLE_RANGES or y_name not in _SWEEPABLE_RANGES:
        raise ValueError(f"x_name/y_name must each be one of {sorted(_SWEEPABLE_RANGES)}.")
    if x_name == y_name:
        raise ValueError("x_name and y_name must differ.")

    x_range = _SWEEPABLE_RANGES[x_name]
    y_range = _SWEEPABLE_RANGES[y_name]
    x_values = np.linspace(x_range.min_value, x_range.max_value, n_x)
    y_values = np.linspace(y_range.min_value, y_range.max_value, n_y)

    grids = {metric: np.zeros((n_y, n_x)) for metric in OUTPUT_METRICS}
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            coords = {x_name: float(x_val), y_name: float(y_val)}
            result = simulate(
                coords["rf_power_w"], coords["pressure_mtorr"], noise_level=noise_level
            )
            for metric in OUTPUT_METRICS:
                grids[metric][i, j] = getattr(result, metric)

    return PairedSweepResult(
        x_name=x_name, y_name=y_name, x_values=x_values, y_values=y_values, metric_grid=grids
    )


def paired_sweep_heatmap(sweep: PairedSweepResult, metric: str) -> go.Figure:
    """2D heatmap / response-surface approximation for one metric [FE-1.5.3]."""
    fig = go.Figure(
        go.Heatmap(x=sweep.x_values, y=sweep.y_values, z=sweep.metric_grid[metric], colorscale="Viridis")
    )
    fig.update_layout(
        title=f"{metric} response surface",
        xaxis_title=sweep.x_name,
        yaxis_title=sweep.y_name,
    )
    return fig
