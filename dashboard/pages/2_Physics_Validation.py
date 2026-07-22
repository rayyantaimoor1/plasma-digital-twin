"""Physics Validation panel: literature-benchmark validation of the digital twin's
physics core against published reference values (Sub-Module 1.6).

Promoted out of 1_Digital_Twin.py to its own top-level page (FUTURE.md item 4) since
this is the platform's primary evidence that the twin behaves like a real argon CCP
discharge, not merely a section buried under the sensitivity analysis. Presentation
only: every number rendered here comes straight out of `BenchmarkResult` — no new
physics computation happens on this page.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.backend import render_sidebar
from digital_twin.physics_validation import (
    density_vs_power_validation_plot,
    run_literature_benchmarks,
    te_vs_pressure_validation_plot,
)

# Status colours (fixed, not themed) — good/critical only, since a benchmark check
# has no intermediate "warning" state, just pass or fail.
_COLOR_GOOD = "#0ca30c"
_COLOR_CRITICAL = "#d03b3b"
_COLOR_BAND = "rgba(137, 135, 129, 0.25)"  # neutral tolerance-band fill

st.set_page_config(page_title="Physics Validation", page_icon="🔬", layout="wide")
config = render_sidebar()

st.title("🔬 Physics Validation — Literature Benchmarking (Sub-Module 1.6)")
st.caption(
    "Semi-quantitative benchmark of the digital twin's physics core against an "
    "independently-sourced, citable reference (Turner 2014), with tolerance bands "
    "stated before the checks were run. Pass/fail is reported honestly — not every "
    "reference point is expected to pass, and failures are shown, not hidden. See "
    "the module docstring in `digital_twin/physics_validation.py` for the full "
    "methodology and why each check is isolated the way it is."
)

results = run_literature_benchmarks()
report = pd.DataFrame([
    {
        "name": r.name,
        "quantity": r.quantity,
        "computed_value": r.computed_value,
        "reference_value": r.reference_value,
        "unit": r.unit,
        "deviation_pct": r.deviation_pct,
        "tolerance_pct": r.tolerance_pct,
        "passed": r.passed,
    }
    for r in results
])
n_pass = int(report["passed"].sum())
n_total = len(report)
pass_rate_pct = 100.0 * n_pass / n_total

st.divider()

# --- pass-rate hero gauge + deviation-vs-tolerance bar chart ---
col_gauge, col_bar = st.columns([1, 2])

with col_gauge:
    st.markdown("**Reference checks within tolerance**")
    # Gauge bar colour follows the status palette based on how many checks passed —
    # green once most checks clear their band, red if fewer than half do.
    gauge_color = _COLOR_GOOD if pass_rate_pct >= 50.0 else _COLOR_CRITICAL
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pass_rate_pct,
        number={"suffix": "%", "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 50], "color": "rgba(208, 59, 59, 0.12)"},
                {"range": [50, 100], "color": "rgba(12, 163, 12, 0.12)"},
            ],
        },
    ))
    gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10))
    st.plotly_chart(gauge_fig, use_container_width=True, key="pv_pass_rate_gauge")
    st.caption(f"{n_pass} / {n_total} checks passed")

with col_bar:
    st.markdown("**Deviation vs. tolerance, per literature check**")
    names = report["name"].tolist()
    deviations = report["deviation_pct"].tolist()
    tolerances = report["tolerance_pct"].tolist()
    passed_flags = report["passed"].tolist()

    bar_fig = go.Figure()
    # Tolerance band drawn first (as a semi-transparent bar centred on zero, one
    # per check since tolerance width differs: +-15% for the direct ratio check,
    # +-30% for solved/compound quantities), then the deviation bars on top.
    bar_fig.add_trace(go.Bar(
        x=names, y=[2.0 * t for t in tolerances], base=[-t for t in tolerances],
        marker_color=_COLOR_BAND, name="±tolerance band", hoverinfo="skip",
    ))
    bar_fig.add_trace(go.Bar(
        x=names, y=deviations,
        marker_color=[_COLOR_GOOD if p else _COLOR_CRITICAL for p in passed_flags],
        name="deviation %",
        text=[f"{d:+.1f}%" for d in deviations], textposition="outside",
    ))
    bar_fig.add_hline(y=0, line_color="#898781", line_width=1)
    bar_fig.update_layout(
        barmode="overlay",
        yaxis_title="Deviation from reference (%)",
        xaxis_title="Benchmark check",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40),
    )
    bar_event = st.plotly_chart(
        bar_fig, use_container_width=True,
        on_select="rerun", key="pv_deviation_bar",  # click-to-drill (U4 pattern)
    )

# Drill-in: clicking a bar shows that check's full description, citation, and
# computed-vs-reference values. View-only — the clicked bar's x is a check name
# already present in `results`.
bar_points = bar_event.selection.points
if bar_points:
    picked_name = bar_points[0].get("x")
    picked = next((r for r in results if r.name == picked_name), None)
    if picked is not None:
        st.markdown(f"**{picked.name}** — selected from the chart above")
        st.write(picked.description)
        d1, d2, d3 = st.columns(3)
        d1.metric("Computed", f"{picked.computed_value:.4g} {picked.unit}")
        d2.metric("Reference", f"{picked.reference_value:.4g} {picked.unit}")
        d3.metric(
            "Deviation vs tolerance",
            f"{picked.deviation_pct:+.1f}%",
            help=f"tolerance band: ±{picked.tolerance_pct:.0f}%",
        )
        st.caption(f"Source: {picked.source}")
else:
    st.caption("Click a bar above to see that check's citation and full computed-vs-reference detail.")

st.divider()

# --- full detail table ---
st.subheader("Full benchmark detail")
st.dataframe(report, use_container_width=True)

st.divider()

# --- trend-curve validation plots (FE-1.6.1) ---
st.subheader("Trend-curve comparisons")
st.caption(
    "The digital twin's continuous sweep against the literature reference point(s), "
    "checking shape as well as magnitude."
)
st.plotly_chart(te_vs_pressure_validation_plot(), use_container_width=True, key="pv_te_vs_pressure")
st.plotly_chart(density_vs_power_validation_plot(), use_container_width=True, key="pv_density_vs_power")
