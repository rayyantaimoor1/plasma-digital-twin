"""Digital Twin panel: full simulation output, parameter sensitivity (1.5), and
literature-benchmark physics validation (1.6)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.backend import render_sidebar
from digital_twin.chamber_config import ChamberParameters
from digital_twin.physics_engine import simulate
from digital_twin.physics_validation import benchmark_summary_table, te_vs_pressure_validation_plot
from digital_twin.sensitivity_analysis import (
    paired_sweep_heatmap,
    run_full_oat_analysis,
    run_paired_sweep,
    sensitivity_bar_chart,
    sensitivity_ranking,
)

st.set_page_config(page_title="Digital Twin", page_icon="🌀", layout="wide")
config = render_sidebar()

st.title("🌀 Digital Twin — Simulation, Sensitivity & Validation")

# --- full simulation output vector (Sub-Module 1.2) ---
st.subheader("Full simulation output (Sub-Module 1.2)")
result = simulate(config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v)
st.dataframe(pd.DataFrame([result.to_dict()]).T.rename(columns={0: "value"}), use_container_width=True)

st.divider()

# --- parameter sensitivity (Sub-Module 1.5) ---
st.subheader("Parameter sensitivity analysis (Sub-Module 1.5)")


@st.fragment
def _render_sensitivity(oat, sweep) -> None:
    """Metric picker + sensitivity charts, scoped as a fragment so changing the
    output metric re-runs ONLY this block — not the whole page (the full simulation
    vector above and the physics-validation section below stay put). This is the
    responsiveness fix in DASHBOARD_UX_REVIEW.md U2. The sweeps are computed once in
    the page body and passed in, so a metric switch just re-draws the existing data;
    a genuine operating-point change (sidebar) triggers a normal full rerun that
    recomputes them. Purely presentational: identical inputs → identical charts."""
    output_metric = st.selectbox(
        "Output metric",
        ["process_quality", "uniformity_index", "reactivity_index", "etch_rate_nm_min",
         "electron_temperature_ev", "plasma_density_m3"],
        index=0,
        key="dt_output_metric",  # persist the choice across reruns / page navigation (UX U1)
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sensitivity ranking** (which parameter matters most)")
        ranking = sensitivity_ranking(oat, output_metric)
        st.dataframe(pd.DataFrame(ranking, columns=["parameter", "sensitivity"]), use_container_width=True)
    with col2:
        st.plotly_chart(sensitivity_bar_chart(oat, output_metric), use_container_width=True)

    st.markdown("**Paired sweep heatmap** — RF power vs pressure")
    st.plotly_chart(paired_sweep_heatmap(sweep, output_metric), use_container_width=True)


oat = run_full_oat_analysis(ChamberParameters(config.rf_power_w, config.pressure_mtorr), n_points=15)
sweep = run_paired_sweep("rf_power_w", "pressure_mtorr", n_x=12, n_y=12)
_render_sensitivity(oat, sweep)

st.divider()

# --- physics validation against literature (Sub-Module 1.6) ---
st.subheader("Physics validation vs published reference values (Sub-Module 1.6)")
st.caption(
    "Semi-quantitative benchmark against independently-sourced reference values with a "
    "stated tolerance band. Pass/fail is reported honestly — not every reference point "
    "is expected to pass, and the failures are shown, not hidden."
)
report = benchmark_summary_table()
st.dataframe(report, use_container_width=True)
n_pass = int(report["passed"].sum())
st.metric("Reference points within tolerance", f"{n_pass} / {len(report)}")
st.plotly_chart(te_vs_pressure_validation_plot(), use_container_width=True)
