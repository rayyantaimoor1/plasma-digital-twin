"""Digital Twin panel: full simulation output and parameter sensitivity (1.5).

Literature-benchmark physics validation (1.6) has its own dedicated page —
see 2_Physics_Validation.py."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.backend import get_oat_analysis, get_paired_sweep, render_sidebar
from digital_twin.physics_engine import simulate
from digital_twin.sensitivity_analysis import (
    paired_sweep_heatmap,
    parameter_effect_curve,
    sensitivity_bar_chart,
    sensitivity_ranking,
)

st.set_page_config(page_title="Digital Twin", page_icon="🌀", layout="wide")
config = render_sidebar()

st.title("🌀 Digital Twin — Simulation & Sensitivity")

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
        bar_event = st.plotly_chart(
            sensitivity_bar_chart(oat, output_metric), use_container_width=True,
            on_select="rerun", key="dt_sensitivity_bar",  # click a bar to drill in (UX U4)
        )
    # Drill-in: clicking a parameter's bar shows that parameter's full effect curve.
    # View-only — the clicked bar's category is a parameter name already in `oat`.
    bar_points = bar_event.selection.points
    if bar_points:
        picked_param = bar_points[0].get("x")
        if picked_param in oat:
            st.markdown(f"**Effect curve for `{picked_param}`** (selected from the bar chart)")
            st.plotly_chart(parameter_effect_curve(oat[picked_param], output_metric),
                            use_container_width=True, key="dt_effect_curve")

    st.markdown("**Paired sweep heatmap** — RF power vs pressure")
    st.plotly_chart(paired_sweep_heatmap(sweep, output_metric), use_container_width=True)


# Cached on the operating point (OAT) / computed once (paired sweep) — see U3 in
# DASHBOARD_UX_REVIEW.md. Passing them into the fragment means a metric switch just
# re-draws them; only a real operating-point change recomputes the OAT.
oat = get_oat_analysis(config.rf_power_w, config.pressure_mtorr)
sweep = get_paired_sweep()
_render_sensitivity(oat, sweep)
