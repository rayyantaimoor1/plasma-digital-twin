"""Trends & Correlation panel: session trend analysis (2.3) and multi-parameter
correlation analysis (2.4)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.backend import get_correlation_sweep, open_db, render_sidebar
from ai_module.correlation_analysis import (
    correlation_heatmap,
    correlation_insights,
    correlation_matrix,
    key_relationship_plots,
    parallel_coordinates_plot,
    scatter_with_regression,
)
from ai_module.trend_analysis import (
    TRACKED_METRICS,
    analyze_session_trends,
    session_records_to_trend_frame,
)

st.set_page_config(page_title="Trends & Correlation", page_icon="📈", layout="wide")
render_sidebar()

st.title("📈 Trends & Correlation Analysis")

# --- trend analysis (Sub-Module 2.3) ---
st.subheader("Session trend analysis (Sub-Module 2.3)")
db = open_db()
records = db.list_sessions()
db.close()

if len(records) >= 5:
    frame = session_records_to_trend_frame(list(reversed(records)))  # chronological
    st.caption(f"Trends across {len(records)} stored sessions.")
else:
    st.info(
        f"Only {len(records)} saved session(s) — showing a synthetic demo sequence "
        "(stable → degradation → recovery → instability). Save more sessions from the "
        "Session History panel to analyse real trends."
    )
    n = 28
    base = np.concatenate([
        np.full(7, 0.55), np.linspace(0.55, 0.28, 7), np.linspace(0.28, 0.6, 7),
        0.45 + 0.12 * np.array([(-1) ** i for i in range(7)]),
    ])
    rng = np.random.default_rng(0)
    q = base + rng.normal(0, 0.015, n)
    frame = pd.DataFrame({
        "run_index": np.arange(n), "process_quality": q, "reactivity_index": q,
        "uniformity_index": q, "electron_temperature_ev": 3.0 + q,
    })

metric = st.selectbox("Trend metric", TRACKED_METRICS, index=1, key="tc_trend_metric")  # persist across navigation (UX U1)
reports = analyze_session_trends(frame, metrics=[metric])
report = reports[metric]
st.plotly_chart(report.figure, use_container_width=True)
s = report.summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean", f"{s.mean:.3f}")
c2.metric("Std", f"{s.std:.3f}")
c3.metric("Trend slope", f"{s.trend_direction_coefficient:+.4f}")
c4.metric("Detected events", len(report.events))
for event in report.events:
    st.write(f"• {event.label}")

st.divider()

# --- correlation analysis (Sub-Module 2.4) ---
st.subheader("Multi-parameter correlation analysis (Sub-Module 2.4)")
sweep = get_correlation_sweep()
corr = correlation_matrix(sweep)
heatmap_event = st.plotly_chart(
    correlation_heatmap(corr), use_container_width=True,
    on_select="rerun", key="tc_corr_heatmap",  # click a cell to drill into that pair (UX U4)
)
# Drill-in: clicking a heatmap cell shows that variable pair's scatter + regression.
# View-only — the cell's x/y are column names already present in `sweep`.
hm_points = heatmap_event.selection.points
if hm_points:
    vx, vy = hm_points[0].get("x"), hm_points[0].get("y")
    if vx in sweep.columns and vy in sweep.columns:
        drill_fig, drill_reg = scatter_with_regression(sweep, vx, vy)
        # Unique key: the drilled pair can equal a key-relationship scatter below
        # (e.g. rf_power_w → reactivity_index), and two unkeyed identical figures
        # collide on Streamlit's auto-generated element id.
        st.plotly_chart(drill_fig, use_container_width=True, key="tc_corr_drill")
        st.caption(f"Selected pair **{vx} → {vy}**: R²={drill_reg.r_squared:.3f}, "
                   f"slope={drill_reg.slope:.3g}, p={drill_reg.p_value:.2g}")

st.markdown("**Key parameter–output relationships** — box/lasso-select points in any "
            "scatter to cross-filter the parallel coordinates below")
plots = key_relationship_plots(sweep)
plot_cols = st.columns(len(plots))
# The scatter selections ARE the shared cross-filter state (UX U5). A nonce baked
# into each widget key lets the Clear button reset every scatter at once. All keys
# are tc_-prefixed and read only on this page, so the filter can never affect
# another page's charts (no cross-page state leak).
nonce = st.session_state.setdefault("tc_xfilter_nonce", 0)
selected_rows: set[int] = set()
for i, (col, ((x, y), (fig, reg))) in enumerate(zip(plot_cols, plots.items())):
    ev = col.plotly_chart(fig, use_container_width=True,
                          on_select="rerun", key=f"tc_scatter_{i}_{nonce}")
    col.caption(f"{x} → {y}: R²={reg.r_squared:.3f}")
    for pt in ev.selection.points:
        if pt.get("curve_number", 0) == 0:  # trace 0 = observations, not the fit line
            ridx = pt.get("point_index", pt.get("point_number"))
            if ridx is not None:
                selected_rows.add(int(ridx))

# Apply the shared subset to the parallel coordinates below. View-only: this selects
# a subset of the already-computed sweep rows, it never re-simulates anything, so the
# heatmap and insights (kept on the full sweep) remain the stable reference.
filtered = sorted(r for r in selected_rows if 0 <= r < len(sweep))
if filtered:
    fc1, fc2 = st.columns([3, 1])
    fc1.caption(f"🔎 Cross-filter active: **{len(filtered)} of {len(sweep)}** points selected "
                "above — the parallel coordinates below show only these.")
    if fc2.button("Clear selection", key="tc_clear_xfilter"):
        st.session_state["tc_xfilter_nonce"] = nonce + 1
        st.rerun()

st.markdown("**Parallel coordinates**" + (" (cross-filtered)" if filtered else ""))
pc_frame = sweep.iloc[filtered] if filtered else sweep
st.plotly_chart(parallel_coordinates_plot(pc_frame), use_container_width=True, key="tc_parcoords")

st.markdown("**Automated correlation insights**")
insights = correlation_insights(sweep)
for line in insights.narration:
    st.write(f"• {line}")
