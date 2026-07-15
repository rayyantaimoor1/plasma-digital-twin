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

metric = st.selectbox("Trend metric", TRACKED_METRICS, index=1)
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
st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

st.markdown("**Key parameter–output relationships**")
plots = key_relationship_plots(sweep)
plot_cols = st.columns(len(plots))
for col, ((x, y), (fig, reg)) in zip(plot_cols, plots.items()):
    col.plotly_chart(fig, use_container_width=True)
    col.caption(f"{x} → {y}: R²={reg.r_squared:.3f}")

st.markdown("**Parallel coordinates**")
st.plotly_chart(parallel_coordinates_plot(sweep), use_container_width=True)

st.markdown("**Automated correlation insights**")
insights = correlation_insights(sweep)
for line in insights.narration:
    st.write(f"• {line}")
