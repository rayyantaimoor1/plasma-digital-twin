"""Anomaly Monitor panel: relationship-level anomaly detection (2.2) with a live
severity read-out, a session anomaly timeline, an SPC control chart, and the
event log."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.backend import get_anomaly_detector, render_sidebar
from ai_module.anomaly_detection import (
    AnomalySeverity,
    SPCMonitor,
    anomaly_timeline_plot,
    generate_anomalous_data,
    generate_normal_operating_data,
    load_anomaly_events,
    store_anomaly_event,
)
from digital_twin.dataset_generation import FEATURE_COLUMNS
from digital_twin.physics_engine import simulate

st.set_page_config(page_title="Anomaly Monitor", page_icon="🚨", layout="wide")
config = render_sidebar()

st.title("🚨 Anomaly Monitor — Physics-Relationship Violations")
st.caption(
    "Anomalies are physics-relationship violations (observed outputs inconsistent "
    "with the logged inputs), not out-of-range values — detected by Isolation Forest "
    "on the physics residual, catching faults a naive range check misses entirely."
)

detector = get_anomaly_detector()

# --- live status for the current configuration ---
result = simulate(config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v)
current_features = pd.DataFrame([{c: getattr(result, c) for c in FEATURE_COLUMNS}])
severity = detector.severity(current_features)[0]
score = float(detector.anomaly_score(current_features)[0])
root_cause = detector.root_cause(current_features)[0]

st.subheader("Current operating point")
c1, c2, c3 = st.columns(3)
icon = {AnomalySeverity.NORMAL: "🟢", AnomalySeverity.WARNING: "🟠", AnomalySeverity.CRITICAL: "🔴"}[severity]
c1.metric("Severity", f"{icon} {severity.value}")
c2.metric("Anomaly score", f"{score:.3f}", help="lower = more anomalous")
c3.metric("Root-cause hint", root_cause if severity != AnomalySeverity.NORMAL else "—")

if severity != AnomalySeverity.NORMAL and st.button("Log this anomaly event"):
    store_anomaly_event(
        session_id=None, rf_power_w=config.rf_power_w, pressure_mtorr=config.pressure_mtorr,
        anomaly_score=score, severity=severity, root_cause=root_cause,
    )
    st.success("Anomaly event logged to the SQLite event store.")

st.divider()

# --- demonstration sequence: normal runs with injected physics-violation faults ---
st.subheader("Detection demonstration (Sub-Module 2.2)")
st.caption("A short run sequence mixing normal operating points with injected fault "
           "conditions, scored by the detector.")
normal = generate_normal_operating_data(power_step_w=50.0, pressure_step_mtorr=5.0, replicates=1).sample(8, random_state=0)
anomalies = generate_anomalous_data(n_samples=4, seed=3)[FEATURE_COLUMNS]
sequence = pd.concat([normal[FEATURE_COLUMNS], anomalies], ignore_index=True)
scores = detector.anomaly_score(sequence)
severities = detector.severity(sequence)
timeline = anomaly_timeline_plot(
    list(range(len(sequence))), list(scores), severities,
    detector.warning_threshold, detector.critical_threshold,
)
st.plotly_chart(timeline, use_container_width=True)

st.divider()

# --- SPC control chart on process quality (FE-2.2.4) ---
st.subheader("Statistical process control (SPC) on process quality")
baseline_quality = generate_normal_operating_data(power_step_w=50.0, pressure_step_mtorr=5.0, replicates=1)
baseline_vals = np.array([
    simulate(row.rf_power_w, row.pressure_mtorr).process_quality
    for row in baseline_quality.itertuples()
])
monitor = SPCMonitor.from_baseline(baseline_vals)
# sequential runs including a deliberate excursion
seq_quality = np.concatenate([baseline_vals[:12], [0.02, 0.9], baseline_vals[12:18]])
flags = monitor.out_of_control(seq_quality)
spc = go.Figure()
spc.add_trace(go.Scatter(y=seq_quality, mode="lines+markers", name="process quality",
                          marker=dict(color=["red" if f else "steelblue" for f in flags])))
spc.add_hline(y=monitor.center_line, line_dash="dot", annotation_text="center")
spc.add_hline(y=monitor.upper_control_limit, line_dash="dash", line_color="red", annotation_text="UCL")
spc.add_hline(y=monitor.lower_control_limit, line_dash="dash", line_color="red", annotation_text="LCL")
spc.update_layout(title="Shewhart control chart (±3σ)", xaxis_title="Run index", yaxis_title="Process quality")
st.plotly_chart(spc, use_container_width=True)
st.metric("Out-of-control runs flagged", int(flags.sum()))

st.divider()

# --- event log (FE-2.2.3) ---
st.subheader("Logged anomaly events")
events = load_anomaly_events()
if len(events):
    st.dataframe(events, use_container_width=True)
else:
    st.info("No anomaly events logged yet. Set an anomalous operating point and use "
            "'Log this anomaly event' above.")

st.divider()

# --- optional deep-learning comparison (FE-2.2.6) ---
st.subheader("Classical vs deep-learning detector (optional, FE-2.2.6)")
st.caption(
    "Isolation Forest (classical, on physics residuals) vs a PyTorch autoencoder "
    "(deep learning, learning the normal manifold from raw features) on the same "
    "injected relationship violations — the autoencoder matches the classical "
    "method without the manual residual feature engineering."
)
try:
    from dashboard.backend import get_autoencoder_comparison

    comp = get_autoencoder_comparison()
    st.dataframe(pd.DataFrame([
        {"detector": "Isolation Forest (classical)", "recall": f"{comp.isolation_forest_recall:.1%}",
         "false positive rate": f"{comp.isolation_forest_fpr:.1%}"},
        {"detector": "Autoencoder (deep learning)", "recall": f"{comp.autoencoder_recall:.1%}",
         "false positive rate": f"{comp.autoencoder_fpr:.1%}"},
        {"detector": "Range check (naive baseline)", "recall": f"{comp.range_check_recall:.1%}",
         "false positive rate": "—"},
    ]), use_container_width=True)
except ImportError:
    st.info(
        "PyTorch is not installed. This optional comparison needs the deep-learning "
        "extra: `pip install -r requirements-optional.txt`, then reload."
    )
