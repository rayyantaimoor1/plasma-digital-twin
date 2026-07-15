"""Phase 3 - unified Streamlit dashboard entry point (overview / control room).

Run from the project root:  streamlit run dashboard/app.py

This is the landing panel: the shared chamber-configuration sidebar plus a live
"control room" summary that runs the current operating point through the digital
twin and gives the headline AI verdicts (suitability class, anomaly status,
semiconductor best-fit). The specialised panels live in the pages/ directory and
are reachable from Streamlit's page navigation.
"""
import pathlib
import sys

# Put the project root on sys.path so `import digital_twin` / `import ai_module`
# resolve when Streamlit runs this file (whose own directory is dashboard/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard.backend import get_anomaly_detector, get_classifiers, render_sidebar
from ai_module.anomaly_detection import AnomalySeverity
from ai_module.classification import ClassifierKind, classify_configuration
from ai_module.suitability_analysis import classify_suitability
from digital_twin.physics_engine import simulate

st.set_page_config(page_title="Plasma Digital Twin", page_icon="⚛️", layout="wide")

st.title("⚛️ AI-Assisted Plasma Process Experimentation & Digital Twin")
st.caption(
    "A software-only virtual RF capacitively coupled argon plasma laboratory with an "
    "AI analytics layer. Use the sidebar to set the operating point; use the page "
    "navigation (left) for the specialised analysis panels."
)

config = render_sidebar()

# --- run the current operating point through the physics engine ---
result = simulate(config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v)

st.subheader("Live operating point")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Electron temperature", f"{result.electron_temperature_ev:.2f} eV")
c2.metric("Plasma density", f"{result.plasma_density_m3:.2e} m⁻³")
c3.metric("Process quality", f"{result.process_quality:.3f}")
c4.metric("Defect probability", f"{result.defect_probability:.3f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Ion flux", f"{result.ion_flux_m2s:.2e} m⁻²s⁻¹")
c6.metric("Ion energy", f"{result.ion_energy_ev:.1f} eV")
c7.metric("Reactivity index", f"{result.reactivity_index:.3f}")
c8.metric("Uniformity index", f"{result.uniformity_index:.3f}")

st.divider()
st.subheader("AI control-room verdict")

col_left, col_mid, col_right = st.columns(3)

with col_left:
    st.markdown("**Suitability classification** (Sub-Module 2.1)")
    classifiers = get_classifiers()
    classification = classify_configuration(
        config.rf_power_w, config.pressure_mtorr, classifiers[ClassifierKind.RANDOM_FOREST]
    )
    st.metric("Random Forest verdict", classification.predicted_class,
              help=f"confidence {classification.confidence:.0%}")

with col_mid:
    st.markdown("**Anomaly status** (Sub-Module 2.2)")
    detector = get_anomaly_detector()
    features = {
        "rf_power_w": config.rf_power_w, "pressure_mtorr": config.pressure_mtorr,
        "electron_temperature_ev": result.electron_temperature_ev,
        "plasma_density_m3": result.plasma_density_m3,
        "reactivity_index": result.reactivity_index,
        "uniformity_index": result.uniformity_index,
        "etch_rate_nm_min": result.etch_rate_nm_min,
    }
    import pandas as pd
    severity = detector.severity(pd.DataFrame([features]))[0]
    icon = {AnomalySeverity.NORMAL: "🟢", AnomalySeverity.WARNING: "🟠", AnomalySeverity.CRITICAL: "🔴"}[severity]
    st.metric("Relationship-anomaly severity", f"{icon} {severity.value}")

with col_right:
    st.markdown("**Best semiconductor fit** (Sub-Module 2.5)")
    scorecard = classify_suitability(
        config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v
    )
    best = scorecard.best_application()
    st.metric("Best-fit application", best.value,
              help=f"{scorecard.ratings[best].overall_compliance_pct:.0f}% window compliance")

st.divider()
st.info(
    "Panels: **Digital Twin** (simulation, sensitivity, physics validation) · "
    "**AI Analytics** (classification, SHAP, uncertainty) · **Anomaly Monitor** · "
    "**Trends & Correlation** · **Suitability & Recommendations** · **Session History** · "
    "**Model Training**. Open them from the navigation on the left."
)
