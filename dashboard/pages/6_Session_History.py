"""Session History panel: persistent experiment records and multi-session
comparison (Sub-Module 1.4)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.backend import get_anomaly_detector, get_classifiers, open_db, render_sidebar
from ai_module.classification import ClassifierKind, classify_configuration
from ai_module.suitability_analysis import SemiconductorApplication, compare_sessions_for_application
from digital_twin.chamber_config import ChamberParameters, ExperimentMode
from digital_twin.dataset_generation import FEATURE_COLUMNS
from digital_twin.physics_engine import simulate

st.set_page_config(page_title="Session History", page_icon="🗄️", layout="wide")
config = render_sidebar()

st.title("🗄️ Session History & Comparison (Sub-Module 1.4)")

# --- save the current operating point as a session ---
st.subheader("Save current experiment")
mode_choice = st.selectbox("Tag with experiment mode", ["(none)"] + [m.value for m in ExperimentMode],
                           key="sh_mode_tag")  # persist across navigation (UX U1)
if st.button("💾 Save current configuration as a session"):
    result = simulate(config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v)
    params = ChamberParameters(rf_power_w=config.rf_power_w, pressure_mtorr=config.pressure_mtorr)
    mode = None if mode_choice == "(none)" else ExperimentMode(mode_choice)
    db = open_db()
    session_id = db.create_session(params, result, mode=mode)
    # attach AI results (classification + anomaly status) to the stored session
    classification = classify_configuration(
        config.rf_power_w, config.pressure_mtorr, get_classifiers()[ClassifierKind.RANDOM_FOREST]
    )
    import pandas as pd
    features = pd.DataFrame([{c: getattr(result, c) for c in FEATURE_COLUMNS}])
    is_anom = bool(get_anomaly_detector().is_anomaly(features)[0])
    db.update_ai_results(session_id, suitability_classification=classification.predicted_class,
                         anomaly_count=int(is_anom))
    db.close()
    st.success(f"Saved session {session_id[:8]}… — {classification.predicted_class}")

st.divider()

# --- stored sessions summary (FE-1.4.4) ---
st.subheader("Stored sessions")
db = open_db()
summary = db.summary_report()
records = db.list_sessions()
db.close()

if len(summary) == 0:
    st.info("No sessions saved yet. Save the current configuration above to begin building history.")
else:
    st.dataframe(summary, use_container_width=True)

    # --- comparative suitability across sessions (FE-2.5.4) ---
    st.subheader("Comparative suitability across sessions (Sub-Module 2.5)")
    app_choice = st.selectbox("Target application", [a.value for a in SemiconductorApplication],
                              key="sh_target_app")  # persist across navigation (UX U1)
    ranked = compare_sessions_for_application(records, SemiconductorApplication(app_choice))
    st.dataframe(ranked, use_container_width=True)
