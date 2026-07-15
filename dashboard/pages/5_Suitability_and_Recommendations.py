"""Suitability & Recommendations panel: semiconductor process suitability (2.5)
and the counterfactual recommendation engine (2.6)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.backend import get_classifiers, render_sidebar
from ai_module.classification import ClassifierKind
from ai_module.recommendation_engine import recommend, store_recommendation
from ai_module.suitability_analysis import (
    SemiconductorApplication,
    all_application_defect_estimates,
    classify_suitability,
)

st.set_page_config(page_title="Suitability & Recommendations", page_icon="🎯", layout="wide")
config = render_sidebar()

st.title("🎯 Semiconductor Suitability & Recommendations")

# --- suitability scorecard (Sub-Module 2.5) ---
st.subheader("Semiconductor process suitability (Sub-Module 2.5)")
scorecard = classify_suitability(config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v)
st.caption(f"Ion bombardment energy at this operating point: **{scorecard.ion_energy_ev:.1f} eV**")
st.dataframe(scorecard.to_dataframe(), use_container_width=True)
best = scorecard.best_application()
st.metric("Best-fit application", best.value,
          help=f"{scorecard.ratings[best].overall_compliance_pct:.0f}% window compliance")

with st.expander("Application-specific defect-risk confidence intervals (FE-2.5.3)"):
    estimates = all_application_defect_estimates(config.rf_power_w, config.pressure_mtorr, n_bootstrap=80)
    rows = [
        {"application": app.value, "point_estimate": e.point_estimate,
         "ci_lower": e.ci_lower, "ci_upper": e.ci_upper}
        for app, e in estimates.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()

# --- counterfactual recommendations (Sub-Module 2.6) ---
st.subheader("Counterfactual recommendations (Sub-Module 2.6)")
st.caption(
    "Every recommendation is re-run through the digital twin — the reported effect "
    "is a real re-simulated outcome, never unverified advisory text (principle #5)."
)
use_classifier = st.checkbox("Rank using the ML classifier's class-transition prediction", value=True)
classifier = get_classifiers()[ClassifierKind.RANDOM_FOREST] if use_classifier else None
recommendations = recommend(config, classifier=classifier)

for i, rec in enumerate(recommendations, 1):
    with st.container(border=True):
        st.markdown(f"**{i}. {rec.action_text}**  &nbsp; (quality {rec.quality_delta:+.3f})")
        st.write(rec.prediction_text)
        st.caption(rec.rationale)
        if st.button("Log this recommendation", key=f"log_rec_{i}"):
            store_recommendation(rec, session_id=None)
            st.success("Recommendation logged with its quantified predicted effect.")
