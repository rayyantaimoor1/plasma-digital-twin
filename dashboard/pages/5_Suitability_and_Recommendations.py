"""Suitability & Recommendations panel: semiconductor process suitability (2.5)
and the counterfactual recommendation engine (2.6)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.backend import get_classifiers, get_defect_estimates, render_sidebar
from ai_module.classification import ClassifierKind
from ai_module.recommendation_engine import recommend, store_recommendation
from ai_module.suitability_analysis import (
    SUITABILITY_WINDOWS,
    SemiconductorApplication,
    classify_suitability,
)

st.set_page_config(page_title="Suitability & Recommendations", page_icon="🎯", layout="wide")
config = render_sidebar()

st.title("🎯 Semiconductor Suitability & Recommendations")

# --- suitability scorecard (Sub-Module 2.5) ---
st.subheader("Semiconductor process suitability (Sub-Module 2.5)")
scorecard = classify_suitability(config.rf_power_w, config.pressure_mtorr, rf_voltage_v=config.rf_voltage_v)
st.caption(f"Ion bombardment energy at this operating point: **{scorecard.ion_energy_ev:.1f} eV**")
score_df = scorecard.to_dataframe()
# Row selection drills into one application's window + this point's compliance —
# view-only (DASHBOARD_UX_REVIEW.md U7). on_select only reads the picked row; the
# scorecard itself is unchanged.
sel = st.dataframe(
    score_df, use_container_width=True,
    on_select="rerun", selection_mode="single-row", key="sr_scorecard_table",
)
best = scorecard.best_application()
st.metric("Best-fit application", best.value,
          help=f"{scorecard.ratings[best].overall_compliance_pct:.0f}% window compliance")

picked = sel.selection.rows
if picked:
    app = SemiconductorApplication(score_df.iloc[picked[0]]["application"])
    window = SUITABILITY_WINDOWS[app]
    rating = scorecard.ratings[app]
    with st.container(border=True):
        st.markdown(f"**{app.value}** — selected application detail")
        d1, d2, d3 = st.columns(3)
        d1.metric("Ion-energy window", f"{window.ion_energy_min_ev:.0f}–{window.ion_energy_max_ev:.0f} eV")
        d2.metric("Ion-energy compliance", f"{rating.ion_energy_compliance_pct:.0f}%")
        d3.metric("Defect compliance", f"{rating.defect_compliance_pct:.0f}%")
        st.caption(window.rationale)

with st.expander("Application-specific defect-risk confidence intervals (FE-2.5.3)"):
    estimates = get_defect_estimates(config.rf_power_w, config.pressure_mtorr, n_bootstrap=80)
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
@st.fragment
def _render_recommendations(config) -> None:
    """Ranking-mode toggle + recommendation cards, scoped as a fragment so toggling
    the classifier ranking (or clicking a "Log" button) re-runs ONLY this block —
    not the suitability scorecard and defect-risk bootstrap above it
    (DASHBOARD_UX_REVIEW.md U2). Purely presentational: the same config and toggle
    state produce the same re-simulated recommendations."""
    use_classifier = st.checkbox("Rank using the ML classifier's class-transition prediction", value=True,
                                 key="sr_use_classifier")  # persist across navigation (UX U1)
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


_render_recommendations(config)
