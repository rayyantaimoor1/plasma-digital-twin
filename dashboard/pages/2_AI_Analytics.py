"""AI Analytics panel: suitability classification with side-by-side model
comparison and SHAP explainability (2.1), plus the conformal-prediction trust
layer (2.8)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.backend import get_classifiers, get_shap_background, get_uncertainty_layer, load_dataset, render_sidebar
from ai_module.classification import ClassifierKind, compare_classifiers, explain_configuration, global_feature_importance
from ai_module.uncertainty_quantification import assess_configuration
from digital_twin.dataset_generation import features_and_labels

st.set_page_config(page_title="AI Analytics", page_icon="🧠", layout="wide")
config = render_sidebar()

st.title("🧠 AI Analytics — Classification, Explainability & Trust")

classifiers = get_classifiers()
background = get_shap_background()

# --- side-by-side model comparison (FE-2.1.3) ---
st.subheader("Side-by-side model comparison (Sub-Module 2.1)")
comparison = compare_classifiers(config.rf_power_w, config.pressure_mtorr, classifiers)
cols = st.columns(3)
for col, (name, res) in zip(cols, comparison.results.items()):
    col.metric(name.replace("_", " ").title(), res.predicted_class, help=f"confidence {res.confidence:.0%}")
if comparison.all_agree:
    st.success(f"All three models agree: **{comparison.majority_class}**")
else:
    st.warning(f"Models disagree — majority verdict: **{comparison.majority_class}**")

st.divider()

# --- SHAP explainability (FE-2.1.2) ---
st.subheader("SHAP explainability (Sub-Module 2.1)")


@st.fragment
def _render_shap(rf_power_w, pressure_mtorr, classifiers, background) -> None:
    """Model picker + SHAP charts, scoped as a fragment so switching the explained
    model re-runs ONLY this block, not the conformal/uncertainty section below it
    (which is independent of the model choice) — DASHBOARD_UX_REVIEW.md U2. Purely
    presentational: the same model at the same operating point yields the same SHAP
    contributions."""
    model_choice = st.selectbox(
        "Model to explain", [k.value for k in ClassifierKind], index=1,
        key="ai_model_choice",  # persist the choice across reruns / page navigation (UX U1)
    )
    kind = ClassifierKind(model_choice)
    explanation = explain_configuration(rf_power_w, pressure_mtorr, classifiers[kind], background)
    st.caption(f"Per-prediction feature contributions for the **{explanation.predicted_class}** prediction:")
    contrib = pd.Series(explanation.feature_contributions).sort_values(key=abs, ascending=False)
    st.bar_chart(contrib)

    X, _y = features_and_labels(load_dataset())
    importance = global_feature_importance(classifiers[kind], X.sample(60, random_state=1), background)
    st.caption("Global feature importance (mean |SHAP value| across the dataset):")
    st.bar_chart(importance)


_render_shap(config.rf_power_w, config.pressure_mtorr, classifiers, background)

st.divider()

# --- conformal trust layer (Sub-Module 2.8) ---
st.subheader("Uncertainty & trust layer (Sub-Module 2.8)")
st.caption(
    "MAPIE conformal prediction: a calibrated prediction SET (size 1 = confident, "
    "size 2+ = the model genuinely cannot distinguish those classes at this "
    "confidence level) and a calibrated defect-probability interval."
)
layer = get_uncertainty_layer()
report = assess_configuration(config.rf_power_w, config.pressure_mtorr, layer.classifier, layer.regressor)

c1, c2, c3 = st.columns(3)
c1.metric("Conformal prediction set", ", ".join(report.predicted_classes))
c2.metric("Defect probability", f"{report.defect_point_estimate:.3f}",
          help=f"90% interval [{report.defect_interval_lower:.3f}, {report.defect_interval_upper:.3f}]")
c3.metric("Confidence flag", "⚠️ LOW" if report.is_low_confidence else "✅ High")
if report.is_low_confidence:
    st.warning(
        "This configuration is flagged **low confidence** — either the prediction set "
        "is ambiguous (multiple classes) or the defect interval is wide."
    )

with st.expander("Empirical coverage (verified, not assumed)"):
    cov = layer.coverage
    st.write(f"Target confidence level: {cov.target_confidence_level:.0%}")
    st.write(f"Empirical classification coverage: {cov.empirical_classification_coverage:.1%}")
    st.write(f"Empirical regression coverage: {cov.empirical_regression_coverage:.1%}")
    st.write(f"Mean prediction-set size: {cov.mean_prediction_set_size:.2f}")
