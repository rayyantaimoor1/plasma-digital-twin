"""Model Training panel: cross-validation, cross-model feature importance, and the
statistical-significance evaluation that operationalises BO-2 / principle #3
(Sub-Modules 2.1 & 2.7)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.backend import get_classifiers, get_evaluation_report, get_shap_background, load_dataset, render_sidebar
from ai_module.training import feature_importance_comparison_plot, run_cross_validation
from digital_twin.dataset_generation import features_and_labels

st.set_page_config(page_title="Model Training", page_icon="🔬", layout="wide")
render_sidebar()

st.title("🔬 Model Training & Evaluation (Sub-Modules 2.1 & 2.7)")

dataset = load_dataset()
st.caption(f"Trained on {len(dataset)} synthetic samples with hidden confounders (Sub-Module 1.3).")

# --- dual-split evaluation + significance vs baseline (principle #3) ---
st.subheader("Performance & statistical significance vs baseline")
st.caption(
    "Success is a statistically significant improvement over a logistic-regression "
    "baseline (McNemar's exact test), not an absolute accuracy figure — reported "
    "honestly whatever the result (CLAUDE.md principle #3)."
)
report = get_evaluation_report()
metrics_df = pd.DataFrame([
    {"classifier": m.classifier, "split": m.split_name, "accuracy": round(m.accuracy, 3),
     "precision": round(m.precision_macro, 3), "recall": round(m.recall_macro, 3),
     "f1": round(m.f1_macro, 3)}
    for m in report.metrics
])
st.dataframe(metrics_df, use_container_width=True)

sig_df = pd.DataFrame([
    {"ensemble": t.ensemble, "split": t.split_name,
     "ensemble_acc": round(t.ensemble_accuracy, 3), "baseline_acc": round(t.baseline_accuracy, 3),
     "p_value": round(t.p_value, 4), "significant": t.significant}
    for t in report.significance_tests
])
st.dataframe(sig_df, use_container_width=True)
if sig_df["significant"].any():
    st.success("At least one ensemble model significantly outperforms the baseline.")
else:
    st.info(
        "On this dataset, no ensemble model significantly outperforms the logistic "
        "baseline (p ≥ 0.05). This is reported honestly rather than tuned away — the "
        "hidden confounders make the task genuinely hard."
    )

st.divider()

# --- k-fold cross-validation (FE-2.7.3) ---
st.subheader("k-fold cross-validation (Sub-Module 2.7)")


@st.fragment
def _render_cross_validation(dataset) -> None:
    """k selector + CV table, scoped as a fragment so changing k re-runs ONLY the
    cross-validation, not the expensive SHAP feature-importance comparison below it
    (which is independent of k) — DASHBOARD_UX_REVIEW.md U2. Purely presentational:
    the same dataset and k give the same fold metrics."""
    k = st.slider("Number of folds (k)", 3, 8, 5, key="mt_kfolds")  # persist across navigation (UX U1)
    cv = run_cross_validation(dataset, k=k)
    st.dataframe(cv.aggregate_table(), use_container_width=True)


_render_cross_validation(dataset)

st.divider()

# --- cross-model feature importance comparison (FE-2.7.4) ---
st.subheader("Cross-model feature importance (Sub-Module 2.7)")
st.caption("Same SHAP-based measure for all three models, so the bars are comparable.")
X, _y = features_and_labels(dataset)
fig = feature_importance_comparison_plot(get_classifiers(), X.sample(50, random_state=2), get_shap_background())
st.plotly_chart(fig, use_container_width=True)
