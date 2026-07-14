"""Tests for Sub-Module 2.6 - counterfactual recommendation engine.

The single most important test here is test_every_recommendation_is_grounded_in_
real_resimulation: it enforces CLAUDE.md non-negotiable principle #5 by re-running
the digital twin on each recommended configuration and checking the reported
before/after numbers were genuinely produced by that re-simulation, not fabricated
advisory text.
"""
import warnings

import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from digital_twin.chamber_config import PRESSURE_MAX_MTORR, RF_POWER_MAX_W, RF_POWER_MIN_W
from digital_twin.dataset_generation import generate_dataset
from digital_twin.physics_engine import simulate
from ai_module.classification import ClassifierKind, train_classifiers
from ai_module.recommendation_engine import (
    Configuration,
    load_recommendations,
    recommend,
    store_recommendation,
    verify_recommendation,
)


@pytest.fixture(scope="module")
def classifier():
    df = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=8)
    return train_classifiers(df)[ClassifierKind.RANDOM_FOREST]


MARGINAL_BASELINE = Configuration(rf_power_w=250.0, pressure_mtorr=18.0)


# ---------------------------------------------------------------------------
# Principle #5 - every recommendation is a real re-simulation (FE-2.6.1)
# ---------------------------------------------------------------------------
def test_every_recommendation_is_grounded_in_real_resimulation() -> None:
    """Re-running the digital twin on each recommended configuration must
    reproduce the reported after-values exactly - proving the quantified effect
    came from a genuine counterfactual re-simulation, not invented text."""
    recommendations = recommend(MARGINAL_BASELINE)
    assert len(recommendations) >= 1
    for rec in recommendations:
        result = simulate(
            rec.recommended.rf_power_w, rec.recommended.pressure_mtorr,
            rf_voltage_v=rec.recommended.rf_voltage_v,
        )
        assert result.process_quality == pytest.approx(rec.quality_after, abs=1e-9)
        assert result.uniformity_index == pytest.approx(rec.uniformity_after, abs=1e-9)
        assert result.defect_probability == pytest.approx(rec.defect_after, abs=1e-9)
        assert result.ion_energy_ev == pytest.approx(rec.ion_energy_after, abs=1e-9)


def test_baseline_values_match_a_real_simulation_of_the_baseline() -> None:
    recommendations = recommend(MARGINAL_BASELINE)
    baseline_result = simulate(MARGINAL_BASELINE.rf_power_w, MARGINAL_BASELINE.pressure_mtorr)
    for rec in recommendations:
        assert rec.quality_before == pytest.approx(baseline_result.process_quality, abs=1e-9)


def test_quality_delta_is_after_minus_before() -> None:
    for rec in recommend(MARGINAL_BASELINE):
        assert rec.quality_delta == pytest.approx(rec.quality_after - rec.quality_before, abs=1e-12)


def test_prediction_text_reports_the_quantified_before_and_after() -> None:
    rec = recommend(MARGINAL_BASELINE)[0]
    assert f"{rec.quality_before:.3f}" in rec.prediction_text
    assert f"{rec.quality_after:.3f}" in rec.prediction_text
    assert "process quality" in rec.prediction_text


# ---------------------------------------------------------------------------
# Candidate generation and ranking (FE-2.6.2, FE-2.6.3)
# ---------------------------------------------------------------------------
def test_produces_at_least_three_recommendations_for_mid_envelope() -> None:
    """FE-2.6.1 requires at least three actionable recommendations per cycle."""
    recs = recommend(Configuration(rf_power_w=150.0, pressure_mtorr=10.0), max_recommendations=3)
    assert len(recs) == 3


def test_recommendations_ranked_by_score_descending() -> None:
    recs = recommend(MARGINAL_BASELINE, max_recommendations=5)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_max_recommendations_is_respected() -> None:
    assert len(recommend(MARGINAL_BASELINE, max_recommendations=2)) == 2
    assert len(recommend(MARGINAL_BASELINE, max_recommendations=1)) == 1


def test_recommended_configs_are_distinct() -> None:
    recs = recommend(MARGINAL_BASELINE, max_recommendations=5)
    keys = [(r.recommended.rf_power_w, r.recommended.pressure_mtorr, r.recommended.rf_voltage_v) for r in recs]
    assert len(keys) == len(set(keys))


def test_top_recommendation_for_low_uniformity_reduces_pressure_and_improves() -> None:
    """At the high-pressure marginal baseline, uniformity is low; the best
    re-simulated move should reduce pressure and genuinely raise quality
    (best-practice rule validated by the physics, not just proposed)."""
    top = recommend(MARGINAL_BASELINE)[0]
    assert top.recommended.pressure_mtorr < MARGINAL_BASELINE.pressure_mtorr
    assert top.quality_delta > 0
    assert top.improves


def test_action_text_matches_the_actual_parameter_change() -> None:
    for rec in recommend(MARGINAL_BASELINE, max_recommendations=5):
        if rec.recommended.rf_power_w != rec.baseline.rf_power_w:
            assert "RF power" in rec.action_text
        elif rec.recommended.pressure_mtorr != rec.baseline.pressure_mtorr:
            assert "pressure" in rec.action_text


def test_improves_property_matches_quality_delta_sign() -> None:
    for rec in recommend(MARGINAL_BASELINE, max_recommendations=6):
        assert rec.improves == (rec.quality_delta > 0.0)


# ---------------------------------------------------------------------------
# Envelope clamping
# ---------------------------------------------------------------------------
def test_no_recommendation_exceeds_the_power_envelope() -> None:
    recs = recommend(Configuration(rf_power_w=RF_POWER_MAX_W, pressure_mtorr=10.0), max_recommendations=6)
    for rec in recs:
        assert RF_POWER_MIN_W <= rec.recommended.rf_power_w <= RF_POWER_MAX_W


def test_no_recommendation_exceeds_the_pressure_envelope() -> None:
    recs = recommend(Configuration(rf_power_w=150.0, pressure_mtorr=PRESSURE_MAX_MTORR), max_recommendations=6)
    for rec in recs:
        assert rec.recommended.pressure_mtorr <= PRESSURE_MAX_MTORR


def test_baseline_at_corner_still_produces_recommendations() -> None:
    """A baseline pinned at an envelope corner loses some candidates to clamping
    but must still yield recommendations from the remaining valid directions."""
    recs = recommend(Configuration(rf_power_w=RF_POWER_MIN_W, pressure_mtorr=1.0))
    assert len(recs) >= 1


# ---------------------------------------------------------------------------
# Classifier-enhanced scoring (FE-2.6.3)
# ---------------------------------------------------------------------------
def test_without_classifier_class_fields_are_none() -> None:
    for rec in recommend(MARGINAL_BASELINE):
        assert rec.class_before is None
        assert rec.class_after is None


def test_with_classifier_class_fields_are_populated(classifier) -> None:
    for rec in recommend(MARGINAL_BASELINE, classifier=classifier):
        assert rec.class_before is not None
        assert rec.class_after is not None


def test_classifier_class_improvement_affects_score(classifier) -> None:
    """A recommendation that lifts the predicted class should get a score bonus
    beyond its raw quality delta (the ML-scoring layer doing real work)."""
    recs = recommend(MARGINAL_BASELINE, classifier=classifier, max_recommendations=6)
    lifted = [r for r in recs if r.class_after is not None and r.class_before is not None
              and r.class_after != r.class_before]
    for rec in lifted:
        # score exceeds the pure quality delta when the class moved up
        from ai_module.recommendation_engine import _CLASS_RANK
        if _CLASS_RANK[rec.class_after] > _CLASS_RANK[rec.class_before]:
            assert rec.score > rec.quality_delta


# ---------------------------------------------------------------------------
# Recommendation history + verification (FE-2.6.4)
# ---------------------------------------------------------------------------
def test_store_and_load_recommendation(tmp_path) -> None:
    rec = recommend(MARGINAL_BASELINE)[0]
    rec_id = store_recommendation(rec, session_id="s1", db_path=tmp_path / "rec.db")
    history = load_recommendations("s1", db_path=tmp_path / "rec.db")
    assert len(history) == 1
    assert history.iloc[0]["recommendation_id"] == rec_id
    assert history.iloc[0]["quality_after"] == pytest.approx(rec.quality_after)
    assert history.iloc[0]["action_text"] == rec.action_text


def test_load_recommendations_filters_by_session(tmp_path) -> None:
    db = tmp_path / "rec.db"
    rec = recommend(MARGINAL_BASELINE)[0]
    store_recommendation(rec, session_id="A", db_path=db)
    store_recommendation(rec, session_id="B", db_path=db)
    assert len(load_recommendations("A", db_path=db)) == 1
    assert len(load_recommendations(db_path=db)) == 2


def test_applying_the_recommendation_exactly_verifies_true(tmp_path) -> None:
    db = tmp_path / "rec.db"
    rec = recommend(MARGINAL_BASELINE)[0]
    rec_id = store_recommendation(rec, db_path=db)
    assert verify_recommendation(rec_id, rec.recommended, db_path=db) is True


def test_applying_a_worse_change_verifies_false(tmp_path) -> None:
    db = tmp_path / "rec.db"
    # baseline where reducing pressure helps; "apply" a higher pressure instead (worse).
    rec = recommend(MARGINAL_BASELINE)[0]
    assert rec.quality_delta > 0  # the recommendation is a genuine improvement
    rec_id = store_recommendation(rec, db_path=db)
    worse = Configuration(rf_power_w=250.0, pressure_mtorr=20.0)
    assert verify_recommendation(rec_id, worse, db_path=db) is False


def test_verify_updates_the_stored_row(tmp_path) -> None:
    db = tmp_path / "rec.db"
    rec = recommend(MARGINAL_BASELINE)[0]
    rec_id = store_recommendation(rec, db_path=db)
    verify_recommendation(rec_id, rec.recommended, db_path=db)

    row = load_recommendations(db_path=db).iloc[0]
    assert row["applied"] == 1
    assert row["prediction_verified"] == 1
    assert row["observed_quality_after"] == pytest.approx(rec.quality_after, abs=1e-9)


def test_verify_unknown_recommendation_raises(tmp_path) -> None:
    db = tmp_path / "rec.db"
    store_recommendation(recommend(MARGINAL_BASELINE)[0], db_path=db)  # ensure table exists
    with pytest.raises(KeyError):
        verify_recommendation("no-such-id", MARGINAL_BASELINE, db_path=db)
