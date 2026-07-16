# AI-Assisted Plasma Process Experimentation & Digital Twin Analytics Platform

[![tests](https://github.com/rayyantaimoor1/plasma-digital-twin/actions/workflows/tests.yml/badge.svg)](https://github.com/rayyantaimoor1/plasma-digital-twin/actions/workflows/tests.yml)

Final Year Project — BSCS, COMSATS University Islamabad (2023–2027).

A software-only virtual plasma laboratory for low-pressure RF capacitively coupled (CCP)
argon plasma, paired with an AI analytics dashboard. The Digital Twin solves a
volume-averaged (0D) global discharge model (particle + power balance equations, per
Lieberman & Lichtenberg) to derive electron temperature and plasma density — nothing is
hand-fit. The AI Module trains on a synthetic dataset with genuine hidden confounders so
that classification accuracy reflects a real predictive task rather than a deterministic
function inversion.

Full specification: [`FYP_Scope_Document_v2_0_1.docx`](FYP_Scope_Document_v2_0_1.docx).
Non-negotiable technical decisions distilled from that spec: [`CLAUDE.md`](CLAUDE.md).

## Team

| Name | Registration No. | Role |
|---|---|---|
| Muhammad Rayyan Taimoor Khan | CIIT/FA23-BCS-072/ISB | Digital Twin & AI Core |
| Sadeem Ur Rehman | CIIT/FA23-BCS-127/ISB | Dashboard & Analytics |

Supervisor: Dr. Majid Iqbal.

## Project layout

```
digital_twin/   Sub-Modules 1.1-1.6 — physics engine, dataset generation, sensitivity, validation
ai_module/      Sub-Modules 2.1-2.8 — classification, anomaly detection, recommendations
dashboard/      Streamlit + Plotly UI unifying both modules (Phase 3)
data/           Generated SQLite DB, CSV datasets, MLflow tracking store (gitignored, reproducible)
tests/          pytest suite — one test file per module
```

## Setup

Requires Python 3.12+ (this project currently runs on 3.14.5; XGBoost 3.x and SHAP 0.5x
both require >= 3.12 to install).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

`requirements.txt` is pinned via `pip freeze` (on Windows); the single Windows-only
transitive dependency (`pywin32`) carries a `sys_platform == "win32"` marker so the
same file installs cleanly on Linux/macOS too.

## Continuous integration

Every push and pull request to `main` runs the pytest suite on GitHub Actions
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) — because all seeds are
fixed and recorded (FE-1.3.7), a CI run reproduces the same dataset, models, and metrics
as a local run.

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

The first load trains the models and builds the conformal uncertainty layer once
(cached thereafter). Set the operating point in the sidebar; the specialised panels
(Digital Twin, AI Analytics, Anomaly Monitor, Trends & Correlation, Suitability &
Recommendations, Session History, Model Training) are in the left-hand navigation.

## Running tests

```bash
pytest                 # full suite
pytest -m "not dashboard"   # skip the slow Streamlit AppTest smoke tests
pytest --cov=digital_twin --cov=ai_module --cov-report=term-missing   # with coverage
```

Coverage stands at **100% of testable code** across every module in `digital_twin/` and
`ai_module/` (`if __name__ == "__main__":` demo blocks are excluded via
`pyproject.toml`'s `[tool.coverage.report]` — each one is a manual entry point run via
`python -m <module>`, independently verified during development, not unit-test material).

## Status

- **Phase 0** — repo skeleton, environment, pinned `requirements.txt`. Done.
- **Sub-Module 1.2** — 0D global discharge physics engine
  ([`digital_twin/physics_engine.py`](digital_twin/physics_engine.py)). Done.
  Particle balance solves for electron temperature via `scipy.optimize.brentq`; power
  balance solves for plasma density. The driven-electrode sheath is computed from the
  collisionless Child–Langmuir law when an RF voltage is supplied
  (`simulate(..., rf_voltage_v=...)`), and falls back to a labelled estimate otherwise.
  Run it standalone with `python digital_twin/physics_engine.py` to see a sample sweep.
- **Sub-Module 1.1** — chamber parameter configuration engine
  ([`digital_twin/chamber_config.py`](digital_twin/chamber_config.py)). Done.
  Parameter ranges, three experiment mode presets (Stable Plasma / Exploratory
  Sweep / Stress Test), and physics-grounded validation.
- **Sub-Module 1.4** — multi-session experiment management
  ([`digital_twin/session_manager.py`](digital_twin/session_manager.py)). Done.
  SQLite-backed session persistence, retrieval, up-to-5 overlay comparison, and
  summary reports.
- **Sub-Module 1.5** — parameter sensitivity and sweep analysis
  ([`digital_twin/sensitivity_analysis.py`](digital_twin/sensitivity_analysis.py)). Done.
  One-at-a-time sweeps, sensitivity ranking, and Plotly figures (bar chart, effect
  curves, paired-sweep heatmap).
- **Sub-Module 1.3** — synthetic dataset generation with hidden confounders
  ([`digital_twin/dataset_generation.py`](digital_twin/dataset_generation.py)). Done.
  Wall-temperature drift, electrode aging, and gas-purity variance shift the label but
  are withheld from the features (confounder strengths documented a priori in the module
  docstring). Soft quartile labelling, region + random splits, CSV export, SQLite storage.
  Run `python -m digital_twin.dataset_generation` to see the confounding in action.
- **Sub-Module 1.6** — physics validation and literature benchmarking
  ([`digital_twin/physics_validation.py`](digital_twin/physics_validation.py)). Done.
  Benchmarks the physics engine against independently-sourced reference values (Turner,
  2014, "Global Models" lecture — an argon global-model worked example cross-checked
  against a PIC simulation), with stated tolerance bands and honest pass/fail reporting
  (5/7 checks currently pass; 2 fail and are reported as such, not hidden). Run
  `python -m digital_twin.physics_validation` to see the full benchmark report.

Phase 1 (Digital Twin) is complete. Phase 2 (AI Module) in progress:

- **Sub-Module 2.1** — plasma process suitability classification engine
  ([`ai_module/classification.py`](ai_module/classification.py)). Done.
  Random Forest and XGBoost benchmarked against a logistic-regression baseline,
  with McNemar's exact test (`scipy.stats.binomtest`) reporting whether each
  ensemble model is statistically significantly better than the baseline —
  reported honestly per non-negotiable principle #3, whatever the result. On the
  current dataset neither ensemble is significant (p > 0.05 on both splits) — a
  real, reported outcome, not tuned to look better. Also: SHAP per-prediction and
  global feature importance, side-by-side model comparison with agreement
  flagging, and dual-split (region/random) evaluation per FE-1.3.5. Run
  `python -m ai_module.classification` to see a full evaluation report.

- **Sub-Module 2.7** — AI model training, evaluation, and retraining interface
  ([`ai_module/training.py`](ai_module/training.py)). Done.
  Model registry (training params, dataset size, timestamp, metrics), stratified
  k-fold cross-validation, a cross-model feature-importance comparison (same
  SHAP-based measure for all three models, so the bars are actually comparable),
  and MLflow local-file-store logging of every training run (`data/mlruns/`).
  Retraining from accumulated stored datasets pulls from Sub-Module 1.3's dataset
  storage specifically — not Sub-Module 1.4's session log, which lacks
  confounder-based labels and would be circular to train on. Run
  `python -m ai_module.training` to see a full training + cross-validation report.
- **Sub-Module 2.2** — real-time anomaly detection and process instability monitor
  ([`ai_module/anomaly_detection.py`](ai_module/anomaly_detection.py)). Done.
  Anomalies are physics-relationship violations, not out-of-range values (principle
  #4): faults are injected as in-range points whose observed outputs are inconsistent
  with the logged inputs, and Isolation Forest detects them on the physics RESIDUAL
  (observed − digital-twin-predicted), not the raw features. Demonstrated contrast:
  ~93% Isolation Forest recall vs ~0% for a naive range check on the same anomalies.
  Includes severity levels, root-cause indication, SQLite event logging, an SPC
  control chart on process quality, and a Plotly anomaly timeline. Run
  `python -m ai_module.anomaly_detection` to see the detection-vs-range-check report.
  - **FE-2.2.6 (optional, deep-learning comparison)**
    ([`ai_module/autoencoder_detector.py`](ai_module/autoencoder_detector.py)). Done.
    A lightweight PyTorch autoencoder (reconstruction error = anomaly score) as a
    second, deep-learning detector. Trained on the raw features, it *learns* the
    normal manifold directly and matches Isolation Forest (~91% vs ~93% recall)
    **without** the manual residual feature engineering the classical method needed
    — the honest classical-vs-deep-learning finding FE-2.2.6 asks for. PyTorch is
    optional and kept out of the base install (see `requirements-optional.txt`);
    `pip install -r requirements-optional.txt`, then
    `python -m ai_module.autoencoder_detector` for the comparison, or `pytest -m torch`.
- **Sub-Module 2.3** — plasma trend analysis and monitoring engine
  ([`ai_module/trend_analysis.py`](ai_module/trend_analysis.py)). Done.
  Tracks reactivity, process quality, uniformity, and electron temperature across
  sequential runs in a session; EMA smoothing separates systematic trend from
  noise; degradation/recovery (monotonic run) and instability (oscillation) events
  are detected on the smoothed series and annotated on the trend chart; summary
  statistics include a linear trend-direction coefficient. Run
  `python -m ai_module.trend_analysis` to see a synthetic session's detected events.
- **Sub-Module 2.4** — multi-parameter correlation and relationship analysis
  ([`ai_module/correlation_analysis.py`](ai_module/correlation_analysis.py)). Done.
  Pearson correlation heatmap over the full simulation output vector, scatter +
  regression for the three key relationships (power/reactivity, pressure/uniformity,
  density/quality), Plotly parallel coordinates, and template-generated correlation
  narration that flags the genuinely weak, physically-expected density-quality
  relationship (r≈0.20, verified) alongside the strongest observed relationships.
  Run `python -m ai_module.correlation_analysis` to see the narration output.
- **Sub-Module 2.5** — semiconductor process suitability analysis
  ([`ai_module/suitability_analysis.py`](ai_module/suitability_analysis.py)). Done.
  Maps ion bombardment energy to four application categories (cleaning, surface
  treatment, thin-film deposition, plasma etching) via windows whose ORDERING is
  cited to the plasma-processing literature consensus (cleaning < treatment ≈
  deposition < etching in required ion energy — Lieberman & Lichtenberg Ch. 1);
  the exact eV boundaries are openly-stated round calibrations, not misattributed
  to false precision. Verified consequence: etching is essentially unreachable
  without an applied RF voltage (9% suitable across the default envelope vs 63–86%
  for the other three) — a genuine physical result, since real RIE tools are
  distinguished from gentler CCP processes by their driven, biased sheath.
  Compliance scorecards, bootstrap confidence intervals on application-specific
  defect risk, and ranking of stored sessions by suitability. Run
  `python -m ai_module.suitability_analysis` to see the etching-vs-cleaning contrast.
- **Sub-Module 2.6** — intelligent, counterfactual recommendation engine
  ([`ai_module/recommendation_engine.py`](ai_module/recommendation_engine.py)). Done.
  Every recommendation is counterfactual (principle #5): a best-practice rule layer
  proposes candidate parameter adjustments, but each is re-run through the digital
  twin and reported with its REAL predicted numeric outcome (e.g. "reduce chamber
  pressure from 18 mTorr to 12.6 mTorr → process quality 0.209 → 0.226"), never
  bare advisory text. Candidates are ranked by re-simulated quality gain, optionally
  boosted by the classifier's predicted suitability-class transition; a SQLite
  history log records each recommendation's predicted effect and can verify whether
  an applied change delivered it. Run `python -m ai_module.recommendation_engine`
  to see three quantified recommendations and a verification.
- **Sub-Module 2.8** — uncertainty quantification and trust layer
  ([`ai_module/uncertainty_quantification.py`](ai_module/uncertainty_quantification.py)). Done.
  MAPIE 1.4.1 split-conformal prediction replaces point estimates with calibrated
  guarantees: classification gets prediction SETS (a set of size 1 is an
  unambiguous call, size 2+ means the model genuinely can't distinguish those
  classes at this confidence level), and a defect-probability regressor — trained
  on Sub-Module 1.3's confounded `true_defect_probability`, a genuine
  irreducible-uncertainty target — gets calibrated intervals instead of a single
  number. Coverage is verified empirically on a held-out test set (measured
  92.0% classification / 90.3% regression against a 90% target), not assumed.
  Verified example: 150 W/10 mTorr (a real class-boundary condition) is flagged
  low-confidence with a 2-class set, while 280 W/1.5 mTorr gets a single-class,
  high-confidence call. Run `python -m ai_module.uncertainty_quantification` to
  see the coverage report and the trust-layer contrast.

- **Phase 3 — Streamlit dashboard integration**
  ([`dashboard/`](dashboard/)). Done.
  A multipage Streamlit app wiring every Digital Twin (1.1–1.6) and AI (2.1–2.8)
  sub-module into one interface: a shared chamber-configuration sidebar drives a
  live control-room overview plus seven analysis panels (Digital Twin, AI Analytics,
  Anomaly Monitor, Trends & Correlation, Suitability & Recommendations, Session
  History, Model Training). Heavy work (dataset generation, model training, conformal
  calibration) is cached so it runs once. Every page is smoke-tested via Streamlit's
  `AppTest` harness (`tests/test_dashboard.py`), which runs each panel headless and
  fails on any exception. Launch with `streamlit run dashboard/app.py`.

- **Phase 4 — full pytest coverage pass**. Done.
  Audited every source module's public API and every `raise` site (18 across 9
  modules) against its test file. Found and closed two genuine gaps: the physics
  engine's rate-coefficient/energy-loss building blocks (`excitation_rate_coeff`,
  `elastic_rate_coeff`, `collisional_energy_loss`, `ion_mean_free_path`,
  `total_energy_per_pair`) were previously exercised only indirectly through
  `solve_electron_temperature`/`solve_plasma_density` — now directly tested,
  including a permanent regression test for the Lieberman & Lichtenberg Fig. 3.17
  reference check that had only existed as an ad hoc verification script during
  development; and two untested error paths in the anomaly detector (scoring
  before `.fit()`, an unrecognised fault type). All other raise sites were
  confirmed already covered 1:1. **344 tests passing, 0 failures.**

See [`CLAUDE.md`](CLAUDE.md) for the full module build order and non-negotiable
technical principles — read it before starting any new module.
