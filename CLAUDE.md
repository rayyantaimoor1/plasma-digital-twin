# AI-Assisted Plasma Process Experimentation & Digital Twin Analytics Platform

## Project context
Final Year Project, COMSATS University Islamabad, BSCS (2023-2027).
- Muhammad Rayyan Taimoor Khan (CIIT/FA23-BCS-072/ISB) — Digital Twin & AI core lead.
- Sadeem Ur Rehman (CIIT/FA23-BCS-127/ISB) — Dashboard & Analytics lead.
- Supervisor: Dr. Majid Iqbal.

Full specification: `FYP_Scope_Document_v2_0.docx` in this repo — read it before starting
any new module. This file is a persistent summary of the non-negotiable technical
decisions from that document; the docx is the source of truth for detail.

Goal: a software-only virtual plasma laboratory (low-pressure RF capacitively coupled
argon plasma) plus an AI analytics dashboard, submitted to compete for first place in
the department's FYP evaluation. Rayyan has limited hands-on coding experience —
favor clear, well-commented code over clever code, and explain WHY for any non-obvious
physics or statistics decision in comments, not just WHAT the code does. Both students
need to be able to explain and defend every part of this codebase without notes.

## Tech stack (do not substitute without asking first)
- Python 3.12+ — this is a hard floor, not a suggestion: current XGBoost (3.x) and
  SHAP (0.5x) both require Python >= 3.12 to install.
- Streamlit (dashboard) + Plotly (visualisation) + SQLite (storage)
- Scikit-learn, XGBoost, SHAP, MAPIE (optional), MLflow (local tracking only — use
  only the classic `mlflow.log_param` / `mlflow.log_metric` API with a local
  `mlruns/` file store; ignore MLflow's newer GenAI/agent-tracing features, they are
  not relevant here)
- pytest for all tests; GitHub Actions CI is optional but encouraged once Phase 0 exists
- SciPy — both for the physics engine's root-finding and for `scipy.stats`
  significance testing in the ML evaluation

Pin exact versions in `requirements.txt` via `pip freeze` once installed — don't
hand-copy version floors from the proposal document, they will already be stale.

## Non-negotiable technical principles
These fix specific flaws that were caught and deliberately corrected during proposal
review. Do not quietly reintroduce any of them for convenience:

1. **Physics engine (Sub-Module 1.2) must be a real 0D (volume-averaged) global
   discharge model** — solved via particle balance + power balance equations
   (Lieberman & Lichtenberg, *Principles of Plasma Discharges and Materials
   Processing*, Ch. 10), using `scipy.optimize` root-finding for electron
   temperature (Te), then solving for plasma density (n_e). This is NOT a hand-fit
   parametric curve. Every derived output (reactivity, uniformity, etch rate,
   process quality index) must come from Te/n_e, never be set independently.

2. **Synthetic dataset generation (Sub-Module 1.3) must include hidden confounders**
   — simulated wall-temperature drift, electrode aging, gas-purity variance — that
   affect the process quality index but are NOT passed to the classifier as input
   features. Confounder strengths and noise levels must be fixed from physical
   reasoning and documented BEFORE any model is trained, with the random seed
   recorded. Never tune these parameters after seeing model accuracy — that
   reintroduces exactly the circularity this design is meant to prevent.

3. **Classification success (Sub-Module 2.1) is measured as a statistically
   significant improvement over a logistic-regression baseline** (McNemar's test or
   a paired bootstrap via `scipy.stats`), not an absolute accuracy percentage.
   Report whatever accuracy results honestly, even if it's not a round number.

4. **Anomaly detection (2.2): anomalies are physics-relationship violations**
   (simulated faults/drift that break the expected parameter-to-output
   relationship), not simple out-of-range parameter values — a plain range check
   would already catch the latter, which would defeat the point of using Isolation
   Forest at all.

5. **Recommendations (2.6) must be counterfactual** — re-run the suggested
   parameter change through the actual digital twin and report the real predicted
   numeric outcome. Never present advisory text without a re-simulated, quantified
   effect behind it.

6. **Physics validation (1.6) is semi-quantitative** — compare the model's output
   at a few reference operating points against published reference values from the
   cited literature, with a stated percentage tolerance band. "The trend looks
   right" is not sufficient on its own.

7. **Keep Random Forest / XGBoost as the primary classifiers.** Do not add deep
   learning to the core classification task — tree ensembles are empirically
   stronger than deep learning on tabular data structurally similar to this
   project's (Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022). PyTorch/TensorFlow is
   scoped ONLY for the optional autoencoder anomaly-detection comparison in 2.2 —
   it is not required, and should not creep into 2.1.

## Module build order
- **Phase 0** — repo skeleton, virtual environment, `requirements.txt`, SQLite
  schema design. Do this first, and set up `pytest` from the start.
- **Phase 1 (Digital Twin)** — build in this order: 1.2 (physics core) -> 1.1 ->
  1.3 -> 1.4 -> 1.5 -> 1.6. Nothing else can be tested realistically until 1.2
  and 1.3 exist, since they generate the data everything downstream consumes.
- **Phase 2 (AI Module)** — 2.1 -> 2.7 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.6 ->
  2.8 (optional, only if time allows).
- **Phase 3** — Streamlit dashboard integration across both modules.
- **Phase 4** — full pytest coverage pass, then report writing.

Build and test one sub-module at a time. Don't move to the next until the current
one runs standalone with at least a basic pytest file covering it. Commit to git
after each working sub-module, with a commit message that names the sub-module.

## Conventions
- Type hints on all function signatures.
- Docstrings explain WHY for anything physics- or statistics-related — this code
  will be defended live in front of an evaluation committee, so comments should
  help a student explain the reasoning, not just restate what the line does.
- One test file per module under `tests/`, using `pytest`.
- Ask before adding a new dependency that isn't already in the tools list above.
