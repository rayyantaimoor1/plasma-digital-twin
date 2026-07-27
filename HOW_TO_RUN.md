# How to Run This Project

A complete, step-by-step guide to installing and running the AI-Assisted Plasma
Process Experimentation & Digital Twin Analytics Platform from a clean machine.
Every command below is specific to this repository's actual files — not generic
Python-project boilerplate.

---

## 1. Prerequisites

- **Python 3.12 or newer.** This is a hard floor, not a suggestion: `xgboost==3.3.0`
  and `shap==0.52.0` in [`requirements.txt`](requirements.txt) both require Python
  >= 3.12 to install. This project was developed and pinned against **Python
  3.14.5**; anything 3.12+ should work, but 3.14.x is the most tested.
- **git**, to clone the repository.
- No database server, no Docker, no external services. Persistence is a local
  SQLite file created automatically on first use; there is nothing else to stand up.

Check your Python version before starting:

```bash
python --version
```

On some systems the interpreter is only available as `python3`, not `python` — use
whichever one reports 3.12+.

---

## 2. Clone the repository

```bash
git clone https://github.com/rayyantaimoor1/plasma-digital-twin.git
cd plasma-digital-twin
```

(If you already have the code locally without git history, just `cd` into the
project root — the folder containing [`requirements.txt`](requirements.txt) and
[`CLAUDE.md`](CLAUDE.md) — and skip to Step 3.)

---

## 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash / this project's own dev setup):**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Your shell prompt should now be prefixed with `(.venv)`. Every command from here on
assumes the virtual environment is active — if you open a new terminal, re-activate
it first.

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

Notes specific to this file:

- [`requirements.txt`](requirements.txt) is a **fully pinned** `pip freeze` output
  (exact `==` versions for every package, direct and transitive) — not a loose
  floor list. Installing from it should resolve quickly and deterministically.
- **If you ever install packages individually instead of from this file** (e.g.
  adding `streamlit`, `xgboost`, `shap`, `mlflow` one at a time by hand), pip's
  resolver can fall into a long backtracking loop trying to satisfy all of their
  cross-version constraints together. Installing straight from the pinned
  `requirements.txt` avoids this entirely — that's the reason the file is pinned
  rather than left as loose version ranges.
- One line in the file, `pywin32==312 ; sys_platform == "win32"`, carries a platform
  marker: `pip` installs it on Windows and **skips it automatically** on
  macOS/Linux, so the same file works on all three platforms without edits.
- **PyTorch is deliberately not in this file.** It is only needed for one optional
  feature (Step 8 below); the core platform never imports it.

This installs everything needed for the physics engine, the AI module (scikit-learn,
XGBoost, SHAP, MAPIE), the dashboard (Streamlit, Plotly), and MLflow's local
tracking store.

---

## 5. Verify the install

Run one physics-engine demo directly — no dataset generation or model training
required, so this is the fastest possible sanity check:

```bash
python digital_twin/physics_engine.py
```

You should see a small printed table (RF power, pressure, electron temperature,
plasma density, ion flux, ion energy, process quality, defect probability) for a
4×4 sweep of operating points. If this prints without errors, the core physics
engine and its dependencies (`numpy`, `scipy`) are working correctly.

---

## 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

This opens the app in your browser (Streamlit prints the local URL, typically
`http://localhost:8501`). [`dashboard/app.py`](dashboard/app.py) is the landing
page — a live control-room view (current operating point, suitability
classification, anomaly status, best semiconductor-process fit). The specialised
panels live in [`dashboard/pages/`](dashboard/pages) and appear automatically in
Streamlit's left-hand page navigation:

| Page | File | Covers |
|---|---|---|
| Digital Twin | `1_Digital_Twin.py` | Full simulation output, sensitivity analysis (1.5) |
| Physics Validation | `2_Physics_Validation.py` | Literature-benchmark physics validation (1.6) |
| AI Analytics | `3_AI_Analytics.py` | Classification, SHAP explainability (2.1) |
| Anomaly Monitor | `4_Anomaly_Monitor.py` | Physics-relationship anomaly detection (2.2) |
| Trends and Correlation | `5_Trends_and_Correlation.py` | Session trend tracking (2.3), correlation analysis (2.4) |
| Suitability and Recommendations | `6_Suitability_and_Recommendations.py` | Semiconductor suitability (2.5), counterfactual recommendations (2.6) |
| Session History | `7_Session_History.py` | Multi-session persistence and comparison (1.4) |
| Model Training | `8_Model_Training.py` | Model registry, cross-validation, MLflow history (2.7) |

**First load is the slowest step in this whole guide** — it generates the synthetic
training dataset, trains all three classifiers, fits the anomaly detector, and
builds the conformal uncertainty layer, all of which are then cached
(`st.cache_data` / `st.cache_resource` in [`dashboard/backend.py`](dashboard/backend.py))
so every later interaction is fast. Use the sidebar to set RF power and chamber
pressure (or pick an experiment-mode preset); every page reads that same shared
operating point.

To stop the dashboard, press `Ctrl+C` in the terminal it's running in.

### Double-click launchers (Windows)

For a live demo or viva, where typing commands is awkward, three batch files at the
project root do the same thing on a double-click:

| File | Starts |
|---|---|
| [`run_streamlit.bat`](run_streamlit.bat) | The Streamlit dashboard alone (this section) |
| [`run_reactor.bat`](run_reactor.bat) | The Reactor Control Room alone (section 6b) |
| [`run_both.bat`](run_both.bat) | Both, each in its own window, opening both in the browser |

Each checks that the virtual environment exists first and pauses on exit, so an error
stays readable instead of the window closing instantly.

---

## 6b. Run the Reactor Control Room (optional companion app)

A separate, presentation-oriented UI built over the **same** functions as the dashboard
— see [`reactor_control_room/README.md`](reactor_control_room/README.md). It is *not*
part of the graded core stack; its backend is a thin JSON wrapper that performs no
physics or AI of its own, so both UIs are structurally guaranteed to show identical
numbers for identical inputs.

Its extra dependencies are scoped to their own file, deliberately kept out of the core
`requirements.txt`:

```bash
pip install -r reactor_control_room/requirements.txt
```

Then, from the project root:

```bash
uvicorn reactor_control_room.backend.app:app
```

Open <http://127.0.0.1:8000/> for the control room, or <http://127.0.0.1:8000/docs> for
the interactive API. Four pages: **Reactor View** (chamber + controls + 10 output
gauges), **AI Verdict** (classification with SHAP explanations, anomaly status,
best-fit application), **Physics Validation**, and **Session Replay**.

Session Replay reads saved runs from the same SQLite store as the dashboard. If it is
empty, seed a spread of real `simulate()` runs:

```bash
python reactor_control_room/seed_demo_sessions.py
```

> **First load is slow, twice.** The first classification request trains the models
> (~20 s), and the first SHAP request additionally fits the explainer (~25 s). Both are
> cached afterwards — later requests are ~1–2 s. Open the app once before a live demo so
> the audience never waits.

---

## 7. Run individual sub-modules standalone (no dashboard)

Every module in [`digital_twin/`](digital_twin) and [`ai_module/`](ai_module) has a
`python -m <module>` demo block that prints real output to the terminal — useful for
inspecting one sub-module in isolation, or for the FYP defence. Run these from the
project root:

```bash
python -m digital_twin.dataset_generation          # generate a dataset, show confounding in action
python -m digital_twin.physics_validation          # benchmark vs. literature reference values
python -m ai_module.classification                 # train + evaluate all 3 classifiers, McNemar test
python -m ai_module.training                       # full training + cross-validation report
python -m ai_module.anomaly_detection               # anomaly detection vs. naive range-check
python -m ai_module.trend_analysis                  # synthetic session trend/event detection
python -m ai_module.correlation_analysis            # correlation heatmap + narration
python -m ai_module.suitability_analysis            # semiconductor suitability scorecards
python -m ai_module.recommendation_engine           # counterfactual recommendations + verification
python -m ai_module.uncertainty_quantification      # conformal prediction coverage report
```

(`digital_twin/physics_engine.py` is the one module with no imports from elsewhere
in this project, so it can also be run as a direct script:
`python digital_twin/physics_engine.py`, matching Step 5 above. Every other module
imports across the `digital_twin`/`ai_module` package boundary and must be run with
the `-m package.module` form shown here — a direct `python path/to/file.py` on any
of them fails with `ModuleNotFoundError`.)

---

## 8. (Optional) Install the deep-learning autoencoder extra

One optional feature — the FE-2.2.6 PyTorch autoencoder anomaly detector, a
classical-vs-deep-learning comparison — needs an extra dependency kept out of the
core install:

```bash
pip install -r requirements.txt -r requirements-optional.txt
```

Then run its own demo or the dashboard's comparison table (Anomaly Monitor page):

```bash
python -m ai_module.autoencoder_detector
```

If you skip this step, the core platform (including the dashboard) still runs
normally — [`dashboard/backend.py`](dashboard/backend.py) imports `torch` lazily
and the Anomaly Monitor page shows an install hint instead of the comparison table.

---

## 9. Run the test suite

```bash
pytest                                  # full suite
```

Useful variations, matching this project's actual pytest markers
(defined in [`pyproject.toml`](pyproject.toml)):

```bash
pytest -m "not dashboard and not torch"   # skip the slow Streamlit AppTest pages + optional torch tests
                                           # (this is exactly what CI runs — see .github/workflows/tests.yml)
pytest -m dashboard                       # only the Streamlit AppTest smoke tests
pytest -m torch                           # only the optional autoencoder tests (needs Step 8 first)
pytest tests/test_physics_engine.py       # one module's test file
pytest --cov=digital_twin --cov=ai_module --cov-report=term-missing   # with coverage
```

All random seeds used anywhere in this project are fixed and recorded (see
[`CLAUDE.md`](CLAUDE.md) principle #2 and `DEFAULT_SEED` in
[`digital_twin/dataset_generation.py`](digital_twin/dataset_generation.py)), so a
test run reproduces the same datasets, models, and metrics every time, on any
machine.

Expect the full suite (including the dashboard and torch-marked tests, if PyTorch
is installed) to take several minutes — most of that time is spent training real
Random Forest / XGBoost models and fitting the Isolation Forest detector across
many test fixtures, deliberately with no mocking (see the module docstrings in
`tests/` for why).

---

## 10. Where generated data goes

Nothing above requires any manual setup of storage. The first thing that touches
the database or MLflow creates it automatically:

- `data/experiments.db` — SQLite database (sessions, anomaly events, recommendation
  history, stored datasets). Created on first write.
- `data/mlruns/` — local MLflow file-store tracking directory (Sub-Module 2.7).
  Created on first training run logged to MLflow.
- `data/*.csv` — only created if you explicitly export a dataset to CSV.

None of these are committed to git (see [`.gitignore`](.gitignore) and
[`data/README.md`](data/README.md)) — they're regenerated deterministically from
the fixed seeds, so deleting `data/experiments.db` and rerunning the dashboard or
tests is always safe and just starts from a clean database.

---

## Quick reference: full clean-machine sequence

```bash
git clone https://github.com/rayyantaimoor1/plasma-digital-twin.git
cd plasma-digital-twin
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\Activate.ps1 on Windows PowerShell
pip install -r requirements.txt
python digital_twin/physics_engine.py   # sanity check
pytest -m "not dashboard and not torch" # run the test suite
streamlit run dashboard/app.py          # launch the dashboard
```
