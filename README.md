# AI-Assisted Plasma Process Experimentation & Digital Twin Analytics Platform

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

## Running tests

```bash
pytest
```

## Status

- **Phase 0** — repo skeleton, environment, pinned `requirements.txt`. Done.
- **Sub-Module 1.2** — 0D global discharge physics engine
  ([`digital_twin/physics_engine.py`](digital_twin/physics_engine.py)). Done.
  Particle balance solves for electron temperature via `scipy.optimize.brentq`; power
  balance solves for plasma density; sheath model adds a power-dependent RF
  self-bias voltage on top of the DC floating sheath. Run it standalone with
  `python digital_twin/physics_engine.py` to see a sample sweep.
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

See [`CLAUDE.md`](CLAUDE.md) for the full module build order and non-negotiable technical
principles — read it before starting any new module. Sub-Module 1.3 (synthetic dataset
generation with hidden confounders) remains, ahead of 1.6 (physics validation).
