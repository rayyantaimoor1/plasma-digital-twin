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

Phase 0 (repo skeleton) in progress. See [`CLAUDE.md`](CLAUDE.md) for the full module
build order and non-negotiable technical principles — read it before starting any new
module.
