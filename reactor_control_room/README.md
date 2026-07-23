# Reactor Control Room (companion app)

A standalone, cinematic companion piece for the viva demo (FUTURE.md item 1),
**separate** from the graded Streamlit dashboard and **not** part of the core
stack locked in `CLAUDE.md`. A small FastAPI backend wraps the project's existing
functions and serves a full-screen, four-page frontend that renders them as a
"reactor control room".

## The correctness guarantee

The backend is a **thin JSON wrapper** over the project's existing functions. It
does **no** physics or AI computation of its own — every endpoint calls one
already-tested function and returns its output serialized to JSON:

| Endpoint | Wraps |
|---|---|
| `GET /api/simulate?rf_power_w=&pressure_mtorr=&rf_voltage_v=` | `digital_twin.physics_engine.simulate` |
| `GET /api/classify?rf_power_w=&pressure_mtorr=` | `ai_module.classification.classify_configuration` (Random Forest) |
| `GET /api/suitability?rf_power_w=&pressure_mtorr=&n_bootstrap=&rf_voltage_v=` | `ai_module.suitability_analysis.all_application_defect_estimates` |
| `GET /api/suitability-scorecard?rf_power_w=&pressure_mtorr=&rf_voltage_v=` | `ai_module.suitability_analysis.classify_suitability` |
| `GET /api/anomaly?rf_power_w=&pressure_mtorr=&fault=` | `ai_module.anomaly_detection.PlasmaAnomalyDetector` (+ `inject_anomaly`) |
| `GET /api/physics-validation` | `digital_twin.physics_validation.run_literature_benchmarks` |
| `GET /api/sessions` · `GET /api/sessions/{id}` | `ExperimentDatabase.list_sessions` / `get_session` |
| `GET /api/system/stats` | `psutil` (host CPU/RAM; GPU reported honestly as idle) |

Because the Streamlit dashboard imports the *same* functions, the two UIs are
structurally guaranteed to show identical numbers for identical inputs — there is
only ever one place the math runs. `tests/test_reactor_backend.py` proves each
endpoint returns exactly what the underlying function returns.

## The frontend

Static HTML/CSS/JS in [`frontend/`](frontend) — no build step, no framework —
served by the backend at `/` (same origin, so `/api/*` needs no CORS). It is
built from a **shared component library** ([`components.js`](frontend/components.js):
chamber, gauge, comparison bar, sparkline) matching the approved mockup in
[`../mockups/reactor-control-room/`](../mockups/reactor-control-room). Every
number shown is **fetched** ([`api.js`](frontend/api.js) is the only seam to the
backend) — nothing is computed in JavaScript; the components only render values
they are handed.

Four pages (hash-routed): **Reactor View** (chamber + sliders + 10 output
gauges), **AI Verdict** (classification / anomaly / best-fit donuts), **Physics
Validation** (pass-rate gauge + source cards + comparison bars), **Session
Replay** (run log + gauges + sparklines). A live CPU/RAM strip runs on every page.

[`components-demo.html`](frontend/components-demo.html) is a dev showcase that
renders every component from live backend data.

## Install & run

The extra dependencies (`fastapi`, `uvicorn`, `psutil`) are scoped to this
companion app in [`requirements.txt`](requirements.txt) — the project's core
`requirements.txt` is untouched.

```bash
pip install -r reactor_control_room/requirements.txt
# from the project root:
uvicorn reactor_control_room.backend.app:app --reload
# then open http://127.0.0.1:8000/         (the control room)
#           http://127.0.0.1:8000/docs      (interactive API)
```

### Demo sessions for Session Replay

The Session Replay page reads saved runs from the shared session store
(`data/experiments.db`, gitignored). If it's empty, seed a spread of real
`simulate()` runs (genuine engine outputs, tagged by experiment mode):

```bash
python reactor_control_room/seed_demo_sessions.py   # idempotent
```

Delete `data/experiments.db` to clear them. (These also appear in the dashboard's
Session History, since it is the same store.)

## Tests

```bash
pytest tests/test_reactor_backend.py -v
```
