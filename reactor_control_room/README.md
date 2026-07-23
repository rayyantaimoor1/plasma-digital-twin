# Reactor Control Room (companion app)

A standalone, cinematic companion piece for the viva demo (FUTURE.md item 1),
**separate** from the graded Streamlit dashboard and **not** part of the core
stack locked in `CLAUDE.md`. Only the backend exists so far; the frontend is a
later step.

## The correctness guarantee

The backend is a **thin JSON wrapper** over the project's existing functions. It
does **no** physics or AI computation of its own — every endpoint calls one
already-tested function and returns its output serialized to JSON:

| Endpoint | Wraps |
|---|---|
| `GET /api/simulate?rf_power_w=&pressure_mtorr=&rf_voltage_v=` | `digital_twin.physics_engine.simulate` |
| `GET /api/classify?rf_power_w=&pressure_mtorr=` | `ai_module.classification.classify_configuration` (Random Forest) |
| `GET /api/suitability?rf_power_w=&pressure_mtorr=&n_bootstrap=&rf_voltage_v=` | `ai_module.suitability_analysis.all_application_defect_estimates` |
| `GET /api/physics-validation` | `digital_twin.physics_validation.benchmark_summary_table` |
| `GET /api/sessions` · `GET /api/sessions/{id}` | `ExperimentDatabase.list_sessions` / `get_session` |
| `GET /api/system/stats` | `psutil` (host CPU/RAM; GPU reported honestly as idle) |

Because the Streamlit dashboard imports the *same* functions, the two UIs are
structurally guaranteed to show identical numbers for identical inputs — there is
only ever one place the math runs. `tests/test_reactor_backend.py` proves each
endpoint returns exactly what the underlying function returns.

## Install & run

The extra dependencies (`fastapi`, `uvicorn`, `psutil`) are scoped to this
companion app in [`requirements.txt`](requirements.txt) — the project's core
`requirements.txt` is untouched.

```bash
pip install -r reactor_control_room/requirements.txt
# from the project root:
uvicorn reactor_control_room.backend.app:app --reload
# then open http://127.0.0.1:8000/docs for the interactive API
```

## Tests

```bash
pytest tests/test_reactor_backend.py -v
```
