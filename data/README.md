# data/

Generated artifacts live here at runtime and are not committed to git (see `.gitignore`):

- `experiments.db` — SQLite database of simulation runs, sessions, and AI analysis results.
- `*.csv` — exported synthetic datasets from Sub-Module 1.3 (parameter sweeps + labels).
- `mlruns/` — local MLflow tracking store for model training runs (Sub-Module 2.7).

This folder is kept in git via this README so the expected layout is visible; the
generated files themselves are reproducible from a fixed random seed (see FE-1.3.7).
