# Efficiency & Simplification Review

A second-pass, **read-only** review of the whole repository, separate from
`AUDIT_REPORT.md`. The brief was: find places that could be made faster, lighter,
or simpler **without changing any output or behaviour** — reduced time complexity,
reduced memory, fewer redundant computations, or a data structure/dependency
heavier than the task needs. For every finding I state the current approach, the
proposed change, and **explicitly confirm whether the output stays numerically
identical**.

**Nothing in this document has been applied.** It is a findings report for us to
review together and prioritise, exactly like the audit.

Reviewed at commit `c3d0511` (after the calibration change). All numbers below are
**measured**, not estimated — the measurement commands are noted so they can be
re-run.

---

## The one fact that frames everything

Every finding below traces back to a single measured property of the physics
engine:

```
single simulate()                      = 185 us
  solve_electron_temperature() (brentq) = 156 us   <- 84% of the whole cost
```

`solve_electron_temperature` ([physics_engine.py:359](digital_twin/physics_engine.py:359))
is a Brent root-find through an exponential rate coefficient, and it is **84% of a
simulate() call**. It is also a **pure function of `(pressure, geometry)` only** —
independent of RF power (this is the global model's signature prediction, stated in
the module docstring). So:

- Any workload that **repeats a pressure** (or sweeps power at fixed pressure) is
  re-running the expensive 84% for an answer it already has.
- Everything else (`n_e`, sheath, indices, noise) is the cheap 16%.

That single fact is why the redundancy findings are worth acting on and why they
are all **numerically identical** — they remove repeat evaluations of deterministic
pure functions, never change a formula.

---

## Summary (ranked by impact)

| # | Location | What's redundant | Measured effect | Output identical? | Effort |
|---|----------|------------------|-----------------|-------------------|--------|
| F1 | `physics_engine.solve_electron_temperature` | Te re-solved for a pressure already solved | **−82%** on a fixed-pressure power sweep | ✅ identical | Low |
| F2 | `dataset_generation.generate_dataset` | "measured" row re-solves the physics `nominal` already has | **1870 → 990 solves (−47%)** on the default dataset | ✅ identical | Low–Med |
| F3 | `suitability_analysis.all_application_defect_estimates` | 4 applications each re-run the *same* bootstrap | **320 → 80 solves (−75%)** at `n_bootstrap=80` | ✅ identical | Low |
| F4 | `anomaly_detection.PlasmaAnomalyDetector` | predicted outputs computed 2× in `fit`, again per call, per duplicate row | **fit 2520 → ~420 (−83%)**; dashboard point 3 calls → 1 | ✅ identical | Med |
| F5 | `recommendation_engine._evaluate_candidate` | baseline re-classified for every candidate | **11× → 1×** baseline classify+simulate | ✅ identical | Low |
| F6 | `physics_engine.simulate` | `_discharge_geometry_factors` run 4–5× per call | ~a few % per simulate, universal | ✅ identical | Low–Med |
| F7 | `classification.train_classifiers` | uncalibrated `explainer_model` fit even when SHAP is never used | ~2× RF/XGB fit in eval/CV/McNemar paths | ✅ identical (trade-off: memory) | Med |
| F8 | `trend_analysis.analyze_session_trends` | EMA computed twice per metric | negligible | ✅ identical | Low |

Out-of-scope items that **would change output** (tuning, not simplification) are
listed separately at the end so they are not mistaken for free wins.

---

## F1 — Memoise the electron-temperature solve *(highest leverage)*

**Where:** [`solve_electron_temperature`](digital_twin/physics_engine.py:359), called
once inside every [`simulate`](digital_twin/physics_engine.py:745).

**Current approach.** Every `simulate(power, pressure)` runs a fresh `brentq`
root-find for Te, even when the pressure (and geometry) is one already solved
microseconds earlier. Because Te is 84% of the cost and depends only on
`(pressure, geometry)`, a fixed-pressure power sweep pays that 84% on every point
for an identical answer. Measured:

```
41-point fixed-pressure power sweep:          4.57 ms
same sweep with Te solve memoised:            0.83 ms   (-82%)
```

This shape is common in the code: `run_oat_sweep("rf_power_w", ...)`
([sensitivity_analysis.py:82](digital_twin/sensitivity_analysis.py:82)) holds
pressure fixed across all 15 points; `run_paired_sweep`'s inner loop
([sensitivity_analysis.py:188](digital_twin/sensitivity_analysis.py:188)) holds
pressure fixed across each row; `density_vs_power_validation_plot`
([physics_validation.py:402](digital_twin/physics_validation.py:402)) sweeps 40
powers at one pressure; and every module that re-simulates the same operating
envelope (anomaly, correlation, dataset) repeats pressures constantly.

**Proposed change.** Wrap the Te solve in a small pressure/geometry-keyed cache
(`functools.lru_cache` on a `(pressure_mtorr, geometry)` key, or an explicit dict).
`ChamberGeometry` is a frozen dataclass, so it is hashable and usable as a key
directly.

**Numerically identical?** **Yes, exactly.** `brentq` with fixed `xtol/rtol` on a
deterministic function returns the same root for the same inputs — memoising a pure
function returns the cached value bit-for-bit. Nothing downstream can tell the
difference.

**Caveats / honest scope.**
- The win only materialises when the *same* `(pressure, geometry)` recurs. It fully
  covers fixed-pressure sweeps and repeated envelope simulations. It does **not**
  help the dataset-generation *confounded* path (F2), where wall-temp drift makes
  `geometry.gas_temp_k` continuous, so every key is unique — that path is addressed
  by F2 instead.
- Float keys rely on the *same float value* recurring. In the sweeps above the
  pressure literal is reused each iteration, so hits are exact; this is not
  float-tolerance matching, just identity of a repeated value.
- If using `lru_cache`, set a `maxsize` (pressures are always a small grid, so the
  cache stays tiny) to avoid unbounded growth in a long-running dashboard.

**Effort:** Low (one wrapper, no call-site changes). Highest impact-per-line in the
review.

---

## F2 — Don't re-solve the physics the dataset generator already has

**Where:** [`generate_dataset`](digital_twin/dataset_generation.py:264-282).

**Current approach.** Per recipe the loop computes `nominal = simulate(power,
pressure)` once (line 267), then for each replicate calls
`measured = simulate(power, pressure, noise_level=…, seed=…)` (line 276). But
`simulate` with noise does exactly two things: (1) the **same deterministic solve**
as `nominal`, then (2) a post-hoc multiplicative noise pass
([`_apply_noise`](digital_twin/physics_engine.py:671), seeded by its own RNG). Step
(1) is identical to `nominal` every single time and is thrown away. Measured on the
default dataset:

```
generate_dataset default: 880 rows, 1870 physics solves
  nominal   110   (1 per recipe)
  measured  880   <- fully redundant deterministic solve, only noise differs
  confounded 880  (genuinely different: perturbed power & gas temp)
=> 1870 -> 990 solves  (-47%)
```

**Proposed change.** Reuse the already-computed `nominal` deterministic result and
apply only the noise step for each replicate, e.g. expose a small
`apply_measurement_noise(result, power, pressure, noise_level, rng)` (a thin public
wrapper over the existing `_apply_noise`) and call it with
`np.random.default_rng(noise_seed)`.

**Numerically identical?** **Yes, exactly**, and I checked the RNG bookkeeping
specifically because that is the only place it could drift:
- The deterministic base of `measured` is a pure function of `(power, pressure,
  DEFAULT_GEOMETRY)` — identical to `nominal`. ✅
- The noise is drawn from `simulate`'s *own* `np.random.default_rng(noise_seed)`, a
  separate stream. Re-applying `_apply_noise` with `default_rng(noise_seed)`
  reproduces the identical draws. ✅
- The main generation `rng` still draws `wall_temp → age → impurity → noise_seed` in
  the same order (we don't remove the `noise_seed` draw), so every subsequent
  confounder sample and the final label jitter are unchanged. ✅

The `confounded` solve (line 280) is **not** removable — it uses a perturbed power
and a different `gas_temp_k`, so it is a genuinely different physics point.

**Effort:** Low–Medium (expose one wrapper, adjust one call site). High value
because dataset generation is the single most-run expensive operation in the repo
(every classifier test and the dashboard's first load go through it).

---

## F3 — Share the bootstrap across the four suitability applications

**Where:** [`all_application_defect_estimates`](ai_module/suitability_analysis.py:271)
→ [`application_defect_estimate`](ai_module/suitability_analysis.py:221).

**Current approach.** `all_application_defect_estimates` calls
`application_defect_estimate` once per application; each call runs its own
`n_bootstrap` noisy `simulate` calls at the **same** `(power, pressure)`. Measured:

```
all_application_defect_estimates(150,10, n_bootstrap=80): 320 physics solves
  = 4 applications x 80  ->  but all 4 use the SAME seed and SAME (power,pressure)
```

Crucially, every call defaults to the same `seed`, so the four applications draw
the **identical** sequence of noisy simulations; only the deterministic
`deviation`/`risk_multiplier` (from each window's centre) differs afterward. The
320 simulations are 80 distinct results computed four times.

**Proposed change.** In `all_application_defect_estimates`, run the `n_bootstrap`
noisy simulations **once**, capture each result's `(ion_energy_ev,
defect_probability)`, and apply each application's window arithmetic to that shared
sample set. (Optionally combine with F1/F2: the deterministic solve is also
identical across all 80 bootstrap iterations, so this collapses to ~1 real solve +
80 cheap noise passes.)

**Numerically identical?** **Yes, exactly.** Same seed ⇒ same
`rng.integers` sequence ⇒ same per-iteration simulate seeds ⇒ byte-identical
`(ion_energy, defect)` samples for every application today; computing them once and
reusing them yields the same numbers. The per-application `deviation`,
`risk_multiplier`, median and quantiles are unchanged.

**Effort:** Low. This is the dashboard's Suitability expander
([5_Suitability_and_Recommendations.py:34](dashboard/pages/5_Suitability_and_Recommendations.py:34),
`n_bootstrap=80` → 320 solves per render), so it is a directly felt responsiveness
win.

---

## F4 — Anomaly detector recomputes physics predictions repeatedly

**Where:** [`PlasmaAnomalyDetector`](ai_module/anomaly_detection.py:221),
specifically `_predicted_outputs` / `relative_residuals` / `_residual_summary`.

Three overlapping redundancies, all in computing the twin's *predicted* outputs
(`_feature_row` → `simulate`), which are a pure function of `(power, pressure,
geometry)`:

**(a) `fit` computes the predictions twice.**
[`fit`](ai_module/anomaly_detection.py:263) calls `self.relative_residuals(normal_df)`
(line 266) for the std, then `self._residual_summary(normal_df)` (line 271), which
calls `_standardized_residuals` → `relative_residuals` **again**. Measured:

```
detector.fit on 1260 normal rows: 2520 physics solves   (= 1260 x 2)
```

**(b) Predictions aren't deduplicated by operating point.** The normal set has
`replicates=3`, so of those 1260 rows only **420 are unique** `(power, pressure)`
pairs; the predicted (noiseless) output is identical within a replicate group.

```
unique (power,pressure) pairs in the 1260-row normal set: 420
=> predictions needed: 420, not 1260 (or 2520)
```

**(c) The dashboard calls three methods that each recompute.** The Anomaly Monitor
page runs `detector.severity(x)`
([3_Anomaly_Monitor.py:42](dashboard/pages/3_Anomaly_Monitor.py:42)),
`detector.anomaly_score(x)` (line 43) and `detector.root_cause(x)` (line 44) on the
same row — each independently recomputing the residuals/predictions. Three physics
passes where one would do.

**Proposed change.** (a) In `fit`, compute `relative_residuals(normal_df)` once and
derive both the std and the summary from it. (b) In `_predicted_outputs`, compute
predictions on the **unique** `(power, pressure)` pairs and map back (a
`drop_duplicates` + merge, or a per-pair cache — F1's Te cache already removes most
of this cost for free). (c) Optionally expose one method returning
`(score, severity, root_cause)` together, or memoise per input, so the dashboard's
three calls share one physics pass.

**Numerically identical?** **Yes, exactly** for (a) and (b): predicted outputs are a
deterministic pure function of the logged inputs; computing them once, or once per
unique input, yields identical residuals, identical Isolation-Forest scores, and
identical thresholds. (c) is identical provided the same input rows are passed
(they are).

**Effort:** Medium (touches the residual pipeline), but (a) alone is a two-line win
and F1 already covers much of (b).

---

## F5 — Recommendation engine re-classifies the baseline for every candidate

**Where:** [`_evaluate_candidate`](ai_module/recommendation_engine.py:186), called
per candidate from [`recommend`](ai_module/recommendation_engine.py:267).

**Current approach.** For each candidate, `_evaluate_candidate` classifies **both**
the baseline (lines 209–211) and the candidate (lines 212–214). `class_before` is
the baseline's class — identical for every candidate — but it is recomputed each
time, and `classify_configuration` internally runs a `simulate` + `predict_proba`.
Measured candidate count for a typical voltage-biased baseline:

```
distinct candidates evaluated: 11
=> baseline classified 11 times (simulate + predict_proba each) when 1 would do
```

**Proposed change.** Classify the baseline once in `recommend` (it already computes
`baseline_result` once at line 260) and pass `class_before` into
`_evaluate_candidate`.

**Numerically identical?** **Yes, exactly.** `class_before` is a deterministic
function of the fixed baseline; computing it once versus eleven times gives the same
value, so `class_delta`, `score`, ranking and `prediction_text` are unchanged.

**Effort:** Low (hoist one computation up one level, add one parameter).

---

## F6 — `simulate` recomputes the geometry factors 4–5× per call

**Where:** [`_discharge_geometry_factors`](digital_twin/physics_engine.py:340),
invoked separately by `solve_electron_temperature`, `solve_plasma_density`,
`sheath_and_ion_energy` (and `child_langmuir_sheath` when RF voltage is supplied),
then again in `simulate` step 4. Measured:

```
single simulate(rf_voltage=None): _discharge_geometry_factors called 4 times
single simulate(rf_voltage=300):  _discharge_geometry_factors called 5 times
```

`n_g`, `h_l`, `h_R`, volume and area depend only on `(pressure, geometry)`; they are
recomputed 4–5× per `simulate`. Each is cheap versus the Te solve (a few `sqrt`),
so this is the smallest of the redundancy findings, but it is on the universal path.

**Proposed change.** Compute the factors once at the top of `simulate` and pass them
into the helpers (add an optional pre-computed-factors parameter, defaulting to
`None` → compute, so the helpers stay independently callable and their tests
unchanged).

**Numerically identical?** **Yes, exactly** — same pure function, same inputs,
computed once instead of five times.

**Effort:** Low–Medium (thread an optional argument through four helpers). Lower
priority than F1 since F1's Te cache already removes the dominant repeat; this only
saves the cheap 16% overhead.

---

## F7 — Calibration fits an extra model that most call paths never use

**Where:** [`train_classifiers`](ai_module/classification.py) — introduced by the
calibration change committed as `c3d0511`. **Flagging my own recent code honestly.**

**Current approach.** Each ensemble is now fit **twice**: once wrapped in
`CalibratedClassifierCV` (used by `predict`/`predict_proba`) and once as a plain
`explainer_model` (used only by `shap_values`, because `TreeExplainer` can't see
inside the calibration wrapper). But several hot paths never call SHAP:
`run_full_evaluation`, `run_cross_validation`, the McNemar evaluation, and
`train_and_log` all only `predict`/`evaluate`. For them the `explainer_model` fit is
pure overhead — roughly a 2× RF/XGB training cost in the CV loop, which retrains all
three per fold.

**Proposed change (options).**
1. **Lazy fit** — fit `explainer_model` on first `shap_values()` call. Trade-off:
   the classifier must retain the training arrays to fit later, costing memory.
2. **Opt-in** — a `with_explainer: bool = True` flag on `train_classifiers`; pass
   `False` from the SHAP-free paths (CV, McNemar). No memory cost, but call sites
   must know whether they'll need SHAP.

**Numerically identical?** **Yes** — the `explainer_model` is fit with identical
hyperparameters, data and seed whether eager or lazy, so SHAP values are unchanged;
predictions never used it. The only trade-off is compute-vs-memory (option 1) or a
small API flag (option 2), not any numerical difference.

**Effort:** Medium. **Note:** this is a genuine regression-in-the-making from the
calibration work — worth deciding on now while it's fresh, but it does not affect any
result, only training time on the eval/CV paths.

---

## F8 — Trend analysis smooths each metric twice

**Where:** [`analyze_session_trends`](ai_module/trend_analysis.py:248).

**Current approach.** Per metric it computes `smoothed = ema_smooth(...)` (line 256)
for event detection, then calls `trend_chart(..., smoothing_window=…)` (line 259),
which computes the **same** EMA again internally
([trend_analysis.py:221](ai_module/trend_analysis.py:221)).

**Proposed change.** Let `trend_chart` accept an already-smoothed series (optional
param), passing the one from line 256.

**Numerically identical?** **Yes** — `ewm(...).mean()` is deterministic; computing it
once vs twice is identical. Impact is negligible (EMA is cheap and session series are
short), included only for completeness of the "redundant computation" sweep.

---

## Considered and deliberately **excluded** (these would change output)

The brief was explicitly "without changing output or behaviour," so these are noted
only so they aren't confused with the free wins above — each **would** change results
and therefore needs a separate accuracy discussion, not a silent efficiency edit:

- **`IsolationForest(n_estimators=300)`** ([anomaly_detection.py:228](ai_module/anomaly_detection.py:228))
  on a 3-feature summary could likely use fewer trees, but any change alters the
  forest and thus the scores/recall. Out of scope.
- **Bootstrap `n_bootstrap=200`** ([suitability_analysis.py:226](ai_module/suitability_analysis.py:226))
  and **`n_estimators=200`** for RF/XGB are accuracy/stability knobs; lowering them
  changes numbers. (F3 makes the *existing* 200 far cheaper without touching it —
  that's the in-scope win.)
- **Replacing pandas with numpy** in the residual/summary paths would micro-optimise
  but risks subtle dtype/index differences; not worth the correctness risk for a
  student-defended codebase, and not a bottleneck next to the physics solve.

## No structural bloat found

Data structures are already lightweight and appropriate: `SimulationResult` and the
config objects are plain dataclasses; SQLite is stdlib `sqlite3` with a small fixed
schema (no ORM); Plotly figures are built without a running server; `torch` is
correctly quarantined to the optional autoencoder. There is no heavier-than-needed
dependency or container to remove — the efficiency story here is entirely about
**not repeating the physics solve**, which F1–F6 address.

---

## Suggested order, if we act on these

1. **F1** (Te memoisation) — one wrapper, biggest and broadest win, zero risk.
2. **F2** (dataset measured-reuse) — halves the most-run expensive operation.
3. **F3** (share suitability bootstrap) — direct dashboard responsiveness.
4. **F4a** (fix the double `relative_residuals` in `fit`) — two-line win.
5. **F5** (hoist baseline classification) — one-line hoist.
6. **F7** (decide lazy vs opt-in explainer) — prevents the calibration change from
   quietly doubling CV training time.
7. **F6 / F4b / F8** — smaller cleanups, do if we're touching those files anyway.

Every one of F1–F8 is verifiable by the existing pytest suite: because they are all
numerically identical, **all 363 tests must still pass unchanged** after each — which
is exactly the regression guard we'd want before trusting any of them.
