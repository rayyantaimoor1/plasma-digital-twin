"""Sub-Module 1.3 - Synthetic Plasma Dataset Generation and Management.

Generates the labelled dataset the AI Module (Phase 2) trains on. Its defining
feature - CLAUDE.md non-negotiable principle #2 - is that the label depends on
HIDDEN CONFOUNDERS that are deliberately withheld from the classifier's features,
so that the visible parameters do not fully determine the outcome. Without this,
the label would be a near-deterministic function of the inputs and the reported
classification accuracy (BO-2) would be meaningless.

================================================================================
CAUSAL STRUCTURE (why this is a genuine prediction problem, not function inversion)
================================================================================

  Recipe (VISIBLE, operator-set):        rf_power_w, pressure_mtorr
        |                                         |
        |  physics engine at the recipe           |  physics engine at the
        |  (nominal, no confounders)              |  CONFOUNDER-PERTURBED condition
        v                                         v
  Observable features  <-- measurement      True process quality  <-- gas-purity
  (Te, n_e, reactivity,     noise                 (confounded)         penalty
   uniformity, etch)                                    |
        |                                               v
        |                                     soft quartile labelling
        v                                               |
  FEATURES given to classifier            <----/         v
                                                      LABEL (suitability)

  Hidden confounders (NOT features):  wall_temp_drift, electrode_age, gas_purity

The observable features are computed from the recipe at NOMINAL conditions (plus
measurement noise), so the confounders never leak into the features - they are
genuinely hidden. The label is computed from the CONFOUNDED true quality. The gap
between "what the recipe predicts" and "what actually happened" is irreducible,
confounder-driven error the classifier cannot eliminate. Replicates of the same
recipe therefore carry DIFFERENT labels - the clearest possible demonstration that
the visible features do not determine the outcome.

================================================================================
CONFOUNDER STRENGTHS AND NOISE LEVELS - FIXED A PRIORI FROM PHYSICAL REASONING
================================================================================
Per FE-1.3.7, these are chosen from physics BEFORE any classifier is trained, and
are NOT to be tuned afterward to hit an accuracy target. Each value below is
justified physically, not by its effect on model performance. A fixed random seed
(DEFAULT_SEED) is recorded for every generation run so the dataset is reproducible.

1. WALL-TEMPERATURE DRIFT  ->  routed through neutral gas temperature (real physics)
   Physical reasoning: RF power dissipated in the discharge heats the electrodes
   and chamber walls; over a run, and between runs without full thermal
   re-equilibration, the effective neutral gas temperature drifts above the 300 K
   baseline. Measured wall/gas temperatures in lab CCPs commonly reach 350-450 K
   under sustained power. We model the drift as added Kelvin on the gas temperature,
   which enters the physics rigorously via n_g = p / (k_B * T_g): a hotter wall
   lowers the neutral density, shifting the whole global-model operating point. We
   route the confounder through this genuine physical channel rather than inventing
   an arbitrary penalty; its job is to inject real, physics-based unexplained
   variance, not to assert a sign of "better/worse."
     Distribution: Normal(mean = +40 K, sd = 25 K), clipped to [0, 150] K
     (walls heat, never cool below ambient; +150 K is a plausible hot-run bound).
     Resulting effect: ~4-12% shift in process quality across the drift range.

2. ELECTRODE AGING  ->  routed through effective absorbed power (real physics)
   Physical reasoning: over hundreds of hours, electrodes erode (sputtering) and
   accumulate deposits, degrading RF coupling efficiency. A 10-20% coupling loss is
   realistic for an aged electrode still in service before scheduled maintenance
   (beyond ~20% it would typically be flagged and replaced). We model age in [0, 1]
   (a fleet of chambers at random points in their maintenance cycle) reducing the
   effective absorbed power: P_eff = P * (1 - 0.20 * age). Lower absorbed power
   lowers n_e (power balance) and hence reactivity and quality.
     Distribution: Uniform[0, 1].  Max coupling loss: 20%.
     Resulting effect: up to ~-9% process quality at full aging.

3. GAS-PURITY VARIANCE  ->  direct quality penalty (genuinely outside argon physics)
   Physical reasoning: process argon is nominally 99.999%, but real base pressure,
   small leaks, and outgassing introduce O2 / N2 / H2O impurities at the 0.1-1%
   level. Electronegative impurities (O2, H2O) attach electrons and seed
   particulates, degrading etch/deposition quality in ways an argon-ONLY 0D model
   cannot represent - so this confounder is applied as a documented external quality
   penalty rather than through the (argon-only) engine. This is the honest choice:
   the effect is real but lies outside the engine's chemistry.
     Distribution: impurity_fraction ~ HalfNormal(sd = 0.004), clipped to [0, 0.03].
     Penalty: true_quality *= (1 - 10 * impurity_fraction)  (a 1% impurity costs
     ~10% quality; up to ~30% at the 3% worst-contamination bound). The 10x
     sensitivity reflects how strongly electronegative contamination degrades a
     sensitive plasma process.

MEASUREMENT NOISE on the observable features: 5% relative, applied via the physics
engine's heteroscedastic noise model (FE-1.2.5), representing realistic
Langmuir-probe / OES diagnostic precision. Separate from, and additional to, the
confounder-driven label uncertainty.

LABEL SOFTNESS (FE-1.3.3): the four suitability classes are the quartiles of the
true (confounded) process-quality distribution - a fixed, pre-registered labelling
rule. Boundaries are made SOFT by adding small Gaussian jitter (sd = 0.02 on the
0-1 quality scale) to the quality before thresholding, so samples near a class
boundary are assigned probabilistically rather than by a razor-sharp cut.

None of the numbers above were chosen by looking at classifier accuracy. If the
resulting accuracy is unimpressive, that is reported honestly (BO-2).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from digital_twin.physics_engine import ChamberGeometry, simulate
from digital_twin.session_manager import DEFAULT_DB_PATH

# ---------------------------------------------------------------------------
# Reproducibility: one fixed, recorded seed (FE-1.3.7).
# ---------------------------------------------------------------------------
DEFAULT_SEED = 20260710  # recorded generation seed; change only with a new dataset id

# ---------------------------------------------------------------------------
# Confounder strengths - see the module docstring for the physical justification
# of every number here. Grouped into a frozen config so the exact values used are
# stored in the dataset metadata (FE-1.3.6) and can never silently drift.
# ---------------------------------------------------------------------------
BASE_GAS_TEMP_K = 300.0


@dataclass(frozen=True)
class ConfounderConfig:
    """A priori confounder distribution parameters (FE-1.3.7). Frozen and recorded."""
    wall_temp_drift_mean_k: float = 40.0
    wall_temp_drift_sd_k: float = 25.0
    wall_temp_drift_max_k: float = 150.0

    electrode_aging_max_coupling_loss: float = 0.20

    gas_impurity_sd: float = 0.004
    gas_impurity_max: float = 0.03
    gas_impurity_quality_sensitivity: float = 10.0

    feature_measurement_noise: float = 0.05
    label_jitter_sd: float = 0.02


DEFAULT_CONFOUNDER_CONFIG = ConfounderConfig()

# ---------------------------------------------------------------------------
# Column contract. The classifier (Sub-Module 2.1) MUST train on FEATURE_COLUMNS
# only and MUST NOT touch CONFOUNDER_COLUMNS or the quality columns - those are
# recorded purely for transparency, analysis, and physics validation (1.6).
# process_quality / defect_probability are excluded from features because the label
# is derived from them (including them would be label leakage).
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "rf_power_w",
    "pressure_mtorr",
    "electron_temperature_ev",
    "plasma_density_m3",
    "reactivity_index",
    "uniformity_index",
    "etch_rate_nm_min",
]
LABEL_COLUMN = "suitability"
CONFOUNDER_COLUMNS = ["wall_temp_drift_k", "electrode_age", "gas_purity"]
ANALYSIS_COLUMNS = ["nominal_process_quality", "true_process_quality"] + CONFOUNDER_COLUMNS

# Suitability classes, ascending in quality; index = quartile bucket 0..3.
SUITABILITY_CLASSES = ["Unsuitable", "Marginal", "Acceptable", "Optimal"]

# Region-based split boundary (FE-1.3.5): the held-out high-power / high-pressure
# corner used to test EXTRAPOLATION to an unseen operating region.
POWER_EXTRAPOLATION_THRESHOLD_W = 250.0
PRESSURE_EXTRAPOLATION_THRESHOLD_MTORR = 17.0


# ---------------------------------------------------------------------------
# Confounder sampling (all draws come from a single seeded RNG stream)
# ---------------------------------------------------------------------------
def _sample_wall_temp_drift(rng: np.random.Generator, cfg: ConfounderConfig) -> float:
    drift = rng.normal(cfg.wall_temp_drift_mean_k, cfg.wall_temp_drift_sd_k)
    return float(np.clip(drift, 0.0, cfg.wall_temp_drift_max_k))


def _sample_electrode_age(rng: np.random.Generator) -> float:
    return float(rng.uniform(0.0, 1.0))


def _sample_gas_impurity(rng: np.random.Generator, cfg: ConfounderConfig) -> float:
    # Half-normal: |N(0, sd)| gives mostly-pure argon with an occasional dirty run.
    impurity = abs(rng.normal(0.0, cfg.gas_impurity_sd))
    return float(np.clip(impurity, 0.0, cfg.gas_impurity_max))


def _true_quality_with_confounders(
    rf_power_w: float,
    pressure_mtorr: float,
    wall_temp_drift_k: float,
    electrode_age: float,
    gas_impurity: float,
    cfg: ConfounderConfig,
) -> float:
    """Process quality at the confounder-perturbed condition (the LABEL basis).

    Wall temperature and electrode aging act through the real physics (gas
    temperature and effective absorbed power); gas impurity is a documented
    external penalty (outside argon-only chemistry). See module docstring.
    """
    effective_gas_temp = BASE_GAS_TEMP_K + wall_temp_drift_k
    effective_power = rf_power_w * (1.0 - cfg.electrode_aging_max_coupling_loss * electrode_age)
    geometry = ChamberGeometry(gas_temp_k=effective_gas_temp)

    confounded = simulate(effective_power, pressure_mtorr, geometry=geometry)
    purity_penalty = 1.0 - cfg.gas_impurity_quality_sensitivity * gas_impurity
    return float(np.clip(confounded.process_quality * max(0.0, purity_penalty), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Dataset generation (FE-1.3.1, FE-1.3.2, FE-1.3.3)
# ---------------------------------------------------------------------------
def _default_recipe_grid(
    power_step_w: float, pressure_step_mtorr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Systematic parameter sweep with user-defined step sizes (FE-1.3.1)."""
    powers = np.arange(50.0, 300.0 + 1e-9, power_step_w)
    pressures = np.arange(1.0, 20.0 + 1e-9, pressure_step_mtorr)
    return powers, pressures


def generate_dataset(
    power_step_w: float = 25.0,
    pressure_step_mtorr: float = 2.0,
    replicates_per_recipe: int = 8,
    confounder_config: ConfounderConfig = DEFAULT_CONFOUNDER_CONFIG,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate the labelled synthetic dataset.

    A grid of (power, pressure) recipes is swept (FE-1.3.1); each recipe is
    replicated `replicates_per_recipe` times, and every replicate draws its own
    hidden confounders (FE-1.3.2). This is what makes identical recipes carry
    different labels. Observable features are the recipe's NOMINAL physics with
    measurement noise; the label is the soft-quartile suitability of the CONFOUNDED
    true quality (FE-1.3.3).

    Returns a DataFrame with FEATURE_COLUMNS + [LABEL_COLUMN] + ANALYSIS_COLUMNS.
    """
    rng = np.random.default_rng(seed)
    cfg = confounder_config
    powers, pressures = _default_recipe_grid(power_step_w, pressure_step_mtorr)

    rows: list[dict[str, float]] = []
    for power in powers:
        for pressure in pressures:
            # Nominal physics for this recipe (deterministic; shared by replicates).
            nominal = simulate(float(power), float(pressure))
            for _ in range(replicates_per_recipe):
                wall_temp = _sample_wall_temp_drift(rng, cfg)
                age = _sample_electrode_age(rng)
                impurity = _sample_gas_impurity(rng, cfg)

                # Observable features: nominal physics + heteroscedastic measurement
                # noise (engine's own noise model). Recipe columns stay exact.
                noise_seed = int(rng.integers(0, 2**32 - 1))
                measured = simulate(
                    float(power), float(pressure),
                    noise_level=cfg.feature_measurement_noise, seed=noise_seed,
                )
                true_quality = _true_quality_with_confounders(
                    float(power), float(pressure), wall_temp, age, impurity, cfg
                )
                rows.append({
                    "rf_power_w": float(power),
                    "pressure_mtorr": float(pressure),
                    "electron_temperature_ev": measured.electron_temperature_ev,
                    "plasma_density_m3": measured.plasma_density_m3,
                    "reactivity_index": measured.reactivity_index,
                    "uniformity_index": measured.uniformity_index,
                    "etch_rate_nm_min": measured.etch_rate_nm_min,
                    "nominal_process_quality": nominal.process_quality,
                    "true_process_quality": true_quality,
                    "wall_temp_drift_k": wall_temp,
                    "electrode_age": age,
                    "gas_purity": 1.0 - impurity,
                })

    frame = pd.DataFrame(rows)
    frame[LABEL_COLUMN] = _soft_quartile_labels(
        frame["true_process_quality"].to_numpy(), rng, cfg.label_jitter_sd
    )
    # Order columns: features, label, then analysis/ground-truth columns.
    return frame[FEATURE_COLUMNS + [LABEL_COLUMN] + ANALYSIS_COLUMNS]


def _soft_quartile_labels(
    true_quality: np.ndarray, rng: np.random.Generator, jitter_sd: float
) -> list[str]:
    """Assign suitability classes as soft quartiles of the true-quality distribution.

    The class boundaries are the 25/50/75th percentiles of the true (confounded)
    quality - a fixed, pre-registered rule that gives ~balanced classes and is
    clearly not tuned to any classifier. Boundaries are SOFTENED (FE-1.3.3) by
    adding Gaussian jitter to the quality before thresholding, so a sample near a
    boundary lands in one class or the neighbour probabilistically.
    """
    thresholds = np.quantile(true_quality, [0.25, 0.50, 0.75])
    jittered = true_quality + rng.normal(0.0, jitter_sd, size=true_quality.shape)
    buckets = np.digitize(jittered, thresholds)  # 0..3
    return [SUITABILITY_CLASSES[b] for b in buckets]


def features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into (X, y) for the AI Module, enforcing the feature/confounder contract."""
    return frame[FEATURE_COLUMNS].copy(), frame[LABEL_COLUMN].copy()


# ---------------------------------------------------------------------------
# Train/test splits (FE-1.3.5)
# ---------------------------------------------------------------------------
def region_based_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by parameter-space region to test EXTRAPOLATION (FE-1.3.5).

    The test set is the held-out high-power / high-pressure corner the model never
    trains on; the train set is the rest. This measures generalisation to an unseen
    operating region, which a random split (interpolation) cannot reveal.
    """
    in_test_region = (
        (frame["rf_power_w"] > POWER_EXTRAPOLATION_THRESHOLD_W)
        | (frame["pressure_mtorr"] > PRESSURE_EXTRAPOLATION_THRESHOLD_MTORR)
    )
    train = frame[~in_test_region].reset_index(drop=True)
    test = frame[in_test_region].reset_index(drop=True)
    return train, test


def random_split(
    frame: pd.DataFrame, test_fraction: float = 0.2, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard shuffled split to test INTERPOLATION within seen regions (FE-1.3.5)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    n_test = int(round(len(frame) * test_fraction))
    test_idx, train_idx = order[:n_test], order[n_test:]
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# Export and persistence (FE-1.3.4, FE-1.3.6)
# ---------------------------------------------------------------------------
def export_csv(frame: pd.DataFrame, path: Path | str) -> Path:
    """Export the full dataset (all columns) to CSV for offline analysis (FE-1.3.4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


_DATASETS_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    config_json TEXT NOT NULL
);
"""


def store_dataset_to_db(
    frame: pd.DataFrame,
    confounder_config: ConfounderConfig = DEFAULT_CONFOUNDER_CONFIG,
    seed: int = DEFAULT_SEED,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Persist the dataset + its generation metadata to SQLite (FE-1.3.6).

    Metadata (seed, timestamp, full confounder config) goes in `datasets`; the rows
    go in `dataset_rows` tagged with the dataset_id, so the exact data-generating
    configuration is auditable alongside the data itself.
    """
    dataset_id = str(uuid.uuid4())
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_DATASETS_METADATA_SCHEMA)
        conn.execute(
            "INSERT INTO datasets (dataset_id, created_at, n_samples, seed, config_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                dataset_id,
                datetime.now(timezone.utc).isoformat(),
                len(frame),
                seed,
                json.dumps(asdict(confounder_config)),
            ),
        )
        stored = frame.copy()
        stored.insert(0, "dataset_id", dataset_id)
        stored.to_sql("dataset_rows", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()
    return dataset_id


def load_dataset_from_db(dataset_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Reload a stored dataset by id (rows only, without the dataset_id tag column)."""
    conn = sqlite3.connect(db_path)
    try:
        frame = pd.read_sql_query(
            "SELECT * FROM dataset_rows WHERE dataset_id = ?", conn, params=(dataset_id,)
        )
    finally:
        conn.close()
    if frame.empty:
        raise KeyError(f"No dataset stored with id {dataset_id!r}")
    return frame.drop(columns=["dataset_id"])


if __name__ == "__main__":
    # Standalone demo: generate a small dataset and show the confounding in action.
    # Run as a module (cross-package import):  python -m digital_twin.dataset_generation
    df = generate_dataset(power_step_w=50.0, pressure_step_mtorr=5.0, replicates_per_recipe=6)
    print(f"Generated {len(df)} samples, seed={DEFAULT_SEED}\n")
    print("Label distribution:")
    print(df[LABEL_COLUMN].value_counts(), "\n")
    # Same recipe, different labels => visible features do not determine the outcome.
    grp = df.groupby(["rf_power_w", "pressure_mtorr"])[LABEL_COLUMN].nunique()
    print(f"Recipes with >1 distinct label (confounding): {int((grp > 1).sum())} / {len(grp)}")
