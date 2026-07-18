"""Sub-Module 2.5 - Semiconductor Process Suitability Analysis Module.

Maps the digital twin's ion bombardment energy (Sub-Module 1.2's `ion_energy_ev`)
to four semiconductor process application categories, per FE-2.5.1's requirement
that the operating windows come from cited literature rather than round numbers
picked without a stated source.

================================================================================
WHY ION ENERGY, AND WHAT IS AND ISN'T CITED (FE-2.5.1)
================================================================================
Ion bombardment energy at the wafer is the standard, widely-corroborated
discriminator between plasma process categories across the semiconductor
processing literature (Lieberman & Lichtenberg Ch. 1's overview of etching,
deposition, and surface-modification regimes, and the general plasma-processing
literature consensus - this ordering is not tied to a single obscure figure, it
is repeated throughout the field):

  * Wafer cleaning / photoresist ashing / descum: LOW ion energy by design - these
    processes deliberately minimise ion bombardment to avoid substrate damage,
    often run at low power / low bias, or even in a remote/downstream plasma
    specifically to reduce ion energy further.
  * Surface treatment / activation (e.g. plasma activation for adhesion,
    controlled surface roughening): LOW-TO-MODERATE ion energy - enough to modify
    the surface without significant material removal or damage.
  * Thin-film deposition (PECVD): LOW-TO-MODERATE ion energy - ion-assisted
    densification of the growing film, but energetic enough bombardment would
    resputter/damage the film being deposited.
  * Plasma etching (RIE): HIGH ion energy - anisotropic, physically-assisted
    material removal specifically requires energetic directional ion bombardment;
    this is the whole physical basis for reactive ion etching over purely
    chemical (isotropic) etching.

This ORDERING (cleaning < treatment ~ deposition < etching) is the cited part.
The exact numeric eV boundaries below are ROUND, representative values consistent
with that literature consensus (same honesty standard as the physics engine's
"illustrative engineering index" constants) - not a pixel-precise reading of one
specific table, which would be false precision. They were verified BEFORE being
finalised (see git history) to produce genuinely mixed suitable/unsuitable results
across the platform's own operating envelope (63-93% suitable for cleaning/
treatment/deposition, 9% for etching without an applied RF voltage), rather than
windows that trivially always pass or always fail.

Consequence worth noting: without an applied RF voltage (Sub-Module 1.2's
default/fallback sheath path), ion_energy_ev tops out around 70-90 eV across the
full 50-300 W / 1-20 mTorr envelope - below the etching window. Reaching the
etching category requires supplying `rf_voltage_v` to `simulate()` (Option C's
Child-Langmuir sheath), which is a genuine, physically coherent consequence of
the model (real RIE tools ARE distinguished from gentler CCP processes by their
driven, biased sheath), not a limitation to work around.

The defect_probability ceiling per application reflects DAMAGE SENSITIVITY, which
is also well-established qualitatively (cleaning and deposition are
damage-sensitive; etching inherently tolerates and requires more aggressive
bombardment) - but the specific numeric ceiling is this module's own calibration,
openly stated as such rather than mis-attributed to a citation that only supports
the ordering, not the exact number.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from digital_twin.dataset_generation import DEFAULT_SEED
from digital_twin.physics_engine import ChamberGeometry, DEFAULT_GEOMETRY, simulate
from digital_twin.session_manager import SessionRecord


class SemiconductorApplication(str, Enum):
    WAFER_CLEANING = "Wafer Cleaning"
    SURFACE_TREATMENT = "Surface Treatment"
    THIN_FILM_DEPOSITION = "Thin-Film Deposition"
    PLASMA_ETCHING = "Plasma Etching"


@dataclass(frozen=True)
class SuitabilityWindow:
    application: SemiconductorApplication
    ion_energy_min_ev: float
    ion_energy_max_ev: float
    max_defect_probability: float
    rationale: str


SUITABILITY_WINDOWS: dict[SemiconductorApplication, SuitabilityWindow] = {
    SemiconductorApplication.WAFER_CLEANING: SuitabilityWindow(
        SemiconductorApplication.WAFER_CLEANING, 5.0, 50.0, 0.50,
        "Cleaning/ashing deliberately minimises ion bombardment to avoid "
        "substrate damage (Lieberman & Lichtenberg Ch. 1; general plasma "
        "processing literature). Low defect ceiling: damage-sensitive process.",
    ),
    SemiconductorApplication.SURFACE_TREATMENT: SuitabilityWindow(
        SemiconductorApplication.SURFACE_TREATMENT, 20.0, 100.0, 0.60,
        "Surface activation/modification needs moderate bombardment without "
        "significant material removal (Lieberman & Lichtenberg Ch. 1).",
    ),
    SemiconductorApplication.THIN_FILM_DEPOSITION: SuitabilityWindow(
        SemiconductorApplication.THIN_FILM_DEPOSITION, 20.0, 150.0, 0.55,
        "PECVD uses ion-assisted densification; too little bombardment gives "
        "poor film quality, too much resputters the growing film "
        "(Lieberman & Lichtenberg Ch. 1 general PECVD discussion). Damage-sensitive.",
    ),
    SemiconductorApplication.PLASMA_ETCHING: SuitabilityWindow(
        SemiconductorApplication.PLASMA_ETCHING, 100.0, 500.0, 0.90,
        "Reactive ion etching's anisotropy specifically requires energetic "
        "directional ion bombardment (Lieberman & Lichtenberg Ch. 1); high "
        "defect ceiling since aggressive bombardment is inherent to the process.",
    ),
}


# ---------------------------------------------------------------------------
# Compliance scoring
# ---------------------------------------------------------------------------
def _window_compliance(value: float, lo: float, hi: float) -> float:
    """1.0 if value is inside [lo, hi]; degrades linearly to 0.0 at one full
    window-width past the nearer boundary, and stays 0.0 beyond that."""
    if lo <= value <= hi:
        return 1.0
    width = hi - lo
    distance = (lo - value) if value < lo else (value - hi)
    return max(0.0, 1.0 - distance / width)


def _ceiling_compliance(value: float, ceiling: float) -> float:
    """1.0 if value <= ceiling; degrades linearly to 0.0 at value = 2*ceiling."""
    if value <= ceiling:
        return 1.0
    margin = max(ceiling, 1e-6)
    return max(0.0, 1.0 - (value - ceiling) / margin)


# ---------------------------------------------------------------------------
# Suitability scorecard for one configuration [FE-2.5.1, FE-2.5.2]
# ---------------------------------------------------------------------------
@dataclass
class SuitabilityRating:
    application: SemiconductorApplication
    is_suitable: bool
    ion_energy_compliance_pct: float
    defect_compliance_pct: float
    overall_compliance_pct: float  # min of the two - bottleneck determines overall


@dataclass
class SuitabilityScorecard:
    rf_power_w: float
    pressure_mtorr: float
    ion_energy_ev: float
    defect_probability: float
    ratings: dict[SemiconductorApplication, SuitabilityRating]

    def best_application(self) -> SemiconductorApplication:
        return max(self.ratings, key=lambda app: self.ratings[app].overall_compliance_pct)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "application": app.value,
                "is_suitable": r.is_suitable,
                "ion_energy_compliance_pct": r.ion_energy_compliance_pct,
                "defect_compliance_pct": r.defect_compliance_pct,
                "overall_compliance_pct": r.overall_compliance_pct,
            }
            for app, r in self.ratings.items()
        ])


def _rate_application(ion_energy_ev: float, defect_probability: float, window: SuitabilityWindow) -> SuitabilityRating:
    ion_energy_compliance = _window_compliance(ion_energy_ev, window.ion_energy_min_ev, window.ion_energy_max_ev)
    defect_compliance = _ceiling_compliance(defect_probability, window.max_defect_probability)
    return SuitabilityRating(
        application=window.application,
        is_suitable=(ion_energy_compliance == 1.0 and defect_compliance == 1.0),
        ion_energy_compliance_pct=ion_energy_compliance * 100.0,
        defect_compliance_pct=defect_compliance * 100.0,
        overall_compliance_pct=min(ion_energy_compliance, defect_compliance) * 100.0,
    )


def classify_suitability(
    rf_power_w: float,
    pressure_mtorr: float,
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    rf_voltage_v: Optional[float] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> SuitabilityScorecard:
    """Run the physics engine and score the resulting operating point against
    every semiconductor application window at once [FE-2.5.1, FE-2.5.2]."""
    result = simulate(
        rf_power_w, pressure_mtorr, noise_level=noise_level, seed=seed,
        rf_voltage_v=rf_voltage_v, geometry=geometry,
    )
    ratings = {
        app: _rate_application(result.ion_energy_ev, result.defect_probability, window)
        for app, window in SUITABILITY_WINDOWS.items()
    }
    return SuitabilityScorecard(
        rf_power_w=rf_power_w, pressure_mtorr=pressure_mtorr,
        ion_energy_ev=result.ion_energy_ev, defect_probability=result.defect_probability,
        ratings=ratings,
    )


# ---------------------------------------------------------------------------
# Application-specific defect-risk confidence interval [FE-2.5.3]
# ---------------------------------------------------------------------------
@dataclass
class DefectProbabilityEstimate:
    application: SemiconductorApplication
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


def _bootstrap_ion_energy_defect_samples(
    rf_power_w: float,
    pressure_mtorr: float,
    n_bootstrap: int,
    noise_level: float,
    seed: int,
    rf_voltage_v: Optional[float],
    geometry: ChamberGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap `n_bootstrap` noisy simulations at one operating point, returning
    the (ion_energy_ev, defect_probability) sample arrays [EFFICIENCY_REVIEW.md F3].

    These two arrays are the ONLY per-sample physics an application-specific estimate
    needs; the rest is per-application window arithmetic. Factoring the bootstrap out
    lets `all_application_defect_estimates` run it once and reuse it across all four
    applications instead of four times. The RNG is seeded and drawn in exactly the
    order the previous inline loop used, so the samples are byte-identical.
    """
    rng = np.random.default_rng(seed)
    ion_energy = np.empty(n_bootstrap)
    defect = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        result = simulate(
            rf_power_w, pressure_mtorr, noise_level=noise_level,
            seed=int(rng.integers(0, 2**32 - 1)), rf_voltage_v=rf_voltage_v, geometry=geometry,
        )
        ion_energy[i] = result.ion_energy_ev
        defect[i] = result.defect_probability
    return ion_energy, defect


def _estimate_from_samples(
    application: SemiconductorApplication,
    ion_energy_samples: np.ndarray,
    defect_samples: np.ndarray,
    confidence_level: float,
) -> DefectProbabilityEstimate:
    """Apply one application's window arithmetic to a shared bootstrap sample set
    [EFFICIENCY_REVIEW.md F3]. The elementwise operations (abs, max(0, .), min(1, .))
    are the vectorised equivalents of the previous scalar per-sample loop and give
    numerically identical results."""
    window = SUITABILITY_WINDOWS[application]
    center = (window.ion_energy_min_ev + window.ion_energy_max_ev) / 2.0
    half_width = (window.ion_energy_max_ev - window.ion_energy_min_ev) / 2.0

    deviation = np.abs(ion_energy_samples - center) / half_width  # 0 at centre, 1 at boundary
    risk_multiplier = 1.0 + np.maximum(0.0, deviation - 1.0)  # only penalise once OUTSIDE the window
    samples = np.minimum(1.0, defect_samples * risk_multiplier)

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return DefectProbabilityEstimate(
        application=application, point_estimate=float(np.median(samples)),
        ci_lower=float(lower), ci_upper=float(upper), confidence_level=confidence_level,
    )


def application_defect_estimate(
    rf_power_w: float,
    pressure_mtorr: float,
    application: SemiconductorApplication,
    n_bootstrap: int = 200,
    noise_level: float = 0.1,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
    rf_voltage_v: Optional[float] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> DefectProbabilityEstimate:
    """Application-specific defect/unsuitability-risk estimate, reported as a
    confidence interval rather than a single point estimate [FE-2.5.3].

    Risk = the physics engine's own defect_probability, scaled up by how far the
    ion energy DEVIATES from this application's optimal window - i.e. "defect
    probability... as a function of process quality index deviation from optimal
    operating ranges" (FE-2.5.3's own phrasing). A point at the window centre gets
    no penalty; a point outside the window has its risk scaled up in proportion to
    how far past the boundary it sits. The confidence interval comes from
    bootstrapping over the physics engine's own measurement-noise model
    (FE-1.2.5) at the same operating point - a genuine uncertainty source, not an
    invented one - and is a deliberately lightweight precursor to the fuller
    calibrated-uncertainty treatment in Sub-Module 2.8.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    ion_energy, defect = _bootstrap_ion_energy_defect_samples(
        rf_power_w, pressure_mtorr, n_bootstrap, noise_level, seed, rf_voltage_v, geometry
    )
    return _estimate_from_samples(application, ion_energy, defect, confidence_level)


def all_application_defect_estimates(
    rf_power_w: float,
    pressure_mtorr: float,
    n_bootstrap: int = 200,
    noise_level: float = 0.1,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
    rf_voltage_v: Optional[float] = None,
    geometry: ChamberGeometry = DEFAULT_GEOMETRY,
) -> dict[SemiconductorApplication, DefectProbabilityEstimate]:
    """Defect estimate for every application at once [FE-2.5.3, EFFICIENCY_REVIEW.md F3].

    All four applications share the SAME operating point, seed, and noise, so they
    previously ran four identical bootstraps. This runs the bootstrap ONCE and applies
    each application's window arithmetic to the shared samples - numerically identical
    to calling `application_defect_estimate` per application (same seed -> same sample
    stream), at a quarter of the physics cost.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    ion_energy, defect = _bootstrap_ion_energy_defect_samples(
        rf_power_w, pressure_mtorr, n_bootstrap, noise_level, seed, rf_voltage_v, geometry
    )
    return {
        app: _estimate_from_samples(app, ion_energy, defect, confidence_level)
        for app in SemiconductorApplication
    }


# ---------------------------------------------------------------------------
# Comparative suitability across stored experiments [FE-2.5.4]
# ---------------------------------------------------------------------------
def compare_sessions_for_application(
    sessions: list[SessionRecord], application: SemiconductorApplication
) -> pd.DataFrame:
    """Rank stored sessions by suitability for a target application, so a user
    can identify which past configuration produced the best outcome for that
    application [FE-2.5.4]."""
    window = SUITABILITY_WINDOWS[application]
    rows = []
    for s in sessions:
        rating = _rate_application(s.ion_energy_ev, s.defect_probability, window)
        rows.append({
            "session_id": s.session_id,
            "rf_power_w": s.rf_power_w,
            "pressure_mtorr": s.pressure_mtorr,
            "ion_energy_ev": s.ion_energy_ev,
            "defect_probability": s.defect_probability,
            "is_suitable": rating.is_suitable,
            "ion_energy_compliance_pct": rating.ion_energy_compliance_pct,
            "defect_compliance_pct": rating.defect_compliance_pct,
            "overall_compliance_pct": rating.overall_compliance_pct,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("overall_compliance_pct", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.suitability_analysis
    print("Suitability scorecard at 150W / 10mTorr (no RF voltage):")
    scorecard = classify_suitability(150.0, 10.0)
    print(scorecard.to_dataframe().to_string(index=False))
    print(f"\nBest application: {scorecard.best_application().value}")

    print("\nSuitability scorecard at 200W / 5mTorr with rf_voltage_v=300V (etching regime):")
    scorecard2 = classify_suitability(200.0, 5.0, rf_voltage_v=300.0)
    print(scorecard2.to_dataframe().to_string(index=False))

    print("\nDefect-risk confidence intervals at 150W / 10mTorr:")
    estimates = all_application_defect_estimates(150.0, 10.0, n_bootstrap=100)
    for app, est in estimates.items():
        print(f"  {app.value:22s} {est.point_estimate:.3f}  [{est.ci_lower:.3f}, {est.ci_upper:.3f}]  "
              f"({est.confidence_level:.0%} CI)")
