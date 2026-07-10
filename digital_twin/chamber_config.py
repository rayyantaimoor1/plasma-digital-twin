"""Sub-Module 1.1 - Virtual Plasma Chamber and Parameter Configuration Engine.

Defines the user-settable inputs to the digital twin, their validated operating
ranges, named "experiment mode" presets, physically-grounded validation, and a
reset-to-default function.

This module is deliberately UI-agnostic: FE-1.1.1 calls for "sliders, numeric
inputs, and preset mode selectors," but building the actual widgets is a Streamlit
concern that belongs to Phase 3. What belongs here, now, is the DATA a future
Streamlit UI would be built from - `ParameterRange` carries exactly the
(min, max, default, step, unit) a slider needs - so the dashboard can wire itself
up later without this module importing Streamlit or knowing anything about it.

Electron temperature and plasma density are deliberately NOT settable fields
anywhere in this module: per the FYP scope revision, they are solved outputs of
the physics engine (Sub-Module 1.2), never free inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from digital_twin.physics_engine import simulate

# ---------------------------------------------------------------------------
# Operating envelope (FYP Scope Document, Section 6). Gas flow rate and exposure
# time are the "optional" inputs FE-1.1.1 mentions; the physics engine (1.2) does
# not currently consume them (argon composition and exposure duration do not enter
# the steady-state particle/power balance), so they are recorded for the session
# log / future use but do not affect simulate()'s output.
# ---------------------------------------------------------------------------
RF_POWER_MIN_W = 50.0
RF_POWER_MAX_W = 300.0
PRESSURE_MIN_MTORR = 1.0
PRESSURE_MAX_MTORR = 20.0
GAS_FLOW_MIN_SCCM = 5.0
GAS_FLOW_MAX_SCCM = 100.0
EXPOSURE_MIN_S = 1.0
EXPOSURE_MAX_S = 600.0


@dataclass(frozen=True)
class ParameterRange:
    """Slider/numeric-input metadata for one configurable parameter [FE-1.1.1].

    This is exactly the (min, max, default, step, unit) tuple a Streamlit
    `st.slider` or `st.number_input` would be built from in Phase 3.
    """
    name: str
    min_value: float
    max_value: float
    default: float
    step: float
    unit: str


RF_POWER_RANGE = ParameterRange("rf_power_w", RF_POWER_MIN_W, RF_POWER_MAX_W, 100.0, 5.0, "W")
PRESSURE_RANGE = ParameterRange("pressure_mtorr", PRESSURE_MIN_MTORR, PRESSURE_MAX_MTORR, 10.0, 0.5, "mTorr")
GAS_FLOW_RANGE = ParameterRange("gas_flow_sccm", GAS_FLOW_MIN_SCCM, GAS_FLOW_MAX_SCCM, 20.0, 1.0, "sccm")
EXPOSURE_RANGE = ParameterRange("exposure_time_s", EXPOSURE_MIN_S, EXPOSURE_MAX_S, 60.0, 1.0, "s")


@dataclass
class ChamberParameters:
    """User-settable inputs for one experiment configuration [FE-1.1.1]."""
    rf_power_w: float = RF_POWER_RANGE.default
    pressure_mtorr: float = PRESSURE_RANGE.default
    gas_flow_sccm: Optional[float] = GAS_FLOW_RANGE.default
    exposure_time_s: Optional[float] = EXPOSURE_RANGE.default


class ExperimentMode(str, Enum):
    """Three predefined experiment modes aligned to educational use cases [FE-1.1.3]."""
    STABLE_PLASMA = "Stable Plasma"
    EXPLORATORY_SWEEP = "Exploratory Sweep"
    STRESS_TEST = "Stress Test"


@dataclass(frozen=True)
class ExperimentModePreset:
    """A named operating point plus the sub-range of the envelope it represents."""
    mode: ExperimentMode
    rf_power_w: float
    pressure_mtorr: float
    rf_power_bounds: tuple[float, float]
    pressure_bounds: tuple[float, float]
    description: str


EXPERIMENT_MODE_PRESETS: dict[ExperimentMode, ExperimentModePreset] = {
    ExperimentMode.STABLE_PLASMA: ExperimentModePreset(
        mode=ExperimentMode.STABLE_PLASMA,
        rf_power_w=100.0,
        pressure_mtorr=10.0,
        rf_power_bounds=(80.0, 150.0),
        pressure_bounds=(8.0, 12.0),
        description=(
            "Mid-range power and pressure representative of a well-behaved, "
            "low-defect discharge. Recommended starting point for new users, and "
            "the platform's default reset state."
        ),
    ),
    ExperimentMode.EXPLORATORY_SWEEP: ExperimentModePreset(
        mode=ExperimentMode.EXPLORATORY_SWEEP,
        rf_power_w=RF_POWER_RANGE.default,
        pressure_mtorr=PRESSURE_RANGE.default,
        rf_power_bounds=(RF_POWER_RANGE.min_value, RF_POWER_RANGE.max_value),
        pressure_bounds=(PRESSURE_RANGE.min_value, PRESSURE_RANGE.max_value),
        description=(
            "Full operating envelope enabled, for parameter sweeps and "
            "sensitivity analysis (Sub-Module 1.5)."
        ),
    ),
    ExperimentMode.STRESS_TEST: ExperimentModePreset(
        mode=ExperimentMode.STRESS_TEST,
        rf_power_w=280.0,
        pressure_mtorr=1.5,
        rf_power_bounds=(250.0, RF_POWER_RANGE.max_value),
        pressure_bounds=(PRESSURE_RANGE.min_value, 3.0),
        description=(
            "High-power / low-pressure corner of the envelope: maximises RF "
            "self-bias voltage and ion bombardment energy (see Sub-Module 1.2's "
            "sheath model), deliberately probing the highest-defect-risk regime "
            "for educational demonstration."
        ),
    ),
}


def parameters_for_mode(mode: ExperimentMode) -> ChamberParameters:
    """Build a ChamberParameters at a preset mode's default operating point."""
    preset = EXPERIMENT_MODE_PRESETS[mode]
    return ChamberParameters(rf_power_w=preset.rf_power_w, pressure_mtorr=preset.pressure_mtorr)


def default_parameters() -> ChamberParameters:
    """Reset to validated default Stable Plasma operating conditions [FE-1.1.4]."""
    return parameters_for_mode(ExperimentMode.STABLE_PLASMA)


# ---------------------------------------------------------------------------
# Validation [FE-1.1.2]
# ---------------------------------------------------------------------------
# Above this predicted defect_probability, a configuration is flagged with a
# warning (not blocked). Grounded in the physics engine's own output rather than
# an independently invented rule, consistent with non-negotiable principle #1:
# every derived judgement traces back to Te/n_e, never set independently.
DEFECT_WARNING_THRESHOLD = 0.55


@dataclass
class ValidationResult:
    """Result of validating a ChamberParameters configuration [FE-1.1.2].

    `errors` are hard failures (out of the defined operating envelope) that
    should block simulation. `warnings` are physically-grounded cautions (the
    physics engine predicts elevated defect risk at this combination) that do
    not block simulation but should be surfaced to the user before they run it.
    """
    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_parameters(params: ChamberParameters) -> ValidationResult:
    """Flag physically inconsistent parameter combinations [FE-1.1.2]."""
    errors: list[str] = []
    warnings: list[str] = []

    for param_range, value in (
        (RF_POWER_RANGE, params.rf_power_w),
        (PRESSURE_RANGE, params.pressure_mtorr),
    ):
        if not (param_range.min_value <= value <= param_range.max_value):
            errors.append(
                f"{param_range.name} = {value} {param_range.unit} is outside the "
                f"operating envelope [{param_range.min_value}, {param_range.max_value}] "
                f"{param_range.unit}."
            )

    if params.gas_flow_sccm is not None and params.gas_flow_sccm <= 0:
        errors.append("gas_flow_sccm must be positive.")
    if params.exposure_time_s is not None and params.exposure_time_s <= 0:
        errors.append("exposure_time_s must be positive.")

    if not errors:
        # Only safe to query the physics engine once both inputs are in-range
        # (simulate() raises ValueError on non-positive power/pressure).
        result = simulate(params.rf_power_w, params.pressure_mtorr)
        if result.defect_probability > DEFECT_WARNING_THRESHOLD:
            warnings.append(
                f"This configuration is predicted to carry a high defect probability "
                f"({result.defect_probability:.2f}) - consider reducing RF power or "
                f"raising pressure toward a more stable operating point."
            )

    return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)
