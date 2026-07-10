"""Tests for Sub-Module 1.1 - virtual chamber and parameter configuration engine."""
import pytest

from digital_twin.chamber_config import (
    DEFECT_WARNING_THRESHOLD,
    EXPERIMENT_MODE_PRESETS,
    GAS_FLOW_RANGE,
    EXPOSURE_RANGE,
    PRESSURE_RANGE,
    RF_POWER_RANGE,
    ChamberParameters,
    ExperimentMode,
    default_parameters,
    parameters_for_mode,
    validate_parameters,
)
from digital_twin.physics_engine import simulate


# ---------------------------------------------------------------------------
# Parameter ranges [FE-1.1.1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "param_range", [RF_POWER_RANGE, PRESSURE_RANGE, GAS_FLOW_RANGE, EXPOSURE_RANGE]
)
def test_parameter_ranges_are_internally_consistent(param_range) -> None:
    assert param_range.min_value < param_range.max_value
    assert param_range.min_value <= param_range.default <= param_range.max_value
    assert param_range.step > 0


def test_default_parameters_within_operating_envelope() -> None:
    params = ChamberParameters()
    assert RF_POWER_RANGE.min_value <= params.rf_power_w <= RF_POWER_RANGE.max_value
    assert PRESSURE_RANGE.min_value <= params.pressure_mtorr <= PRESSURE_RANGE.max_value


def test_default_parameters_feed_physics_engine_without_error() -> None:
    """A default ChamberParameters must be directly usable by Sub-Module 1.2."""
    params = ChamberParameters()
    result = simulate(params.rf_power_w, params.pressure_mtorr)
    assert result.electron_temperature_ev > 0
    assert result.plasma_density_m3 > 0


def test_electron_temperature_and_density_are_not_settable_fields() -> None:
    """Non-negotiable principle #1: Te/n_e must be solved, never free inputs."""
    field_names = {f for f in ChamberParameters.__dataclass_fields__}
    assert "electron_temperature_ev" not in field_names
    assert "plasma_density_m3" not in field_names


# ---------------------------------------------------------------------------
# Reset to default [FE-1.1.4]
# ---------------------------------------------------------------------------
def test_reset_returns_stable_plasma_defaults() -> None:
    reset = default_parameters()
    stable_preset = EXPERIMENT_MODE_PRESETS[ExperimentMode.STABLE_PLASMA]
    assert reset.rf_power_w == pytest.approx(stable_preset.rf_power_w)
    assert reset.pressure_mtorr == pytest.approx(stable_preset.pressure_mtorr)


def test_reset_parameters_are_valid() -> None:
    result = validate_parameters(default_parameters())
    assert result.is_valid
    assert result.errors == []


# ---------------------------------------------------------------------------
# Experiment mode presets [FE-1.1.3]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", list(ExperimentMode))
def test_every_mode_preset_is_within_global_bounds(mode) -> None:
    params = parameters_for_mode(mode)
    assert RF_POWER_RANGE.min_value <= params.rf_power_w <= RF_POWER_RANGE.max_value
    assert PRESSURE_RANGE.min_value <= params.pressure_mtorr <= PRESSURE_RANGE.max_value


@pytest.mark.parametrize("mode", list(ExperimentMode))
def test_every_mode_preset_is_valid_with_no_errors(mode) -> None:
    params = parameters_for_mode(mode)
    result = validate_parameters(params)
    assert result.is_valid
    assert result.errors == []


def test_all_three_experiment_modes_are_defined() -> None:
    assert set(EXPERIMENT_MODE_PRESETS.keys()) == {
        ExperimentMode.STABLE_PLASMA,
        ExperimentMode.EXPLORATORY_SWEEP,
        ExperimentMode.STRESS_TEST,
    }


def test_stress_test_preset_is_the_high_power_low_pressure_corner() -> None:
    stress = EXPERIMENT_MODE_PRESETS[ExperimentMode.STRESS_TEST]
    stable = EXPERIMENT_MODE_PRESETS[ExperimentMode.STABLE_PLASMA]
    assert stress.rf_power_w > stable.rf_power_w
    assert stress.pressure_mtorr < stable.pressure_mtorr


def test_exploratory_sweep_preset_spans_the_full_envelope() -> None:
    sweep = EXPERIMENT_MODE_PRESETS[ExperimentMode.EXPLORATORY_SWEEP]
    assert sweep.rf_power_bounds == (RF_POWER_RANGE.min_value, RF_POWER_RANGE.max_value)
    assert sweep.pressure_bounds == (PRESSURE_RANGE.min_value, PRESSURE_RANGE.max_value)


# ---------------------------------------------------------------------------
# Validation [FE-1.1.2]
# ---------------------------------------------------------------------------
def test_validate_parameters_accepts_a_reasonable_configuration() -> None:
    params = ChamberParameters(rf_power_w=120.0, pressure_mtorr=9.0)
    result = validate_parameters(params)
    assert result.is_valid
    assert result.errors == []


def test_validate_parameters_flags_out_of_range_power() -> None:
    params = ChamberParameters(rf_power_w=RF_POWER_RANGE.max_value + 50.0, pressure_mtorr=10.0)
    result = validate_parameters(params)
    assert not result.is_valid
    assert any("rf_power_w" in e for e in result.errors)


def test_validate_parameters_flags_out_of_range_pressure() -> None:
    params = ChamberParameters(rf_power_w=100.0, pressure_mtorr=PRESSURE_RANGE.min_value - 0.5)
    result = validate_parameters(params)
    assert not result.is_valid
    assert any("pressure_mtorr" in e for e in result.errors)


def test_validate_parameters_flags_negative_gas_flow() -> None:
    params = ChamberParameters(gas_flow_sccm=-5.0)
    result = validate_parameters(params)
    assert not result.is_valid


def test_validate_parameters_flags_negative_exposure_time() -> None:
    params = ChamberParameters(exposure_time_s=-1.0)
    result = validate_parameters(params)
    assert not result.is_valid


def test_validate_parameters_allows_missing_optional_inputs() -> None:
    """gas_flow_sccm / exposure_time_s are optional per FE-1.1.1 and may be None."""
    params = ChamberParameters(gas_flow_sccm=None, exposure_time_s=None)
    result = validate_parameters(params)
    assert result.is_valid


def test_stress_test_configuration_triggers_defect_warning() -> None:
    """The high-power/low-pressure corner is grounded physics, not an arbitrary
    rule: it must actually cross the physics engine's own defect threshold."""
    stress_params = parameters_for_mode(ExperimentMode.STRESS_TEST)
    sim = simulate(stress_params.rf_power_w, stress_params.pressure_mtorr)
    assert sim.defect_probability > DEFECT_WARNING_THRESHOLD

    result = validate_parameters(stress_params)
    assert result.is_valid  # still in-range, just risky
    assert result.warnings, "expected a defect-risk warning at the Stress Test preset"


def test_stable_plasma_configuration_does_not_trigger_defect_warning() -> None:
    stable_params = parameters_for_mode(ExperimentMode.STABLE_PLASMA)
    result = validate_parameters(stable_params)
    assert result.is_valid
    assert result.warnings == []
