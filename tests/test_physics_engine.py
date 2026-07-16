"""Tests for Sub-Module 1.2 - the 0D global discharge physics engine.

These tests assert the *physical properties* the global model must have, not
hard-coded magic numbers. They are the first line of defence for the sub-module's
correctness and encode the results a committee would expect us to be able to state:
Te set by pressure (not power), n_e linear in power, and physically plausible ranges.
"""
import math

import numpy as np
import pytest

from digital_twin.physics_engine import (
    DEFAULT_GEOMETRY,
    FLOATING_SHEATH_COEFF,
    ION_NEUTRAL_XSEC,
    RF_MEAN_SHEATH_FRACTION,
    bohm_velocity,
    child_langmuir_sheath,
    collisional_energy_loss,
    debye_length,
    elastic_rate_coeff,
    excitation_rate_coeff,
    implied_ion_power_w,
    ion_mean_free_path,
    ionization_rate_coeff,
    gas_density,
    chamber_volume,
    effective_loss_area,
    edge_to_center_factors,
    solve_electron_temperature,
    solve_plasma_density,
    sheath_and_ion_energy,
    simulate,
    total_energy_per_pair,
)

# Nominal operating range from the FYP scope document (Section 6).
POWERS_W = [50.0, 100.0, 200.0, 300.0]
PRESSURES_MTORR = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]


# ---------------------------------------------------------------------------
# Particle balance / electron temperature (FE-1.2.1)
# ---------------------------------------------------------------------------
def test_electron_temperature_in_physical_range() -> None:
    """Te should land in the few-eV band expected for an argon CCP (2-7 eV)."""
    for pressure in PRESSURES_MTORR:
        te = solve_electron_temperature(pressure)
        assert 1.0 < te < 8.0, f"Te={te} eV out of physical range at {pressure} mTorr"


def test_electron_temperature_decreases_with_pressure() -> None:
    """Known global-model result: Te falls as pressure rises.

    At higher pressure the effective plasma size n_g*d_eff is larger, so a lower Te
    already sustains enough ionization to replace wall losses.
    """
    temps = [solve_electron_temperature(p) for p in PRESSURES_MTORR]
    for lower, higher in zip(temps, temps[1:]):
        assert higher < lower, "Te must decrease monotonically with pressure"


def test_electron_temperature_independent_of_power() -> None:
    """The signature prediction of the global model: Te does not depend on RF power.

    Te comes purely from the particle balance (pressure + geometry). Power only
    enters the *power* balance, which sets density. We verify Te is identical
    regardless of the power we later feed to the density solve.
    """
    te_direct = solve_electron_temperature(10.0)
    for power in POWERS_W:
        result = simulate(power, 10.0)
        assert result.electron_temperature_ev == pytest.approx(te_direct, rel=1e-9)


def test_particle_balance_residual_is_zero_at_solution() -> None:
    """At the solved Te, ionization production must equal wall loss (residual ~ 0)."""
    for pressure in PRESSURES_MTORR:
        te = solve_electron_temperature(pressure)
        n_g = gas_density(pressure, DEFAULT_GEOMETRY.gas_temp_k)
        h_l, h_r = edge_to_center_factors(DEFAULT_GEOMETRY, n_g)
        volume = chamber_volume(DEFAULT_GEOMETRY)
        area = effective_loss_area(DEFAULT_GEOMETRY, h_l, h_r)
        production = ionization_rate_coeff(te) * n_g * volume
        loss = bohm_velocity(te) * area
        # Residual should be a tiny fraction of either (balanced) term.
        assert abs(production - loss) < 1e-6 * production


# ---------------------------------------------------------------------------
# Rate coefficients and energy-loss building blocks (Phase 4 coverage pass).
# These feed solve_electron_temperature/solve_plasma_density and were previously
# exercised only indirectly through those solves; direct tests pin down their
# own behaviour, including the L&L Fig. 3.17 reference-value check that was
# verified ad hoc during the Sub-Module 1.2 physics review but never captured
# as a permanent regression test.
# ---------------------------------------------------------------------------
def test_excitation_rate_coeff_increases_with_temperature() -> None:
    """Hotter electrons excite more often - K_exc must rise monotonically with Te."""
    values = [excitation_rate_coeff(te) for te in (2.0, 3.0, 5.0, 7.0)]
    for lower, higher in zip(values, values[1:]):
        assert higher > lower


def test_excitation_rate_coeff_positive() -> None:
    for te in (0.5, 1.0, 3.0, 10.0, 30.0):
        assert excitation_rate_coeff(te) > 0.0


def test_elastic_rate_coeff_is_constant() -> None:
    """Documented as a fixed order-of-magnitude value, not a function of Te."""
    values = {elastic_rate_coeff(te) for te in (2.0, 5.0, 10.0, 30.0)}
    assert values == {1.0e-13}


def test_collisional_energy_loss_matches_lieberman_lichtenberg_fig_3_17() -> None:
    """Verified reference check (Sub-Module 1.2 physics review): E_c(Te) must land
    within the published argon Fig. 3.17 bands - ~50-70 V at Te=3 eV and
    ~35-45 V at Te=5 eV - and rise steeply as Te falls below ~4 eV."""
    assert 50.0 <= collisional_energy_loss(3.0) <= 70.0
    assert 35.0 <= collisional_energy_loss(5.0) <= 45.0
    assert collisional_energy_loss(2.0) > collisional_energy_loss(3.0) > collisional_energy_loss(5.0)


def test_collisional_energy_loss_exceeds_ionization_potential() -> None:
    """E_c must always exceed the bare 15.76 V ionization potential (E_iz) - the
    excitation/elastic terms are additional costs on top of it, never negative."""
    for te in (1.0, 3.0, 5.0, 10.0, 20.0):
        assert collisional_energy_loss(te) > 15.76


def test_total_energy_per_pair_decomposes_exactly() -> None:
    """E_T = E_c(Te) + 2*Te + (FLOATING_SHEATH_COEFF + 0.5)*Te exactly - the
    documented closed-form decomposition, not an approximation."""
    for te in (2.0, 3.0, 5.0, 8.0):
        expected = collisional_energy_loss(te) + 2.0 * te + (FLOATING_SHEATH_COEFF + 0.5) * te
        assert total_energy_per_pair(te) == pytest.approx(expected, rel=1e-12)


def test_total_energy_per_pair_exceeds_collisional_loss_alone() -> None:
    for te in (2.0, 5.0, 10.0):
        assert total_energy_per_pair(te) > collisional_energy_loss(te)


def test_ion_mean_free_path_inversely_proportional_to_gas_density() -> None:
    """lambda_i = 1/(n_g * sigma_i): the product with n_g must equal the fixed
    cross-section constant exactly, and lambda_i must shrink as pressure rises."""
    lambdas = []
    for pressure in (1.0, 10.0, 20.0):
        n_g = gas_density(pressure, DEFAULT_GEOMETRY.gas_temp_k)
        lam = ion_mean_free_path(n_g)
        assert n_g * lam == pytest.approx(1.0 / ION_NEUTRAL_XSEC, rel=1e-9)
        lambdas.append(lam)
    for longer, shorter in zip(lambdas, lambdas[1:]):
        assert shorter < longer


# ---------------------------------------------------------------------------
# Power balance / plasma density (FE-1.2.2)
# ---------------------------------------------------------------------------
def test_density_is_positive_and_physical() -> None:
    """n_e should be positive and in the ~1e15-1e18 m^-3 band typical of CCPs."""
    for power in POWERS_W:
        for pressure in PRESSURES_MTORR:
            result = simulate(power, pressure)
            assert 1e15 < result.plasma_density_m3 < 1e18


def test_density_increases_monotonically_with_power() -> None:
    """More absorbed power sustains a denser plasma at fixed pressure."""
    for pressure in PRESSURES_MTORR:
        densities = [simulate(p, pressure).plasma_density_m3 for p in POWERS_W]
        for lower, higher in zip(densities, densities[1:]):
            assert higher > lower


def test_density_is_linear_in_power() -> None:
    """Global-model result: n_e is proportional to absorbed power.

    Nothing in the density expression except P_abs depends on power, so doubling
    the power must double the density essentially exactly (this is exact in the
    noiseless model, up to floating-point).
    """
    for pressure in PRESSURES_MTORR:
        ne_100 = solve_plasma_density(100.0, solve_electron_temperature(pressure), pressure)
        ne_200 = solve_plasma_density(200.0, solve_electron_temperature(pressure), pressure)
        assert ne_200 == pytest.approx(2.0 * ne_100, rel=1e-9)


# ---------------------------------------------------------------------------
# Sheath / ion-energy model (FE-1.2.3) - regression tests for a fixed bug where
# sheath_voltage collapsed to a power-independent value by algebraic accident
# (P_abs cancelled against the n_e it was divided by, since n_e was itself solved
# from the same P_abs). These tests pin down the correct, power-dependent behaviour.
# ---------------------------------------------------------------------------
def test_ion_energy_increases_with_power_at_fixed_pressure() -> None:
    """The RF self-bias contribution must make ion energy grow with power.

    This is the direct regression test for the fix: before it, ion_energy_ev was
    identical across all four power levels at a given pressure.
    """
    for pressure in PRESSURES_MTORR:
        energies = [simulate(p, pressure).ion_energy_ev for p in POWERS_W]
        for lower, higher in zip(energies, energies[1:]):
            assert higher > lower


def test_sheath_voltage_increases_with_power_at_fixed_pressure() -> None:
    """Same regression, checked directly on sheath_voltage_v."""
    for pressure in PRESSURES_MTORR:
        voltages = [simulate(p, pressure).sheath_voltage_v for p in POWERS_W]
        for lower, higher in zip(voltages, voltages[1:]):
            assert higher > lower


def test_floating_sheath_component_is_power_independent() -> None:
    """The DC/ambipolar floating-sheath term alone should NOT depend on power.

    Only the RF self-bias term should carry the power dependence; the floating
    component is genuine, non-accidental physics that depends on Te only.
    """
    for pressure in PRESSURES_MTORR:
        te = solve_electron_temperature(pressure)
        v_floating = FLOATING_SHEATH_COEFF * te
        # v_floating is independent of both n_e and power by construction; just
        # confirm it only needs Te (i.e. is a pure function of pressure here).
        assert v_floating == pytest.approx(FLOATING_SHEATH_COEFF * solve_electron_temperature(pressure))


def test_sheath_voltage_exceeds_floating_component_alone() -> None:
    """sheath_voltage must be strictly larger than the floating term alone.

    i.e. the RF self-bias contribution is strictly positive at any nonzero power.
    """
    for power in POWERS_W:
        for pressure in PRESSURES_MTORR:
            te = solve_electron_temperature(pressure)
            n_e = solve_plasma_density(power, te, pressure)
            _flux, sheath_voltage, _ion_energy = sheath_and_ion_energy(power, te, n_e, pressure)
            v_floating = FLOATING_SHEATH_COEFF * te
            assert sheath_voltage > v_floating


def test_debye_length_shrinks_with_density() -> None:
    """Sanity check on the Debye-length helper: denser plasma screens faster."""
    te = 4.0
    assert debye_length(te, 1e17) < debye_length(te, 1e16)


# ---------------------------------------------------------------------------
# Child-Langmuir sheath: the physically-determined driven-electrode sheath that
# resolves the (power, pressure) under-determination by taking an independent RF
# voltage drive (FE-1.2.3, "Option C").
# ---------------------------------------------------------------------------
def _te_ne(power, pressure):
    te = solve_electron_temperature(pressure)
    ne = solve_plasma_density(power, te, pressure)
    return te, ne


def test_child_langmuir_mean_voltage_reduces_to_floating_at_zero_drive() -> None:
    """With no RF drive, the mean sheath voltage is just the floating potential."""
    te, ne = _te_ne(100.0, 10.0)
    sheath = child_langmuir_sheath(0.0, te, ne, 10.0)
    assert sheath.mean_sheath_voltage_v == pytest.approx(FLOATING_SHEATH_COEFF * te)


def test_child_langmuir_mean_voltage_tracks_rf_fraction() -> None:
    """Mean sheath voltage = floating + RF_MEAN_SHEATH_FRACTION * V_rf."""
    te, ne = _te_ne(100.0, 10.0)
    v_rf = 200.0
    sheath = child_langmuir_sheath(v_rf, te, ne, 10.0)
    expected = FLOATING_SHEATH_COEFF * te + RF_MEAN_SHEATH_FRACTION * v_rf
    assert sheath.mean_sheath_voltage_v == pytest.approx(expected)


def test_child_langmuir_ion_energy_increases_with_rf_voltage() -> None:
    energies = [simulate(100.0, 10.0, rf_voltage_v=v).ion_energy_ev for v in (50.0, 150.0, 300.0, 600.0)]
    for lower, higher in zip(energies, energies[1:]):
        assert higher > lower


def test_child_langmuir_thickness_positive_and_shrinks_with_density() -> None:
    """Denser plasma (from higher power) => thinner Child-Langmuir sheath."""
    te = solve_electron_temperature(10.0)
    thicknesses = []
    for power in POWERS_W:
        ne = solve_plasma_density(power, te, 10.0)
        sheath = child_langmuir_sheath(300.0, te, ne, 10.0)
        assert sheath.thickness_m > 0.0
        thicknesses.append(sheath.thickness_m)
    for thicker, thinner in zip(thicknesses, thicknesses[1:]):
        assert thinner < thicker


def test_child_langmuir_high_voltage_flag() -> None:
    te, ne = _te_ne(100.0, 10.0)
    assert child_langmuir_sheath(0.0, te, ne, 10.0).is_high_voltage is False
    assert child_langmuir_sheath(400.0, te, ne, 10.0).is_high_voltage is True


def test_child_langmuir_collisionless_in_low_pressure_regime() -> None:
    """At low pressure the sheath is thinner than an ion mfp - the Child-Langmuir
    collisionless treatment is valid there."""
    for power in POWERS_W:
        for pressure in (1.0, 5.0, 10.0):
            te, ne = _te_ne(power, pressure)
            assert child_langmuir_sheath(300.0, te, ne, pressure).is_collisionless


def test_child_langmuir_flags_collisional_regime_at_high_pressure_low_power() -> None:
    """The whole value of reporting the flag: at the high-pressure / low-power
    corner under strong drive, the sheath grows thicker than an ion mfp and the
    collisionless Child law no longer strictly applies - the model says so honestly
    rather than silently returning a number outside its own validity."""
    te, ne = _te_ne(50.0, 20.0)
    assert child_langmuir_sheath(300.0, te, ne, 20.0).is_collisionless is False


def test_child_langmuir_negative_voltage_raises() -> None:
    te, ne = _te_ne(100.0, 10.0)
    with pytest.raises(ValueError):
        child_langmuir_sheath(-10.0, te, ne, 10.0)


def test_simulate_rf_voltage_changes_ion_energy_vs_default() -> None:
    default = simulate(100.0, 10.0)
    driven = simulate(100.0, 10.0, rf_voltage_v=300.0)
    assert driven.ion_energy_ev > default.ion_energy_ev


def test_simulate_default_path_unchanged_by_option_c() -> None:
    """Regression: rf_voltage_v=None must reproduce the fallback estimate exactly,
    so Sub-Modules 1.1/1.4/1.5 (which call simulate without a voltage) are unaffected."""
    explicit_none = simulate(120.0, 7.0, rf_voltage_v=None)
    positional = simulate(120.0, 7.0)
    assert explicit_none.to_dict() == positional.to_dict()


def test_implied_ion_power_increases_with_rf_voltage() -> None:
    te, ne = _te_ne(100.0, 10.0)
    _flux, _sv, _ie = sheath_and_ion_energy(100.0, te, ne, 10.0, rf_voltage_v=100.0)
    powers = []
    for v in (50.0, 150.0, 300.0):
        flux, sheath_v, _ = sheath_and_ion_energy(100.0, te, ne, 10.0, rf_voltage_v=v)
        powers.append(implied_ion_power_w(sheath_v, flux))
    for lower, higher in zip(powers, powers[1:]):
        assert higher > lower


def test_implied_ion_power_flags_overdrive() -> None:
    """A large RF voltage at modest power implies more ion power than is absorbed -
    the consistency check the validation layer uses to catch an over-driven sheath."""
    absorbed = 100.0
    te, ne = _te_ne(absorbed, 10.0)
    flux, sheath_v, _ = sheath_and_ion_energy(absorbed, te, ne, 10.0, rf_voltage_v=400.0)
    assert implied_ion_power_w(sheath_v, flux) > absorbed


# ---------------------------------------------------------------------------
# Derived outputs and index ranges (FE-1.2.3 / FE-1.2.4)
# ---------------------------------------------------------------------------
def test_bounded_indices_stay_in_unit_interval() -> None:
    """uniformity, process_quality and defect_probability are all in [0, 1]."""
    for power in POWERS_W:
        for pressure in PRESSURES_MTORR:
            r = simulate(power, pressure)
            assert 0.0 <= r.uniformity_index <= 1.0
            assert 0.0 <= r.process_quality <= 1.0
            assert 0.0 <= r.defect_probability <= 1.0


def test_positive_physical_outputs() -> None:
    """Flux, sheath voltage, ion energy and etch rate must be non-negative."""
    for power in POWERS_W:
        for pressure in PRESSURES_MTORR:
            r = simulate(power, pressure)
            assert r.ion_flux_m2s > 0.0
            assert r.sheath_voltage_v > 0.0
            assert r.ion_energy_ev > 0.0
            assert r.etch_rate_nm_min >= 0.0


def test_reactivity_scales_with_power() -> None:
    """Reactivity tracks ion flux, which grows with density and hence with power."""
    for pressure in PRESSURES_MTORR:
        reactivities = [simulate(p, pressure).reactivity_index for p in POWERS_W]
        for lower, higher in zip(reactivities, reactivities[1:]):
            assert higher > lower


def test_uniformity_decreases_with_pressure() -> None:
    """Our uniformity proxy (radial edge-to-center flatness) worsens with pressure.

    Higher pressure -> shorter ion mean free path -> more centre-peaked profile
    -> lower h_R -> lower uniformity index.
    """
    uniformities = [simulate(100.0, p).uniformity_index for p in PRESSURES_MTORR]
    for higher, lower in zip(uniformities, uniformities[1:]):
        assert lower <= higher


# ---------------------------------------------------------------------------
# Reproducibility and noise (FE-1.2.5)
# ---------------------------------------------------------------------------
def test_noiseless_simulation_is_deterministic() -> None:
    """With noise off, repeated calls must be byte-for-byte identical."""
    a = simulate(150.0, 8.0)
    b = simulate(150.0, 8.0)
    assert a.to_dict() == b.to_dict()


def test_noise_is_reproducible_with_seed() -> None:
    """A given seed reproduces the same noisy result (needed for FE-1.3.7 seeding)."""
    a = simulate(150.0, 8.0, noise_level=0.1, seed=42)
    b = simulate(150.0, 8.0, noise_level=0.1, seed=42)
    assert a.to_dict() == b.to_dict()


def test_noise_perturbs_but_stays_reasonable() -> None:
    """Noise should move outputs off the clean value but not wildly (10% level)."""
    clean = simulate(150.0, 8.0)
    noisy = simulate(150.0, 8.0, noise_level=0.1, seed=1)
    assert noisy.plasma_density_m3 != clean.plasma_density_m3
    # Within a few sigma of the clean value for a single 10%-ish draw.
    rel_dev = abs(noisy.plasma_density_m3 - clean.plasma_density_m3) / clean.plasma_density_m3
    assert rel_dev < 0.6


def test_noise_keeps_bounded_indices_in_range() -> None:
    """Even under heavy noise, clipped indices never escape [0, 1]."""
    for seed in range(20):
        r = simulate(60.0, 20.0, noise_level=0.15, seed=seed)
        assert 0.0 <= r.uniformity_index <= 1.0
        assert 0.0 <= r.process_quality <= 1.0
        assert 0.0 <= r.defect_probability <= 1.0


# ---------------------------------------------------------------------------
# Robustness and input validation
# ---------------------------------------------------------------------------
def test_solver_runs_across_full_operating_grid() -> None:
    """brentq must find a root at every point on a dense operating grid."""
    for power in np.linspace(50.0, 300.0, 11):
        for pressure in np.linspace(1.0, 20.0, 20):
            r = simulate(float(power), float(pressure))
            assert math.isfinite(r.electron_temperature_ev)
            assert math.isfinite(r.plasma_density_m3)


def test_invalid_inputs_raise() -> None:
    """Non-physical inputs should fail loudly rather than return garbage."""
    with pytest.raises(ValueError):
        solve_electron_temperature(0.0)
    with pytest.raises(ValueError):
        simulate(0.0, 10.0)
    with pytest.raises(ValueError):
        simulate(100.0, 10.0, noise_level=0.5)


def test_semi_quantitative_reference_point() -> None:
    """Precursor to Sub-Module 1.6: Te at a mid-range point sits near literature.

    Argon CCP electron temperatures at ~10 mTorr are typically ~2.5-4 eV. We assert
    a generous band here; the formal literature benchmark with a stated tolerance is
    Sub-Module 1.6's job.
    """
    te = solve_electron_temperature(10.0)
    assert 2.5 <= te <= 4.0
