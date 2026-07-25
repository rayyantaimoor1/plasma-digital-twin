"""Measure computational performance and physics outputs, before vs after the
rejected K_iz refit, WITHOUT modifying the repository.

The candidate fit is applied by monkey-patching `ionization_rate_coeff` in memory
only, exactly as make_figures.py does, so the engine on disk is never touched.

Run from the project root:
    ./.venv/Scripts/python.exe error_fixing_attempts/measure_before_after.py
"""
from __future__ import annotations

import math
import pathlib
import statistics
import sys
import timeit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from digital_twin import physics_engine as pe

CURRENT = pe.ionization_rate_coeff


def refit(te_v: float) -> float:
    """The rejected refit to Voronov (1997), for comparison only."""
    return 3.5744e-14 * te_v ** 0.39170 * math.exp(-15.7288 / te_v)


def use(k_func) -> None:
    """Swap in a K_iz implementation AND clear the dependent cache.

    `solve_electron_temperature` is @lru_cache'd on (pressure, geometry). Without
    clearing it, a patched K_iz would silently return the previously cached Te and
    every downstream number would be a meaningless mix of old and new. (Verified:
    omitting this made Te appear unchanged.)
    """
    pe.ionization_rate_coeff = k_func
    pe.solve_electron_temperature.cache_clear()


def bench(label: str, n_repeat: int = 9, n_calls: int = 500) -> float:
    """Median wall time per simulate() call, in microseconds.

    Warms up first so the Te cache is populated: `use()` clears it, and timing a
    cold cache would charge the whole root-find to whichever variant happened to
    be measured first (this initially inflated the 'before' figure ~2x).
    """
    for _ in range(300):
        pe.simulate(150.0, 10.0)
    timer = timeit.Timer(lambda: pe.simulate(150.0, 10.0))
    samples = timer.repeat(repeat=n_repeat, number=n_calls)
    per_call_us = [s / n_calls * 1e6 for s in samples]
    med = statistics.median(per_call_us)
    print(f"  {label:<28} median {med:8.1f} us/call   "
          f"(min {min(per_call_us):.1f}, max {max(per_call_us):.1f})")
    return med


def snapshot(label: str) -> dict:
    r = pe.simulate(150.0, 10.0)
    d = {
        "K_iz(3eV)": pe.ionization_rate_coeff(3.0),
        "K_exc/K_iz(3eV)": pe.excitation_rate_coeff(3.0) / pe.ionization_rate_coeff(3.0),
        "E_c(3eV) [V]": pe.collisional_energy_loss(3.0),
        "E_T(3eV) [eV]": pe.total_energy_per_pair(3.0),
        "solved Te [eV]": r.electron_temperature_ev,
        "n_e [m^-3]": r.plasma_density_m3,
    }
    print(f"  {label}")
    for k, v in d.items():
        print(f"    {k:<20} {v:.6g}")
    return d


if __name__ == "__main__":
    print("=" * 68)
    print("PHYSICS OUTPUTS")
    print("=" * 68)
    use(CURRENT)
    before = snapshot("BEFORE (current engine, retained)")
    use(refit)
    after = snapshot("AFTER  (rejected refit, in memory only)")
    use(CURRENT)

    print()
    print("=" * 68)
    print("COMPUTATIONAL PERFORMANCE  (simulate(150 W, 10 mTorr))")
    print("=" * 68)
    # Warm up so the first-call import/JIT costs don't skew the first measurement.
    for _ in range(200):
        pe.simulate(150.0, 10.0)
    t_before = bench("BEFORE (current engine)")
    use(refit)
    t_after = bench("AFTER  (rejected refit)")
    use(CURRENT)
    t_again = bench("BEFORE (re-measured)")

    print()
    print(f"  delta (after vs before)      {100.0 * (t_after - t_before) / t_before:+.1f}%")
    print(f"  before-vs-before spread      {100.0 * (t_again - t_before) / t_before:+.1f}%"
          "   <- measurement noise floor")
    print()
    print("  Both fits share the identical functional form A*Te^n*exp(-E/Te), so the")
    print("  operation count per call is the same; any difference is measurement noise")
    print("  plus a possible change in root-find iteration count.")
