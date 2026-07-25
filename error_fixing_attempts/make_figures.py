"""Generate the figures for the K_iz refit post-mortem report / presentation.

Every number plotted here is COMPUTED, not transcribed: the old and refitted
Arrhenius forms and Voronov's published fit are all evaluated live, and the
energy-loss curves are computed by importing the project's own physics_engine and
temporarily swapping in the candidate K_iz. The Sub-Module 1.6 benchmark
before/after table is the one exception - those are the recorded outputs of the
experiment run (the refit itself was reverted, so they cannot be recomputed
without re-applying it).

Run from the project root:
    ./.venv/Scripts/python.exe error_fixing_attempts/make_figures.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from digital_twin import physics_engine as pe

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- palette (colour-blind-safe pairing, distinct in greyscale by linestyle too) ---
C_OLD = "#2a78d6"      # blue   - the engine's current (kept) fit
C_NEW = "#eb6834"      # orange - the rejected refit
C_REF = "#333333"      # dark   - Voronov reference
C_PASS = "#0ca30c"
C_FAIL = "#d03b3b"
C_BAND = "#0ca30c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
})


# ---------------------------------------------------------------------------
# The three K_iz candidates
# ---------------------------------------------------------------------------
def k_old(te):
    """The engine's current fit - unattributed 'representative' parameterization."""
    return 2.34e-14 * te ** 0.59 * np.exp(-17.44 / te)


def k_new(te):
    """The rejected refit to Voronov (1997), fitted on 1-10 eV in log space."""
    return 3.5744e-14 * te ** 0.39170 * np.exp(-15.7288 / te)


def k_voronov(te):
    """Voronov (1997) ADNDT 65, 1 - Ar I parameters, converted cm^3/s -> m^3/s."""
    A, P, X, K, dE = 5.99e-8, 1.0, 0.1360, 0.26, 15.8
    U = dE / te
    return A * (1.0 + P * np.sqrt(U)) / (X + U) * U ** K * np.exp(-U) * 1e-6


# Published K_iz(3 eV) anchors already in the Sub-Module 1.6 benchmark [m^3/s]
ANCHORS = [
    ("Turner\n(2014)", 2.20e-16),
    ("Jimenez-\nRedondo (2014)", 1.914e-16),
    ("Chabert\n(2011)", 1.494e-16),
]

TE = np.linspace(1.0, 10.0, 400)


# ---------------------------------------------------------------------------
# Fig 1 - K_iz(Te): the refit is faithful to its source; the old fit is not
# ---------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(TE, k_voronov(TE), color=C_REF, lw=3.2, alpha=0.35,
                label="Voronov (1997) — refit target")
    ax.semilogy(TE, k_new(TE), color=C_NEW, lw=1.9, ls="--",
                label="Refitted fit (rejected)")
    ax.semilogy(TE, k_old(TE), color=C_OLD, lw=1.9,
                label="Current engine fit (kept)")
    ax.axvline(3.0, color="#888", lw=0.9, ls=":")
    ax.annotate("benchmark\nevaluated here", xy=(3.0, 4e-19), xytext=(4.1, 2e-20),
                fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))
    ax.set_xlabel("Electron temperature $T_e$  (eV)")
    ax.set_ylabel("$K_{iz}$  (m$^3$/s)")
    ax.set_title("Ionization rate coefficient: the refit reproduces its source almost exactly\n"
                 "(the two dashed/grey curves overlie); the current fit sits well below it",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_kiz_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 - relative deviation from Voronov: quantifies "faithful" vs "not"
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    dev_old = 100.0 * (k_old(TE) - k_voronov(TE)) / k_voronov(TE)
    dev_new = 100.0 * (k_new(TE) - k_voronov(TE)) / k_voronov(TE)
    ax.axhline(0, color=C_REF, lw=1.2, alpha=0.6)
    ax.plot(TE, dev_old, color=C_OLD, lw=1.9, label="Current engine fit (kept)")
    ax.plot(TE, dev_new, color=C_NEW, lw=1.9, ls="--", label="Refitted fit (rejected)")
    ax.set_xlabel("Electron temperature $T_e$  (eV)")
    ax.set_ylabel("Deviation from Voronov  (%)")
    ax.set_title("Fit fidelity to the cited source: refit within ±0.15%, current fit −88% to −13%",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    for te_mark in (1.0, 3.0, 10.0):
        d = 100.0 * (k_old(te_mark) - k_voronov(te_mark)) / k_voronov(te_mark)
        ax.plot([te_mark], [d], "o", color=C_OLD, ms=4.5)
        ax.annotate(f"{d:+.0f}%", xy=(te_mark, d), xytext=(4, -11),
                    textcoords="offset points", fontsize=8, color=C_OLD)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_fit_fidelity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 - K_iz(3 eV) vs the three benchmark anchors: source-vs-source spread
# ---------------------------------------------------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [a[0] for a in ANCHORS] + ["Current fit\n(kept)", "Refit\n(rejected)"]
    vals = [a[1] for a in ANCHORS] + [float(k_old(3.0)), float(k_new(3.0))]
    colors = ["#8a8a8a"] * 3 + [C_OLD, C_NEW]
    bars = ax.bar(labels, np.array(vals) * 1e16, color=colors, width=0.62)
    lo = min(a[1] for a in ANCHORS) * 1e16
    hi = max(a[1] for a in ANCHORS) * 1e16
    ax.axhspan(lo, hi, color="#8a8a8a", alpha=0.13, zorder=0)
    ax.annotate(f"published spread\n{lo:.2f}–{hi:.2f}  (~40% apart)",
                xy=(2.6, hi), xytext=(2.55, hi + 0.42), fontsize=8, color="#555")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1e16 + 0.05,
                f"{v*1e16:.2f}", ha="center", fontsize=8.5)
    ax.set_ylabel("$K_{iz}$(3 eV)   ($\\times 10^{-16}$ m$^3$/s)")
    ax.set_title("At 3 eV the refit overshoots every published anchor, while the current fit undershoots them\n"
                 "The three anchors already disagree with each other by ~40%",
                 fontsize=10.5, loc="left")
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_anchor_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 - THE DECISIVE FIGURE: E_c vs the L&L Fig. 3.17 published band.
# Computed by importing the real engine and swapping K_iz.
# ---------------------------------------------------------------------------
def _e_c_with(k_func, te_grid):
    """Compute the project's own collisional_energy_loss with K_iz swapped."""
    original = pe.ionization_rate_coeff
    try:
        pe.ionization_rate_coeff = lambda te: float(k_func(te))
        return np.array([pe.collisional_energy_loss(float(t)) for t in te_grid])
    finally:
        pe.ionization_rate_coeff = original


def fig4():
    te = np.linspace(2.0, 10.0, 240)
    ec_old = _e_c_with(k_old, te)
    ec_new = _e_c_with(k_new, te)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhspan(50, 70, color=C_BAND, alpha=0.14, zorder=0)
    ax.annotate("Lieberman & Lichtenberg Fig. 3.17\npublished argon band, 50–70 V",
                xy=(6.4, 60), fontsize=8.5, color="#0a7a0a", ha="left", va="center")
    ax.plot(te, ec_old, color=C_OLD, lw=2.0, label="Current engine fit (kept)")
    ax.plot(te, ec_new, color=C_NEW, lw=2.0, ls="--", label="Refitted fit (rejected)")
    # Annotate at EXACTLY 3 eV (the benchmark point), not the nearest grid sample.
    for kf, col in ((k_old, C_OLD), (k_new, C_NEW)):
        v3 = float(_e_c_with(kf, [3.0])[0])
        ax.plot([3.0], [v3], "o", color=col, ms=6, zorder=5)
        ax.annotate(f"{v3:.1f} V", xy=(3.0, v3), xytext=(8, 6),
                    textcoords="offset points", fontsize=9, color=col, fontweight="bold")
    ax.axvline(3.0, color="#888", lw=0.9, ls=":")
    ax.set_xlabel("Electron temperature $T_e$  (eV)")
    ax.set_ylabel("Collisional energy loss per e–ion pair  $E_c$  (V)")
    ax.set_title("WHY THE REFIT WAS REJECTED: it pushes $E_c$ out of the published band\n"
                 "This is an INDEPENDENT literature check that passed before the change",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_ylim(20, 90)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_energy_loss_band.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 - ROOT CAUSE: E_c depends only on the ratio K_exc/K_iz
# ---------------------------------------------------------------------------
def fig5():
    te = np.linspace(2.0, 10.0, 240)
    ratio_old = np.array([pe.excitation_rate_coeff(float(t)) for t in te]) / k_old(te)
    ratio_new = np.array([pe.excitation_rate_coeff(float(t)) for t in te]) / k_new(te)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(te, ratio_old, color=C_OLD, lw=2.0, label="Current: both fits unattributed")
    ax.plot(te, ratio_new, color=C_NEW, lw=2.0, ls="--",
            label="Refit: cited $K_{iz}$ + uncited $K_{exc}$")
    # Annotate at EXACTLY 3 eV (the benchmark point), not the nearest grid sample.
    for kf, col in ((k_old, C_OLD), (k_new, C_NEW)):
        r3 = pe.excitation_rate_coeff(3.0) / float(kf(3.0))
        ax.plot([3.0], [r3], "o", color=col, ms=6, zorder=5)
        ax.annotate(f"{r3:.2f}", xy=(3.0, r3), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, color=col, fontweight="bold")
    ax.axvline(3.0, color="#888", lw=0.9, ls=":")
    ax.set_xlabel("Electron temperature $T_e$  (eV)")
    ax.set_ylabel("$K_{exc} / K_{iz}$   (dimensionless)")
    ax.set_title("ROOT CAUSE: $E_c$ sees only the RATIO $K_{exc}/K_{iz}$\n"
                 "Refitting one coefficient alone collapsed it 3.77 → 1.73 at 3 eV",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ratio_coupling.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 - Sub-Module 1.6 benchmark, before vs after (recorded experiment output)
# ---------------------------------------------------------------------------
BENCH = [
    # (short label, tolerance %, before dev %, after dev %)
    ("Ion mean free path $d/\\lambda_i$", 15, 2.9, 2.9),
    ("$T_e$ vs global model", 30, 9.6, -4.8),
    ("$T_e$ vs PIC", 30, -8.7, -20.7),
    ("$K_{iz}$(3eV) vs Turner", 30, -39.2, 32.0),
    ("$K_{iz}$(3eV) vs Jim.-Redondo", 30, -30.2, 51.7),
    ("$K_{iz}$(3eV) vs Chabert", 30, -10.5, 94.5),
    ("$E_T$ per pair (3eV)", 30, 18.7, -16.6),
    ("$n_e$ vs PIC line-avg", 30, 41.7, 101.7),
    ("$n_e$ vs PIC centre", 30, 4.4, 48.6),
]


def fig6():
    labels = [b[0] for b in BENCH]
    tol = np.array([b[1] for b in BENCH], dtype=float)
    before = np.array([b[2] for b in BENCH], dtype=float)
    after = np.array([b[3] for b in BENCH], dtype=float)
    y = np.arange(len(BENCH))
    h = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    # tolerance band per check
    for i, t in enumerate(tol):
        ax.barh(i, 2 * t, left=-t, height=0.86, color="#8a8a8a", alpha=0.15, zorder=0)
        ax.text(t + 2, i - 0.34, f"±{t:.0f}%", fontsize=7, color="#777", va="center")

    cols_b = [C_PASS if abs(v) <= t else C_FAIL for v, t in zip(before, tol)]
    cols_a = [C_PASS if abs(v) <= t else C_FAIL for v, t in zip(after, tol)]
    ax.barh(y + h / 2, before, height=h, color=cols_b, label="Before (kept fit)")
    ax.barh(y - h / 2, after, height=h, color=cols_a, alpha=0.55, hatch="///",
            edgecolor="white", linewidth=0.4, label="After refit (rejected)")
    ax.axvline(0, color=C_REF, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Deviation from published reference  (%)   —   green = inside tolerance, red = outside")
    ax.set_title("Sub-Module 1.6 benchmark: 6/9 passing before → 4/9 after the refit\n"
                 "Solid = before (kept), hatched = after (rejected)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlim(-60, 130)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_benchmark_before_after.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7 - the propagation chain, as a simple annotated flow
# ---------------------------------------------------------------------------
def fig7():
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    ax.axis("off")
    boxes = [
        ("Refit $K_{iz}$ alone\n(faithful to Voronov)", C_NEW),
        ("Ratio $K_{exc}/K_{iz}$\n3.77 → 1.73", C_FAIL),
        ("$E_c$ 61.5 → 36.8 V\nOUT of 50–70 V band", C_FAIL),
        ("$E_T$ 83 → 58 eV\n$n_e \\propto 1/E_T$", C_FAIL),
        ("Density checks\n+42% → +102%", C_FAIL),
    ]
    n = len(boxes)
    for i, (txt, col) in enumerate(boxes):
        x = i / n
        ax.add_patch(plt.Rectangle((x + 0.008, 0.30), 1.0 / n - 0.035, 0.40,
                                   transform=ax.transAxes, facecolor=col, alpha=0.15,
                                   edgecolor=col, linewidth=1.4, zorder=2))
        ax.text(x + (1.0 / n - 0.027) / 2 + 0.008, 0.50, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=8.6, zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x + 1.0 / n - 0.020, 0.50), xytext=(x + 1.0 / n - 0.030, 0.50),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.6))
    ax.text(0.008, 0.86, "How one coefficient's refit propagated into a validation failure",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold")
    ax.text(0.008, 0.13,
            "The two original fits are wrong in the SAME direction, so their ratio — and hence $E_c$ — "
            "came out plausible for the wrong reasons.",
            transform=ax.transAxes, fontsize=8.3, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_propagation_chain.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print("wrote", f.__name__)
    # Sanity values quoted in the report / slides
    print()
    print("--- values quoted in the deliverables ---")
    print(f"K_iz(3eV) current  = {float(k_old(3.0)):.4e} m^3/s")
    print(f"K_iz(3eV) refit    = {float(k_new(3.0)):.4e} m^3/s")
    print(f"K_iz(3eV) Voronov  = {float(k_voronov(3.0)):.4e} m^3/s")
    print(f"E_c(3eV) current   = {_e_c_with(k_old, [3.0])[0]:.2f} V")
    print(f"E_c(3eV) refit     = {_e_c_with(k_new, [3.0])[0]:.2f} V")
    r_old = pe.excitation_rate_coeff(3.0) / float(k_old(3.0))
    r_new = pe.excitation_rate_coeff(3.0) / float(k_new(3.0))
    print(f"K_exc/K_iz current = {r_old:.3f}")
    print(f"K_exc/K_iz refit   = {r_new:.3f}")
    mx = np.max(np.abs(100.0 * (k_new(TE) - k_voronov(TE)) / k_voronov(TE)))
    print(f"max |refit - Voronov| on 1-10 eV = {mx:.4f}%")
    print("figures ->", OUT)
