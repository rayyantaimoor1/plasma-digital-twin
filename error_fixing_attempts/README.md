# Error-Fixing Attempts

Records of deliberate attempts to fix known limitations in the physics engine —
**including the ones that were rejected.** Keeping the failures documented is the
point: each one carries a reusable constraint that is not obvious from the code
alone, and each is defensible in the FYP evaluation.

## Contents

| File | What it is |
|---|---|
| [Kiz_Refit_Attempt_Report.docx](Kiz_Refit_Attempt_Report.docx) | Full written post-mortem (~11 pages): source verification, pre-registered methodology, results, root-cause analysis, the revert decision, and an anticipated-questions section for the viva. |
| [Kiz_Refit_Attempt_Presentation.pptx](Kiz_Refit_Attempt_Presentation.pptx) | 15-slide companion deck for presenting this one decision on its own, with speaker notes on every slide. |
| [make_figures.py](make_figures.py) | Generates all nine figures. Every plotted value is **computed live** from the project's own `physics_engine`, not transcribed — so the analysis is reproducible. |
| [measure_before_after.py](measure_before_after.py) | Measures physics outputs and `simulate()` runtime before vs after, by monkey-patching the candidate fit **in memory only** — the engine on disk is never touched. |
| [figures/](figures) | The generated PNGs, embedded in both documents above. |

Both documents cover, in order: **why** a fix was attempted, **how** the approach and
source were chosen, **what** was done during the attempt, **why** it could not be kept
and was reverted, and a closing **before/after comparison of engine credibility and
performance** — with figures and tables throughout.

## Attempt 1 — Argon ionization rate coefficient K_iz  (2026-07-25)

**Goal.** Three of the nine Sub-Module 1.6 validation checks fail, all traceable
to `ionization_rate_coeff` being an unattributed "representative" Arrhenius fit
with no citable origin. The aim was to replace it with a properly sourced fit.

**Attempted.** Refit against Voronov (1997), *Atomic Data and Nuclear Data
Tables* **65**, 1, DOI 10.1006/adnd.1997.0732 — a genuinely independent source
whose argon parameters were verified first-hand from the published machine-readable
table. Methodology was pre-registered: fit window fixed at Te = 1–10 eV *before*
fitting, one fit in log space, no post-hoc tuning, benchmark and tests untouched.

**Result: REVERTED.** The fit itself was excellent (0.148% max error vs source;
the fitted threshold recovered argon's 15.76 V ionization potential to 0.2%
without being imposed). But it broke an **independent** literature check that
previously passed — `E_c`(3 eV) fell 61.5 V → 36.8 V, outside the 50–70 V band in
Lieberman & Lichtenberg Fig. 3.17 — and the benchmark dropped 6/9 → 4/9.

**The reusable finding.** `E_c` depends only on the **ratio** `K_exc / K_iz`. Both
original fits are inaccurate in the *same* direction, so their ratio — and hence
`E_c` — came out plausible for the wrong reasons. Refitting one coefficient alone
destroys that cancellation, making a cited `K_iz` beside an uncited `K_exc`
*internally inconsistent* and measurably worse than two consistently inaccurate
fits.

**Acceptance gate for any future attempt:** refit `K_iz` **and** `K_exc` together
from comparable sources, and confirm `E_c`(3 eV) lands inside 50–70 V **before**
consulting the Sub-Module 1.6 benchmark at all. See `FUTURE.md` item 1 and the
warning comment above the rate coefficients in
[physics_engine.py](../digital_twin/physics_engine.py).

### Net effect on the engine

| | Before (retained) | After refit (rejected) |
|---|---|---|
| K_iz provenance | unattributed | **cited** (the one gain) |
| Internal consistency of the coefficient pair | matched | **mixed** |
| Independent L&L Fig. 3.17 check | **PASS** (61.56 V) | **FAIL** (36.84 V) |
| Sub-Module 1.6 checks | 6 / 9 | 4 / 9 |
| Test suite (non-dashboard) | 366 passed, 0 failed | 363 passed, 3 failed |
| `simulate()` runtime | ~24–27 µs/call | no measurable change |

One dimension improved, four degraded. Provenance is a property of the
*documentation*; agreement with published physics is a property of the *model*.
Trading the second for the first is the wrong trade for a physics engine — which is
why the change was reverted despite achieving exactly what it set out to achieve.

## Reproducing the figures

```bash
./.venv/Scripts/python.exe error_fixing_attempts/make_figures.py
```

Rebuilding the `.docx` / `.pptx` requires Node with the `docx` and `pptxgenjs`
packages; the generator scripts were run from a scratch directory and are not
checked in, since the rendered documents are the deliverables.
