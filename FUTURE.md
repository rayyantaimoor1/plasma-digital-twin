# Future Work Backlog

Live holding list so ideas don't get lost between sessions. The previous version
of this file (items 1-5) is fully resolved - see "Completed" at the bottom for the
one-line record.

---

## 1. Rate-coefficient provenance: K_iz and K_exc must be re-sourced AS A PAIR

**Status: attempted, reverted, root cause understood. Do not retry naively.**

`ionization_rate_coeff` (K_iz) and `excitation_rate_coeff` (K_exc) in
[physics_engine.py](digital_twin/physics_engine.py) are both *unattributed*
"representative" Arrhenius fits. Their constants have no traceable published
origin, which is a genuine (if honest) weakness in the validation story.

### What was tried (2026-07-25)

Refit K_iz alone against a properly sourced, independent, verified reference:

> G. S. Voronov, "A practical fit formula for ionization rate coefficients of
> atoms and ions by electron impact: Z = 1-28," *Atomic Data and Nuclear Data
> Tables* **65**, 1 (1997). DOI 10.1006/adnd.1997.0732.
> Ar I parameters (dE=15.8 eV, A=5.99e-8 cm^3/s, P=1, X=0.1360, K=0.26, valid
> 1 eV - 20 keV), read first-hand from the machine-readable table at
> <https://www.pa.uky.edu/~verner/col.html>.

Procedure was methodologically clean: Te fit window fixed at **1-10 eV before any
fitting**, fit performed once in log space, no post-hoc tuning. Resulting fit
`K_iz = 3.5744e-14 * Te^0.39170 * exp(-15.7288/Te)` reproduced Voronov to 0.15%
max relative error (R^2 = 0.99999999), and the fitted threshold (15.7288 V)
independently landed within 0.2% of argon's true ionization potential (15.76 V) -
a good sign the fit itself was sound.

### Why it was reverted

1. **It broke an independent literature check that previously PASSED.**
   `test_collisional_energy_loss_matches_lieberman_lichtenberg_fig_3_17`:
   E_c(3 eV) fell 61.5 V -> 36.8 V, outside L&L Fig. 3.17's published argon band
   of 50-70 V. That is a real regression, not merely a benchmark number moving.
2. Sub-Module 1.6 benchmark went **6/9 -> 4/9** passing.

### THE ACTUAL LESSON (this is the reusable knowledge)

**E_c depends only on the RATIO K_exc/K_iz**, via
`E_c = E_IZ + E_EXC*(K_exc/K_iz) + (elastic term)`. Refitting one coefficient
while leaving the other as the old unattributed fit collapsed that ratio from
3.765 to 1.733 at 3 eV. E_T then dropped 83 -> 58 eV, and since **n_e is
proportional to 1/E_T**, both density checks blew out (+41.7% -> +101.7%, and
+4.4% -> +48.6%). Separately, a larger K_iz means a lower Te suffices to balance
ionization against loss, so solved Te shifted 3.29 -> 2.86 eV, moving *away* from
the PIC reference.

Critically: the two old fits were **wrong in the same direction**, so their ratio
was roughly right and E_c came out plausible *for the wrong reasons*. Fixing one
in isolation unmasked that. A cited K_iz paired with an uncited K_exc is
**internally inconsistent - worse than two consistently-wrong fits.**

### If picked up again

- Source K_iz **and** K_exc from comparable references, refit **both**, and check
  `E_c(Te)` against L&L Fig. 3.17 (50-70 V at 3 eV) *as an acceptance gate*
  before looking at the Sub-Module 1.6 benchmark at all.
- Voronov covers **ionization only** - a separate source is needed for excitation.
- Note Voronov may be the wrong source family for this window regardless: it is a
  general-purpose astrophysical fit spanning Z=1-28 and 1 eV-20 keV, optimised
  across that whole range, and at 3 eV it gives 2.90e-16 m^3/s - *above all three*
  plasma-processing anchors already in the benchmark (Turner 2.20, Jimenez-Redondo
  1.91, Chabert 1.49, all e-16). Prefer a source from the low-temperature
  plasma-processing literature for the 2-6 eV CCP regime.
- Keep the discipline that worked: fix the fit window before fitting, fit once,
  never tune constants after seeing benchmark results.

## 2. A cleanly-independent 3rd/4th reference source for Sub-Module 1.6

Sub-Module 1.6 currently benchmarks against three sources (Turner 2014;
Jimenez-Redondo et al. 2014; Chabert & Braithwaite 2011 via Powis et al. 2025).
Only the first two are genuinely independent of the Lieberman & Lichtenberg /
Godyak lineage the engine itself descends from - Chabert is flagged in-code as a
same-lineage consistency anchor, not an independent measurement.

**Hard constraint, non-negotiable:** any added reference value must come from an
actually looked-up, verifiable, citable published source - never a number recalled
from memory. Verify by reading the source document and quoting the table/equation.

Note the train/test separation: **Voronov (1997) can now only be used as a refit
target OR a benchmark anchor, never both.** If item 1 is ever completed using
Voronov, it is consumed and cannot also serve as a fourth anchor.

Fetching notes for this environment: WebFetch cannot parse compressed-stream PDFs
(arXiv `/pdf`, journal PDFs) and there is no poppler locally, so saved PDFs can't
be rendered either. What works: PubMed Central full-text HTML, arXiv **native**
HTML (`arxiv.org/html/<id>vN`, ~2023+ papers only), plain-text data files fetched
with `curl`. MDPI article pages return HTTP 403.

## 3. Optional CSV exports not yet added

Item 3 of the old backlog is done for the tables that mattered (Session History's
comparative-suitability ranking; Model Training's evaluation-metrics,
significance-test and cross-validation tables). Two were listed as optional and
were **not** added - pick up only if wanted:
- Correlation sweep table ([5_Trends_and_Correlation.py](dashboard/pages/5_Trends_and_Correlation.py))
- Suitability scorecard table ([6_Suitability_and_Recommendations.py](dashboard/pages/6_Suitability_and_Recommendations.py))

## 4. Reactor Control Room - possible enhancements

The companion app ([reactor_control_room/](reactor_control_room)) is built and
working (backend + 4 pages). Ideas discussed but not built:
- **SHAP feature-contribution bars on the AI Verdict page** - explain *why* the
  classifier chose a class, not just what it chose. `explain_configuration` in
  [classification.py](ai_module/classification.py) already exists, so this is
  another thin wrapper endpoint. Cost: needs a cached SHAP background sample and
  adds latency per operating-point change.
- **Show the XGBoost verdict alongside Random Forest** - both are already trained
  in `get_classifiers()`, so it is one extra `classify_configuration` call and no
  new endpoint. Gives an at-a-glance "do the two models agree?" signal.
- CORS on the backend is currently wide open (`allow_origins=["*"]`, read-only
  GETs, local demo API). Tighten to specific localhost origins if that ever
  matters.

---

## Completed (previous backlog items 1-5)

- **Reactor Control Room companion app** - FastAPI backend wrapping existing
  functions only (never reimplementing physics/AI in JS), shared component library
  (chamber / gauge / comparison bar / sparkline), and all 4 pages (Reactor View, AI
  Verdict, Physics Validation, Session Replay). Design reference kept in
  [mockups/reactor-control-room/](mockups/reactor-control-room).
- **One-click launchers** - `run_streamlit.bat`, `run_reactor.bat`, `run_both.bat`.
  (All use `ping` not `timeout` for startup delays - `timeout` needs a real console
  input handle and fails in some launch contexts.)
- **CSV export** - Session History ranking + Model Training metrics / significance
  / cross-validation tables.
- **Physics Validation promoted to its own dashboard page** -
  [2_Physics_Validation.py](dashboard/pages/2_Physics_Validation.py), with pass-rate
  gauge, deviation-vs-tolerance bars and click-to-drill citations.
- **Two additional Sub-Module 1.6 reference sources** - Jimenez-Redondo et al.
  (2014) and Chabert via Powis et al. (2025); benchmark went from 1 to 3 sources.
  See item 2 above for what remains.
