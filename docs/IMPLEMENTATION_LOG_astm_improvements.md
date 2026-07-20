# Implementation log — astm branch improvement campaign

Running log for the review-driven improvements (plan: `docs/ASTM_BRANCH_REVIEW.md`,
execution order §5). Newest entries at the bottom. Every deviation from the plan is
called out with **DEVIATION**. Test runner used during development: the pixi env of the
companion inverseDesignSAF repo
(`.../inverseDesignSAF/.pixi/envs/default/bin/python`, numpy/pandas/scipy/jax) — the
repo's own conda env instructions are unchanged.

---

## 2026-07-14 — ASTM-1: Derived Cetane Number v1 (COMPLETE)

**Added**
- `tools/build_dcn_table.py` → `gcmTableData/dcn.csv` (89 bins, schema mirrors
  `das_2018_ysi.csv`: DCN, DCN_err, provenance `Source` tag per row).
- `source/FuelLib.py`: `_dcn_mix(phi, dcn_i)` pure helper (JAX-portable),
  constructor loading (`self.dcn_pure/dcn_err/dcn_source`, bin→formula fallback
  lookup identical to YSI), and `fuel.dcn(Yi, T_ref=288.15)` — linear blending in
  **liquid volume fraction** (mass → volume via per-compound `density(T_ref)`).
- `tests/test_dcn.py` (9 tests: table completeness/bounds/anchors/family-ordering/
  provenance; mixture validation vs published NJFCP DCNs; blending monotonicity;
  volume-vs-mass basis guard; helper purity).
- `tests/test_jax_compat.py`: `_dcn_mix` jit+grad test (gradient == dcn vector).
- `tutorials/astmProperties.py`: DCN row + reference values.

**Data policy (important):** the 14 seed DCNs are hand-transcribed literature
anchors tagged `literature_seed_VERIFY_vs_NREL_compendium` — they are pending
line-by-line verification against the NREL Compendium of Experimental Cetane
Numbers (NREL/TP-5400-67585). Anchors: n-C7 53.8, n-C10 65.8, n-C12 73.7,
n-C16 100 (definition), MCH 21, butylcyclohexane 47, decalin 36, toluene 9,
butylbenzene 14, hexylbenzene 26, naphthalene/1-MN 3, tetralin 11, JP-10-analog
C10-tricyclo 21. Everything else is derived (family fit / offset rule /
holdlargest) and tagged accordingly. Global cap 105 (IQT response flattens
above n-C16; the linear n-alkane fit would otherwise reach 134 at n-C23).

**Mixture validation** (published targets: A-1 48.8, A-2 48.3, A-3 39.2, C-1 17.1;
sources: Edwards "Reference Jet Fuels for Combustion Testing" and NREL
fy24osti/89581 — fuel identities A-1=posf10264/JP-8, A-2=posf10325/Jet A,
A-3=posf10289/JP-5, C-1=posf11498/Gevo ATJ confirmed via NJFCP overview papers):

| fuel | predicted | measured | Δ |
|---|---|---|---|
| posf10264 (A-1) | 51.6 | 48.8 | +2.8 |
| posf10325 (A-2) | 51.4 | 48.3 | +3.1 |
| posf10289 (A-3) | 50.8 | 39.2 | +11.6 |
| posf11498 (C-1) | 56.1 | 17.1 | +39.0 (deliberate `expectedFailure`) |

**DEVIATION 1 (calibrated iso-alkane archetype).** The first build used the
2-methylalkane offset (n-alkane − 12) matching the bins' named reference
compounds; A-1/A-2 then over-predicted by ~9 (family diagnosis: iso mean DCN 66,
mono-cyclo mean 60+). Real jet "Cx-Isoparaffin" bins are multi-methyl-branched,
so the offset was changed to **n-alkane − 25 ± 10**, tagged
`offset_rule_jet_multibranched_CALIBRATED` — explicitly a calibrated archetype
choice (informed by matching A-1/A-2), not a blind prediction. The mono-cyclo
extrapolation was likewise changed from a single steep two-point slope (8.7/C,
unphysical past C13) to piecewise: steep MCH→butylcyclohexane, then the n-alkane
slope (remote-ring CH2 additions behave like chain CH2). After both: A-1/A-2
within ~3.

**A-3 residual (+11.6):** JP-5 is cycloparaffin-rich and the cyclo-family DCNs
rest on two seeds + offsets. Documented tolerance ±15 in the test; tightening
requires more measured cycloalkane DCNs (data, not code).

**C-1 (+39.0) — the designed failure.** The C12-Isoparaffin bin's reference
compound is 2-methylundecane (lightly branched); real C-1 is
~2,2,4,6,6-pentamethylheptane (DCN ~17–24). `test_dcn.py` carries this as
`expectedFailure` with instructions to flip it to a passing assert when ASTM-4
re-maps the posf11498 decomposition. This is the quantitative demonstration of
review finding §2.2.

**Also found (flagged, NOT fixed here):** `tutorials/astmProperties.py` fuel-type
labels disagree with NJFCP designations (posf10264 labeled "Jet A" vs NJFCP JP-8;
posf10289 "JP-8" vs JP-5; posf10325 "JP-5" vs Jet A; posf11498 "HEFA-SPK" vs Gevo
ATJ — its 78 wt% C12-isoparaffin gcData is unambiguously ATJ). The associated
freeze/flash reference values may be attached to the wrong fuels. Left untouched
with a warning comment; audit scheduled under ASTM-4.

**Tests:** full suite green (`test_accuracy`, `test_api`, `test_pure_components`,
`test_unifac`, `test_jax_compat` [1 pre-existing expectedFailure: psat jit],
`test_dcn` [1 designed expectedFailure: C-1], `test_source_docstrings`), exit 0.

---

## 2026-07-14 — ASTM-2: Family-resolved fusion thermodynamics (COMPLETE)

**DEVIATION from plan (data access):** the plan's first choice was transcribing the
Naef 2019 group contributions into the reserved `naef_*` columns. That paper's group
tables are not accessible for faithful transcription (transcribing 78 coefficients
from memory would be fabricating data), so the plan's named fallback was implemented:
**per-family linear fusion-entropy correlations**, `dSfus = A + B*(n_C − C_ref)`
(`gcmTableData/fusion_families.csv`). The n-alkane series is anchored to NIST
dHfus/Tm data (C7 77 / C10 118 / C12 140 / C16 183 J/mol/K; odd-even alternation
not modeled); other families are anchored where single-compound data exists
(MCH 46, decalin ~50, toluene 37, naphthalene 54) and estimated otherwise —
provenance in the CSV's `Source_note`. Family classification reuses the DCN/YSI bin
taxonomy (`self.bin_family`, bin→formula lookup); Walden 56.5 remains only as the
fallback for unclassified compounds; global floor 20 J/mol/K.

**Result, per-compound:** dHfus now within ~5-15% of NIST (n-C7 13.5 vs 14.0,
n-C12 32.3 vs 36.8, n-C16 49.3 vs 53.4 kJ/mol) — the residual gap is exactly the
CG Tm under-prediction (dHfus = dSfus·Tm), scheduled for ASTM-3.

**CRITICAL FINDING — alpha and Walden were a coupled pair.** Swapping in physical
dSfus with the historic `alpha=0.25` pushed every freeze point UP by +26..+41 K
(posf10325: 259 K vs 226 ref). Root cause, from the small-x expansion of eq 21's
mixing term: `alpha*dS_mix ≈ −alpha·R·ln(x)`, while classical ideal SLE
(`ln x = −dHfus/R (1/T − 1/Tm)`) requires exactly `−R·ln(x)` — i.e. **alpha = 1**
for an ideal solution. The historic 0.25 under-weighted the depression by 4x and
was compensated by Walden's ~2.5x-too-small dSfus. Changed the `freeze_point`
default to `alpha=1.0` with the full reasoning in the docstring (0.25 remains
callable and is documented as Walden-paired).

**Freeze validation after the change (alpha=1, physical dSfus):**
pure compounds now satisfy the exact identity freeze(pure) == CG Tm (heptane
175.5 vs Tm 175.6), so the pure-compound error IS the CG Tm error (decane −26 K,
dodecane −25 K vs NIST). Mixtures inherit it: posf10264 201.7 (ref 226),
posf10325 208.5 (226), posf10289 202.6 (219), posf11498 222.5 (240) — uniformly
~−16..−24 K, i.e. the freeze model is now *structurally correct with a known
input bias*, unlike before where two wrong constants canceled unpredictably.
`tests/test_fusion.py` (7 tests) asserts the NIST dHfus agreement, family
ordering, the pure-compound identity, and an INTERIM mixture band [ref−30, ref+3]
explicitly marked for tightening to ~±8 K when ASTM-3 lands.

**Also noted:** `test_pure_components`' YSI block is circular (reference file's YSI
column was built from the same Volume-2 table the model reads → 0.00% MAPE by
construction). Its other property columns (LHV/Cl/FP/freeze vs NIST) are real
comparisons. And its `FreezePoint_K` column IS per-compound experimental Tm for
~30 compounds — the exact anchor data ASTM-3 needs, already in-repo.

**Tests:** full suite + test_fusion green, exit 0.
