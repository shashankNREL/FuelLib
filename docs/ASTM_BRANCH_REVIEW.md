# Brutal review: FuelLib `astm` branch — accuracy, SAF generalizability, and a DCN plan

**Subject:** FuelLib branch `astm` (commit `1df3155`, frozen in `reference/FuelLib.py`):
the five ASTM property models (LHV, YSI, flash, freeze, liquid Cp), the extended GCM
table, the YSI table pipeline, and the UNIFAC 2.0 activity layer.

**Evidence base:** the whole branch was frozen, re-tested, and ported to JAX in this repo;
every property was parity-validated and gradient-checked. Data claims below (zero Naef
columns, YSI source counts, ATJ reference compounds) were verified against the snapshot
files, not assumed.

**Status: review + plan only. No code changes. Awaiting approval.**

---

## 1. Credit where due

This branch is the best-engineered part of FuelLib:
- Module-level pure array helpers (`_lhv_hess`, `_boehm2022_iter`, ...) with thin method
  wrappers — deliberately JAX-portable, with `test_jax_compat.py` enforcing it (mostly).
- `tests/test_pure_components.py`: per-family MAPE regression against NIST for ~30
  reference compounds. This is real validation discipline; most GC codebases have none.
- The YSI table carries per-bin provenance tags (`measured_*` / `extrapolated_*` / ...).
- The extended-table architecture (`Nij @ get_ext_row(...)`) makes adding a new
  per-compound property a data change, not a code change — which is exactly why the DCN
  proposal in §4 is cheap.

Now the brutal part.

---

## 2. Brutal findings

### 2.1 The Naef fusion columns are empty — freeze point runs on a placeholder (CRITICAL for SAF)
`gcmExtendedTable.csv` contains `naef_Cp_sol_298, naef_Cp_liq_298, naef_dHfus,
naef_dSfus` columns — **verified all-zero**. The code silently falls back to Walden's
rule: `dSfus = 56.5 J/mol/K` for *every* compound, `dHfus = 56.5·Tm`.

Reality check: n-C12 has ΔHfus ≈ 36.8 kJ/mol at Tm = 263.6 K; Walden gives 14.9 kJ/mol —
**2.5× too low**. Globular branched isomers (the ATJ backbone) err in the *opposite*
direction (plastic-crystal phases, ΔSfus far below Walden). Because the freeze model is a
max-over-compounds, the winning compound's fusion error IS the fuel's freeze error — and
for paraffinic SAFs (HEFA, FT, ATJ), freeze point is the *binding* D7566 constraint.
The docstring's own estimate ("10–20 K worse than Naef") is the difference between a
usable and unusable freeze model for SAF screening.

### 2.2 Reference-compound fidelity: the ATJ smoking gun (CRITICAL for SAF)
`posf11498_init.csv` (NJFCP C-1, a Gevo-type ATJ) is 78.3 wt% "C12-Isoparaffin" whose
reference compound is **2-methylundecane** — a lightly-branched archetype. Real C-1 is
dominated by **2,2,4,6,6-pentamethylheptane**. Consequences of that single mapping:
- Tb: ~483 K (2-methylundecane) vs ~451 K (pentamethylheptane) → +30 K bias in every
  volatility-driven property (D86, flash, psat).
- DCN: ~70 (2-methylundecane) vs ~17 (pentamethylheptane) — not an error bar, a
  different fuel.
- Freeze: mono-methyl vs penta-methyl crystallization behavior differs qualitatively.

The bin taxonomy carries one lightly-branched archetype per (family, C#) because that is
what POSF fuels look like. **Any SAF whose isomer distribution differs from Jet-A
convention is silently remapped onto Jet-A-like molecules.** The infrastructure already
supports per-fuel decompositions (each fuel has its own `groupDecompositionData` CSV) —
this is a data fix, not an architecture fix, but until it's made, "works for SAF" claims
are unfounded for exactly the fuels (ATJ, HEFA-SPK) people will ask about.

### 2.3 CG Tm (and Tb) should not be trusted where they matter most
Melting point depends on molecular symmetry and crystal packing — information group
counts fundamentally cannot see (no odd–even alternation for n-alkanes, no globular
anomalies). The CG form `Tm = 102.425·ln(ΣN·tmk)` feeds both the freeze solve *and*
`dHfus = dSfus·Tm`, compounding §2.1. Same story, milder, for Tb (indane: CG 504 K vs
exp 450 K) which feeds psat → flash and D86. Both `_init.csv` files already name a
reference compound per bin — experimental Tb/Tm anchors are one CSV column away.

### 2.4 YSI: 61% of the table is not a measurement
Verified source counts over the 89 bins: **35 measured, 33 extrapolated, 16
hold-largest, 5 cross-filled**. The per-family linear-in-carbon extrapolation is
reasonable for interpolation but is doing >half the work, the `YSI_err` column is unused
downstream, and `fuel.ysi()` raises `NotImplementedError` when any NaN-YSI compound has
nonzero mass — hostile to optimization loops (inverseDesignSAF had to fill + flag at
build time). No per-bin uncertainty ever reaches the user.

### 2.5 The JAX-portability claim is only ~70% true
`test_jax_compat.py` marks `_psat_lee_kesler` `@expectedFailure` under jit (np/jnp
dispatch pending), and `_fp_liaw_ideal_iter` / `_boehm2022_iter` / `_freeze_max_over_j`
contain `float()` casts, Python `max()`, and a per-compound list comprehension that break
tracing. All were rewritten in the inverseDesignSAF port (bracketed+IFT flash solve,
vmapped freeze, guards via `jnp.where`) — back-porting is copy-paste work, and would make
the branch's stated design contract actually hold.

### 2.6 Assorted brittleness (small, cheap to fix)
- Heteroatom guard in `heat_of_combustion` hardcodes group index ranges (15..51, 54..77)
  — derive from table metadata or a named constant.
- `activity()` raises `FileNotFoundError` for fuels without a UNIFAC decomposition —
  fine, but the error surfaces mid-distillation rather than at construction.
- UNIFAC runs the full 113-subgroup basis; hydrocarbon fuels use ~10. Compressing to
  used columns at `__init__` is exact and was measured at ~60× on the activity kernel.
- `mixing_rule` is an O(n²) Python double loop (off the ASTM critical path, still ugly).
- Liquid Cp (Ruzicka-Domalski) drift documented at +2% (−10 °C) → +13% (160 °C) — the
  docstring diagnosis (iso/cyclo d(Cp)/dT contributions) is good; the fix is still open.
- Mixture-level validation only asserts on conventional fuels; `hefa-*`, `jet-a`, and
  `posf11498` data files exist on the branch but nothing asserts against them.

---

## 3. Low-hanging improvements, ranked (accuracy × SAF-generalizability ÷ effort)

| # | Action | Effort | Payoff |
|---|--------|--------|--------|
| 1 | **Populate the Naef fusion columns** (transcribe Naef 2019 group values into the already-plumbed `naef_*` columns; keep Walden as fallback). If Naef stays inaccessible: per-family ΔSfus constants fitted to NIST ΔHfus data — even a two-parameter (n-alkane / branched) split beats one Walden constant. Validate per-family vs NIST fusion data. | 1–2 d | Freeze point becomes credible for paraffinic SAFs — the binding spec |
| 2 | **Anchor Tb and Tm per bin to reference-compound NIST values** (add `exp_Tb`, `exp_Tm` columns; prefer over CG at construction; keep CG for unmatched bins). One data change improves flash, freeze, psat, and D86 simultaneously. | 1–2 d | Largest single accuracy lever across four properties |
| 3 | **Fix SAF decompositions**: re-map posf11498 bins to true ATJ isomers (pentamethylheptane family), audit `hefa-*` bins the same way, and add mixture-level CI assertions for one HEFA + one ATJ against published D7566/NJFCP data. | 2–3 d | Turns "works for POSF" into "works for SAF", with proof |
| 4 | **YSI hygiene**: propagate per-bin σ (measured `YSI_err` vs inflated σ for extrapolated/crossfilled), replace the NaN raise with warn+fill+flag. | 0.5 d | Honest sooting channel; optimizer-safe |
| 5 | **UNIFAC subgroup compression** at construction (exact; ~60× on γ). | 0.5 d | Speed for every VLE consumer |
| 6 | **Make JAX-portability true**: np/jnp shim for psat, de-`float()` the flash/freeze helpers (port back from inverseDesignSAF), clear the `expectedFailure`. | 0.5–1 d | Contract honesty; enables direct reuse |
| 7 | Metadata-driven heteroatom guard; construction-time UNIFAC file check; O(n²) `mixing_rule` → einsum. | 0.5 d | Robustness |

Items 1–3 are data work more than code work — which is exactly why they're low-hanging:
the architecture already supports them.

---

## 4. Derived cetane number (DCN): yes, and it fits the existing pattern exactly

DCN (ASTM D6890/D7668) is the *most valuable* channel you could add, for two reasons:
(a) it's a D7566/D4054 screening property, and (b) — measured in the inverseDesignSAF
noise tests — the current 11 channels **cannot distinguish paraffin branching classes**
(iso-alkanes swung 30→53 wt% under 1σ target noise). DCN is nearly orthogonal to every
existing channel precisely because it is savage about branching: n-C12 ≈ 73,
pentamethylheptane ≈ 17, aromatics < 10. One DCN row in the Jacobian collapses the
worst degeneracy in the inverse problem.

### v1 — lookup table + linear volume blending (recommended start, ~2–3 days)
Mirror the YSI implementation piece for piece:
1. **Data**: NREL *Compendium of Experimental Cetane Numbers* (Yanowitz, Ratcliff,
   McCormick et al., NREL/TP-5400-67585 + updates) — hundreds of pure-HC IQT DCNs, and
   it's an in-house NREL product. Supplement with Dahmen & Marquardt (2015) where needed.
2. **`tools/build_dcn_table.py`** ← clone of `build_ysi_table.py`: match the 89
   refCompounds (name → formula fallback), per-family carbon-number extrapolation,
   `Source` tags, `DCN_err` column from IQT reproducibility (±~1 DCN at 40–55, worse at
   extremes).
3. **`fuel.dcn(Yi)`**: linear **volume-fraction** blending
   `DCN = Σ φᵢ·DCNᵢ`, φᵢ from Yi/ρᵢ(15 °C) normalized — the literature default and the
   same differentiable dot-product shape as `ysi()`. Pure helper `_dcn_mix(phi, dcn_i)`
   + thin wrapper, jit test in `test_jax_compat.py`, rows in
   `pureComponentReference.csv`, per-family MAPE in `test_pure_components.py`.
4. **Mixture validation**: NJFCP measured DCNs (approx.: A-2/posf10325 ≈ 48.8,
   A-1/posf10264 ≈ 48.3, A-3/posf10289 ≈ 39.2, C-1/posf11498 ≈ 17.4 — confirm from
   Edwards 2020 tables before committing). **Prediction:** C-1 will fail badly until
   §3-item-3 (real ATJ isomers) is done — i.e., DCN doubles as the sharpest diagnostic
   for reference-compound fidelity. That's a feature: assert A-fuels tightly, xfail C-1
   with a comment until the decomposition fix lands.

### v2 — nonlinear blending (only if binary-blend residuals demand it)
Cetane blending is mildly non-linear (aromatic antagonism, improver-like synergies).
Ghosh-style β-weighted blending, `DCN = Σφᵢβᵢ·DCNᵢ / Σφᵢβᵢ` with per-family β calibrated
on binary-blend data, stays a smooth rational function — still trivially differentiable,
+1 day once blend data is in hand.

### v3 — GC-QSPR fallback for unmeasured compounds
A group-contribution DCN model (e.g., Dahmen-Marquardt GC-QSPR or NREL's Kubic 2017)
evaluated from the existing `Nij` fills table gaps with something better than
family-linear extrapolation; measured values always take precedence. This is the same
`Nij @ row` pattern as everything else in the branch.

### Inverse-design integration (one afternoon, in inverseDesignSAF)
Add `dcn` to `FuelTables` (per-bin constants) + a `dcn` channel in `forward_map.py`
(volume-weighted dot product), σ ≈ 1–2 DCN. Re-run the noise study — the expected
result is the iso-alkane swing collapsing from ~23 wt% to a few wt%, which would be the
quantitative proof that the channel was worth adding.

---

## 5. Suggested execution order

1. DCN v1 (table + linear blend + tests) — independent of everything else, immediate
   inverse-design payoff.
2. Naef fusion transcription (or per-family ΔSfus interim) + freeze validation.
3. Tb/Tm reference anchoring.
4. SAF decomposition fixes (posf11498, hefa-*) + mixture-level SAF CI assertions —
   then re-check DCN on C-1 as the acceptance test.
5. YSI σ propagation + NaN policy.
6. UNIFAC compression + JAX-portability backports + small brittleness fixes.

*Written 2026-07-14. Verified facts: Naef columns all-zero; YSI sources 35/33/16/5
(measured/extrapolated/holdlargest/crossfill); posf11498 = 78.3 wt% C12-Isoparaffin →
"2-methyl undecane"; NJFCP/Edwards DCN values marked "confirm before committing".*
