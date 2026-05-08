# Distillation Model: Proposed Improvements

This document records proposed improvements to the D86 distillation simulation
code (`distillation.py`, `distillation_rk2_condenser.py`,
`distillation_rk2_condenser_optE.py`, `distillation_condenser_v2.py`).
Improvements are split into two categories: **computational speed** and
**code simplification**.  No code has been changed; this is a reference for
future work.

---

## 1. Computational Speed

Each RK2 step calls `_compute_derivatives_condenser_v2` twice.  Per call the
execution path is:

| Step | Operation | Cost |
|------|-----------|------|
| Stage 1 | Bisect on bubble-point residual | ~15–20 K-value evaluations |
| Stage 2 | Bisect (max 40 iter) × Rachford–Rice SS (max 40 iter each) | ~50–200 K-value evaluations |
| Stage 2b | One Rachford–Rice flash (condenser) | ~5–40 K-value evaluations |

Each K-value evaluation runs the full UNIFAC activity calculation for all
compounds, making Stage 2 the dominant cost.  At dt = 2 s and a target rate
of 4.5 mL/min, a full 100 mL run requires approximately 650 steps
(~1300 derivative evaluations) and on the order of 100 000–200 000 K-value
calls total.

### 1.1 Warm-start the Stage 2 bisect bracket (highest priority)

**Current behaviour.**  `solve_stage2_flash` searches for the neck
temperature $T_2$ over the full bracket `[T_room + 5, T1]`, which can span
200 K or more.  Bisect with `xtol = 0.1 K` therefore requires
$\log_2(200/0.1) \approx 11$ iterations even in the best case, each calling
a full Rachford–Rice flash.

**Proposed change.**  Carry the previous step's $T_2$ in state and initialise
the bisect bracket as `[T2_prev - delta, T2_prev + delta]` (e.g., `delta =
15 K`).  If the residual does not straddle a sign change in the narrow bracket,
fall back to the full range.

**Expected impact.**  Neck temperature changes by at most a few Kelvin per
step.  The narrow bracket should bracket in one check, reducing bisect
iterations from ~11 to ~2–3.  This alone could cut Stage 2 cost by 3–5×,
giving a wall-time reduction of roughly 50–70 % per run.

**Implementation effort.**  Two lines: thread `T2_prev` into
`solve_stage2_flash` (or `solve_stage2_flash_v2`) as an optional keyword
argument with default `None`, and use it to tighten the initial bracket when
available.

---

### 1.2 Reduce maximum iterations in Rachford–Rice and Stage 2 bisect

**Current behaviour.**
- `solve_rachford_rice`: `max_iter = 40` successive-substitution iterations.
- `solve_stage2_flash` bisect: `maxiter = 40`.

**Observed behaviour.**  For UNIFAC K-values (not SRK), successive
substitution typically converges in 5–8 iterations.  The remaining 32–35
iterations are wasted guard against a worst case that rarely occurs.  The
Stage 2 bisect with `xtol = 0.1 K` likewise converges in well under 20
iterations across the bracket.

**Proposed change.**  Reduce both limits:
- `solve_rachford_rice`: `max_iter = 15`.
- Stage 2 bisect: `maxiter = 20`.

Add a convergence diagnostic (a counter or warning) to verify in practice that
the tighter limits are never hit for the fuels of interest.

**Expected impact.**  Approximately 2× speedup on Stage 2 and the condenser
flash, with negligible accuracy loss for UNIFAC.

**Implementation effort.**  Two keyword-argument default changes.

---

### 1.3 Replace Stage 2 bisect+RR with a single-pass approximation (optional)

**Current behaviour.**  `solve_stage2_flash` solves for $T_2$ by iterating a
full Rachford–Rice flash inside a bisect loop — a nested solve.

**Proposed change.**  Approximate the vapour fraction $V_\text{fl}$ directly
from the energy balance, assuming constant $\Delta H_\text{vap}$ and $C_p$
over the neck:

$$
V_\text{fl} \approx 1 - \frac{h A (T_1 - T_\text{room})}{D_1 \Delta H_\text{vap}(T_1)}
$$

Then compute $T_2$ from a single energy-balance solve and run **one**
Rachford–Rice flash at the resulting $T_2$ to obtain compositions.  This
replaces the nested bisect+RR with two property evaluations and one flash.

**Expected impact.**  Eliminates the inner RR calls entirely from Stage 2,
reducing its cost to that of a single RR solve (~40 K-value calls).  Combined
with improvement 1.1, the total per-step cost could approach that of the
original no-condenser driver.

**Accuracy trade-off.**  The approximation assumes $Q_\text{loss} \ll D_1
\Delta H_\text{vap}$, which holds when neck heat loss is small relative to
the vapour enthalpy.  Under extreme conditions (very low flow, very long neck)
this may not hold.  Validation against the full bisect solution is recommended
before adoption.

**Implementation effort.**  Medium — new function or optional code path inside
`solve_stage2_flash`.

---

## 2. Code Simplification

### 2.1 Fix `stall_vol_tol_mL` default in the simulation drivers

**Problem.**  Every simulation driver (`run_d86_simulation`,
`run_d86_simulation_rk2_condenser`, `run_d86_simulation_rk2_condenser_optE`,
`run_d86_simulation_rk2_condenser_optE_v2`) defaults
`stall_vol_tol_mL = 1e-4 mL`.  A trial with a near-zero distillation rate
(e.g., 0.001 mL/min) advances the distillate volume by 0.0001 mL every
6 seconds, continuously resetting the stall timer and preventing termination.
In practice this caused a calibration run to hang for 10+ hours.

**Proposed change.**  Change the default to `stall_vol_tol_mL = 0.5 mL` in
all four drivers.  At the target rate of 4.5 mL/min the distillate advances
22 mL in the 300 s stall window — far above the threshold.  Only trials with
a rate below ~0.1 mL/min (2 % of target) will stall.

**Implementation effort.**  One line per driver (four files).

---

### 2.2 Merge `solve_stage2_flash_v2` into `solve_stage2_flash` via a flag

**Problem.**  `solve_stage2_flash_v2` (in `distillation_condenser_v2.py`) is
identical to `solve_stage2_flash` (in `distillation.py`) except for one line:
the latent heat $\Delta H_\text{vap}$ is evaluated at the reflux composition
`liq_comp` rather than the feed `z`.  Maintaining two near-identical functions
means any future bug fix or enhancement must be applied in two places.

**Proposed change.**  Add a boolean keyword argument to the original:

```python
def solve_stage2_flash(..., lv_at_reflux: bool = False) -> tuple:
```

When `lv_at_reflux=True`, evaluate $\Delta H_\text{vap}$ at `liq_comp` (the
Issue 6 fix); when `False`, use the original behaviour.

**Implementation effort.**  Small refactor — add the flag, move four lines,
delete `solve_stage2_flash_v2`.

---

### 2.3 Replace large return tuples with a named structure

**Problem.**  `_compute_derivatives_condenser` returns an 11-element tuple;
`_compute_derivatives_condenser_v2` returns a 14-element tuple.  Callers must
unpack by position, so a reordering silently breaks everything.  The tuples
also carry ambiguous names (`Dtot`, `Dliq`, `ccomp`, etc.) that require
reading the source to interpret.

**Proposed change.**  Replace with a `dataclass` or `typing.NamedTuple`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class DerivativeResult:
    dN_dt:       np.ndarray  # mol/s per component
    T1:          float       # pot bubble-point temperature, K
    T2:          float       # neck temperature, K
    R2:          float       # reflux molar flow, mol/s
    D2:          float       # forward vapour to condenser, mol/s
    vapor_comp2: np.ndarray  # neck-exit vapour composition
    T_cond:      float       # condenser outlet temperature, K
    D_cond_tot:  float       # total condenser outlet flow, mol/s
    D_cond_liq:  float       # condensed liquid flow, mol/s
    cond_comp:   np.ndarray  # condenser outlet composition
    reflux_comp: np.ndarray  # neck reflux composition
    D1:          float       # vapour from pot, mol/s
    vapor_comp1: np.ndarray  # pot vapour composition
```

Callers use `result.T2` instead of `result[2]`.

**Implementation effort.**  Medium — define the dataclass, update
`_compute_derivatives_condenser_v2` and its two call sites in the driver loop.

---

### 2.4 Remove the `n_air` legacy parameter from `solve_stage3_cstr`

**Problem.**  `solve_stage3_cstr` accepts `n_air_old` (a historical moles-of-air
carry variable) and returns it unchanged.  The parameter has not been used in
any computation since the CSTR was refactored to the lumped-capacitance form.
Every call site carries the dead variable through.

**Proposed change.**  Remove `n_air_old` from the signature and return only
`T_D86`.  Update call sites to drop the second return value and the `n_air`
state variable.

**Implementation effort.**  Small — two-line signature change, update three
call sites across two drivers.

---

### 2.5 Consolidate to one canonical driver

**Problem.**  There are four simulation drivers, each a near-complete copy of
the main loop with minor variations:

| Driver | File | Integration | Controller |
|--------|------|-------------|------------|
| `run_d86_simulation` | `distillation.py` | Euler | PI |
| `run_d86_simulation_rk2_condenser` | `distillation_rk2_condenser.py` | RK2 | PI |
| `run_d86_simulation_rk2_condenser_optE` | `distillation_rk2_condenser_optE.py` | RK2 | Option E |
| `run_d86_simulation_rk2_condenser_optE_v2` | `distillation_condenser_v2.py` | RK2 | Option E + fixes |

The distillate volume calculation, pot volume update, stall detection, and
recording logic are duplicated verbatim.  Any bug in shared logic (such as
`stall_vol_tol_mL`) must be fixed in four places.

**Proposed change.**  Designate `run_d86_simulation_rk2_condenser_optE_v2` as
the canonical driver.  Retain the others for backward compatibility but mark
them deprecated.  Longer term, extract the shared loop body into a single
internal function parameterised by the derivative function and controller type.

**Implementation effort.**  Large — involves refactoring the main loop, but
does not change any physics.  Best done incrementally once the physics are
stable.

---

### 2.6 Reconcile bisect tolerances across the codebase

**Problem.**  Bisect tolerances are set inconsistently and without documented
rationale:

| Call site | `xtol` | `maxiter` |
|-----------|--------|-----------|
| Stage 1 bubble-point | `1e-4 K` | default (100) |
| Stage 2 neck flash | `0.1 K` | 40 |
| Rachford–Rice inner bisect for V | `1e-8` | 200 |
| Condenser (original, now replaced) | `0.1 K` | 40 |

The inner RR bisect uses a tolerance of `1e-8` with `maxiter=200`, making it
far tighter than the outer SS loop that only requires `rtol=1e-5` on K-values.

**Proposed change.**  Adopt a single documented tolerance policy:
- Temperature solves (Stage 1, Stage 2): `xtol = 0.05 K`, `maxiter = 30`.
- Rachford–Rice V bisect: `xtol = 1e-6`, `maxiter = 50` (consistent with K-value SS tolerance).

Document the rationale in a module-level docstring.

**Implementation effort.**  Small — change four constant values and add a
comment.

---

## Priority Summary

| Priority | Item | Effort | Expected Benefit |
|----------|------|--------|-----------------|
| 1 | Warm-start Stage 2 bisect bracket (§1.1) | 2 lines | 50–70 % wall-time reduction |
| 2 | Reduce RR and bisect `max_iter` (§1.2) | 2 lines | ~2× additional speedup |
| 3 | Fix `stall_vol_tol_mL` default in drivers (§2.1) | 4 lines | eliminates infinite-run bug |
| 4 | Merge `solve_stage2_flash_v2` via flag (§2.2) | small | single source of truth |
| 5 | Replace return tuples with dataclass (§2.3) | medium | maintainability |
| 6 | Remove `n_air` from Stage 3 (§2.4) | small | clarity |
| 7 | Reconcile bisect tolerances (§2.6) | small | consistency |
| 8 | Single-pass Stage 2 approximation (§1.3) | medium | large speedup, needs validation |
| 9 | Consolidate to one driver (§2.5) | large | long-term health |
