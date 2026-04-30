# Tasks Completed

This document records the concrete work that has landed on the
`copilot/jax-integration-adjoint-study` branch. Each item maps back to a
workstream in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).

## Summary

- **New subpackage:** `source/jax_diff/` (5 implementation modules + `__init__.py`).
- **New test suite:** `tests/jax_diff/` with **73 tests** (64 fast + 9 slow).
- **Hardening:** new `jax` pytest marker, package README, this docs set.
- **Numpy reference untouched** — the production code paths and their
  existing test suite continue to pass without modification.

## W1 — Infrastructure ✅

- [x] Created `source/jax_diff/__init__.py` with explicit re-exports of
      every public symbol used downstream.
- [x] `properties_jax.FuelTables` NamedTuple with frozen `jnp.ndarray`
      snapshots of all per-compound arrays needed by the property and
      VLE routines.
- [x] `build_fuel_tables(fuel)` factory that snapshots a numpy `fuel`
      object once at startup so per-call traces don't touch numpy.
- [x] Float64 enabled at import time
      (`jax.config.update("jax_enable_x64", True)` in `__init__.py`).
- [x] `pytest.ini` registers a `jax` marker.
- [x] `tests/jax_diff/conftest.py` provides shared fuel fixtures
      (`posf10325`, etc.) with module-level scope.

## W2 — Pure-property ports ✅

Implemented in `source/jax_diff/properties_jax.py`:

- [x] `psat` (Lee–Kesler vapor pressure, vmap over compounds).
- [x] `cp_molar` and `latent_heat_vaporization`.
- [x] `molar_liquid_vol` and pure-component `density`.
- [x] `surface_tension` and `viscosity_kinematic`.
- [x] Mixture properties: `mixture_density`,
      `mixture_kinematic_viscosity`, `mixture_surface_tension`.
- [x] Composition conversions: `mass2X`, `Y2X`, `X2Y`,
      `mean_molecular_weight`.
- [x] Full UNIFAC `activity` with `vmap`-over-compounds and safe-log
      boundary handling for zero mole fractions.

Validation in `tests/jax_diff/test_w2_properties.py`:

- [x] Bit-for-bit regression vs. numpy at multiple temperatures and
      compositions (rtol ≤ 1e-10 on every property).
- [x] FD-vs-AD gradient consistency at interior and boundary compositions.
- [x] JIT-compatibility smoke tests.

## W3 — Algebraic blocks with IFT custom VJPs ✅

Implemented in `source/jax_diff/algebraic_jax.py` with the shared
internal helper `source/jax_diff/_implicit.py::implicit_scalar_root`:

- [x] `solve_rr(z, T, P, ft)` — isothermal Rachford–Rice flash returning
      `(V, xi, yi)`. Forward: bisection-then-Newton in `lax.fori_loop`.
      Backward: implicit-function theorem (memory independent of
      iteration count).
- [x] `solve_bubble_point(Xi, P, ft)` — `T₁` such that `Σ Kᵢ xᵢ = 1`,
      with custom VJP via IFT on the bubble-point residual.
- [x] `solve_stage2(D1, T1, vapor_in, h_coeff, P, ft)` — outer
      bisection on `T₂` with an inner RR flash at each guess. The two
      custom VJPs compose cleanly.

Validation:

- [x] `tests/jax_diff/test_w3_rr.py` — RR forward correctness,
      `jax.grad` and `jax.jacrev` finiteness, JIT compatibility.
- [x] `tests/jax_diff/test_w3_bubble.py` — bubble-point correctness
      against numpy, gradient finiteness.
- [x] `tests/jax_diff/test_w3_stage2.py` — Stage-2 forward correctness
      and end-to-end gradient through the nested solver.

## W4 — Differentiable distillation driver (demonstrator) ✅

Implemented in `source/jax_diff/distillation_jax.py`:

- [x] `run_distillation(N0, ft, h_coeff, D1_per_mole, n_steps, dt)` —
      Heun (RK2) integrator inside `lax.scan` with a frozen-state
      done-mask so component depletion is handled smoothly.
- [x] `extract_astm_cuts(dist_vol_trace, T2_trace, fractions)` —
      differentiable interpolation of cut temperatures from the
      simulated `T₂(V)` curve.

Validation:

- [x] `tests/jax_diff/test_w4_ode.py` — end-to-end forward run,
      `jax.grad` of a scalar functional of the distillate trace
      returns finite gradients.
- [x] Slow tests marked with the `slow` pytest marker so the fast
      suite (`pytest -m "not slow"`) stays under a minute.

> ⚠️ This driver is an autodiff **demonstrator**, not a production
> D86 reproducer. See [`REMAINING_WORK.md`](./REMAINING_WORK.md).

## W5 — Multi-output forward map ✅

Implemented in `source/jax_diff/inverse_jax.py`:

- [x] `ObservationConfig` dataclass with toggles for each channel and
      configurable Tref lists, cut fractions, and per-channel σ.
- [x] `forward_observations(w, ft, cfg, *, cut_curve_T=None,
      cut_curve_V=None)` returning `(y, sigma)` for weighted
      least-squares.
- [x] Channels: ASTM cuts at arbitrary fractions, density at multiple
      Trefs, kinematic viscosity at multiple Trefs, surface tension,
      mean molecular weight.
- [x] Differentiable interpolation accepting an externally-supplied
      `T_D86(V)` curve so the calibrated numpy driver can drive
      JAX-side inversion.

Validation in `tests/jax_diff/test_w5_forward.py`:

- [x] Channel masking — toggling each option on/off changes `y`
      length consistently.
- [x] Gradients of every channel are finite under `jax.jacrev`.
- [x] σ vector is strictly positive on every channel.

## W6 — Inverse problem ✅

Implemented in `source/jax_diff/inverse_jax.py`:

- [x] `softmax_simplex(θ) = w` parametrisation.
- [x] `tikhonov_loss(θ, y_meas, sigma, fwd, *, w_prior, L1, lam1, L2,
      lam2, lam3)` — full quadratic form covering data weighting,
      prior pull, smoothness, and ridge.
- [x] `first_difference_operator(n, perm=argsort(Tb))` — canonical L₂
      built on the boiling-point axis.
- [x] `jacobian_svd(fwd, theta)` returning `(U, s, Vt)` for the adjoint
      study.
- [x] `posterior_covariance(fwd, theta, sigma, *, L2, lam2, lam3)`
      returning `(Σ_θ, σ_w)` for per-component 1σ bars.

Validation in `tests/jax_diff/test_w6_inverse.py`:

- [x] Twin-experiment recovery: forward → add Gaussian noise →
      Adam-optimise softmax-parametrised loss → recovered composition
      sits at the noise floor.
- [x] L₂ regularisation lifts the smallest non-trivial Hessian
      eigenvalue on the cuts-only kerosene problem (regularisation is
      doing the right thing).
- [x] `posterior_covariance` returns finite, positive σ_w for every
      component on the same case.

## W7 — Hardening and documentation ✅

- [x] `pytest.ini` updated with the `jax` marker.
- [x] `source/jax_diff/README.md` — module-by-module reference,
      quick-start, test mapping.
- [x] `docs/jax_diff/` — this docs set
      (`IMPLEMENTATION_PLAN.md`, `TASKS_COMPLETED.md`,
      `REMAINING_WORK.md`, `CAPABILITY_USAGE.md`).
- [x] Code-review feedback incorporated: dead variable in
      `distillation_jax` removed
      (commit `2834c87`).

## Test inventory

```
tests/jax_diff/test_w2_properties.py  — pure-property regression + AD
tests/jax_diff/test_w3_rr.py          — Rachford-Rice algebraic block
tests/jax_diff/test_w3_bubble.py      — bubble-point algebraic block
tests/jax_diff/test_w3_stage2.py      — nested Stage-2 solver
tests/jax_diff/test_w4_ode.py         — RK2 scan + end-to-end gradient
tests/jax_diff/test_w5_forward.py     — multi-output observation map
tests/jax_diff/test_w6_inverse.py     — twin-experiment + adjoint study
```

Run subsets via:

```bash
pytest tests/jax_diff/ -m "not slow"   # fast (≈ 50 s, 64 tests)
pytest tests/jax_diff/                 # full   (≈ 80 s, 73 tests)
pytest tests/ -m "not d86"             # numpy reference, still green
```
