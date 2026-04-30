# Remaining Work

This document lists items from the design plan that did **not** land on
the `copilot/jax-integration-adjoint-study` branch and the rationale for
each. Items are grouped by priority for follow-up work.

## High priority

### 1. Production-grade differentiable D86 driver (W4 follow-up)

**Status:** deferred. The current `run_distillation` is a working
autodiff demonstrator and is not a calibrated D86 reproducer.

**What is missing relative to `source/distillation_rk2.py`:**

- PI controller on the heater duty driving the column-neck temperature
  ramp.
- CSTR thermometer dynamics (lag between true neck temperature and the
  sensed `T_D86`).
- Cut-by-cut adaptive timestep selection.
- `jax.checkpoint` rematerialisation around the inner step so reverse-
  mode memory doesn't blow up at hundreds of timesteps.

**Why it was deferred:** the algebraic blocks (W3) and the multi-output
forward map (W5) are the rate-limiting differentiable pieces; the
inverse problem (W6) can be exercised today by feeding a numpy-computed
`T_D86(V)` curve into `forward_observations(..., cut_curve_T=,
cut_curve_V=)`, so the inverse harness is already useful without the
production driver.

**Recommended next step:** port `run_d86_simulation_rk2` step-by-step,
keeping the inner step JIT-friendly and wrapping it in
`jax.checkpoint`. Add a regression test that the JAX driver matches
the numpy `T_D86(V)` to the integrator tolerance on `posf10325`.

### 2. L-curve / GCV regularisation tuning driver (W6.2)

**Status:** building blocks present, sweep harness missing.

**What is missing:**

- A driver that sweeps `λ₂` (and optionally `λ₁`, `λ₃`) on a log grid,
  re-optimises the softmax-parametrised loss at each grid point, and
  records `(‖W(y − y_meas)‖, ‖L₂ w‖)` pairs for the L-curve plot.
- A GCV / discrepancy-principle helper that picks `λ` automatically.

**Why it was deferred:** mechanical to write once the science of which
regularisers to combine is settled. `tikhonov_loss` and
`jacobian_svd` already expose everything needed.

### 3. Diagnostic notebook (W6.3)

**Status:** data-producing functions exist; matplotlib glue missing.

**Recommended deliverable:** a Jupyter notebook under `tutorials/` that
on a kerosene case shows:

1. The L-curve and the chosen `λ`.
2. The singular-value spectrum from `jacobian_svd`.
3. The right-singular vectors for the smallest singular values
   (null-space directions of the data).
4. Recovered composition with 1σ bars from `posterior_covariance`,
   overlaid on the truth.
5. The recovered `T_D86(V)` curve overlaid on the data.

## Medium priority

### 4. HMC / SMC posterior over θ via numpyro

**Status:** out of scope for this branch.

**What is missing:** a numpyro / blackjax model definition that wraps
the same `forward_observations` map and runs NUTS over θ. The forward
map already returns finite gradients under `jax.jit` and has been
exercised at twin-experiment scale, so no source changes are expected.

### 5. Vapor-phase non-ideality (SRK)

**Status:** explicitly out of scope by plan decision.

If reintroduced later, the natural place is a new pair of functions
`fugacity_vapor_jax` / `K_value_jax` that replace the current ideal-vapor
shortcut inside `solve_rr` / `solve_bubble_point`. The existing
implicit-VJP pattern carries through unchanged.

## Low priority / nice-to-have

### 6. Vectorisation over multiple fuels in a single forward call

The current `FuelTables` is a single-fuel snapshot. A `vmap`-friendly
batched version would make sensitivity studies across a fuel database
faster, but is purely a performance refactor.

### 7. GPU / TPU validation

All tests have only been run on CPU. The implementation uses pure
`jax.numpy` and `lax` primitives, so GPU/TPU should "just work", but
this has not been verified.

### 8. Integration with the existing `tutorials/` directory

The `source/jax_diff/README.md` quick-start could be promoted to a
notebook under `tutorials/` and cross-linked from `docs/index.rst`.

### 9. CI coverage for the JAX suite

Currently the JAX tests live under `tests/jax_diff/` but no CI
configuration explicitly opts them in. Adding a `jax` job to the CI
matrix (or running the full suite by default) would catch regressions
earlier.

## Known limitations of the current code

These are not bugs but documented behavioural choices that downstream
users should be aware of:

- `run_distillation` uses a simple constant `D1_per_mole` removal rate
  per timestep rather than tracking distillate volume against the
  pot's instantaneous density. This is fine for the differentiability
  demonstration; it should not be used for quantitative D86
  prediction.
- `extract_astm_cuts` interpolates linearly in `(V, T₂)` space. If a
  cut fraction lies outside the simulated range it returns the
  endpoint temperature (saturating behaviour) — this is differentiable
  but biased. For real data, the cut fractions and the simulated range
  must overlap.
- `softmax_simplex` is invariant to a constant shift of θ. The Hessian
  has one structural zero eigenvalue along the all-ones direction; the
  `lam3 ‖θ‖²` ridge term in `tikhonov_loss` exists primarily to lift
  this. Don't set `lam3 = 0` unless you handle the gauge another way.
