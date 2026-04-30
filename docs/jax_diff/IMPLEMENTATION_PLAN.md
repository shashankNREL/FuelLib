# Implementation Plan — Differentiable JAX Port of FuelLib + Tikhonov Inverse

This document captures the design plan that drove the work on the
`copilot/jax-integration-adjoint-study` branch. It is a record of the plan
*as designed*; the actual completion status of each item is tracked in
[`TASKS_COMPLETED.md`](./TASKS_COMPLETED.md) and
[`REMAINING_WORK.md`](./REMAINING_WORK.md).

## 1. Motivation

The D86 distillation simulator in `source/distillation.py` /
`source/distillation_rk2.py` is a DAE system with three nested algebraic
blocks at every time step (bubble point, Stage-2 column-neck energy
balance, Rachford-Rice flash) plus an ODE on pot composition. For
calibration and inverse-problem work we need:

1. **Smooth, differentiable forward simulation** so that
   `∂y/∂w` — the Jacobian of mixture observables with respect to
   composition — is available for gradient-based optimisers and
   uncertainty quantification.
2. **Tractable adjoints** so that calibrating against
   ASTM cuts + density + viscosity + surface tension + MW does not
   require finite differences over a 20–80-component composition
   vector.
3. **Regularisation for ill-posed problems**, since kerosene surrogates
   typically carry many more components than measurable observables and
   the inverse problem is rank-deficient.

The numpy code paths are the production reference and remain untouched.
The JAX port lives in a sibling subpackage `source/jax_diff/` and is
free to reorganise the math for autodiff convenience as long as it
matches the numpy code numerically (rtol ≤ 1e-10 on every property).

## 2. Scope decisions

- **No SRK.** Vapor-phase non-ideality is omitted; the differentiable
  port assumes ideal vapor and uses UNIFAC for the liquid only. This is
  consistent with the existing `distillation_rk2.py` configuration and
  removes a major source of non-smoothness.
- **No new physics.** Every property must reproduce the numpy
  implementation bit-for-bit at typical operating points.
- **JAX as the only autodiff backend.** No PyTorch, no JAX-on-TF.
- **Implicit-function-theorem (IFT) custom VJPs everywhere a scalar
  root-finder is used.** Reverse-mode memory must be independent of
  iteration count so that the composed `(RR ∘ Stage-2 ∘ scan)` graph
  remains differentiable through hundreds of time steps.
- **Tikhonov regularisation, not priors-as-constraints.** The inverse
  problem uses a quadratic penalty form so that gradients stay
  smooth and so that the same machinery covers ridge, smoothness, and
  prior-pull terms.

## 3. Workstream breakdown

The plan was decomposed into seven workstreams (W1–W7). Each workstream
ships a code module, a dedicated pytest file under `tests/jax_diff/`,
and validation against the numpy reference where applicable.

### W1 — Infrastructure
- New subpackage `source/jax_diff/` with explicit `__init__.py` exports.
- `FuelTables` NamedTuple snapshotting all numpy fuel arrays into
  frozen `jnp.ndarray`s once at startup (`build_fuel_tables`).
- Float64 enabled at import time.
- `pytest.ini` gains a `jax` marker and `tests/jax_diff/conftest.py`
  fixtures for shared fuel objects.

### W2 — Pure-property ports
- JAX implementations of `psat` (Lee-Kesler), `cp_molar`,
  `molar_liquid_vol`, `latent_heat_vaporization`, `surface_tension`,
  `viscosity_kinematic`, mixture density, mixture kinematic
  viscosity, mixture surface tension, composition conversions
  (`mass2X`, `Y2X`, `X2Y`, `mean_molecular_weight`), and the full
  UNIFAC `activity` (`vmap`-over-compounds with safe-log boundary
  handling).
- Bit-for-bit regression tests vs. the numpy reference at a battery of
  temperatures and compositions.
- FD-vs-AD gradient consistency tests at interior and boundary
  compositions.

### W3 — Algebraic blocks with implicit-function-theorem VJPs
- `solve_rr` — isothermal Rachford-Rice flash (vapor fraction `V` ∈ [0, 1]).
- `solve_bubble_point` — finds `T₁` such that `Σ Kᵢ xᵢ = 1`.
- `solve_stage2` — nested Stage-2 column-neck energy balance:
  outer bisection on `T₂` with an inner RR flash at each guess.
- All three forward passes use bisection-then-Newton inside
  `lax.fori_loop`. Backward pass is the implicit-function theorem via
  the shared internal helper `_implicit.implicit_scalar_root`, so
  reverse-mode memory does not depend on iteration count and the three
  custom VJPs compose cleanly.

### W4 — Differentiable distillation driver
- `run_distillation`: Heun (RK2) integrator under `lax.scan`
  with a frozen-state done-mask. The driver demonstrates that the
  composed `(properties ∘ RR ∘ Stage-2 ∘ scan)` graph is end-to-end
  differentiable.
- **Demonstrator only.** The plan calls out a follow-up production
  driver with PI control, CSTR thermometer dynamics, and
  `jax.checkpoint` rematerialisation — that work is deferred (see
  `REMAINING_WORK.md`).

### W5 — Multi-output forward map
- `ObservationConfig` dataclass selecting which channels (cuts at
  arbitrary fractions, density at multiple Trefs, kinematic
  viscosity, surface tension, MW) are active.
- `forward_observations(w, ft, cfg)` stacks a length-M observation
  vector and returns it together with a per-channel σ vector for
  weighted least-squares.
- Differentiable interpolation against externally-supplied
  `T_D86(V)` curves so a numpy distillation result can drive a
  JAX-side cut extraction.

### W6 — Inverse problem
- `softmax_simplex(θ) = w` parametrisation so optimisation lives on an
  unconstrained ℝⁿ but always satisfies `wᵢ ≥ 0`, `Σ wᵢ = 1`.
- `tikhonov_loss` implements
  `‖W(y − y_meas)‖² + λ₁‖L₁(w − w_prior)‖² + λ₂‖L₂ w‖² + λ₃‖θ‖²`.
- `first_difference_operator(n, perm=argsort(Tb))` is the canonical
  smoothness operator (penalises composition jumps along the boiling-
  point axis).
- `jacobian_svd` and `posterior_covariance` deliver the **adjoint
  study** outputs: the singular spectrum of `J*` (effective rank of the
  data), the right-singular vectors (null-space directions the
  observations cannot resolve), and per-component 1σ error bars from
  the Gauss-Newton Hessian.

### W7 — Hardening and documentation
- `pytest.ini` marker, README under `source/jax_diff/`, this docs set.

## 4. Acceptance criteria

The plan is considered "delivered for the W1–W3, W5–W7 scope" when:

1. Every numpy property has a JAX twin matching to `rtol ≤ 1e-10`.
2. Every algebraic block returns finite forward values **and** finite
   gradients under both `jax.grad` and `jax.jacrev` for representative
   compositions and temperatures.
3. The composed graph through `run_distillation` is JIT-able and
   produces a finite reverse-mode gradient end-to-end.
4. A twin-experiment inversion (`forward → add noise → invert`)
   recovers the underlying composition at the noise floor on a
   kerosene case, and `posterior_covariance` returns finite, positive
   1σ values for every component.
5. The full numpy regression suite (`pytest tests/ -m "not d86"`)
   continues to pass.

## 5. Out-of-scope choices

These items were considered and explicitly deferred:

- **Full PI-controlled, CSTR-thermometer RK2 driver in JAX.** The
  numpy reference (`distillation_rk2.run_d86_simulation_rk2`) remains
  the only path that reproduces calibrated D86 curves end-to-end.
- **L-curve / GCV regularisation tuning driver.** The building blocks
  (`tikhonov_loss`, `jacobian_svd`) are present; a sweep harness is
  not.
- **W6.3 diagnostic notebook.** The data-producing functions are in
  place; only the matplotlib glue is missing.
- **HMC posteriors over θ via numpyro.** The same forward map would
  be reusable with no source changes; out of scope for this branch.
- **Ports of SRK / vapor-phase non-ideality.** Out of scope by design.
