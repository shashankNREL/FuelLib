# `source.jax_diff` — Differentiable no-SRK FuelLib in JAX

This subpackage implements the no-SRK plan for a differentiable
algebraic-equation (DAE) solver coupled to an ODE integrator in JAX,
plus a Tikhonov-regularised multi-output inverse problem for fuel-
composition recovery from mixture observables.

## Why this exists

The D86 distillation simulator in `source/distillation.py` /
`source/distillation_rk2.py` is a DAE system with three nested
algebraic blocks at every time step (bubble point, Stage-2 column-neck
energy balance, Rachford-Rice flash) and one ODE on pot composition.
For inverse-problem work we need:

1. **Smooth, differentiable forward simulation** — so that we can
   compute `∂y/∂w` (Jacobian of observables w.r.t. fuel composition)
   for use with gradient-based optimisers and uncertainty
   quantification.
2. **Tractable adjoints** — so that calibrating against ASTM cuts +
   density + viscosity + … doesn't require finite differences over
   thousands of components.
3. **Regularisation for ill-posed problems** — kerosene surrogates
   typically have far more components (20–80) than measurable
   observables (a dozen), making the inverse problem rank-deficient.

## Modules

| Module                 | Purpose                                                                                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| `properties_jax.py`    | JAX ports of `psat`, `Cp`, `molar_liquid_vol`, `latent_heat_vaporization`, surface tension, Dutt viscosity, mixture properties, composition conversions, and full UNIFAC `activity` (vmap'd over compounds) |
| `algebraic_jax.py`     | Differentiable Rachford-Rice flash, bubble-point, and Stage-2 outer T₂ — each wrapped with implicit-function-theorem `custom_vjp` |
| `distillation_jax.py`  | Simplified Heun-RK2 batch distillation driver under `lax.scan` (demonstrates the W4 pattern; the production driver with full PI controller is future work) |
| `inverse_jax.py`       | Multi-output forward observation map, softmax simplex parametrisation, Tikhonov-regularised loss, and adjoint-study utilities (Jacobian SVD, posterior covariance) |
| `_implicit.py`         | Internal: implicit-function-theorem `custom_vjp` helper, plus safe-log / safe-div utilities for boundary-of-simplex compositions |

## Quick start

```python
from source.FuelLib import fuel
import source.jax_diff as jd
import jax, jax.numpy as jnp

f  = fuel("posf10325")
ft = jd.build_fuel_tables(f)

# Forward-simulate observables for a candidate composition.
w  = jnp.asarray(f.Y_0)
y, sigma = jd.forward_observations(w, ft, jd.ObservationConfig())

# Inverse problem: recover w from y_meas with Tikhonov regularisation.
n  = ft.num_compounds
L2 = jd.first_difference_operator(n, perm=jnp.argsort(ft.Tb))

def fwd(theta):
    return jd.forward_observations(jd.softmax_simplex(theta), ft, jd.ObservationConfig())[0]

loss = lambda th: jd.tikhonov_loss(
    th, y, sigma, fwd,
    L2=L2, lam2=1e-2, lam3=1e-4,
)
grad_loss = jax.jit(jax.grad(loss))
```

## Testing

The `tests/jax_diff/` directory contains test modules for each
workstream:

| File                       | Targets   | What it asserts |
|----------------------------|-----------|-----------------|
| `test_w2_properties.py`    | W2        | Bit-for-bit numerical match to numpy `FuelLib`, FD-vs-AD gradient checks, finiteness at the simplex boundary |
| `test_w3_rr.py`            | W3.1      | Rachford-Rice forward agreement, finite gradients, JIT compatibility |
| `test_w3_bubble.py`        | W3.2      | Bubble-point forward agreement, FD-vs-AD, vapor normalisation |
| `test_w3_stage2.py`        | W3.3      | Stage-2 nested DAE physical sanity + gradient finiteness |
| `test_w4_ode.py`           | W4        | Full RK2/scan distillation: shape, done-mask, end-to-end gradient |
| `test_w5_forward.py`       | W5        | Multi-output forward map shape, channel mask, monotone cuts, Jacobian sanity |
| `test_w6_inverse.py`       | W6        | Twin experiment recovery, FD-vs-AD on loss, conditioning improvement under L₂, posterior covariance positivity, SVD spectrum |

Run with:

```bash
pytest tests/jax_diff/        # all jax-port tests
pytest tests/jax_diff/ -m "not slow"   # skip the W4 distillation test
```

## Tikhonov regularisation in one paragraph

With ``M`` observables (~12 — eight ASTM cuts plus a few mixture
scalars) and ``n_components ≫ M`` (kerosenes use 20–80), the Jacobian
``J = ∂y/∂w`` is rank-deficient: an ``(n − rank(J))``-dimensional
subspace of compositions fits the data identically.  Tikhonov
regularisation injects external information to make the inverse problem
well-posed:

* **Zero-order** ``λ₁ ‖w − w_prior‖²`` — pull toward an a-priori
  composition (e.g. GCMS),
* **First-order** ``λ₂ ‖L₂ w‖²`` — penalise spikes between adjacent
  components ordered by carbon number / boiling point.  This is the
  dominant lever when no prior is available.
* **Ridge** ``λ₃ ‖θ‖²`` — keeps the softmax parametrisation tame for
  components the data cannot resolve.

The tuning of ``(λ₁, λ₂, λ₃)`` follows the standard L-curve / GCV
recipe — see W6 in the design document.

## Future work (pointers from the design plan)

* W4: full PI-controlled, CSTR-thermometer batch distillation under
  ``lax.scan`` with ``jax.checkpoint`` for memory-bounded reverse mode.
  The simplified driver in `distillation_jax.py` validates the
  composed graph but does not reproduce the calibrated D86 curve.
* W6.3: notebook with L-curve, singular-spectrum, recovered-vs-measured
  diagnostic plots.
