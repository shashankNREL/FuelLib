# Capability Usage Guide — `source.jax_diff`

This is the practical "how do I run this and what can I do with it"
guide for the differentiable JAX port. For a module-by-module
reference see [`source/jax_diff/README.md`](../../source/jax_diff/README.md);
for the design rationale see [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).

## 1. Setup

The code lives in `source/jax_diff/` and requires JAX (and the existing
FuelLib dependencies). Install once:

```bash
pip install jax jaxlib numpy scipy pandas pytest
```

Float64 is enabled automatically when you `import source.jax_diff`
(it calls `jax.config.update("jax_enable_x64", True)` at module load).

## 2. Running the tests

From the repo root:

```bash
# Fast test suite (64 unit tests, ≈ 50 s)
pytest tests/jax_diff/ -m "not slow"

# Full suite including W4 distillation driver (73 tests, ≈ 80 s)
pytest tests/jax_diff/

# A single workstream
pytest tests/jax_diff/test_w6_inverse.py -v

# Existing numpy regression suite still passes alongside
pytest tests/ -m "not d86"
```

## 3. What you can do — five concrete use cases

### (a) Use any FuelLib property in a differentiable, JIT-able way

```python
from source.FuelLib import fuel
import source.jax_diff as jd
import jax, jax.numpy as jnp

f  = fuel("posf10325")
ft = jd.build_fuel_tables(f)        # one-time numpy → JAX snapshot

# Drop-in replacements for fuel.psat / .activity / .mixture_density / etc.
jd.psat(400.0, ft)
jd.activity(jnp.asarray(f.Y2X(f.Y_0)), 400.0, ft)
jd.mixture_density(jnp.asarray(f.Y_0), 313.15, ft)

# All of these are jit-compatible and grad-compatible
rho_grad_fn = jax.jit(jax.grad(lambda Y: jd.mixture_density(Y, 313.15, ft)))
```

Available functions: `psat`, `cp_molar`, `molar_liquid_vol`,
`latent_heat_vaporization`, `surface_tension`, `viscosity_kinematic`,
`density`, `mixture_density`, `mixture_kinematic_viscosity`,
`mixture_surface_tension`, `mass2X` / `Y2X` / `X2Y`,
`mean_molecular_weight`, full UNIFAC `activity`.

### (b) Solve a differentiable VLE block (bubble point / RR flash / Stage-2)

```python
P_atm = 101325.0
Xi    = jnp.asarray(f.Y2X(f.Y_0))

# Bubble-point with ∂T₁/∂(X, P) via implicit function theorem
T1, vapor_comp = jd.solve_bubble_point(Xi, P_atm, ft)
dT1_dX = jax.grad(lambda X: jd.solve_bubble_point(X, P_atm, ft)[0])(Xi)

# Isothermal Rachford-Rice flash with finite ∂V/∂(z, T, P)
flash = jd.solve_rr(Xi, 450.0, P_atm, ft)
print(flash.V, flash.xi, flash.yi)

# Stage-2 nested DAE (RR-inside-T₂ bisection)
stage2 = jd.solve_stage2(D1=jnp.array(1e-3), T1=T1, vapor_in=vapor_comp,
                         h_coeff=jnp.array(2.0), P_atm=jnp.array(P_atm), ft=ft)
print(stage2.T2, stage2.R2, stage2.D2)
```

Each of these is wrapped in a `jax.custom_vjp` that uses the implicit
function theorem, so `jax.grad` / `jax.jacrev` produce finite gradients
without unrolling the iterative solver.

### (c) Run the simplified differentiable distillation driver

```python
N0 = jnp.asarray(f.Y_0) * 0.1 / ft.MW          # 100 g pot, mol per component
final, dist_vol_trace, T2_trace = jd.run_distillation(
    N0, ft, h_coeff=2.0, D1_per_mole=0.05, n_steps=200, dt=0.5,
)

# Extract ASTM cuts differentiably
cuts = jd.extract_astm_cuts(dist_vol_trace, T2_trace,
                            jnp.array([0.05, 0.10, 0.50, 0.90]))

# End-to-end gradient through the entire scan
grad_total_dist = jax.grad(
    lambda N: jd.run_distillation(N, ft, n_steps=20, dt=1.0)[1][-1]
)(N0)
```

> **Note:** this is a *demonstrator* — no PI controller, no CSTR
> thermometer. For curve calibration against measured D86 data, the
> production `source.distillation_rk2.run_d86_simulation_rk2` is still
> the reference. See [`REMAINING_WORK.md`](./REMAINING_WORK.md).

### (d) Build a multi-output observation vector

```python
cfg = jd.ObservationConfig(
    T_refs_density=(288.15, 313.15),
    T_refs_kviscosity=(313.15,),
    cut_fractions=(0.05, 0.10, 0.20, 0.40, 0.50, 0.70, 0.90, 0.95),
    sigma_cut_K=5.0,
    sigma_density=5.0,
    sigma_kviscosity=5e-7,
    # turn channels on/off:
    include_kviscosity=True,
    include_surface_tension=True,
    include_mw_avg=True,
)
y, sigma = jd.forward_observations(jnp.asarray(f.Y_0), ft, cfg)
# y is a length-M observation vector you can feed to the inverse problem
```

If you have a real simulated `T_D86(V)` curve from W4 (or anywhere
else), pass it via `cut_curve_T=` / `cut_curve_V=` and
`forward_observations` will interpolate the cuts differentiably.

### (e) Solve the Tikhonov-regularised inverse problem (the headline capability)

```python
n  = ft.num_compounds
L2 = jd.first_difference_operator(n, perm=jnp.argsort(ft.Tb))   # smoothness on Tᵦ axis

# 1) Pretend you measured y_meas (real data or synthetic)
y_meas, sigma_obs = jd.forward_observations(jnp.asarray(f.Y_0), ft, cfg)

# 2) Define the loss
def fwd(theta):
    return jd.forward_observations(jd.softmax_simplex(theta), ft, cfg)[0]

loss = lambda th: jd.tikhonov_loss(
    th, y_meas, sigma_obs, fwd,
    L2=L2, lam2=1e-2, lam3=1e-4,    # smoothness + ridge
    # w_prior=jnp.asarray(f.Y_0), L1=jnp.eye(n), lam1=1e-3   # if you have a prior
)
grad_loss = jax.jit(jax.grad(loss))

# 3) Optimise with any JAX-friendly optimiser (Adam shown in test_w6_inverse.py;
#    L-BFGS via jaxopt also works without modification)
theta = jnp.zeros(n)
for _ in range(500):
    theta = theta - 0.05 * grad_loss(theta)
w_hat = jd.softmax_simplex(theta)

# 4) Adjoint-study deliverables — the reusable scientific output
U, s, Vt = jd.jacobian_svd(fwd, theta)            # singular spectrum + null-space
Sigma_theta, sigma_w = jd.posterior_covariance(   # 1σ bars per component
    fwd, theta, sigma_obs, L2=L2, lam2=1e-2, lam3=1e-4,
)
```

What the adjoint study tells you:

- **`s`** — the singular-value spectrum of `J*`. Where it falls off a
  cliff is the effective rank of your data, i.e., how many components
  your observations are actually identifying.
- **`Vt`** (rows for small `s`) — the null-space directions:
  composition modes the observations cannot resolve. If a particular
  paraffin–aromatic ratio sits in the null space, *no amount* of
  fitting will recover it from the chosen channels — you need a new
  measurement.
- **`sigma_w`** — per-component 1σ error bars. Components with
  `sigma_w[i]` close to `w_hat[i]` are essentially unidentified.

## 4. Tuning the regularisation (λ₁, λ₂, λ₃)

The plan calls for L-curve / GCV; the building blocks are all there:

```python
losses, sol_norms = [], []
for lam2 in jnp.logspace(-6, 2, 30):
    L = lambda th: jd.tikhonov_loss(th, y_meas, sigma_obs, fwd,
                                    L2=L2, lam2=lam2, lam3=1e-4)
    # ...optimise, then record residual ‖W(y − y_meas)‖ vs. ‖L₂ w‖
```

Plot residual vs. solution norm on log–log axes; the corner is the
recommended `λ`. A turn-key sweep driver is listed under
[`REMAINING_WORK.md`](./REMAINING_WORK.md#2-l-curve--gcv-regularisation-tuning-driver-w62).

## 5. What is *not* yet implemented

See [`REMAINING_WORK.md`](./REMAINING_WORK.md) for the full list. The
short version:

- The full PI-controlled, CSTR-thermometer-coupled RK2 distillation
  driver with `jax.checkpoint`. Current `run_distillation` is a
  working differentiable demonstrator only. For calibration against
  real data, run the numpy `run_d86_simulation_rk2` and pass the
  resulting `T_D86(V)` curve into
  `forward_observations(..., cut_curve_T=, cut_curve_V=)`.
- The L-curve / GCV λ-tuning driver (W6.2).
- The diagnostic notebook (W6.3) — the *data* for these plots all
  comes from the functions above; only the matplotlib glue is missing.

## 6. Where to look for working examples

- `tests/jax_diff/test_w2_properties.py` — bit-for-bit numpy
  regression and FD-vs-AD checks.
- `tests/jax_diff/test_w3_*.py` — gradient finiteness / JIT
  compatibility patterns for each algebraic block.
- `tests/jax_diff/test_w4_ode.py` — driving the simplified RK2 scan
  and getting an end-to-end gradient.
- `tests/jax_diff/test_w5_forward.py` — building observation vectors
  with different channel masks.
- **`tests/jax_diff/test_w6_inverse.py`** — full twin-experiment
  inversion (Adam optimiser, posterior covariance, SVD spectrum,
  conditioning monitor). This is the most useful read-along.
- `source/jax_diff/README.md` — module-by-module reference.
