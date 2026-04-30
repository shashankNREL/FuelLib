# Q1 Controller Design Notes — Keeping Distillation Rate in [4, 5] mL/min

This document collects design options for the heater-power (`Q1`) controller used by
`source/distillation_rk2_condenser.py` to regulate the distillation rate to the ASTM-D86
target band of **4–5 mL/min**, with a particular emphasis on producing a controller that
is friendly to **JAX-based inverse differentiability** (i.e. `jax.grad` / `jax.jacrev`
through the simulator with respect to controller gains, target rate, fuel parameters,
condenser UA, etc.).

The recommendations here are *suggestions only* — no production code is changed by
adding this file. Its purpose is to record the design space and a corresponding test
plan for future work.

---

## Context — what the current controller does

`source/distillation_rk2_condenser.py` runs an RK2 loop. Each step it measures the
instantaneous distillate rate (mL/min), EMA-smooths it, then updates `Q1` via a
velocity-form PI controller with hard saturation and a conditional anti-windup branch:

```
error = target_rate - rate_ema             # 4.5 mL/min target
ΔQ_p  = Kp * (error - prev_error)
ΔQ_i  = 0   if at saturation else Ki * error * dt
Q1    = clip(Q1 + ΔQ_p + ΔQ_i, Q_min, Q_max)
```

The controller is **stateful** (carries `prev_error`, `rate_ema`, `Q1`), uses Python
`if` / `max` / `min`, and lives inside an early-termination `for` loop. That is the
source of the JAX friction: data-dependent branching, Python control flow, and
non-smooth `min`/`max` all break `jit` / `grad` cleanly, even though each is
individually fixable.

The **goal** of the design exercise: a *simpler* controller that

1. keeps the rate in **[4, 5] mL/min**, and
2. is friendly to JAX inverse-mode autodiff (so the whole simulation can be embedded
   in an outer optimization or parameter-inference loop).

---

## Options (simplest → most expressive)

### Option A — Static feed-forward Q1 (no controller at all)

The simplest "controller" is none: pick a constant `Q1` (or a precomputed schedule
`Q1(t)` or `Q1(V)`) that historical runs show keeps the rate near 4.5 mL/min for that
fuel. Calibrate it once via a short outer-loop optimization (Optuna already exists in
`source/optimize_d86_optuna_condenser.py`, or via JAX gradient descent once the inner
solver is JAX).

- **Pros:** trivially JAX-differentiable — `Q1` is just a parameter, no state, no
  branches. Inverse problems become "given measured curve, find `Q1` (or its schedule)."
- **Cons:** no closed-loop rejection of disturbances; fuel-to-fuel re-tuning needed;
  slope of `rate(t)` will drift across the boil curve.
- **Best when:** the user's real need is *differentiability* more than *true rate
  regulation*, or when the fuel is well-characterized.

### Option B — Proportional-only controller on cumulative volume (the simplest closed loop)

Replace rate-tracking with **volume-schedule** tracking. Define a target schedule
`V_target(t) = 4.5 · t / 60` (mL). At each step:

```
Q1 = Q1_nom + Kp * (V_target(t) − V_actual(t))
```

This is a single-line, stateless, branch-free P controller. It *implicitly* averages
the rate (because volume is the integral of rate), so it doesn't need an EMA filter or
velocity form.

- **JAX-friendly:** pure arithmetic, no `if`. Gradient of the trajectory w.r.t. `Kp`,
  `Q1_nom`, `target_rate` is well-defined.
- **Saturation:** replace `clip` with a `softclip`, e.g.
  `Q_min + (Q_max − Q_min) · sigmoid((Q1 − Q_min)/(Q_max − Q_min))`, or `jnp.clip`
  (subgradient is fine for inverse problems but zero at saturation — usually
  acceptable; otherwise a softplus/softclip preserves smoothness).
- **Pros:** ~1 line, stateless, easy to reason about, and the "stay in [4,5]"
  requirement maps directly onto a *band* on the integrated volume.
- **Cons:** a steady offset is possible if `Q1_nom` is wrong; usually that is
  tolerable because the integral action is "free" — error in volume *is* the
  integrated rate error.

### Option C — PI on volume error (still stateless w.r.t. Python branches)

If Option B leaves a residual rate bias, add a small derivative of the volume error
(which is the rate error) without an EMA:

```
e_V    = V_target(t) − V_actual(t)
de_dt  = target_rate − instantaneous_rate           # both mL/min, same units
Q1     = clip_smooth(Q1_nom + Kp·e_V + Kd·de_dt, Q_min, Q_max)
```

- This is a PD on volume = PI on rate, but with **no integrator state** and **no
  anti-windup logic**, because the integral is supplied by the simulator's own
  `V_actual`.
- Pure function of `(t, V_actual, instantaneous_rate)` — trivially `vmap` / `grad`-able.
- Anti-windup is handled by the smooth clip: if `Q1` saturates, the gradient just
  attenuates; no special-case branch.

### Option D — Barrier / penalty controller (no feedback law at all; differentiable by construction)

Don't write a controller; write a **loss** that the outer solver minimizes:

```
L(θ) = Σ_t  softplus(rate(t) − 5)² + softplus(4 − rate(t))²
        + λ · (rate(t) − 4.5)²            # optional centring term
```

where `θ` are the parameters you actually want to invert (e.g. `Q1` profile
coefficients, fuel composition, condenser UA). Drop the inner controller entirely:
instead of regulating, the outer JAX optimizer chooses `Q1(t)` (or `Q1`) so that the
constraint is satisfied. This is the canonical "trajectory optimization" framing.

- **Pros:** cleanest for *inverse differentiability* — the whole problem is one
  differentiable objective, no nested control loop, no piecewise logic.
- **Pros:** the [4, 5] band is encoded directly via the barrier; `softplus` / `huber`
  keeps gradients smooth.
- **Cons:** requires the outer JAX solver (you already have Optuna; an L-BFGS / Adam
  pass over the same parameters with `jax.grad` is straightforward once the simulator
  is JAX).
- **Best when:** differentiability is the actual goal — this is the option that buys
  the most by going JAX-native.

### Option E — Inverse-model (model-based) feed-forward + small P trim

Energy balance gives an analytic estimate of how much `Q1` is needed for a given
vaporization rate:

```
Q1_ff(t) ≈ ṁ_target · ΔH_vap(T1, X)        # plus sensible heat & losses
Q1       = Q1_ff(t) + Kp · (target_rate − rate)
```

`ΔH_vap` and composition are already in `fuel_obj`, so this is a one-liner once you
expose `latent_heat(T, X)` in JAX. Because the feed-forward does most of the work,
`Kp` can be tiny and the closed-loop term rarely saturates — making the whole thing
effectively linear and very gradient-friendly.

- **Pros:** physically motivated, robust across fuels, smallest residual error,
  smallest reliance on the controller for accuracy.
- **Cons:** needs a JAX-callable `ΔH_vap`; slightly more code than Option B.

### Option F — Keep PI, but make it JAX-clean (minimal-change refactor)

If you want to preserve current behavior exactly:

1. Replace the `for`-loop body with a `jax.lax.scan` carry of
   `(N, T2, R2, Q1, prev_error, rate_ema)`.
2. Replace `max(Q_min, min(Q_max, x))` with `jnp.clip(x, Q_min, Q_max)` (subgradient
   OK) or a smooth clip.
3. Replace `if at_upper or at_lower: …` anti-windup with
   `jnp.where(at_saturation, 0.0, Ki*error*dt)` — same semantics, differentiable,
   JIT-able.
4. Replace early-termination `break` with a `lax.scan` that always runs to `n_max`
   plus a mask for the "stop" condition (or `lax.while_loop` if you don't need
   reverse-mode through the loop length; `while_loop` is *not* reverse-mode
   differentiable, so prefer fixed-length `scan`).

This is Option F because it's the most code, not the least — included only as the
"drop-in JAX port of what you have."

---

## Recommendation

For "*simpler* and *JAX-inverse-differentiable*" the sweet spot is **Option B** (or
Option C if you see a steady-state offset), or **Option D** if you're willing to
recast the problem as trajectory optimization. **Option E** is the best if you also
want robustness across fuels with little tuning.

> **Rule of thumb:** track the integrated quantity (**volume**) instead of its
> derivative (**rate**). That single change removes the EMA filter, the velocity
> form, the integrator state, and the anti-windup branch — i.e. almost all of the
> JAX-unfriendly pieces — for free, because the simulator already integrates rate
> into volume for you.

---

## Suggested tests (no production code changes required)

Place under `tests/` next to the existing `tests/test_integration_d86.py`. They
should run on top of whichever option above is chosen; all are black-box and don't
require touching `source/`.

1. **Rate-band test (acceptance).** Run a full `simulate_distillation_condenser` for
   the standard fuel(s) used in `test_integration_d86.py`. Assert that for
   `t > t_warmup` (e.g. 60 s, after the first drop) the windowed rate
   `(dV/dt)·60` stays in `[4, 5]` mL/min for at least X% of samples (e.g. 90%) and
   the median is within `[4.3, 4.7]`.
2. **Steady-state offset test.** Same run; assert
   `|mean(rate[t > warmup]) − 4.5| < 0.2` mL/min.
3. **Setpoint sweep.** Parametrize over
   `target_rate ∈ {3.5, 4.0, 4.5, 5.0, 5.5}` and assert (a) achieved median rate
   tracks setpoint within ±0.3 mL/min, (b) no NaNs, (c) `Q1` stays within
   `[Q_min, Q_max]`.
4. **Saturation behavior.** Force `Q_max` artificially low so the controller
   saturates; assert: no NaNs, `Q1 == Q_max` for the saturated steps, no integrator
   wind-up (i.e. when target later becomes feasible, recovery time < N steps).
5. **Disturbance rejection.** Mid-run, perturb `UA_cond` by ±20% (or `T_bath` by a
   few K). Assert the rate returns inside `[4, 5]` within a tolerance window.
6. **Determinism / reproducibility.** Same inputs → identical trajectory (bit-exact
   for the NumPy path; `jnp.allclose` for the JAX path).
7. **Differentiability tests (only if the JAX path is implemented).**
   - `jax.grad` of a scalar loss (e.g. `Σ (rate − 4.5)²`) w.r.t. `Kp`, `Kd`,
     `target_rate`, `Q1_nom` returns finite, non-NaN values.
   - Finite-difference check: `jax.grad` matches
     `(L(θ + ε) − L(θ − ε)) / (2ε)` to within `1e-3` relative for at least one
     nontrivial parameter.
   - `jax.jit(simulate)` produces the same trajectory as the un-jitted version.
   - `jax.vmap` over a batch of `target_rate` values runs without error and the
     output shapes are correct.
8. **Inverse problem smoke test.** Given a synthetic "measured" trajectory generated
   with `θ_true`, run a few steps of `optax.adam` (or
   `jax.example_libraries.optimizers.adam`) on `‖rate(θ) − rate_true‖²` starting
   from `θ_0 ≠ θ_true` and assert the loss decreases monotonically over, say, 20
   iterations. This is the actual *inverse differentiability* end-to-end check.
9. **Comparison test (regression).** For the existing fuels, assert the new
   controller's D86 curve is within `±2 K` of the current PI-controller baselines
   stored in `tests/baselinePredictions/` (i.e. the simpler controller doesn't
   degrade accuracy).
10. **Stateless property test (Options B / C / D only).** Property-based
    (Hypothesis) test that the controller output is a pure function of its
    arguments — call it twice with identical inputs in a fresh process and assert
    identical outputs; this catches accidental hidden state if someone refactors.

Tests **1–4** and **9** are required regardless of choice. Tests **7–8** are the
ones that actually exercise "JAX-based inverse differentiability" and should be the
gating criteria for declaring the new controller successful.
