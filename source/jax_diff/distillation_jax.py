"""
``distillation_jax`` — Simplified differentiable batch-distillation driver.

This module realises the W4 plan element in *demonstration* form: a
fixed-horizon Heun (RK2) integrator under :func:`jax.lax.scan` that
evolves the pot composition through the Stage-1 + Stage-2 algebraic
blocks at every step.  It is **not** a full port of
``source.distillation_rk2.run_d86_simulation_rk2`` — the production
distillation code includes a PI heat-input controller, a CSTR
thermometer ODE and a number of empirically-tuned heat-loss
coefficients which are out of scope for this minimal differentiable
demonstration.  The plan documents these omissions in §W4 (PI-controller
smoothing) and §5 (risks).

What *is* implemented:

* Fixed-horizon ``lax.scan`` of length ``n_steps``, advancing the pot
  molar composition ``N`` via Heun integration of
  ``dN/dt = R₂ · reflux_comp − D₁ · vapor_comp_in``.
* Stage-1 bubble-point solve at every step (fully differentiable).
* Stage-2 outer T₂ + inner Rachford-Rice solve at every step (fully
  differentiable through the implicit-diff custom_vjp).
* Done-mask: once pot moles drop below a threshold, the RHS is forced
  to zero and the carry freezes.  This replaces the original
  ``while V_pot > min_volume`` early termination.
* Recorded distillate-fraction and column-neck-temperature traces, plus
  an :func:`extract_astm_cuts` helper that performs differentiable
  :func:`jnp.interp` onto user-specified ASTM volume points.

The full RK2 production driver — including the PI controller smoothed
via either ``stop_gradient`` on Q₁ or a ``sigmoid``-based saturation —
is left as the natural next step (see W4 in the design document).  The
machinery here is sufficient to wire the algebraic-DAE solver through a
``lax.scan`` integrator end-to-end and, in particular, validates that
gradients flow correctly through the composed (RR ∘ Stage-2 ∘ RK2)
graph, which is the riskier construction.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import properties_jax as P
from .algebraic_jax import solve_bubble_point, solve_stage2
from .properties_jax import FuelTables


class DistillationState(NamedTuple):
    """Per-step carry of the simplified RK2 driver."""
    N: jnp.ndarray              # pot moles per component, (num_comp,)
    distillate_vol: jnp.ndarray # cumulative distillate volume (m^3), scalar
    T2_log: jnp.ndarray         # last computed Stage-2 temperature, scalar
    done: jnp.ndarray           # bool — true once pot exhausted


def _rhs(N, h_coeff, D1_per_mole, P_atm, ft):
    """
    Right-hand side ``dN/dt`` of the pot-composition ODE.

    The pot evolves according to::

        dN/dt = R₂ · reflux_comp − D₁ · vapor_comp_in

    with ``D₁`` exogenously prescribed (via ``D1_per_mole`` — moles of
    vapour produced per second per mole of pot inventory) — a stand-in
    for the PI-controlled heat input in the production driver.
    """
    n_total = jnp.sum(N)
    n_safe = jnp.where(n_total > 0, n_total, 1.0)
    Xi = jnp.where(n_total > 0, N / n_safe, jnp.zeros_like(N))

    T1, vapor_in = solve_bubble_point(Xi, P_atm, ft)
    D1 = D1_per_mole * n_total
    stage2 = solve_stage2(D1, T1, vapor_in, h_coeff, P_atm, ft)
    dN = stage2.R2 * stage2.reflux_comp - D1 * vapor_in
    return dN, stage2.T2, stage2.D2, vapor_in, T1


def _heun_step(state, dt, h_coeff, D1_per_mole, P_atm, ft, min_moles):
    """Single Heun (RK2) step with a frozen-state done-mask."""
    N = state.N
    is_active = ~state.done

    dN1, T2, D2, _, T1 = _rhs(N, h_coeff, D1_per_mole, P_atm, ft)
    N_pred = N + dt * dN1
    N_pred = jnp.maximum(N_pred, 0.0)
    dN2, T2_b, D2_b, _, _ = _rhs(N_pred, h_coeff, D1_per_mole, P_atm, ft)

    N_new = N + 0.5 * dt * (dN1 + dN2)
    N_new = jnp.maximum(N_new, 0.0)

    # Distillate volume increment: net vapour molar rate × molar liquid
    # volume of the *outgoing distillate* at T₂ × dt.
    # We approximate by the average over the two RK stages, using the
    # mole-averaged molar volume at the mean Stage-2 temperature.
    avg_Vm = jnp.sum(P.molar_liquid_vol(0.5 * (T2 + T2_b), ft)) / ft.num_compounds
    dV = 0.5 * (D2 + D2_b) * avg_Vm * dt

    # Freeze the state if we declared the run done.
    new_state = DistillationState(
        N=jnp.where(is_active, N_new, N),
        distillate_vol=jnp.where(is_active,
                                 state.distillate_vol + dV,
                                 state.distillate_vol),
        T2_log=jnp.where(is_active, T2, state.T2_log),
        done=state.done | (jnp.sum(N_new) < min_moles),
    )
    return new_state, (new_state.distillate_vol, new_state.T2_log)


def run_distillation(
    N0: jnp.ndarray,
    ft: FuelTables,
    *,
    h_coeff: float = 2.0,
    P_atm: float = 101325.0,
    D1_per_mole: float = 0.05,        # 5% pot inventory vaporised / second
    n_steps: int = 200,
    dt: float = 0.5,
    min_moles_frac: float = 1e-3,
):
    """
    Run the simplified differentiable batch distillation.

    :param N0: Initial pot moles per component (shape ``(num_comp,)``).
    :param ft: :class:`FuelTables`.
    :param h_coeff: Stage-2 lumped heat-loss coefficient (W/K).
    :param P_atm: System pressure (Pa).
    :param D1_per_mole: Exogenous vaporisation rate, mol/s per mol of
        pot inventory.  Stands in for the PI-controlled heat input;
        gradients flow through this parameter cleanly so it can be
        calibrated from a measured Q₁(t) trace.
    :param n_steps: Fixed scan horizon length.
    :param dt: Time step (s).
    :param min_moles_frac: Stop fraction (frozen-state done mask) — the
        run is declared finished once pot moles drop below
        ``min_moles_frac · sum(N0)``.

    :returns: ``(state_final, dist_vol_trace, T2_trace)``.  Both traces
        have shape ``(n_steps,)`` and are differentiable wrt every
        scalar parameter and ``N0``.
    """
    min_moles = min_moles_frac * jnp.sum(N0)
    init = DistillationState(
        N=N0,
        distillate_vol=jnp.array(0.0),
        T2_log=jnp.array(298.15),
        done=jnp.array(False),
    )

    def scan_body(state, _):
        return _heun_step(state, dt, h_coeff, D1_per_mole, P_atm, ft, min_moles)

    final, traces = jax.lax.scan(scan_body, init, xs=None, length=n_steps)
    dist_vol_trace, T2_trace = traces
    return final, dist_vol_trace, T2_trace


def extract_astm_cuts(
    dist_vol_trace: jnp.ndarray,
    T2_trace: jnp.ndarray,
    cut_fractions: jnp.ndarray,
):
    """
    Differentiable extraction of ASTM cut temperatures from a recorded
    ``(distillate volume, T₂)`` trace via :func:`jnp.interp`.

    :param cut_fractions: Distillate *fractions* in ``[0, 1]`` at which
        to extract the temperature.
    """
    total = dist_vol_trace[-1]
    total_safe = jnp.where(total > 0, total, 1.0)
    frac = jnp.where(total > 0,
                     dist_vol_trace / total_safe,
                     jnp.zeros_like(dist_vol_trace))
    return jnp.interp(cut_fractions, frac, T2_trace)
