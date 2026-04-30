"""
``algebraic_jax`` — Differentiable algebraic solvers for the no-SRK D86 DAE.

Three algebraic blocks at every time step of the distillation simulator:

============== ====================================================== ===============
g₁(T₁; X) = 0  bubble-point: ``Σ Kᵢ(T₁,X) Xᵢ − 1 = 0``                :func:`solve_bubble_point`
g₂(T₂; ...) = 0 Stage-2 column-neck energy balance                    :func:`solve_stage2`
g₃(V; T,P,z) = 0 Rachford-Rice: ``Σ zᵢ(Kᵢ−1)/(1+V(Kᵢ−1)) = 0``          :func:`solve_rr`
============== ====================================================== ===============

In the **no-SRK** branch the K-values come from
``Kᵢ = γᵢ(X,T) · Pˢᵃᵗᵢ(T) / P`` (UNIFAC + Lee-Kesler), all of which are
smooth functions of the state and parameters, so each ``gⱼ`` is smooth in
its hidden variable.  This file wraps the iterative forward solves with
the implicit-function-theorem helper :func:`._implicit.implicit_scalar_root`
so the reverse-mode graph never unrolls iterations.

Forward solvers
---------------
Each block uses a fixed-iteration **bisection-then-Newton** scheme inside
``jax.lax.fori_loop`` so the forward pass is JIT-compatible and trivially
``vmap``-able over batches of feeds.

* Bisection is guaranteed to converge in ``O(log₂(b−a)/xtol)`` steps and
  is unconditionally stable; it gets us to within ~1e-3 of the root.
* A short Newton tail polishes the answer to machine precision, which is
  important for the IFT backward pass to evaluate the residual partials
  *exactly* at ``g(x*) = 0``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._implicit import implicit_scalar_root, safe_div
from . import properties_jax as P
from .properties_jax import FuelTables


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _k_values_no_srk(T: jnp.ndarray, X: jnp.ndarray, P_atm: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """K-values via modified Raoult's law: ``K = γ(X,T) Pˢᵃᵗ(T) / P``."""
    gamma = P.activity(X, T, ft)
    psat = P.psat(T, ft)
    return gamma * psat / P_atm


# -----------------------------------------------------------------------------
# W3.1 — Rachford-Rice flash
# -----------------------------------------------------------------------------
class FlashResult(NamedTuple):
    """Output of an isothermal flash."""
    V: jnp.ndarray   # vapour fraction, scalar
    xi: jnp.ndarray  # liquid mole fractions  (num_comp,)
    yi: jnp.ndarray  # vapour mole fractions  (num_comp,)


def _rr_residual(V, K, z):
    """``f(V) = Σ zᵢ(Kᵢ−1) / (1 + V(Kᵢ−1))``."""
    Km1 = K - 1.0
    return jnp.sum(z * Km1 / (1.0 + V * Km1))


def _rr_solve_inner(z, K):
    """
    Solve the Rachford-Rice scalar equation ``f(V) = 0`` for fixed K.

    Forward: 60 bisection steps on the Whitson-Brulé interval, followed
    by 8 Newton steps for polish.  Backward: implicit-function theorem.

    The two-phase branch is detected by the sign of ``f(0)`` and ``f(1)``;
    in the trivial single-phase regime we return ``V = 0`` (sub-cooled)
    or ``V = 1`` (super-heated) and the routine still produces sensible
    ``xi``/``yi`` (both equal to ``z``).
    """
    K_max = jnp.max(K)
    K_min = jnp.min(K)
    eps_b = 1.0e-10

    # Whitson-Brulé negative-flash interval.  Guard against pure
    # components (K_max=K_min=1) by snapping to a tiny non-degenerate
    # bracket.
    V_lo = 1.0 / jnp.where(K_max > 1.0 + 1e-12, 1.0 - K_max, -1.0) + eps_b
    V_hi = 1.0 / jnp.where(K_min < 1.0 - 1e-12, 1.0 - K_min,  1.0) - eps_b

    f0 = jnp.sum(z * (K - 1.0))
    K_safe = jnp.where(K > 0, K, 1.0)
    f1 = jnp.sum(jnp.where(K > 0, z * (K - 1.0) / K_safe, 0.0))

    is_subcooled = f0 <= 0.0     # all liquid
    is_superheat = f1 >= 0.0     # all vapour
    is_two_phase = ~(is_subcooled | is_superheat)

    # -- Bisection (forward only) --
    def bisection_body(i, state):
        a, b = state
        m = 0.5 * (a + b)
        fm = _rr_residual(m, K, z)
        fa = _rr_residual(a, K, z)
        same_sign = fa * fm > 0.0
        a_new = jnp.where(same_sign, m, a)
        b_new = jnp.where(same_sign, b, m)
        return (a_new, b_new)

    a0, b0 = V_lo, V_hi
    a_end, b_end = jax.lax.fori_loop(0, 60, bisection_body, (a0, b0))
    V_bis = 0.5 * (a_end + b_end)

    # -- Newton polish --
    def newton_body(i, V):
        Km1 = K - 1.0
        denom = 1.0 + V * Km1
        f = jnp.sum(z * Km1 / denom)
        fp = -jnp.sum(z * Km1 ** 2 / denom ** 2)
        # Damped Newton; clamp step to keep V inside [V_lo+eps, V_hi-eps].
        step = safe_div(f, fp)
        V_new = V - step
        V_new = jnp.clip(V_new, V_lo, V_hi)
        return V_new

    V_polished = jax.lax.fori_loop(0, 8, newton_body, V_bis)

    # Trivial-phase fallback: V = 0 / 1, xᵢ = yᵢ = z.
    V_two = V_polished
    V_final = jnp.where(is_two_phase, V_two,
                        jnp.where(is_subcooled, 0.0, 1.0))
    return V_final


def solve_rr(
    z: jnp.ndarray,
    T: jnp.ndarray,
    P_atm: jnp.ndarray,
    ft: FuelTables,
    n_outer: int = 12,
) -> FlashResult:
    """
    Differentiable Rachford-Rice flash with successive substitution on K.

    The outer loop is *also* implicit-diff'd: we run ``n_outer``
    successive-substitution updates of the K-values, and the gradient of
    the converged composition w.r.t. ``(z, T, P_atm, ft)`` is taken via
    JAX's autodiff through the *fixed-point residual* — the heavy
    iterative loop is avoided entirely in reverse mode by wrapping the
    inner V-solve with implicit-diff.

    For modest ``n_outer`` (default 12 is ample for kerosene-range
    surrogates) the explicit unroll of the outer loop is cheap, so we
    don't pay the cost of a separate fixed-point custom_vjp here.

    :returns: ``FlashResult(V, xi, yi)`` — all differentiable.
    """
    z = jnp.asarray(z)

    # Initial K-values from feed composition.
    K = _k_values_no_srk(T, z, P_atm, ft)

    def step(carry, _):
        K_prev = carry

        # Inner V-solve made differentiable via IFT on the scalar root.
        # Closures capture (K_prev, z); we expose them as `theta` so the
        # custom_vjp can take partials w.r.t. them.
        theta = (K_prev, z)
        residual = lambda V_, th: _rr_residual(V_, th[0], th[1])
        forward = lambda th: _rr_solve_inner(th[1], th[0])
        V_fn = implicit_scalar_root(residual, forward)
        V = V_fn(theta)

        Km1 = K_prev - 1.0
        denom = 1.0 + V * Km1
        x_unnorm = z / denom
        x_sum = jnp.sum(x_unnorm)
        x_sum_safe = jnp.where(x_sum > 0, x_sum, 1.0)
        xi = jnp.where(x_sum > 0,
                       x_unnorm / x_sum_safe,
                       jnp.zeros_like(x_unnorm))

        K_new = _k_values_no_srk(T, xi, P_atm, ft)
        return K_new, None

    K_final, _ = jax.lax.scan(step, K, xs=None, length=n_outer)

    # Final V solve at the converged K.
    theta = (K_final, z)
    residual = lambda V_, th: _rr_residual(V_, th[0], th[1])
    forward = lambda th: _rr_solve_inner(th[1], th[0])
    V_fn = implicit_scalar_root(residual, forward)
    V = V_fn(theta)

    Km1 = K_final - 1.0
    denom = 1.0 + V * Km1
    x_unnorm = z / denom
    x_sum = jnp.sum(x_unnorm)
    x_sum_safe = jnp.where(x_sum > 0, x_sum, 1.0)
    xi = jnp.where(x_sum > 0, x_unnorm / x_sum_safe, jnp.zeros_like(x_unnorm))

    yi_unnorm = K_final * xi
    y_sum = jnp.sum(yi_unnorm)
    y_sum_safe = jnp.where(y_sum > 0, y_sum, 1.0)
    yi = jnp.where(y_sum > 0, yi_unnorm / y_sum_safe, jnp.zeros_like(yi_unnorm))

    # Detect trivial-phase exits and force xi = yi = z so downstream
    # consumers see a smooth limit; gradient comes from the outer
    # branches, which already compose smoothly.
    K_max = jnp.max(K_final)
    K_min = jnp.min(K_final)
    f0 = jnp.sum(z * (K_final - 1.0))
    K_safe = jnp.where(K_final > 0, K_final, 1.0)
    f1 = jnp.sum(jnp.where(K_final > 0, z * (K_final - 1.0) / K_safe, 0.0))
    is_subcool = f0 <= 0.0
    is_superht = f1 >= 0.0
    is_trivial = is_subcool | is_superht
    xi = jnp.where(is_trivial, z, xi)
    yi = jnp.where(is_trivial, z, yi)
    V_clamped = jnp.clip(V, 0.0, 1.0)
    V_out = jnp.where(is_subcool, 0.0,
                      jnp.where(is_superht, 1.0, V_clamped))
    return FlashResult(V=V_out, xi=xi, yi=yi)


# -----------------------------------------------------------------------------
# W3.2 — Bubble-point temperature
# -----------------------------------------------------------------------------
def _bp_residual(T, X, P_atm, ft):
    """``Σ Kᵢ(T,X) Xᵢ − 1`` (the modified-Raoult bubble-point condition)."""
    K = _k_values_no_srk(T, X, P_atm, ft)
    return jnp.sum(K * X) - 1.0


def _bp_forward(theta):
    """Bisection (40 steps) + Newton (8 steps) on a wide bracket."""
    X, P_atm, ft = theta
    T_lo, T_hi = 200.0, 900.0

    def bisection_body(i, state):
        a, b = state
        m = 0.5 * (a + b)
        fm = _bp_residual(m, X, P_atm, ft)
        fa = _bp_residual(a, X, P_atm, ft)
        same_sign = fa * fm > 0.0
        a_new = jnp.where(same_sign, m, a)
        b_new = jnp.where(same_sign, b, m)
        return (a_new, b_new)

    a, b = jax.lax.fori_loop(0, 40, bisection_body, (T_lo, T_hi))
    T_bis = 0.5 * (a + b)

    def newton_body(i, T):
        f = _bp_residual(T, X, P_atm, ft)
        fp = jax.grad(lambda Tx: _bp_residual(Tx, X, P_atm, ft))(T)
        return T - safe_div(f, fp)

    T_polished = jax.lax.fori_loop(0, 8, newton_body, T_bis)
    return T_polished


def solve_bubble_point(
    X: jnp.ndarray,
    P_atm: jnp.ndarray,
    ft: FuelTables,
):
    """
    Bubble-point temperature ``T₁`` and equilibrium vapor composition.

    :returns: ``(T1, vapor_comp)`` — ``T1 ∈ ℝ`` is differentiable wrt
        ``(X, P_atm, ft)`` via implicit diff, and ``vapor_comp = K X /
        Σ(K X)`` is a smooth post-image.
    """
    theta = (X, P_atm, ft)
    residual = lambda T_, th: _bp_residual(T_, th[0], th[1], th[2])
    T1_fn = implicit_scalar_root(residual, _bp_forward)
    T1 = T1_fn(theta)

    K = _k_values_no_srk(T1, X, P_atm, ft)
    vapor_unnorm = K * X
    s = jnp.sum(vapor_unnorm)
    s_safe = jnp.where(s > 0, s, 1.0)
    vapor = jnp.where(s > 0, vapor_unnorm / s_safe, jnp.zeros_like(vapor_unnorm))
    return T1, vapor


# -----------------------------------------------------------------------------
# W3.3 — Stage-2 outer energy balance (nested DAE)
# -----------------------------------------------------------------------------
def _h_lat_at_T(T: jnp.ndarray, X: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Mole-weighted molar latent heat (J/mol)."""
    Lv_kg = P.latent_heat_vaporization(T, ft)
    Hv_mol = Lv_kg * ft.MW
    return jnp.sum(X * Hv_mol)


def _stage2_residual(T2, D1, T1, vapor_in, h_coeff, P_atm, ft):
    """
    Stage-2 column-neck energy balance::

        Q_supp(T₂) − Q_loss(T₂) = 0,
            Q_supp = D₁ · ΔH_vap_in
            Q_loss = h · A · (T₁ − T₂)   +    R₂ · Cp_l · (T₁ − T₂)

    For a clean differentiable port we condense the heat-loss model into
    a single coefficient ``h_coeff`` (W/K) so calibration / inversion
    against the Stage-2 thermometer curve is reduced to a single scalar
    parameter.  This matches the structure already exercised by the
    Optuna calibration drivers and keeps the algebraic block 1-D.

    Inside the residual we additionally perform a Rachford-Rice flash on
    ``vapor_in`` at ``T₂`` to compute the reflux fraction, exactly
    mirroring :func:`source.distillation.solve_stage2_flash` but in
    differentiable form.
    """
    flash = solve_rr(vapor_in, T2, P_atm, ft)
    R2_frac = 1.0 - flash.V       # condensed fraction returned as reflux
    R2 = R2_frac * D1
    deltaH = _h_lat_at_T(T2, flash.xi, ft)
    Cp_l = jnp.sum(flash.xi * P.cp_molar(T2, ft))

    Q_supp = D1 * deltaH
    Q_loss = h_coeff * (T1 - T2) + R2 * Cp_l * (T1 - T2)
    return Q_supp - Q_loss


def _stage2_forward(theta):
    """Bisection (40 steps) on ``[T_room, T₁ − ε]``."""
    D1, T1, vapor_in, h_coeff, P_atm, ft = theta
    T_lo = 200.0
    T_hi = T1 - 1.0e-3

    def body(i, state):
        a, b = state
        m = 0.5 * (a + b)
        fm = _stage2_residual(m, D1, T1, vapor_in, h_coeff, P_atm, ft)
        fa = _stage2_residual(a, D1, T1, vapor_in, h_coeff, P_atm, ft)
        same_sign = fa * fm > 0.0
        a_new = jnp.where(same_sign, m, a)
        b_new = jnp.where(same_sign, b, m)
        return (a_new, b_new)

    a, b = jax.lax.fori_loop(0, 40, body, (T_lo, T_hi))
    return 0.5 * (a + b)


class Stage2Result(NamedTuple):
    """Output of the Stage-2 energy balance."""
    T2: jnp.ndarray         # column-neck temperature, K
    R2: jnp.ndarray         # reflux molar rate, mol/s
    reflux_comp: jnp.ndarray  # reflux composition (num_comp,)
    D2: jnp.ndarray         # net distillate molar rate, mol/s
    vapor_out: jnp.ndarray  # outgoing vapour composition (num_comp,)


def solve_stage2(
    D1: jnp.ndarray,
    T1: jnp.ndarray,
    vapor_in: jnp.ndarray,
    h_coeff: jnp.ndarray,
    P_atm: jnp.ndarray,
    ft: FuelTables,
) -> Stage2Result:
    """
    Stage-2 outer T₂ energy balance with nested Rachford-Rice flash.

    The outer T₂ is implicit-diff'd; the inner RR flash already has its
    own ``custom_vjp`` from :func:`solve_rr`, so JAX composes the two
    correctly.

    :returns: :class:`Stage2Result` — every field differentiable wrt
        ``(D1, T1, vapor_in, h_coeff, P_atm, ft)``.
    """
    theta = (D1, T1, vapor_in, h_coeff, P_atm, ft)
    residual = lambda T_, th: _stage2_residual(T_, *th)
    T2_fn = implicit_scalar_root(residual, _stage2_forward)
    T2 = T2_fn(theta)

    flash = solve_rr(vapor_in, T2, P_atm, ft)
    R2 = (1.0 - flash.V) * D1
    D2 = flash.V * D1
    return Stage2Result(
        T2=T2,
        R2=R2,
        reflux_comp=flash.xi,
        D2=D2,
        vapor_out=flash.yi,
    )
