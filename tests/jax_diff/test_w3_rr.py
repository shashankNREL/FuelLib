"""
Tests for W3.1 — differentiable Rachford-Rice flash
:func:`source.jax_diff.algebraic_jax.solve_rr`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import source.jax_diff as jd
from source.distillation import solve_rachford_rice, solve_stage1_bubble_point


P_ATM = 101325.0


def _two_phase_temperature(fuel_obj, X):
    """Pick a flash temperature inside the two-phase envelope (T_b + 5 K)."""
    T1, _ = solve_stage1_bubble_point(fuel_obj, P_ATM, X)
    return T1 + 5.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solve_rr_matches_numpy_two_phase(kerosene_fuel, kerosene_tables, seed):
    """JAX RR ≡ numpy RR within the SS convergence tolerance."""
    n = kerosene_fuel.num_compounds
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.full(n, 2.0))
    T = _two_phase_temperature(kerosene_fuel, X)

    res = jd.solve_rr(jnp.asarray(X), T, P_ATM, kerosene_tables)
    V_np, x_np, y_np = solve_rachford_rice(kerosene_fuel, X, T, P_ATM)

    np.testing.assert_allclose(float(res.V), V_np, atol=5e-5)
    np.testing.assert_allclose(np.asarray(res.xi), x_np, atol=1e-5)
    np.testing.assert_allclose(np.asarray(res.yi), y_np, atol=1e-5)


def test_solve_rr_subcooled_returns_V_zero(kerosene_fuel, kerosene_tables):
    """Below the bubble point: V should clamp to 0 and xi=yi=z."""
    n = kerosene_fuel.num_compounds
    X = jnp.full((n,), 1.0 / n)
    T1, _ = solve_stage1_bubble_point(kerosene_fuel, P_ATM, np.asarray(X))
    T_below = T1 - 30.0
    res = jd.solve_rr(X, T_below, P_ATM, kerosene_tables)
    assert float(res.V) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.asarray(res.xi), np.asarray(X), atol=1e-12)
    np.testing.assert_allclose(np.asarray(res.yi), np.asarray(X), atol=1e-12)


def test_grad_V_wrt_z_finite(kerosene_tables):
    """``∂V*/∂z`` must be finite for a representative two-phase feed."""
    n = kerosene_tables.num_compounds
    X = jnp.full((n,), 1.0 / n)
    # Choose a T well into two-phase by trial.
    T = 450.0
    fn = lambda zv: jd.solve_rr(zv, T, P_ATM, kerosene_tables).V
    g = jax.grad(fn)(X)
    assert bool(jnp.all(jnp.isfinite(g))), "∂V/∂z had non-finite entries"


def test_grad_V_wrt_T_finite_two_phase(kerosene_fuel, kerosene_tables):
    n = kerosene_fuel.num_compounds
    rng = np.random.default_rng(11)
    X = rng.dirichlet(np.full(n, 3.0))
    T = _two_phase_temperature(kerosene_fuel, X)
    fn = lambda Tv: jd.solve_rr(jnp.asarray(X), Tv, P_ATM, kerosene_tables).V
    g = jax.grad(fn)(jnp.asarray(T))
    assert bool(jnp.isfinite(g))
    assert float(g) > 0.0  # heating ⇒ more vapour


def test_solve_rr_jit_compatible(kerosene_tables):
    n = kerosene_tables.num_compounds
    X = jnp.full((n,), 1.0 / n)
    fn = jax.jit(lambda zv: jd.solve_rr(zv, 450.0, P_ATM, kerosene_tables).V)
    val = fn(X)
    assert 0.0 <= float(val) <= 1.0
