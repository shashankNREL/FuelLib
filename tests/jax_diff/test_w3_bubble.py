"""
Tests for W3.2 — differentiable bubble-point solver
:func:`source.jax_diff.algebraic_jax.solve_bubble_point`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import source.jax_diff as jd
from source.distillation import solve_stage1_bubble_point


P_ATM = 101325.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solve_bubble_point_matches_numpy(kerosene_fuel, kerosene_tables, seed):
    """
    Random Dirichlet liquid compositions; JAX bubble-point T must match
    the brentq-based numpy reference within the bisection x-tolerance.
    """
    n = kerosene_fuel.num_compounds
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.full(n, 2.0))
    T1_np, vap_np = solve_stage1_bubble_point(kerosene_fuel, P_ATM, X)
    T1_jx, vap_jx = jd.solve_bubble_point(jnp.asarray(X), P_ATM, kerosene_tables)
    # Wide bisection [200, 900] over 40 steps: residual at root ~ 1e-3 in T.
    np.testing.assert_allclose(float(T1_jx), T1_np, atol=1e-3)
    np.testing.assert_allclose(np.asarray(vap_jx), vap_np, atol=1e-6)


def test_bubble_point_vapor_composition_is_normalized(kerosene_tables):
    """``Σ vapor_comp = 1`` to machine precision (post-normalisation)."""
    n = kerosene_tables.num_compounds
    X = jnp.full((n,), 1.0 / n)
    _, vap = jd.solve_bubble_point(X, P_ATM, kerosene_tables)
    np.testing.assert_allclose(float(jnp.sum(vap)), 1.0, atol=1e-12)


def test_grad_T1_wrt_Xi_is_finite(kerosene_tables):
    """Reverse-mode gradient ∂T₁/∂Xᵢ must be finite (no NaN, no Inf)."""
    n = kerosene_tables.num_compounds
    X = jnp.full((n,), 1.0 / n)
    g = jax.grad(lambda Xv: jd.solve_bubble_point(Xv, P_ATM, kerosene_tables)[0])(X)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_grad_T1_wrt_Xi_matches_fd(kerosene_fuel, kerosene_tables):
    """Reverse-mode AD vs. central FD on a few component perturbations."""
    n = kerosene_fuel.num_compounds
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.full(n, 5.0))     # interior point
    Xj = jnp.asarray(X)

    fn_jx = lambda Xv: jd.solve_bubble_point(Xv, P_ATM, kerosene_tables)[0]
    g_ad = np.asarray(jax.grad(fn_jx)(Xj))

    h = 1e-5
    # Test only on three random indices to keep the runtime modest.
    idxs = rng.choice(n, size=3, replace=False)
    for i in idxs:
        Xp = X.copy(); Xp[i] += h
        Xm = X.copy(); Xm[i] -= h
        Tp, _ = solve_stage1_bubble_point(kerosene_fuel, P_ATM, Xp)
        Tm, _ = solve_stage1_bubble_point(kerosene_fuel, P_ATM, Xm)
        fd = (Tp - Tm) / (2 * h)
        rel = abs(g_ad[i] - fd) / max(abs(fd), 1e-3)
        assert rel < 5e-3, f"index {i}: AD={g_ad[i]:.4e}  FD={fd:.4e}  rel={rel:.2e}"


def test_grad_T1_wrt_P_finite(kerosene_tables):
    """Gradient wrt pressure: must be negative and finite (lower P ⇒ lower T)."""
    n = kerosene_tables.num_compounds
    X = jnp.full((n,), 1.0 / n)
    fn = lambda Pv: jd.solve_bubble_point(X, Pv, kerosene_tables)[0]
    g = jax.grad(fn)(jnp.asarray(P_ATM))
    assert jnp.isfinite(g)
    # Bubble-point T increases with P.
    assert float(g) > 0.0


def test_solve_bubble_point_jit(kerosene_tables):
    """The solver must compose with ``jax.jit`` without error."""
    n = kerosene_tables.num_compounds
    X = jnp.full((n,), 1.0 / n)
    f_jit = jax.jit(lambda Xv: jd.solve_bubble_point(Xv, P_ATM, kerosene_tables)[0])
    T1 = f_jit(X)
    assert float(T1) > 200.0 and float(T1) < 900.0
