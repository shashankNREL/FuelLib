"""
Tests for W3.3 — Stage-2 outer energy balance with nested Rachford-Rice
flash (:func:`source.jax_diff.algebraic_jax.solve_stage2`).

Note: the JAX Stage-2 routine uses a single-parameter heat-loss model
``Q_loss = h_coeff * (T₁ − T₂)`` (plus the reflux sensible-cooling
term).  This is a *structural* simplification of the full
:func:`source.distillation.solve_stage2_flash`, which has a 2-D
parameter set ``(h, A)``.  The two are equivalent if we treat
``h_coeff = h * A``; for the Tikhonov inverse problem this single scalar
is exactly the calibratable knob, so the simplification is intentional.
The tests below therefore *do not* attempt a numpy regression — they
exercise gradient correctness, JIT compatibility and physical sanity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import source.jax_diff as jd


P_ATM = 101325.0


def _make_stage2_inputs(fuel_obj, ft):
    """Pre-compute realistic ``(D1, T1, vapor_in)`` from the bubble-point."""
    Xi = jnp.asarray(fuel_obj.Y2X(fuel_obj.Y_0))
    T1, vap = jd.solve_bubble_point(Xi, P_ATM, ft)
    D1 = jnp.array(1.0e-3)
    return D1, T1, vap


def test_stage2_returns_T2_below_T1(kerosene_fuel, kerosene_tables):
    D1, T1, vap = _make_stage2_inputs(kerosene_fuel, kerosene_tables)
    res = jd.solve_stage2(D1, T1, vap, jnp.array(2.0), jnp.array(P_ATM), kerosene_tables)
    assert float(res.T2) < float(T1)
    assert float(res.T2) > 200.0
    assert 0.0 <= float(res.R2) <= float(D1) + 1e-12
    np.testing.assert_allclose(float(res.R2 + res.D2), float(D1), rtol=1e-9)


def test_stage2_grad_wrt_h_coeff_finite(kerosene_fuel, kerosene_tables):
    D1, T1, vap = _make_stage2_inputs(kerosene_fuel, kerosene_tables)
    fn = lambda h: jd.solve_stage2(
        D1, T1, vap, h, jnp.array(P_ATM), kerosene_tables
    ).T2
    g = jax.grad(fn)(jnp.array(2.0))
    assert bool(jnp.isfinite(g))
    # Larger h ⇒ more cooling ⇒ T₂ moves further from T₁ → derivative finite.
    assert float(g) != 0.0


def test_stage2_grad_wrt_vapor_in_finite(kerosene_fuel, kerosene_tables):
    D1, T1, vap = _make_stage2_inputs(kerosene_fuel, kerosene_tables)
    fn = lambda v: jd.solve_stage2(
        D1, T1, v, jnp.array(2.0), jnp.array(P_ATM), kerosene_tables
    ).T2
    g = jax.grad(fn)(vap)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_stage2_jit_compatible(kerosene_fuel, kerosene_tables):
    D1, T1, vap = _make_stage2_inputs(kerosene_fuel, kerosene_tables)
    fn = jax.jit(lambda h: jd.solve_stage2(
        D1, T1, vap, h, jnp.array(P_ATM), kerosene_tables
    ).T2)
    T2 = fn(jnp.array(2.0))
    assert 200.0 < float(T2) < float(T1)
