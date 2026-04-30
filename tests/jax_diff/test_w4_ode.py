"""
Tests for W4 — simplified differentiable batch distillation driver
(:func:`source.jax_diff.distillation_jax.run_distillation`).

The production ``source.distillation_rk2.run_d86_simulation_rk2`` has
many calibrated knobs (a PI controller, a CSTR thermometer ODE,
empirical heat-loss coefficients) which are intentionally out of scope
for this minimal differentiable demonstration.  These tests therefore
only verify *physical sanity* and *differentiability* of the simplified
driver, not numerical agreement with the production curve.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import source.jax_diff as jd


pytestmark = pytest.mark.slow


def _make_initial_pot(fuel_obj, ft):
    # Convert mass fractions to a moles vector using a 100 g batch.
    Y = jnp.asarray(fuel_obj.Y_0)
    mass = Y * 0.1                # 100 g pot
    return mass / ft.MW           # mol per component


def test_distillation_run_shape_and_finite(small_fuel, small_tables):
    N0 = _make_initial_pot(small_fuel, small_tables)
    final, dist_vol, T2 = jd.run_distillation(
        N0, small_tables, n_steps=20, dt=1.0,
    )
    assert dist_vol.shape == (20,)
    assert T2.shape == (20,)
    assert bool(jnp.all(jnp.isfinite(dist_vol)))
    assert bool(jnp.all(jnp.isfinite(T2)))
    # Distillate volume must be non-decreasing.
    diffs = jnp.diff(dist_vol)
    assert bool(jnp.all(diffs >= -1e-12))


def test_distillation_done_mask_freezes_state(small_fuel, small_tables):
    """
    If we drive the run hard enough that the pot is exhausted, the done
    mask should freeze the state — final dist_vol is reproduced over the
    last few steps and no NaNs are produced.
    """
    N0 = _make_initial_pot(small_fuel, small_tables)
    final, dist_vol, T2 = jd.run_distillation(
        N0, small_tables, D1_per_mole=0.5, n_steps=50, dt=1.0,
    )
    # When the pot is exhausted, dist_vol stops growing.
    last5 = dist_vol[-5:]
    assert float(jnp.max(last5) - jnp.min(last5)) < 1e-6


def test_distillation_grad_wrt_N0_finite(small_fuel, small_tables):
    """
    Gradient of total distillate volume wrt initial pot composition.

    Differentiability of the composed (RR ∘ Stage-2 ∘ Heun ∘ scan) graph
    is the central deliverable of the JAX port — verify it produces
    finite gradients on a small fuel.
    """
    N0 = _make_initial_pot(small_fuel, small_tables)

    def scalar_loss(N):
        _, dist, _ = jd.run_distillation(N, small_tables, n_steps=10, dt=1.0)
        return dist[-1]

    g = jax.grad(scalar_loss)(N0)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_extract_astm_cuts_shape(small_fuel, small_tables):
    N0 = _make_initial_pot(small_fuel, small_tables)
    _, dist, T2 = jd.run_distillation(N0, small_tables, n_steps=15, dt=1.0)
    cuts = jd.extract_astm_cuts(dist, T2, jnp.array([0.1, 0.5, 0.9]))
    assert cuts.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(cuts)))
