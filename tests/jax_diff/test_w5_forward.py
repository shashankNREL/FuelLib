"""
Tests for W5 — multi-output forward observation map
:func:`source.jax_diff.inverse_jax.forward_observations`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import source.jax_diff as jd
from source.jax_diff.inverse_jax import (
    ObservationConfig,
    forward_observations,
    softmax_simplex,
)


def test_observation_vector_length(kerosene_tables):
    cfg = ObservationConfig()
    w = jnp.asarray(np.full(kerosene_tables.num_compounds, 1.0 / kerosene_tables.num_compounds))
    y, sigma = forward_observations(w, kerosene_tables, cfg)
    # 8 cuts + 1 ρ + 1 ν + 1 σ + 1 MW = 12.
    assert y.shape == (12,)
    assert sigma.shape == (12,)
    assert bool(jnp.all(sigma > 0))


def test_observation_vector_finite_at_uniform_composition(kerosene_tables):
    cfg = ObservationConfig()
    n = kerosene_tables.num_compounds
    w = jnp.full((n,), 1.0 / n)
    y, _ = forward_observations(w, kerosene_tables, cfg)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_observation_jacobian_shape_and_finite(kerosene_tables):
    """Shape ``(M, n)`` and no NaN entries."""
    cfg = ObservationConfig()
    n = kerosene_tables.num_compounds
    theta = jnp.zeros(n)
    fwd = lambda th: forward_observations(softmax_simplex(th), kerosene_tables, cfg)[0]
    J = jax.jacrev(fwd)(theta)
    assert J.shape == (12, n)
    assert bool(jnp.all(jnp.isfinite(J)))


def test_density_and_kinematic_viscosity_are_sane(kerosene_fuel, kerosene_tables):
    """Sanity: typical kerosene ρ ∈ [600, 900] kg/m^3, ν ~ 1–4 mm^2/s @ 40 °C."""
    cfg = ObservationConfig()
    w = jnp.asarray(kerosene_fuel.Y_0)
    y, _ = forward_observations(w, kerosene_tables, cfg)
    # The observation order is: 8 cuts, ρ, ν, σ, MW.
    rho = float(y[8])
    nu = float(y[9])
    assert 600.0 < rho < 900.0, f"ρ outside sanity bracket: {rho}"
    assert 1e-7 < nu < 1e-5, f"ν outside sanity bracket: {nu}"


def test_disable_channels_changes_M(kerosene_tables):
    cfg = ObservationConfig(include_kviscosity=False, include_surface_tension=False)
    n = kerosene_tables.num_compounds
    w = jnp.full((n,), 1.0 / n)
    y, _ = forward_observations(w, kerosene_tables, cfg)
    # 8 cuts + 1 ρ + 1 MW = 10
    assert y.shape == (10,)


def test_cuts_increase_with_volume_fraction(kerosene_tables):
    """ASTM cut temperatures must be monotonically non-decreasing in V%."""
    cfg = ObservationConfig()
    n = kerosene_tables.num_compounds
    w = jnp.full((n,), 1.0 / n)
    y, _ = forward_observations(w, kerosene_tables, cfg)
    cuts = np.asarray(y[:8])
    diffs = np.diff(cuts)
    assert np.all(diffs >= -1e-9), f"cuts not non-decreasing: {diffs}"
