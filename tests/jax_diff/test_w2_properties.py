"""
Tests for the W2 JAX property layer.

Asserts bit-for-bit numerical equivalence between :mod:`source.jax_diff.properties_jax`
and the original numpy methods on :class:`source.FuelLib.fuel`, plus a
finite-difference vs. autodiff gradient check on the mixture density and
the UNIFAC activity routine.

Random compositions are drawn from a Dirichlet (with a fixed seed) so
the boundary safe-log idiom in the JAX activity port is exercised at
realistic compositions.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import source.jax_diff as jd


# Use multiple temperatures spanning the kerosene operating envelope.
T_VALUES = [288.15, 313.15, 400.0, 500.0]


def _dirichlet(n_comp, n_samples, seed):
    rng = np.random.default_rng(seed)
    # Slightly concentrated near uniform — mirrors a real surrogate.
    alpha = np.full(n_comp, 1.0)
    return rng.dirichlet(alpha, size=n_samples)


@pytest.mark.parametrize("T", T_VALUES)
def test_psat_matches_numpy(kerosene_fuel, kerosene_tables, T):
    expected = kerosene_fuel.psat(T)
    got = np.asarray(jd.psat(T, kerosene_tables))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("T", T_VALUES)
def test_cp_molar_matches_numpy(kerosene_fuel, kerosene_tables, T):
    expected = kerosene_fuel.Cp(T)
    got = np.asarray(jd.cp_molar(T, kerosene_tables))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("T", T_VALUES)
def test_molar_liquid_vol_matches_numpy(kerosene_fuel, kerosene_tables, T):
    expected = kerosene_fuel.molar_liquid_vol(T)
    got = np.asarray(jd.molar_liquid_vol(T, kerosene_tables))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("T", T_VALUES)
def test_latent_heat_matches_numpy(kerosene_fuel, kerosene_tables, T):
    expected = kerosene_fuel.latent_heat_vaporization(T)
    got = np.asarray(jd.latent_heat_vaporization(T, kerosene_tables))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("T", T_VALUES)
def test_surface_tension_matches_numpy(kerosene_fuel, kerosene_tables, T):
    expected = kerosene_fuel.surface_tension(T)
    got = np.asarray(jd.surface_tension(T, kerosene_tables))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("T", T_VALUES)
def test_viscosity_kinematic_matches_numpy(kerosene_fuel, kerosene_tables, T):
    expected = kerosene_fuel.viscosity_kinematic(T)
    got = np.asarray(jd.viscosity_kinematic(T, kerosene_tables))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("T", [313.15, 450.0])
def test_activity_matches_numpy(kerosene_fuel, kerosene_tables, T, seed):
    Xs = _dirichlet(kerosene_fuel.num_compounds, n_samples=4, seed=seed)
    for X in Xs:
        expected = kerosene_fuel.activity(X, T)
        got = np.asarray(jd.activity(jnp.asarray(X), T, kerosene_tables))
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("T", T_VALUES)
def test_mixture_properties_match_numpy(kerosene_fuel, kerosene_tables, T):
    Y = kerosene_fuel.Y_0
    Yj = jnp.asarray(Y)
    np.testing.assert_allclose(
        float(jd.mixture_density(Yj, T, kerosene_tables)),
        kerosene_fuel.mixture_density(Y, T),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(jd.mixture_kinematic_viscosity(Yj, T, kerosene_tables)),
        kerosene_fuel.mixture_kinematic_viscosity(Y, T),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(jd.mixture_surface_tension(Yj, T, kerosene_tables)),
        kerosene_fuel.mixture_surface_tension(Y, T),
        rtol=1e-12,
    )


def test_grad_mixture_density_matches_fd(kerosene_fuel, kerosene_tables):
    """Reverse-mode AD vs. central finite differences for ∂ρ_mix/∂Yᵢ."""
    Y = jnp.asarray(kerosene_fuel.Y_0)
    fn = lambda Yv: jd.mixture_density(Yv, 313.15, kerosene_tables)
    g_ad = np.asarray(jax.grad(fn)(Y))

    h = 1e-5
    g_fd = np.zeros_like(g_ad)
    Y_np = np.asarray(Y)
    for i in range(len(Y_np)):
        Yp = Y_np.copy(); Yp[i] += h
        Ym = Y_np.copy(); Ym[i] -= h
        g_fd[i] = (kerosene_fuel.mixture_density(Yp, 313.15)
                   - kerosene_fuel.mixture_density(Ym, 313.15)) / (2 * h)
    rel = np.max(np.abs(g_ad - g_fd)) / (np.max(np.abs(g_fd)) + 1e-30)
    assert rel < 1e-4, f"FD vs AD mismatch on mixture_density grad: {rel:.2e}"


def test_grad_activity_matches_fd_interior(kerosene_fuel, kerosene_tables):
    """
    FD vs AD for ∂(Σ γᵢ)/∂Xⱼ on a strictly-interior composition.

    Boundary X→0 introduces step-functions in numpy via ``where`` masks
    that are not differentiable in the usual sense; the JAX safe-log
    idiom keeps gradients finite (test :func:`test_activity_grad_finite_at_boundary`)
    but FD comparisons must use an interior point.
    """
    rng = np.random.default_rng(7)
    n = kerosene_fuel.num_compounds
    X = rng.dirichlet(np.full(n, 5.0))   # concentrated → interior
    Xj = jnp.asarray(X)
    fn = lambda Xv: jnp.sum(jd.activity(Xv, 400.0, kerosene_tables))
    g_ad = np.asarray(jax.grad(fn)(Xj))

    h = 1e-6
    g_fd = np.zeros_like(g_ad)
    for i in range(n):
        Xp = X.copy(); Xp[i] += h
        Xm = X.copy(); Xm[i] -= h
        g_fd[i] = (np.sum(kerosene_fuel.activity(Xp, 400.0))
                   - np.sum(kerosene_fuel.activity(Xm, 400.0))) / (2 * h)
    rel = np.max(np.abs(g_ad - g_fd)) / (np.max(np.abs(g_fd)) + 1e-30)
    assert rel < 1e-3, f"FD vs AD mismatch on activity grad: {rel:.2e}"


def test_activity_grad_finite_at_boundary(kerosene_fuel, kerosene_tables):
    """At the simplex boundary (Xᵢ=0 for some components) gradients must be finite."""
    n = kerosene_fuel.num_compounds
    X = np.zeros(n); X[:5] = 1.0 / 5
    Xj = jnp.asarray(X)
    fn = lambda Xv: jnp.sum(jd.activity(Xv, 400.0, kerosene_tables))
    g_ad = np.asarray(jax.grad(fn)(Xj))
    assert np.all(np.isfinite(g_ad)), "activity gradient blew up at boundary"
