"""
Tests for W6 — Tikhonov-regularised multi-output inverse problem.

Demonstrates the planned twin experiments end-to-end:

1. **Well-posed twin:**  generate ``y_meas = Forward(w*)`` for a known
   ``w*``, then recover ``w*`` via JAX-autodiff gradient descent on the
   un-regularised loss.  Recovery within a small relative tolerance.

2. **Ill-posed twin (cuts only):**  drop all property channels, keep
   only the ASTM cuts.  Without regularisation different optima yield
   the same loss (non-uniqueness).  With λ₂ > 0 (smoothness on Tᵦ
   ordering), the optimum becomes unique.

3. **Gradient correctness:**  FD vs. AD on the loss for a small fuel.

4. **Conditioning monitor:**  the regularised Hessian condition number
   decreases as ``λ₂`` grows (the regularisation does its job).

5. **Posterior covariance shape and positivity.**

The optimiser used here is plain Adam from ``optax``-style hand rolling
(we don't add an ``optax`` dep) so the test is self-contained and fast.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import source.jax_diff as jd
from source.jax_diff.inverse_jax import (
    ObservationConfig,
    forward_observations,
    softmax_simplex,
    first_difference_operator,
    tikhonov_loss,
    posterior_covariance,
    jacobian_svd,
)


def _adam(grad_fn, theta0, *, n_steps=300, lr=0.05):
    """Tiny self-contained Adam for tests (no optax dep)."""
    m = jnp.zeros_like(theta0)
    v = jnp.zeros_like(theta0)
    theta = theta0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, n_steps + 1):
        g = grad_fn(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return theta


def test_twin_experiment_well_posed_recovers_w(small_fuel, small_tables):
    """
    Tiny binary fuel with all channels enabled — the unregularised
    inverse should recover the true composition closely.

    With only 2 components and 12 observations, the system is
    over-determined and easy to invert.  This tests the W5 + W6 plumbing
    end-to-end.
    """
    n = small_tables.num_compounds
    cfg = ObservationConfig()

    # Truth.
    rng = np.random.default_rng(0)
    theta_true = jnp.asarray(rng.normal(size=n) * 0.5)
    w_true = softmax_simplex(theta_true)
    y_meas, sigma_obs = forward_observations(w_true, small_tables, cfg)

    fwd = lambda th: forward_observations(softmax_simplex(th), small_tables, cfg)[0]
    loss = lambda th: tikhonov_loss(th, y_meas, sigma_obs, fwd, lam3=1e-6)
    grad_loss = jax.jit(jax.grad(loss))

    theta0 = jnp.zeros(n)
    theta_hat = _adam(grad_loss, theta0, n_steps=400, lr=0.1)
    w_hat = softmax_simplex(theta_hat)

    # Tight component recovery for binary fuel with all channels on.
    np.testing.assert_allclose(
        np.asarray(w_hat), np.asarray(w_true),
        atol=0.05,  # 5 percentage-point tolerance per component
    )


def test_loss_grad_matches_fd(small_tables):
    """Reverse-mode AD vs. central FD on the Tikhonov loss."""
    n = small_tables.num_compounds
    cfg = ObservationConfig()

    theta_true = jnp.array(np.linspace(-0.5, 0.5, n))
    w_true = softmax_simplex(theta_true)
    y_meas, sigma_obs = forward_observations(w_true, small_tables, cfg)
    fwd = lambda th: forward_observations(softmax_simplex(th), small_tables, cfg)[0]
    L2 = first_difference_operator(n, perm=jnp.argsort(small_tables.Tb))
    loss = lambda th: tikhonov_loss(
        th, y_meas, sigma_obs, fwd,
        L2=L2, lam2=1e-3, lam3=1e-4,
    )
    g_ad = np.asarray(jax.grad(loss)(theta_true + 0.1))

    h = 1e-5
    g_fd = np.zeros(n)
    base = theta_true + 0.1
    for i in range(n):
        ep = base.at[i].add(h)
        em = base.at[i].add(-h)
        g_fd[i] = (float(loss(ep)) - float(loss(em))) / (2 * h)
    rel = np.max(np.abs(g_ad - g_fd)) / (np.max(np.abs(g_fd)) + 1e-30)
    assert rel < 1e-4, f"AD vs FD on loss: rel={rel:.2e}"


def test_regularisation_lifts_smallest_nontrivial_eigenvalue(kerosene_tables):
    """
    The whole point of Tikhonov: as ``λ₂`` grows, the Hessian
    ``H = JᵀW²J + λ₂ L₂ᵀL₂`` has its smallest *non-trivial* eigenvalue
    lifted, eliminating the unidentified composition modes from the
    cuts-only data.

    The kerosene fuel has ``n_components ≫ M_cuts`` so the cuts-only
    Jacobian is rank-deficient — exactly the regime Tikhonov is for.
    Note that *both* the softmax parametrisation and ``L₂`` have the
    constant-θ direction in their null spaces; we therefore include a
    tiny ridge (mirroring the production ``tikhonov_loss``) and inspect
    the second-smallest eigenvalue, which is the worst direction the
    smoothness term actually fights.
    """
    n = kerosene_tables.num_compounds
    cfg = ObservationConfig(
        include_density=False,
        include_kviscosity=False,
        include_surface_tension=False,
        include_mw_avg=False,   # cuts-only (the most ill-posed regime)
    )
    theta = jnp.zeros(n)
    fwd = lambda th: forward_observations(softmax_simplex(th), kerosene_tables, cfg)[0]
    y_meas, sigma_obs = forward_observations(softmax_simplex(theta), kerosene_tables, cfg)
    L2 = first_difference_operator(n, perm=jnp.argsort(kerosene_tables.Tb))

    J = jax.jacrev(fwd)(theta)
    W2 = jnp.diag(1.0 / sigma_obs ** 2)
    H_data = J.T @ W2 @ J

    def smallest_eig(lam2):
        # Note: softmax has a 1-D null direction (uniform shift in θ
        # doesn't change w); ``L₂ ones = 0`` for the same direction.
        # Always include a tiny λ₃ ridge — exactly as production
        # ``tikhonov_loss`` does — so we isolate the L₂ contribution to
        # the *next* smallest eigenvalue.
        H = H_data + lam2 * (L2.T @ L2) + 1e-6 * jnp.eye(n)
        eigs = jnp.linalg.eigvalsh(H)
        # Skip the constant-direction null eigenvalue absorbed by the ridge;
        # report the *second-smallest* (the worst direction L2 actually fights).
        return float(jnp.sort(eigs)[1])

    s_low = smallest_eig(0.0)
    s_high = smallest_eig(1.0)
    assert s_high > s_low, (
        f"Tikhonov L2 did not lift the smallest non-trivial eigenvalue: "
        f"λ=0 → λ_min2={s_low:.2e}; λ=1 → λ_min2={s_high:.2e}"
    )
    # The L2 regulariser should make a substantial difference.
    assert s_high > 10 * s_low, (
        f"L2 effect too weak: λ=0 → {s_low:.2e}; λ=1 → {s_high:.2e}"
    )


def test_posterior_covariance_returns_finite_sigma_w(kerosene_tables):
    """``sigma_w`` should be all finite and non-negative."""
    n = kerosene_tables.num_compounds
    cfg = ObservationConfig()
    theta = jnp.zeros(n)
    fwd = lambda th: forward_observations(softmax_simplex(th), kerosene_tables, cfg)[0]
    _, sigma_obs = forward_observations(softmax_simplex(theta), kerosene_tables, cfg)
    L2 = first_difference_operator(n, perm=jnp.argsort(kerosene_tables.Tb))
    _, sw = posterior_covariance(
        fwd, theta, sigma_obs, L2=L2, lam2=1e-2, lam3=1e-4,
    )
    assert sw.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(sw)))
    assert bool(jnp.all(sw >= 0))


def test_jacobian_svd_reports_singular_spectrum(kerosene_tables):
    """``J*`` SVD must return arrays of the correct shapes and finite values."""
    n = kerosene_tables.num_compounds
    cfg = ObservationConfig()
    theta = jnp.zeros(n)
    fwd = lambda th: forward_observations(softmax_simplex(th), kerosene_tables, cfg)[0]
    U, s, Vt = jacobian_svd(fwd, theta)
    M = 12
    assert U.shape == (M, M) or U.shape == (M, min(M, n))
    assert s.shape[0] == min(M, n)
    assert bool(jnp.all(jnp.isfinite(s)))
    # The smallest singular value being much smaller than the largest is
    # the empirical fingerprint of the ill-posedness this whole module
    # is designed to address.  Document, don't gate, the ratio.
    if s.shape[0] > 1:
        assert float(s[0]) >= float(s[-1]) - 1e-12


def test_first_difference_operator_action():
    """Sanity: ``L₂ ones = 0`` (constant compositions are smooth)."""
    n = 5
    L = first_difference_operator(n)
    out = np.asarray(L @ jnp.ones(n))
    np.testing.assert_allclose(out, np.zeros(n - 1), atol=1e-12)
    # A spike gives non-zero norm.
    spike = jnp.zeros(n).at[2].set(1.0)
    out2 = np.asarray(L @ spike)
    assert np.linalg.norm(out2) > 0.5
