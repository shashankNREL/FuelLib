"""
``inverse_jax`` — Multi-output forward map and Tikhonov-regularised inverse.

This module implements W5 + W6 of the plan:

W5  Forward map ``y = Forward(w, ft, T_refs)``
    where ``w`` are mass fractions, ``y ∈ ℝ^M`` is a stacked observation
    vector containing optionally:

    * ASTM cuts ``T_D86(V%)`` for ``V% ∈ {5,10,20,40,50,70,90,95}``
      (extracted from a user-supplied curve oracle — for testing we
      provide a cheap proxy from ``T(Vfrac) ≈ T_bubble of vapour
      composition at fraction Vfrac``; the full RK2 driver will plug in
      here once W4 is complete).
    * Mixture density ρ(w, Tref) at one or more reference temperatures.
    * Mixture kinematic viscosity ν(w, Tref).
    * Mixture surface tension σ(w, Tref).
    * Average molecular weight ``MW̄ = Σ wᵢ MWᵢ``.

W6  Tikhonov-regularised loss

    L(θ) = ‖W_obs(y(θ) − y_meas)‖²
         + λ₁ ‖L₁ (w(θ) − w_prior)‖²
         + λ₂ ‖L₂ w(θ)‖²
         + λ₃ ‖θ‖²

    plus utilities for adjoint diagnostics:

    * :func:`first_difference_operator` — first-difference operator over
      a chosen permutation of components (carbon-number / Tᵦ ordering).
    * :func:`softmax_simplex` — re-parametrises ``w`` from an
      unconstrained ``θ ∈ ℝⁿ``.
    * :func:`jacobian_svd` — SVD of ``J* = ∂y/∂θ`` at the optimum,
      reporting singular spectrum and null-space directions.
    * :func:`posterior_covariance` — Σ_w from the regularised Hessian.

Why Tikhonov?
-------------
With ``M`` data points (8 cuts + a few property scalars) and
``n_components ≫ M`` the Jacobian of the forward map is rank-deficient.
The unregularised least-squares problem has an
``(n_components − rank(J))``-dimensional null space — many compositions
fit the data identically.  Tikhonov regularisation injects a smoothness
prior (via ``L₂``) and an a-priori composition (via ``L₁ w_prior``),
making the inverse problem well-posed and the Hessian invertible.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from . import properties_jax as P
from .properties_jax import FuelTables


# -----------------------------------------------------------------------------
# W5 — Multi-output forward map
# -----------------------------------------------------------------------------
class ObservationConfig(NamedTuple):
    """
    Description of which mixture observables are included in ``y(w)`` and
    what their per-output measurement uncertainties are.

    All fields are static (Python lists / tuples or scalars) — the
    config itself is treated as ``static`` under :func:`jax.jit`.
    """

    # Reference temperatures (K) at which to evaluate bulk properties.
    T_refs_density: tuple = (288.15,)
    T_refs_kviscosity: tuple = (313.15,)         # 40 °C — ASTM D445
    T_refs_surface_tension: tuple = (288.15,)

    # Per-output 1-σ uncertainties used for ``W_obs = diag(1/σ_obs)``.
    # Units: K for cuts, kg/m^3 for ρ, m^2/s for ν, N/m for σ, kg/mol for MW.
    sigma_cut_K: float = 5.0
    sigma_density: float = 5.0
    sigma_kviscosity: float = 5e-7
    sigma_surface_tension: float = 5e-4
    sigma_mw_avg: float = 5e-3

    # Whether to include each channel.
    include_cuts: bool = True
    include_density: bool = True
    include_kviscosity: bool = True
    include_surface_tension: bool = True
    include_mw_avg: bool = True

    # ASTM volume points (fraction in [0,1]).
    cut_fractions: tuple = (0.05, 0.10, 0.20, 0.40, 0.50, 0.70, 0.90, 0.95)


def forward_observations(
    w: jnp.ndarray,
    ft: FuelTables,
    cfg: ObservationConfig,
    cut_curve_T: jnp.ndarray = None,
    cut_curve_V: jnp.ndarray = None,
):
    """
    Forward observation map ``y(w)``.

    :param w: Mass fractions on the simplex (shape ``(num_compounds,)``).
    :param ft: :class:`FuelTables` snapshot.
    :param cfg: :class:`ObservationConfig` — selects channels and σ.
    :param cut_curve_T: Optional simulated ``T_D86(V)`` curve (shape ``(N,)``)
        from a JAX distillation driver.  When supplied, the ASTM cuts are
        extracted via differentiable :func:`jnp.interp`.  When ``None``,
        the cut channel is filled by a *proxy* model — the bubble
        temperatures of the residual liquid at each cut fraction (see
        :func:`_proxy_cut_curve`).  The proxy is good enough for
        well-posedness / ill-posedness studies on the inverse problem
        and exercises the full forward graph; replace with the W4 RK2
        driver for production calibrations.
    :param cut_curve_V: Optional matching distillate volume axis for
        ``cut_curve_T``.

    :returns: ``(y, sigma)`` where ``y`` is the concatenated observation
        vector and ``sigma`` is the matching per-output uncertainty
        vector.  Both have shape ``(M,)``.
    """
    pieces = []
    sigmas = []

    if cfg.include_cuts:
        if cut_curve_T is None:
            T_cuts = _proxy_cut_curve(w, ft, jnp.asarray(cfg.cut_fractions))
        else:
            # Differentiable interpolation onto the ASTM cut grid.
            T_cuts = jnp.interp(
                jnp.asarray(cfg.cut_fractions),
                cut_curve_V,
                cut_curve_T,
            )
        pieces.append(T_cuts)
        sigmas.append(jnp.full_like(T_cuts, cfg.sigma_cut_K))

    if cfg.include_density:
        rho = jnp.array([P.mixture_density(w, T, ft) for T in cfg.T_refs_density])
        pieces.append(rho)
        sigmas.append(jnp.full_like(rho, cfg.sigma_density))

    if cfg.include_kviscosity:
        nu = jnp.array(
            [P.mixture_kinematic_viscosity(w, T, ft) for T in cfg.T_refs_kviscosity]
        )
        pieces.append(nu)
        sigmas.append(jnp.full_like(nu, cfg.sigma_kviscosity))

    if cfg.include_surface_tension:
        st = jnp.array(
            [P.mixture_surface_tension(w, T, ft) for T in cfg.T_refs_surface_tension]
        )
        pieces.append(st)
        sigmas.append(jnp.full_like(st, cfg.sigma_surface_tension))

    if cfg.include_mw_avg:
        Xi = P.Y2X(w, ft)
        mw_avg = jnp.sum(Xi * ft.MW)
        pieces.append(jnp.atleast_1d(mw_avg))
        sigmas.append(jnp.atleast_1d(jnp.asarray(cfg.sigma_mw_avg)))

    y = jnp.concatenate(pieces)
    sigma = jnp.concatenate(sigmas)
    return y, sigma


def _proxy_cut_curve(w, ft, cut_fractions):
    """
    Fast proxy for an ASTM-cut curve based on equilibrium bubble points
    at progressive evaporation fractions.

    Used for tests and for unit-level inverse-problem studies — does not
    replace the full Stage-1/Stage-2 RK2 distillation driver, but it
    *does* exercise the differentiable property graph and produces
    monotonically increasing cut temperatures vs. distillate fraction
    for a reasonable surrogate, which is sufficient to verify
    well-posedness / Tikhonov behaviour.

    For each fraction ``f``, we approximate the cut as the bubble
    temperature of the *residual* liquid after lumping the lightest ``f``
    fraction (by boiling temperature) into the distillate.  This does
    *not* call the algebraic solver — it is a closed-form expression in
    component boiling points — keeping the proxy cheap and analytic.
    """
    # Sort components by boiling point (numpy-side ordering is fine —
    # ``ft.Tb`` is a static-shape array; we use ``jnp.argsort`` so the
    # routine remains traceable, but the ordering is deterministic for a
    # given fuel).
    Xi = P.Y2X(w, ft)
    order = jnp.argsort(ft.Tb)
    Xi_sorted = Xi[order]
    Tb_sorted = ft.Tb[order]
    # Cumulative mole fraction is our distillate-fraction proxy.
    cum = jnp.cumsum(Xi_sorted)
    # Differentiable interp from cumulative fraction to Tb.
    return jnp.interp(cut_fractions, cum, Tb_sorted)


# -----------------------------------------------------------------------------
# W6 — Tikhonov-regularised inverse problem
# -----------------------------------------------------------------------------
def softmax_simplex(theta: jnp.ndarray) -> jnp.ndarray:
    """Re-parametrise an unconstrained ``θ ∈ ℝⁿ`` onto the simplex via softmax."""
    return jax.nn.softmax(theta)


def first_difference_operator(n: int, perm: Sequence[int] | None = None) -> jnp.ndarray:
    """
    First-difference operator ``L₂`` of shape ``(n−1, n)``.

    Acting on a vector ``w`` ordered by ``perm`` (default identity), it
    returns ``[w[perm[1]] − w[perm[0]], …, w[perm[n−1]] − w[perm[n−2]]]``,
    so ``‖L₂ w‖²`` penalises spikes between adjacent (by carbon-number /
    boiling point) components.

    :param n: Length of the parameter vector.
    :param perm: Optional permutation specifying the ordering along
        which differences are taken.  If ``None``, the identity ordering
        is used — for fuel inversion the recommended choice is the
        ascending-Tᵦ permutation, see e.g.
        ``perm = np.argsort(fuel.Tb)``.

    :returns: ``(n−1, n)`` JAX array.
    """
    L = jnp.zeros((n - 1, n))
    if perm is None:
        perm_arr = jnp.arange(n)
    else:
        perm_arr = jnp.asarray(perm, dtype=jnp.int32)
    rows = jnp.arange(n - 1)
    L = L.at[rows, perm_arr[:-1]].set(-1.0)
    L = L.at[rows, perm_arr[1:]].set(1.0)
    return L


def tikhonov_loss(
    theta: jnp.ndarray,
    y_meas: jnp.ndarray,
    sigma_obs: jnp.ndarray,
    forward_fn,
    *,
    w_prior: jnp.ndarray = None,
    L1: jnp.ndarray = None,
    L2: jnp.ndarray = None,
    lam1: float = 0.0,
    lam2: float = 0.0,
    lam3: float = 0.0,
) -> jnp.ndarray:
    """
    Generalised Tikhonov objective — see module docstring.

    :param theta: Unconstrained parameter, ``w = softmax(θ)``.
    :param y_meas: Measured observation vector (shape ``(M,)``).
    :param sigma_obs: Per-output 1-σ uncertainties (shape ``(M,)``);
        used to build ``W_obs = diag(1/σ_obs)``.
    :param forward_fn: ``θ → y(θ)`` — ``forward_observations`` curried
        with ``ft`` and ``cfg``.
    :param w_prior: A-priori mass fractions for the zero-order term.
        ``None`` ⇒ no zero-order term (equivalent to ``λ₁ = 0``).
    :param L1: ``(k1, n)`` operator for the zero-order term (default
        identity).
    :param L2: ``(k2, n)`` operator for the first-order term — typically
        :func:`first_difference_operator`.
    :param lam1, lam2, lam3: regularisation weights.  Tune via L-curve /
        GCV.

    :returns: Scalar loss (differentiable in ``θ``).
    """
    w = softmax_simplex(theta)
    y = forward_fn(theta)
    misfit = (y - y_meas) / sigma_obs
    loss = jnp.sum(misfit ** 2)

    if lam1 > 0.0 and w_prior is not None:
        if L1 is None:
            diff = w - w_prior
        else:
            diff = L1 @ (w - w_prior)
        loss = loss + lam1 * jnp.sum(diff ** 2)

    if lam2 > 0.0 and L2 is not None:
        Lw = L2 @ w
        loss = loss + lam2 * jnp.sum(Lw ** 2)

    if lam3 > 0.0:
        loss = loss + lam3 * jnp.sum(theta ** 2)

    return loss


def jacobian_svd(forward_fn, theta_star: jnp.ndarray):
    """
    SVD of ``J* = ∂y/∂θ`` at the optimum.

    Reports the singular spectrum and right-singular vectors so the
    caller can inspect *which composition modes the chosen outputs are
    blind to* — these are the right-singular vectors corresponding to
    the smallest singular values.

    :returns: ``(U, s, Vt)`` from :func:`jnp.linalg.svd`.
    """
    J = jax.jacrev(forward_fn)(theta_star)  # (M, n)
    return jnp.linalg.svd(J, full_matrices=False)


def posterior_covariance(
    forward_fn,
    theta_star: jnp.ndarray,
    sigma_obs: jnp.ndarray,
    *,
    L1: jnp.ndarray = None,
    L2: jnp.ndarray = None,
    lam1: float = 0.0,
    lam2: float = 0.0,
    lam3: float = 0.0,
):
    """
    Linearised posterior covariance ``Σ_θ`` evaluated at ``θ*``.

    Σ_θ ≈ (Jᵀ W² J + λ₁ L₁ᵀL₁ + λ₂ L₂ᵀL₂ + λ₃ I)⁻¹

    The propagated covariance on ``w = softmax(θ)`` (1-σ error bars per
    component) is computed via the linearised Jacobian
    ``∂w/∂θ = diag(w) − wwᵀ`` and returned alongside.

    :returns: ``(Sigma_theta, sigma_w)`` where ``sigma_w[i]`` is the
        1-σ uncertainty on component ``i``'s mass fraction.
    """
    J = jax.jacrev(forward_fn)(theta_star)
    W2 = jnp.diag(1.0 / sigma_obs ** 2)
    H = J.T @ W2 @ J

    n = theta_star.shape[0]
    if lam1 > 0.0:
        if L1 is None:
            H = H + lam1 * jnp.eye(n)
        else:
            H = H + lam1 * (L1.T @ L1)
    if lam2 > 0.0 and L2 is not None:
        H = H + lam2 * (L2.T @ L2)
    if lam3 > 0.0:
        H = H + lam3 * jnp.eye(n)

    Sigma_theta = jnp.linalg.pinv(H)

    w = softmax_simplex(theta_star)
    Jw = jnp.diag(w) - jnp.outer(w, w)  # ∂w/∂θ
    Sigma_w = Jw @ Sigma_theta @ Jw.T
    sigma_w = jnp.sqrt(jnp.clip(jnp.diag(Sigma_w), 0.0, None))
    return Sigma_theta, sigma_w
