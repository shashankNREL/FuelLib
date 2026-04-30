"""
Implicit function theorem helpers for JAX.

For an algebraic constraint ``g(x*; θ) = 0`` solved iteratively in the
forward pass, the implicit function theorem gives the gradient of the
solution ``x*`` with respect to parameters ``θ`` directly from the
*converged* point::

    ∂x*/∂θ = − (∂g/∂x|_*)⁻¹ · ∂g/∂θ|_*

so the reverse-mode graph need not contain any of the iterative-solver
operations (no unrolled bisection / Newton iterations and the associated
quadratic memory blow-up under ``jax.grad``).

The implementation here uses :func:`jax.custom_vjp`.  The forward pass is
the user-supplied iterative solver.  The backward pass evaluates the two
partials of ``g`` *locally* at ``x*`` via :func:`jax.jvp` / :func:`jax.grad`
and applies the IFT formula.

Currently only the **scalar root** case ``x* ∈ ℝ`` is needed — bubble
point, Rachford-Rice and Stage-2 outer T₂ are all 1-D root finds, with a
PyTree of parameters ``θ``.  A vector-valued generalisation is sketched
in the docstring of :func:`implicit_scalar_root` but is not used in this
package.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def implicit_scalar_root(
    residual_fn: Callable[[jax.Array, Any], jax.Array],
    forward_fn: Callable[[Any], jax.Array],
) -> Callable[[Any], jax.Array]:
    """
    Build a differentiable scalar root-finder.

    :param residual_fn: ``g(x, θ) → ℝ`` — the algebraic residual.
    :param forward_fn:  ``θ → x*`` — *non-differentiable* iterative solver
        that returns a converged scalar root of ``g`` at parameters ``θ``.
        Anything is allowed: bisection, Newton, Brent's, hybrid, etc.

    :returns: A function ``θ → x*`` that is forward-equivalent to
        ``forward_fn`` but whose reverse-mode gradient with respect to
        ``θ`` is computed via the implicit function theorem instead of
        differentiating through the forward iterations.

    Vector-valued generalisation
    ----------------------------
    For a vector root ``g(x, θ) = 0`` with ``x ∈ ℝⁿ``, the analogous IFT
    formula uses the Jacobian inverse:
    ``∂x*/∂θ = −(∂g/∂x)⁻¹ ∂g/∂θ``.  In reverse mode the relevant linear
    solve is ``(∂g/∂x)ᵀ λ = co_x`` followed by ``co_θ = −∂g/∂θᵀ λ``.
    The helper below implements the scalar specialisation, which is what
    every algebraic block in this package needs.
    """

    @jax.custom_vjp
    def solve(theta):
        return forward_fn(theta)

    def fwd(theta):
        x_star = forward_fn(theta)
        return x_star, (x_star, theta)

    def bwd(res, cotangent):
        x_star, theta = res
        # ∂g/∂x evaluated at the converged point (scalar).
        dg_dx = jax.grad(residual_fn, argnums=0)(x_star, theta)
        # Pullback of θ ↦ g(x*, θ) at the cotangent  −cotangent / dg_dx
        # (equivalent to applying the IFT formula via vjp).
        _, vjp_theta = jax.vjp(lambda th: residual_fn(x_star, th), theta)
        # ∂x*/∂θ = − ∂g/∂θ / ∂g/∂x  ⇒  cotθ = −cotx · ∂g/∂θ / ∂g/∂x
        scale = -cotangent / dg_dx
        (theta_cot,) = vjp_theta(scale)
        return (theta_cot,)

    solve.defvjp(fwd, bwd)
    return solve


def safe_log(x: jax.Array, eps: float = 1e-300) -> jax.Array:
    """
    Compute :math:`\\log x` with an autodiff-safe fallback at ``x ≤ 0``.

    Uses the ``jnp.log(jnp.where(x>0, x, 1.0))`` idiom so reverse-mode AD
    sees a finite value at the masked branch (the gradient is masked
    further upstream by an outer ``jnp.where`` on the *result*, which is
    the standard JAX recipe for piecewise-defined functions).
    """
    safe = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.log(safe), -jnp.inf)


def safe_div(a: jax.Array, b: jax.Array) -> jax.Array:
    """``a / b`` with the gradient-friendly mask ``where(b!=0, a/b, 0)``."""
    safe_b = jnp.where(jnp.abs(b) > 0.0, b, 1.0)
    return jnp.where(jnp.abs(b) > 0.0, a / safe_b, 0.0)
