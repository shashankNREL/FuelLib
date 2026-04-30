"""
``jax_diff`` — Differentiable, JAX-traceable port of the no-SRK FuelLib physics.

This subpackage implements the plan described in the project's design
document for a *differentiable algebraic-equation (DAE) solver* coupled to an
ODE integrator in JAX, plus a Tikhonov-regularised multi-output inverse
problem mapping mixture observables (ASTM cuts, density, viscosity, surface
tension, ...) back to component weight fractions.

Design notes
------------
* Float64 is enabled at import time — most thermodynamic correlations are
  numerically sensitive (e.g. Lee-Kesler at low Tr) and float32 corrupts
  gradient checks.
* The ``fuel`` object's CSV / pandas I/O stays on numpy.  A thin adapter,
  :func:`build_fuel_tables`, packs the data into a JAX-friendly NamedTuple
  (``FuelTables``) once at startup; nothing inside ``jax_diff`` ever
  touches pandas or scipy at runtime, so all downstream functions are
  ``jax.jit`` / ``jax.grad`` compatible.
* The algebraic blocks (Rachford-Rice, bubble point, Stage-2 energy
  balance) are wrapped with :func:`_implicit.implicit_root` so the
  backward pass uses the **implicit function theorem** instead of
  unrolling the iterative forward solve.  This keeps reverse-mode memory
  cost independent of iteration count.

Modules
-------
``properties_jax``
    JAX ports of ``psat``, ``Cp``, ``molar_liquid_vol``,
    ``latent_heat_vaporization``, ``surface_tension``,
    ``viscosity_kinematic``, ``density``, plus mixture properties and
    composition conversions.  Includes the full UNIFAC ``activity``
    routine vectorised with :func:`jax.vmap` over compounds.
``algebraic_jax``
    Differentiable algebraic solvers — Rachford-Rice flash
    (:func:`solve_rr`), bubble-point temperature
    (:func:`solve_bubble_point`), and Stage-2 column-neck energy balance
    (:func:`solve_stage2`).
``inverse_jax``
    Multi-output forward observation map :func:`forward_observations`
    plus the Tikhonov-regularised loss :func:`tikhonov_loss` and the
    adjoint-study helpers (Jacobian SVD, posterior covariance).
"""
from __future__ import annotations

# Locking float64 here is intentional — it must happen before any other
# `jax.numpy` operation is traced.  Importing this package at the top of
# downstream modules is the supported public entry point.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from .properties_jax import (  # noqa: E402,F401
    FuelTables,
    build_fuel_tables,
    psat,
    cp_molar,
    molar_liquid_vol,
    latent_heat_vaporization,
    surface_tension,
    viscosity_kinematic,
    density,
    mass2X,
    Y2X,
    X2Y,
    mixture_density,
    mixture_kinematic_viscosity,
    mixture_surface_tension,
    mean_molecular_weight,
    activity,
)

from .algebraic_jax import (  # noqa: E402,F401
    solve_rr,
    solve_bubble_point,
    solve_stage2,
)

from .distillation_jax import (  # noqa: E402,F401
    DistillationState,
    run_distillation,
    extract_astm_cuts,
)

from .inverse_jax import (  # noqa: E402,F401
    ObservationConfig,
    forward_observations,
    softmax_simplex,
    first_difference_operator,
    tikhonov_loss,
    posterior_covariance,
    jacobian_svd,
)

__all__ = [
    # properties_jax
    "FuelTables",
    "build_fuel_tables",
    "psat",
    "cp_molar",
    "molar_liquid_vol",
    "latent_heat_vaporization",
    "surface_tension",
    "viscosity_kinematic",
    "density",
    "mass2X",
    "Y2X",
    "X2Y",
    "mixture_density",
    "mixture_kinematic_viscosity",
    "mixture_surface_tension",
    "mean_molecular_weight",
    "activity",
    # algebraic_jax
    "solve_rr",
    "solve_bubble_point",
    "solve_stage2",
    # distillation_jax
    "DistillationState",
    "run_distillation",
    "extract_astm_cuts",
    # inverse_jax
    "ObservationConfig",
    "forward_observations",
    "softmax_simplex",
    "first_difference_operator",
    "tikhonov_loss",
    "posterior_covariance",
    "jacobian_svd",
]
