"""
``properties_jax`` — JAX-traceable port of the no-SRK ``FuelLib.fuel`` properties.

This module wraps the methods on :class:`source.FuelLib.fuel` that the
distillation DAE actually needs (``psat``, ``Cp``, ``molar_liquid_vol``,
``latent_heat_vaporization``, ``surface_tension``,
``viscosity_kinematic``, ``density``, mixture properties, composition
conversions, and full UNIFAC ``activity``) so they are pure functions of
``(state, FuelTables)`` and therefore amenable to ``jax.jit``,
``jax.grad``, ``jax.vmap`` and ``jax.lax.scan``.

Every function in this module accepts a :class:`FuelTables` rather than a
``fuel`` instance.  ``FuelTables`` is a frozen ``NamedTuple`` of
``jnp.ndarray`` properties — building it with :func:`build_fuel_tables`
performs all the pandas / file I/O on the numpy side once at startup.

Numerical notes
---------------
Wherever the original numpy code uses ``np.errstate`` masks plus
``np.where`` to side-step ``log(0)`` or ``divide-by-zero`` issues at the
simplex boundary (``Xᵢ = 0``), the JAX port uses the standard
``log(where(x>0, x, 1.0))`` idiom and a ``where(...)`` on the *result*.
This keeps reverse-mode gradients finite at the boundary, which is
critical for the Tikhonov inverse problem where the simplex parametrisation
can drive components arbitrarily close to zero during optimisation.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._implicit import safe_div, safe_log


class FuelTables(NamedTuple):
    """
    JAX-friendly snapshot of a :class:`source.FuelLib.fuel`.

    All array fields are ``jnp.ndarray`` of dtype ``float64``.  Static
    fields (``num_compounds``, ``num_groups``) are plain Python ints —
    they are ``static_argnums`` for any downstream :func:`jax.jit`.

    Field names mirror the corresponding ``fuel`` attributes so that
    porting downstream consumers is mostly a search-and-replace.
    """

    # Sizes (treated as static under jit).
    num_compounds: int
    num_groups: int

    # Per-compound thermodynamic properties.
    MW: jnp.ndarray            # (num_comp,)  kg/mol
    Tc: jnp.ndarray            # (num_comp,)  K
    Pc: jnp.ndarray            # (num_comp,)  Pa
    Vc: jnp.ndarray            # (num_comp,)  m^3/mol
    Tb: jnp.ndarray            # (num_comp,)  K
    Hv_stp: jnp.ndarray        # (num_comp,)  J/mol
    Lv_stp: jnp.ndarray        # (num_comp,)  J/kg
    omega: jnp.ndarray         # (num_comp,)
    Vm_stp: jnp.ndarray        # (num_comp,)  m^3/mol  at 298 K
    Cp_stp: jnp.ndarray        # (num_comp,)  J/mol/K  at 298 K
    Cp_B: jnp.ndarray          # (num_comp,)  Cp T-correction
    Cp_C: jnp.ndarray          # (num_comp,)  Cp T-correction

    # UNIFAC group-contribution data.
    Nij: jnp.ndarray           # (num_comp, num_groups)  group counts
    Rk: jnp.ndarray            # (num_groups,)
    Qk: jnp.ndarray            # (num_groups,)
    unifac_a: jnp.ndarray      # (num_groups, num_groups)  K

    # Initial mass fractions (taken from gcxgc CSV; can be overridden by
    # callers — included so consumers don't need a separate handle to the
    # source ``fuel`` instance).
    Y_0: jnp.ndarray           # (num_comp,)


def build_fuel_tables(fuel_obj) -> FuelTables:
    """
    Snapshot a :class:`source.FuelLib.fuel` into a JAX-friendly tuple.

    Strictly numpy-side: this performs the only float64 → ``jnp.float64``
    copy needed.  The returned tuple is a plain immutable PyTree leaf and
    can be passed across JIT boundaries freely.
    """
    f = fuel_obj
    n_g = f.num_groups
    return FuelTables(
        num_compounds=int(f.num_compounds),
        num_groups=int(n_g),
        MW=jnp.asarray(f.MW, dtype=jnp.float64),
        Tc=jnp.asarray(f.Tc, dtype=jnp.float64),
        Pc=jnp.asarray(f.Pc, dtype=jnp.float64),
        Vc=jnp.asarray(f.Vc, dtype=jnp.float64),
        Tb=jnp.asarray(f.Tb, dtype=jnp.float64),
        Hv_stp=jnp.asarray(f.Hv_stp, dtype=jnp.float64),
        Lv_stp=jnp.asarray(f.Lv_stp, dtype=jnp.float64),
        omega=jnp.asarray(f.omega, dtype=jnp.float64),
        Vm_stp=jnp.asarray(f.Vm_stp, dtype=jnp.float64),
        Cp_stp=jnp.asarray(f.Cp_stp, dtype=jnp.float64),
        Cp_B=jnp.asarray(f.Cp_B, dtype=jnp.float64),
        Cp_C=jnp.asarray(f.Cp_C, dtype=jnp.float64),
        Nij=jnp.asarray(f.Nij[:, :n_g], dtype=jnp.float64),
        Rk=jnp.asarray(f.Rk[:n_g], dtype=jnp.float64),
        Qk=jnp.asarray(f.Qk[:n_g], dtype=jnp.float64),
        unifac_a=jnp.asarray(f.unifac_a[:n_g, :n_g], dtype=jnp.float64),
        Y_0=jnp.asarray(f.Y_0, dtype=jnp.float64),
    )


# -----------------------------------------------------------------------------
# Composition conversions
# -----------------------------------------------------------------------------
def mean_molecular_weight(Yi: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Mean molecular weight ``Mbar = 1 / Σ(Yᵢ/MWᵢ)`` (kg/mol)."""
    s = jnp.sum(Yi / ft.MW)
    # Match the numpy convention: Mbar = 0 when the mixture is empty.
    return jnp.where(s > 0, 1.0 / jnp.where(s > 0, s, 1.0), 0.0)


def mass2X(mass: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Mole fractions from per-component masses."""
    moles = mass / ft.MW
    s = jnp.sum(moles)
    return jnp.where(s > 0, moles / jnp.where(s > 0, s, 1.0), jnp.zeros_like(moles))


def Y2X(Yi: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Mass → mole fractions."""
    return mass2X(Yi, ft)


def X2Y(Xi: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Mole → mass fractions."""
    mass = Xi * ft.MW
    s = jnp.sum(mass)
    return jnp.where(s > 0, mass / jnp.where(s > 0, s, 1.0), jnp.zeros_like(mass))


# -----------------------------------------------------------------------------
# Pure-component properties
# -----------------------------------------------------------------------------
def psat(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """
    Saturated vapor pressure (Pa) — Lee-Kesler correlation.

    Mirrors :meth:`source.FuelLib.fuel.psat` (default branch) including
    the ``Tr ≤ 1`` clamp.
    """
    Tr = jnp.minimum(T / ft.Tc, 1.0)
    # Avoid log(Tr=0); Tr is always positive within the clamp.
    log_Tr = jnp.log(Tr)
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * log_Tr + 0.169347 * Tr ** 6
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * log_Tr + 0.43577 * Tr ** 6
    return ft.Pc * jnp.exp(f0 + ft.omega * f1)


def cp_molar(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Molar specific heat capacity (J/mol/K)."""
    theta = (T - 298.0) / 700.0
    return ft.Cp_stp + ft.Cp_B * theta + ft.Cp_C * theta ** 2


def molar_liquid_vol(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """
    Molar liquid volume (m^3/mol) with temperature correction.

    Replaces the Python ``for i / if T>Tc[i]`` branching from
    :meth:`source.FuelLib.fuel.molar_liquid_vol` with a vectorised
    ``jnp.where`` so the function is JIT- and grad-friendly.
    """
    Tstp = 298.0
    # Branch 1: T ≤ Tc — physical correction.
    phi_phys = ((1.0 - T / ft.Tc) ** (2.0 / 7.0)) - (
        (1.0 - Tstp / ft.Tc) ** (2.0 / 7.0)
    )
    # Branch 2: T > Tc — supercritical fallback.
    phi_super = -((1.0 - Tstp / ft.Tc) ** (2.0 / 7.0))
    phi = jnp.where(T <= ft.Tc, phi_phys, phi_super)
    z = 0.29056 - 0.08775 * ft.omega
    return ft.Vm_stp * jnp.power(z, phi)


def latent_heat_vaporization(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Per-component latent heat of vaporization (J/kg)."""
    Tr = T / ft.Tc
    Trb = ft.Tb / ft.Tc
    physical = ft.Lv_stp * (((1.0 - Tr) / (1.0 - Trb)) ** 0.38)
    return jnp.where(T <= ft.Tc, physical, jnp.zeros_like(physical))


def surface_tension(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Per-component surface tension (N/m) — Brock-Bird correlation."""
    Tr = T / ft.Tc
    Pc_bar = ft.Pc * 1e-5
    Tbr = ft.Tb / ft.Tc
    Q = 0.1196 * (1.0 + (Tbr * jnp.log(Pc_bar / 1.01325)) / (1.0 - Tbr)) - 0.279
    st_dyn_per_cm = (
        Pc_bar ** (2.0 / 3.0)
        * ft.Tc ** (1.0 / 3.0)
        * Q
        * (1.0 - Tr) ** (11.0 / 9.0)
    )
    return st_dyn_per_cm * 1e-3


def viscosity_kinematic(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Per-component kinematic viscosity (m^2/s) — Dutt's equation (4.23)."""
    T_cels = T - 273.15
    Tb_cels = ft.Tb - 273.15
    rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239.0 - 0.19 * Tb_cels)
    return jnp.exp(rhs) * 1e-6


def density(T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Per-component liquid density ρ = MW/Vm (kg/m^3)."""
    return ft.MW / molar_liquid_vol(T, ft)


# -----------------------------------------------------------------------------
# UNIFAC activity
# -----------------------------------------------------------------------------
def _ln_Gamma_group(Theta: jnp.ndarray, Qk: jnp.ndarray, Psi: jnp.ndarray) -> jnp.ndarray:
    """
    UNIFAC residual-term ``ln Γ_m`` for a group surface-area fraction
    vector ``Theta`` (shape ``(num_groups,)``).
    """
    Theta_Psi = Theta @ Psi
    div = safe_div(Theta, Theta_Psi)
    sum_term = div @ Psi
    log_Theta_Psi = safe_log(Theta_Psi)
    return Qk * (1.0 - log_Theta_Psi - sum_term)


def _ln_Gamma_pure_one(Nij_i: jnp.ndarray, Qk: jnp.ndarray, Psi: jnp.ndarray) -> jnp.ndarray:
    """
    UNIFAC ``ln Γ_m`` for the pure component whose group counts are
    ``Nij_i``.  Designed for ``vmap`` over compounds.
    """
    n_sum = jnp.sum(Nij_i)
    n_sum_safe = jnp.where(n_sum > 0, n_sum, 1.0)
    Xi_norm = jnp.where(n_sum > 0, Nij_i / n_sum_safe, jnp.zeros_like(Nij_i))
    Theta_i = Qk * Xi_norm
    Theta_i_sum = Qk @ Xi_norm
    Theta_i_sum_safe = jnp.where(Theta_i_sum > 0, Theta_i_sum, 1.0)
    Theta_i = jnp.where(
        Theta_i_sum > 0,
        Theta_i / Theta_i_sum_safe,
        jnp.zeros_like(Theta_i),
    )
    return jnp.where(n_sum > 0,
                     _ln_Gamma_group(Theta_i, Qk, Psi),
                     jnp.zeros_like(Qk))


def activity(Xi: jnp.ndarray, T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """
    UNIFAC activity coefficients γᵢ.

    Mirrors :meth:`source.FuelLib.fuel.activity` exactly, including the
    ``γᵢ = 1 for Xᵢ = 0`` post-correction.  The per-compound loop in the
    numpy version is lifted with :func:`jax.vmap` over compounds.
    """
    Nij = ft.Nij
    Rk = ft.Rk
    Qk = ft.Qk
    a_mat = ft.unifac_a

    Psi = jnp.exp(-a_mat / T)
    z = 10.0

    r = Nij @ Rk
    q = Nij @ Qk
    L_vec = (z / 2.0) * (r - q) - (r - 1.0)

    sum_xq = Xi @ q
    sum_xr = Xi @ r
    sum_xL = Xi @ L_vec

    sum_xq_safe = jnp.where(sum_xq > 0, sum_xq, 1.0)
    sum_xr_safe = jnp.where(sum_xr > 0, sum_xr, 1.0)
    theta = jnp.where(sum_xq > 0, (Xi * q) / sum_xq_safe, jnp.zeros_like(Xi))
    phi = jnp.where(sum_xr > 0, (Xi * r) / sum_xr_safe, jnp.zeros_like(Xi))

    # Combinatorial: handle Xi = 0 with safe division / log.
    Xi_safe = jnp.where(Xi > 0, Xi, 1.0)
    phi_over_x = jnp.where(Xi > 0, phi / Xi_safe, 0.0)
    phi_safe = jnp.where(phi > 0, phi, 1.0)
    tht_over_phi = jnp.where(phi > 0, theta / phi_safe, 0.0)

    ln_phi_over_x = jnp.where(phi_over_x > 0, jnp.log(jnp.where(phi_over_x > 0, phi_over_x, 1.0)), 0.0)
    ln_tht_over_phi = jnp.where(tht_over_phi > 0,
                                jnp.log(jnp.where(tht_over_phi > 0, tht_over_phi, 1.0)),
                                0.0)

    ln_gamma_C_full = (
        ln_phi_over_x
        + (z / 2.0) * q * ln_tht_over_phi
        + L_vec
        - (phi / sum_xr_safe) * sum_xL
    )
    # If sum_xr == 0 the mixture is empty / degenerate — fall back to L_vec.
    ln_gamma_C_emp = L_vec
    ln_gamma_C = jnp.where(sum_xr > 0, ln_gamma_C_full, ln_gamma_C_emp)

    gamma_C = jnp.where(Xi > 0, jnp.exp(ln_gamma_C), 1.0)

    # Residual part — group mole fractions and surface-area fractions.
    X_vec = Xi @ Nij
    X_sum = jnp.sum(X_vec)
    X_sum_safe = jnp.where(X_sum > 0, X_sum, 1.0)
    X_vec = jnp.where(X_sum > 0, X_vec / X_sum_safe, jnp.zeros_like(X_vec))

    Theta_vec = Qk * X_vec
    Theta_sum = Qk @ X_vec
    Theta_sum_safe = jnp.where(Theta_sum > 0, Theta_sum, 1.0)
    Theta_vec = jnp.where(Theta_sum > 0,
                          Theta_vec / Theta_sum_safe,
                          jnp.zeros_like(Theta_vec))

    lnGamma_mix = _ln_Gamma_group(Theta_vec, Qk, Psi)

    # vmap over compounds for the per-pure-component group activities.
    lnGamma_pure = jax.vmap(_ln_Gamma_pure_one, in_axes=(0, None, None))(
        Nij, Qk, Psi,
    )  # (num_comp, num_groups)

    ln_gamma_R = jnp.sum(Nij * (lnGamma_mix - lnGamma_pure), axis=1)
    gamma_R = jnp.exp(ln_gamma_R)

    return gamma_C * gamma_R


# -----------------------------------------------------------------------------
# Mixture properties
# -----------------------------------------------------------------------------
def mixture_density(Yi: jnp.ndarray, T: jnp.ndarray, ft: FuelTables) -> jnp.ndarray:
    """Mixture density (kg/m^3)."""
    Vmi = molar_liquid_vol(T, ft)
    return Yi @ (ft.MW / Vmi)


def mixture_kinematic_viscosity(
    Yi: jnp.ndarray, T: jnp.ndarray, ft: FuelTables,
) -> jnp.ndarray:
    """Mixture kinematic viscosity (m^2/s) — Kendall-Monroe mixing rule."""
    nu_i = viscosity_kinematic(T, ft)
    Xi = Y2X(Yi, ft)
    return jnp.sum(Xi * (nu_i ** (1.0 / 3.0))) ** 3


def mixture_surface_tension(
    Yi: jnp.ndarray, T: jnp.ndarray, ft: FuelTables,
) -> jnp.ndarray:
    """Mixture surface tension (N/m) — arithmetic (mole-weighted) mixing."""
    Xi = Y2X(Yi, ft)
    sti = surface_tension(T, ft)
    return jnp.sum(Xi * sti)


# -----------------------------------------------------------------------------
# Validation helper — useful in tests and for `--update-jax-baselines`.
# -----------------------------------------------------------------------------
def numpy_vs_jax_property_table(fuel_obj, T: float, Xi=None) -> dict:
    """
    Return a dict of numpy-vs-jax property values useful for debugging
    JAX-port regressions.  Always allocates new arrays — not for use on
    the hot path.
    """
    ft = build_fuel_tables(fuel_obj)
    if Xi is None:
        Xi = np.asarray(fuel_obj.Y2X(fuel_obj.Y_0))
    Xi_jnp = jnp.asarray(Xi, dtype=jnp.float64)
    return {
        "psat": (np.asarray(fuel_obj.psat(T)), np.asarray(psat(T, ft))),
        "Cp": (np.asarray(fuel_obj.Cp(T)), np.asarray(cp_molar(T, ft))),
        "molar_liquid_vol": (
            np.asarray(fuel_obj.molar_liquid_vol(T)),
            np.asarray(molar_liquid_vol(T, ft)),
        ),
        "latent_heat_vaporization": (
            np.asarray(fuel_obj.latent_heat_vaporization(T)),
            np.asarray(latent_heat_vaporization(T, ft)),
        ),
        "surface_tension": (
            np.asarray(fuel_obj.surface_tension(T)),
            np.asarray(surface_tension(T, ft)),
        ),
        "viscosity_kinematic": (
            np.asarray(fuel_obj.viscosity_kinematic(T)),
            np.asarray(viscosity_kinematic(T, ft)),
        ),
        "activity": (
            np.asarray(fuel_obj.activity(Xi, T)),
            np.asarray(activity(Xi_jnp, T, ft)),
        ),
    }
