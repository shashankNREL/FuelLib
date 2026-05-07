"""
nasa7_vapor_cp — Ideal-gas vapour Cp from NASA7 polynomials (JAX-ready).

Cantera is used **only at initialisation** to parse a mechanism YAML and
extract NASA7 polynomial coefficients.  All runtime Cp evaluation uses
pure NumPy array operations (polynomial arithmetic + ``np.where`` for the
temperature branch), making it directly JAX-traceable when
``import numpy as np`` is replaced with ``import jax.numpy as np``.

Typical usage
-------------
>>> from nasa7_vapor_cp import NASA7VaporCp
>>> nasa7 = NASA7VaporCp("mechanism.yaml", fuel_obj)
>>> cp_mix = nasa7.mixture_cp(T=500.0, Yi=vapor_mole_fractions)

Integration with :mod:`distillation`
------------------------------------
>>> import distillation
>>> from nasa7_vapor_cp import make_vapor_cp_function
>>> distillation.calculate_vapor_heat_capacity = make_vapor_cp_function(
...     "mechanism.yaml", fuel_obj
... )
"""

from __future__ import annotations

import numpy as np

# Universal gas constant  J/(mol·K)
R_UNIVERSAL: float = 8.314462618


class NASA7VaporCp:
    """Pure-array ideal-gas Cp evaluator from NASA7 polynomial coefficients.

    Parameters
    ----------
    mechanism_yaml : str
        Path to the Cantera-format mechanism YAML file.
    fuel_obj : FuelLib.fuel
        Initialised :class:`FuelLib.fuel` object.  Must have a
        ``pelephysics_keys`` attribute (list of Cantera species names,
        one per compound, read from the ``PelePhysics Key`` column in
        the fuel's ``_init.csv``).

    Raises
    ------
    ValueError
        If ``fuel_obj`` lacks ``pelephysics_keys`` or if a key is not
        found in the mechanism YAML.

    Notes
    -----
    * Cantera is imported and used **only inside ``__init__``** to parse
      the YAML and extract polynomial coefficients.  After construction,
      the object holds only NumPy arrays — no Cantera state is retained.
    * The coefficient arrays have shape ``(n_fuel, 7)`` and are indexed
      identically to ``fuel_obj``'s compound ordering, so the ``Yi``
      vector from the distillation simulation can be used directly via
      ``np.dot(Yi, cp_per_species(T))``.
    """

    def __init__(self, mechanism_yaml: str, fuel_obj) -> None:
        # ---- guard ---------------------------------------------------------
        if fuel_obj.pelephysics_keys is None:
            raise ValueError(
                "fuel_obj has no 'pelephysics_keys'.  The fuel's _init.csv "
                "must contain a 'PelePhysics Key' column for NASA7 Cp mapping."
            )

        # ---- import Cantera (init-time only) --------------------------------
        import cantera as ct  # noqa: E402  — deliberate lazy import

        gas = ct.Solution(mechanism_yaml)

        n_fuel: int = fuel_obj.num_compounds
        pele_keys: list[str] = fuel_obj.pelephysics_keys

        # ---- extract NASA7 coefficients for the fuel's compounds only -------
        self.n_fuel: int = n_fuel
        self.T_mid: np.ndarray = np.zeros(n_fuel)
        self.a_lo: np.ndarray = np.zeros((n_fuel, 7))   # low-T coefficients
        self.a_hi: np.ndarray = np.zeros((n_fuel, 7))    # high-T coefficients
        self.species_names: list[str] = []

        for i, key in enumerate(pele_keys):
            key_stripped = key.strip()
            sp_idx = gas.species_index(key_stripped)
            if sp_idx < 0:
                raise ValueError(
                    f"PelePhysics key '{key_stripped}' (FuelLib compound {i}) "
                    f"not found in {mechanism_yaml}"
                )
            sp = gas.species(sp_idx)
            # ``sp.thermo.coeffs`` layout (NASA7):
            #   [T_mid, a_hi[0..6], a_lo[0..6]]   — 15 elements total
            coeffs = sp.thermo.coeffs
            self.T_mid[i] = coeffs[0]
            self.a_hi[i, :] = coeffs[1:8]
            self.a_lo[i, :] = coeffs[8:15]
            self.species_names.append(key_stripped)

        # Molecular weights for optional mass-basis conversion (kg/mol).
        self.MW: np.ndarray = fuel_obj.MW.copy()

        # ---- Cantera is never called again after this point. ----------------

    # ------------------------------------------------------------------
    # Runtime evaluation — pure array ops (JAX-ready)
    # ------------------------------------------------------------------

    def cp_per_species(self, T: float) -> np.ndarray:
        """Ideal-gas Cp for each fuel compound at temperature *T*.

        Parameters
        ----------
        T : float
            Temperature in Kelvin.

        Returns
        -------
        cp : np.ndarray, shape ``(n_fuel,)``
            Cp in **J/mol/K** for each compound.

        Notes
        -----
        The NASA7 Cp/R polynomial is:

        .. math::

            C_p / R = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4

        with separate coefficient sets for ``T < T_mid`` (low) and
        ``T >= T_mid`` (high).  The branch is selected with ``np.where``
        (not a Python ``if``) so that JAX can trace through it.
        """
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T

        # Low-T branch
        cp_lo = (
            self.a_lo[:, 0]
            + self.a_lo[:, 1] * T
            + self.a_lo[:, 2] * T2
            + self.a_lo[:, 3] * T3
            + self.a_lo[:, 4] * T4
        )

        # High-T branch
        cp_hi = (
            self.a_hi[:, 0]
            + self.a_hi[:, 1] * T
            + self.a_hi[:, 2] * T2
            + self.a_hi[:, 3] * T3
            + self.a_hi[:, 4] * T4
        )

        # np.where is JAX-traceable (no Python control flow)
        cp_over_R = np.where(T < self.T_mid, cp_lo, cp_hi)

        return cp_over_R * R_UNIVERSAL  # J/mol/K

    def mixture_cp(self, T: float, Yi: np.ndarray) -> float:
        """Mole-fraction-averaged ideal-gas Cp for the fuel vapour.

        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        Yi : np.ndarray, shape ``(n_fuel,)``
            Vapour-phase mole fractions (same ordering as ``fuel_obj``).

        Returns
        -------
        float
            Mixture Cp in J/mol/K.
        """
        cp_i = self.cp_per_species(T)  # (n_fuel,)  J/mol/K
        return float(np.dot(Yi, cp_i))


# ======================================================================
# Convenience factory
# ======================================================================

def make_vapor_cp_function(mechanism_yaml: str, fuel_obj):
    """Return a callable matching the signature of
    :func:`distillation.calculate_vapor_heat_capacity`.

    Parameters
    ----------
    mechanism_yaml : str
        Path to the Cantera mechanism YAML file.
    fuel_obj : FuelLib.fuel
        Initialised fuel object.

    Returns
    -------
    callable
        ``cp_vap(fuel_obj, T, Yi) -> float``  in J/mol/K.
    """
    nasa7 = NASA7VaporCp(mechanism_yaml, fuel_obj)

    def cp_vap(fuel_obj_, T, Yi):
        return nasa7.mixture_cp(T, Yi)

    return cp_vap
