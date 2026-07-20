import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)
from paths import *

# Standard-state enthalpies of formation at 298.15 K used by the Hess-cycle
# combustion helper below. Product state is gaseous water — this yields the
# net (lower) heating value that ASTM D4809/D3338 measure for aviation fuels.
_HF_CO2_G_JMOL = -393.51e3
_HF_H2O_G_JMOL = -241.83e3


def _psat_lee_kesler(T, Tc, Pc, omega):
    """
    Lee-Kesler vapor pressure per compound.

    Duplicates the closed-form used inside ``fuel.psat`` but keeps it as a
    pure array function so that JAX-friendly helpers below can call it
    without going through the ``fuel`` method dispatch.

    :param T: Temperature in Kelvin (scalar).
    :type T: float
    :param Tc: Critical temperature per compound in K.
    :type Tc: np.ndarray
    :param Pc: Critical pressure per compound in Pa.
    :type Pc: np.ndarray
    :param omega: Acentric factor per compound.
    :type omega: np.ndarray
    :return: Saturation pressure per compound in Pa.
    :rtype: np.ndarray
    """
    Tr = T / Tc
    f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
    f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
    return Pc * np.exp(f0 + omega * f1)


def _fp_alqaheem(Tb):
    """
    Alqaheem-Riazi 2017 pure-component flash point: FP = 0.70 * Tb.

    Reported AAD 1.7 % on 140 hydrocarbons (Alqaheem & Riazi 2017,
    *Energy & Fuels* 31, doi 10.1021/acs.energyfuels.6b02669).

    :param Tb: Per-compound normal boiling point in K.
    :type Tb: np.ndarray
    :return: Per-compound flash point in K.
    :rtype: np.ndarray
    """
    return 0.70 * Tb


def _fp_alibakhshi(Tb, phi_sum):
    """
    Alibakhshi et al. 2015 pure-component flash point.

    Model form: :math:`FP = 12.14 + 0.73 \\cdot NBP + \\sum_i n_i \\phi_i`.
    Reported AAD 5.83 K, AARE 1.61 % on 1533 organics (Alibakhshi,
    Mirshahvalad, Alibakhshi, *Ind. Eng. Chem. Res.* 54, 11230 (2015),
    ``papers/a-modified-group-contribution-method-...pdf``).

    :param Tb: Per-compound normal boiling point in K.
    :type Tb: np.ndarray
    :param phi_sum: Per-compound sum of Alibakhshi group phi_i contributions,
        pre-projected from Alibakhshi's 42 functional groups onto the CG set
        (see ``tools/build_gcm_extended.py::CG_TO_ALIBAKHSHI``).
    :type phi_sum: np.ndarray
    :return: Per-compound flash point in K.
    :rtype: np.ndarray
    """
    return 12.14 + 0.73 * Tb + phi_sum


def _fp_liaw_ideal_iter(Xi, Tf_i, Tc, Pc, omega, n_iter=10):
    """
    Solve the ideal Liaw-Chiu flash-point criterion for a mixture.

    Modified Le Chatelier form (Liaw & Chiu 2006, activity coefficients set
    to unity — reasonable for jet-fuel HC-HC mixtures per Paricaud et al.
    *Fuel* 263, 116534 (2020)):

    .. math::

        \\sum_i \\frac{x_i \\, P_i^{sat}(T)}{P_i^{sat}(T_{fp,i})} = 1

    Solved with a fixed-iteration Newton method on the residual to keep the
    helper JAX-portable (no data-dependent while loops). Initial guess is the
    mole-fraction-weighted average of the pure-compound flash points, which
    is inside the correct basin for typical multi-component fuels.

    :param Xi: Mole fractions of the compounds.
    :type Xi: np.ndarray
    :param Tf_i: Per-compound pure flash point in K.
    :type Tf_i: np.ndarray
    :param Tc: Per-compound critical temperature in K.
    :type Tc: np.ndarray
    :param Pc: Per-compound critical pressure in Pa.
    :type Pc: np.ndarray
    :param omega: Per-compound acentric factor.
    :type omega: np.ndarray
    :param n_iter: Number of Newton iterations (default 10, converges in <=8).
    :type n_iter: int, optional
    :return: Mixture flash point in K.
    :rtype: float
    """
    psat_ref = _psat_lee_kesler(Tf_i, Tc, Pc, omega)

    def residual(T):
        psat_T = _psat_lee_kesler(T, Tc, Pc, omega)
        return float(np.sum(Xi * psat_T / psat_ref) - 1.0)

    T = float(np.sum(Xi * Tf_i))
    dT = 1.0  # K, finite-difference step for the derivative estimate
    for _ in range(n_iter):
        r = residual(T)
        r_up = residual(T + dT)
        dr_dT = (r_up - r) / dT
        step = r / (dr_dT + 1e-30)
        step = float(np.clip(step, -20.0, 20.0))
        T = T - step
    return T


def _boehm2022_iter(x_j, Tm_j, dHfus_j, dSfus_j, dCp_j, alpha=0.25, n_iter=8):
    """
    Solve Boehm et al. 2022 eq 21 for the mixture freeze point ``T_f,mix,j``
    driven by a single high-freeze-point component j.

    :meta private: Source: Boehm, Coburn, Yang, Wanstall, Heyne, *Energy &
    Fuels* 36, 12046 (2022), eq 21, ``papers/blend-prediction-model-...pdf``.
    Fixed-point iteration converges within 5 iterations per Boehm's stated
    convergence; kept at 8 iterations for JAX-friendly fixed loop.

    :param x_j: Mole fraction of component j in the mixture.
    :type x_j: float
    :param Tm_j: Pure freeze (melting) point of j in K.
    :type Tm_j: float
    :param dHfus_j: Enthalpy of fusion of j in J/mol.
    :type dHfus_j: float
    :param dSfus_j: Entropy of fusion of j in J/mol/K.
    :type dSfus_j: float
    :param dCp_j: Solid-minus-liquid heat capacity for j in J/mol/K
        (``Cp_solid - Cp_liq``, negative for typical HCs).
    :type dCp_j: float
    :param alpha: Empirical scaling factor for the mixing entropy term.
        Boehm 2022 fit alpha = 0.25 from bicyclohexyl blend data.
    :type alpha: float, optional
    :param n_iter: Number of fixed-point iterations.
    :type n_iter: int, optional
    :return: Mixture freeze point (T_f,mix,j) in K.
    :rtype: float
    """
    R = 8.31446
    # Boehm eq 20 — mixing entropy of a binary "j vs rest" split.
    # Guard against log(0) at endpoints; x_j is clipped to (1e-6, 1-1e-6).
    x_safe = np.clip(x_j, 1e-6, 1.0 - 1e-6)
    dS_mix = (
        -R / x_safe * ((1.0 - x_safe) * np.log(1.0 - x_safe) + x_safe * np.log(x_safe))
    )
    T = float(Tm_j)  # initial guess: pure freeze point
    for _ in range(n_iter):
        # Clip T strictly positive so log(T/Tm) stays real. The iteration
        # can transiently overshoot into T <= 0 for very dilute components;
        # clamping keeps the iteration in the physically sensible region
        # without changing the converged value.
        T = max(T, 1.0)
        num = dHfus_j + x_safe * dCp_j * (Tm_j - T)
        den = dSfus_j + x_safe * dCp_j * np.log(T / Tm_j) + alpha * dS_mix
        T = num / (den + 1e-30)
    # Final clip: return NaN sentinel if we ended below zero (extreme
    # dilution + poor Walden estimates); the outer max_over_j will drop it.
    return float(T) if T > 0 else float("-inf")


def _freeze_max_over_j(Xi, Tm_i, dHfus_i, dSfus_i, dCp_i, alpha=0.25):
    """
    Return the mixture freeze point as the max over per-component eq-21 solves.

    :meta private: The physical freeze point is the temperature at which the
    first crystal appears on cooling; if compound j freezes at temperature
    T_f,j when at mole fraction x_j, then the mixture freeze point is
    ``max_j T_f,mix,j(x_j)`` — the highest-freezing component wins. Bell 2025
    uses the same outer structure over their control-curve inner solve.

    :param Xi: Mole fractions of the compounds.
    :type Xi: np.ndarray
    :param Tm_i: Per-compound pure freeze point in K.
    :type Tm_i: np.ndarray
    :param dHfus_i: Per-compound enthalpy of fusion in J/mol.
    :type dHfus_i: np.ndarray
    :param dSfus_i: Per-compound entropy of fusion in J/mol/K.
    :type dSfus_i: np.ndarray
    :param dCp_i: Per-compound Cp_solid - Cp_liq in J/mol/K.
    :type dCp_i: np.ndarray
    :param alpha: Empirical mixing-entropy scaling (Boehm 2022, default 0.25).
    :type alpha: float, optional
    :return: Mixture freeze point in K.
    :rtype: float
    """
    Tf_mix_per_j = np.array(
        [
            (
                _boehm2022_iter(
                    Xi[j], Tm_i[j], dHfus_i[j], dSfus_i[j], dCp_i[j], alpha=alpha
                )
                if Xi[j] > 1e-6
                else -np.inf
            )
            for j in range(len(Xi))
        ]
    )
    return float(np.max(Tf_mix_per_j))


def _ysi_mix(Xi, ysi_i):
    """
    Compute mole-fraction-weighted mixture YSI.

    Pure function of arrays — no branching, no in-place mutation. Portable
    to ``jax.numpy`` by swapping the caller's ``np`` for ``jnp``. Backed by
    the observation in Das et al. 2018 (*Combust. Flame* 190, 349, Section
    3.2) that YSI mixes linearly in mole fraction on the unified scale, and
    the older TSI mole-fraction blending rule of Olson-Pickens-Gill 1985.

    :param Xi: Mole fractions of the compounds.
    :type Xi: np.ndarray
    :param ysi_i: Per-compound Unified YSI values.
    :type ysi_i: np.ndarray
    :return: Mole-fraction-weighted mixture YSI (returned as an array of shape
        ``()`` so that JAX tracers propagate; the ``fuel.ysi`` method wrapper
        casts to Python float at the boundary).
    :rtype: float or array-like scalar
    """
    return np.sum(Xi * ysi_i)


def _dcn_mix(phi_i, dcn_i):
    """
    Compute volume-fraction-weighted mixture Derived Cetane Number.

    Pure function of arrays — no branching, no in-place mutation. Portable
    to ``jax.numpy`` by swapping the caller's ``np`` for ``jnp``. Linear
    blending by liquid volume fraction is the standard first-order rule for
    cetane numbers of hydrocarbon blends (e.g., the linear-by-volume
    convention used throughout the NREL Compendium of Experimental Cetane
    Numbers, NREL/TP-5400-67585); known non-linearities (aromatic
    antagonism) are second-order for jet-range HC-HC blends and are left to
    a future Ghosh-style beta-weighted upgrade.

    :param phi_i: Liquid volume fractions of the compounds (sum to 1).
    :type phi_i: np.ndarray
    :param dcn_i: Per-compound DCN values on the ASTM D6890 (IQT) scale.
    :type dcn_i: np.ndarray
    :return: Volume-fraction-weighted mixture DCN (array of shape ``()`` so
        JAX tracers propagate; ``fuel.dcn`` casts to float at the boundary).
    :rtype: float or array-like scalar
    """
    return np.sum(phi_i * dcn_i)


def _cp_liq_rd(T, A, B, D, MW):
    """
    Compute liquid isobaric heat capacity per compound via Ruzicka-Domalski.

    Pure function of arrays — no branching, no in-place mutation. Portable to
    ``jax.numpy`` by swapping the caller's ``np`` for ``jnp``. Model form:
    :math:`C_{p,L}(T)/R = A + B (T/100) + D (T/100)^2`, from Ruzicka &
    Domalski, *J. Phys. Chem. Ref. Data* 22, 597 (1993). The per-compound
    ``A``, ``B``, ``D`` arrays are produced by projecting the Benson-style
    RD group parameters onto the Constantinou-Gani group set at build time
    (see ``tools/build_gcm_extended.py``).

    :param T: Temperature in Kelvin (scalar or broadcastable to MW).
    :type T: float
    :param A: Per-compound RD ``A`` coefficient (dimensionless).
    :type A: np.ndarray
    :param B: Per-compound RD ``B`` coefficient in 1/K.
    :type B: np.ndarray
    :param D: Per-compound RD ``D`` coefficient in 1/K^2.
    :type D: np.ndarray
    :param MW: Per-compound molecular weight in kg/mol.
    :type MW: np.ndarray
    :return: Liquid heat capacity per compound in J/kg/K.
    :rtype: np.ndarray
    """
    R = 8.31446  # gas constant, J/mol/K
    t = T / 100.0
    Cp_mol = R * (A + B * t + D * t * t)  # J/mol/K
    return Cp_mol / MW  # J/kg/K


def _lhv_hess(n_C, n_H, Hf, MW):
    """
    Compute net heat of combustion (LHV) per compound via a Hess cycle.

    Pure function of arrays — no Python branching on numeric inputs, no
    in-place mutation. Trivially portable to ``jax.numpy`` by swapping the
    caller's ``np`` for ``jnp``. Assumes hydrocarbon combustion:
    :math:`C_aH_b + (a + b/4) O_2 \\to a CO_2 + (b/2) H_2O(g)`.

    :param n_C: Number of carbon atoms per compound.
    :type n_C: np.ndarray
    :param n_H: Number of hydrogen atoms per compound.
    :type n_H: np.ndarray
    :param Hf: Standard enthalpy of formation per compound in J/mol.
    :type Hf: np.ndarray
    :param MW: Molecular weight per compound in kg/mol.
    :type MW: np.ndarray
    :return: Net heat of combustion per compound in MJ/kg.
    :rtype: np.ndarray
    """
    dH_comb = n_C * _HF_CO2_G_JMOL + (n_H / 2.0) * _HF_H2O_G_JMOL - Hf
    return -dH_comb * 1e-6 / MW


class fuel:
    """
    Class for handling group contribution calculations of thermodynamic and mixture properties.

    :param name: Name of the mixture as it appears in its gcData file.
    :type name: str
    :param decompName: Name of the groupDecomposition file if different from name. Defaults to None.
    :type decompName: str, optional
    :param fuelDataDir: Directory where the fuel data is stored. Defaults to FuelLib/fuelData.
    :type fuelDataDir: str, optional
    """

    # Number of first and second order groups from Constantinou and Gani
    N_g1 = 78
    N_g2 = 43

    # Boltzmann's constant J/K
    k_B = 1.380649e-23

    def __init__(self, name, decompName=None, fuelDataDir=FUELDATA_DIR):
        """
        Initialize the fuel object and calculate GCM properties.

        :param name: Name of the mixture as it appears in its gcData file.
        :type name: str
        :param decompName: Name of the groupDecomposition file if different from name.
        :type decompName: str, optional
        :param fuelDataDir: Directory where the fuel data is stored.
        :type fuelDataDir: str, optional
        """

        self.name = name
        if decompName is None:
            decompName = name
        if fuelDataDir != FUELDATA_DIR:
            self.fuelDataDir = fuelDataDir
            self.fuelDataGcDir = os.path.join(self.fuelDataDir, "gcData")
            self.fuelDataDecompDir = os.path.join(
                self.fuelDataDir, "groupDecompositionData"
            )
        else:
            self.fuelDataDir = FUELDATA_DIR
            self.fuelDataGcDir = FUELDATA_GC_DIR
            self.fuelDataDecompDir = FUELDATA_DECOMP_DIR

        self.groupDecompFile = os.path.join(self.fuelDataDecompDir, f"{decompName}.csv")
        self.gcxgcFile = os.path.join(self.fuelDataGcDir, f"{name}_init.csv")
        self.gcmTableFile = os.path.join(GCMTABLE_DIR, "gcmTable.csv")

        # Read functional group data for mixture (num_compounds,num_groups)
        df_Nij = pd.read_csv(self.groupDecompFile)
        self.Nij = df_Nij.iloc[:, 1:].to_numpy()
        self.num_compounds = self.Nij.shape[0]
        self.num_groups = self.Nij.shape[1]

        # Classify hydrocarbon by family (used in thermal conductivity)
        # 0: saturated hydrocarbons
        # 1: aromatics
        # 2: cycloparaffins
        # 3: olefins
        self.fam = np.zeros(self.num_compounds, dtype=int)
        aromatics = 10  # starting index for aromatic groups
        num_aromatics = 5
        cyclos = 84  # starting index for membered ring groups
        num_cyclos = 5
        olefins = 4  # starting index for double bound groups
        num_olefins = 6
        for i in range(self.num_compounds):
            # Check if aromatic: does it contain AC's?
            if sum(self.Nij[i, aromatics : aromatics + num_aromatics]) > 0:
                self.fam[i] = 1
            # Check if cycloparaffin: does it contain rings?
            elif sum(self.Nij[i, cyclos : cyclos + num_cyclos]) > 0:
                self.fam[i] = 2
            # Check if olefin: does it contain double bonds?
            elif sum(self.Nij[i, olefins : olefins + num_olefins]) > 0:
                self.fam[i] = 3

        # Read initial liquid composition of mixture and normalize to get mass frac
        df_gcxgc = pd.read_csv(self.gcxgcFile)
        self.compounds = [
            compound.strip() for compound in df_gcxgc["Compound"].to_list()
        ]
        if "PelePhysics Key" in df_gcxgc.columns:
            self.pelephysics_keys = [
                key.strip() for key in df_gcxgc["PelePhysics Key"].to_list()
            ]
        else:
            self.pelephysics_keys = None

        self.Y_0 = df_gcxgc["Weight %"].to_numpy().flatten().astype(float)
        self.Y_0 /= np.sum(self.Y_0)

        # Make sure mixture data is consistent:
        if self.num_groups < self.N_g1:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of columns in {self.groupDecompFile} is less than "
                f"the required number of first-order groups (N_g1 = {self.N_g1})."
            )
        if self.Y_0.shape[0] != self.num_compounds:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of compounds in {self.groupDecompFile} does not "
                f"equal the number of compounds in {self.gcxgcFile}."
            )

        # Read and store GCM table properties
        df_table = pd.read_csv(self.gcmTableFile)
        df_table = df_table.drop(columns=["Units"])

        def get_row(property_name):
            """
            Get property row from GCM table.

            :param property_name: Name of the property to retrieve.
            :type property_name: str
            :return: Property values for all functional groups.
            :rtype: np.ndarray
            :raises ValueError: If property not found in GCM table.
            """
            row = df_table[df_table["Property"] == property_name]
            if row.empty:
                raise ValueError(f"Property '{property_name}' not found in GCM table.")
            return row.iloc[:, 1:].to_numpy().flatten()

        # Table data for functional groups (num_compounds,)
        Tck = get_row("tck")  # critical temperature (1)
        Pck = get_row("pck")  # critical pressure (bar)
        Vck = get_row("vck")  # critical volume (m^3/kmol)
        Tbk = get_row("tbk")  # boiling temperature (1)
        Tmk = get_row("tmk")  # melting point temperature (1)
        hfk = get_row("hfk")  # enthalpy of formation, (kJ/mol)
        gfk = get_row("gfk")  # Gibbs energy (kJ/mol)
        hvk = get_row("hvk")  # latent heat of vaporization (kJ/mol)
        wk = get_row("wk")  # accentric factor (1)
        Vmk = get_row("vmk")  # liquid molar volume fraction (m^3/kmol)
        cpak = get_row("CpAk")  # specific heat values (J/mol/K)
        cpbk = get_row("CpBk")  # specific heat values (J/mol/K)
        cpck = get_row("CpCk")  # specific heat values (J/mol/K)
        mwk = get_row("MW")  # molecular weights (g/mol)

        # --- Compute critical properties at standard temp (num_compounds,)
        # Molecular weights
        self.MW = np.matmul(self.Nij, mwk)  # g/mol
        self.MW *= 1e-3  # Convert to kg/mol

        # T_c (critical temperature)
        self.Tc = 181.128 * np.log(np.matmul(self.Nij, Tck))  # K

        # p_c (critical pressure)
        self.Pc = 1.3705 + (np.matmul(self.Nij, Pck) + 0.10022) ** (-2)  # bar
        self.Pc *= 1e5  # Convert to Pa from bar

        # V_c (critical volume)
        self.Vc = -0.00435 + (np.matmul(self.Nij, Vck))  # m^3/kmol
        self.Vc *= 1e-3  # Convert to m^3/mol

        # T_b (boiling temperature)
        self.Tb = 204.359 * np.log(np.matmul(self.Nij, Tbk))  # K

        # T_m (melting temperature)
        self.Tm = 102.425 * np.log(np.matmul(self.Nij, Tmk))  # K

        # H_f (enthalpy of formation)
        self.Hf = 10.835 + np.matmul(self.Nij, hfk)  # kJ/mol
        self.Hf *= 1e3  # Convert to J/mol

        # G_f (Gibbs free energy)
        self.Gf = -14.828 + np.matmul(self.Nij, gfk)  # kJ/mol
        self.Gf *= 1e3  # Convert to J/mol

        # H_v,stp (enthalpy of vaporization at 298 K)
        self.Hv_stp = 6.829 + (np.matmul(self.Nij, hvk))  # kJ/mol
        self.Hv_stp *= 1e3  # Convert to J/mol

        # omega (accentric factor)
        self.omega = 0.4085 * np.log(np.matmul(self.Nij, wk) + 1.1507) ** (1.0 / 0.5050)

        # V_m (molar liquid volume at 298 K)
        self.Vm_stp = 0.01211 + np.matmul(self.Nij, Vmk)  # m^3/kmol
        self.Vm_stp *= 1e-3  # Convert to m^3/mol

        # C_p,stp (specific heat at 298 K)
        self.Cp_stp = np.matmul(self.Nij, cpak) - 19.7779  # J/mol/K

        # Temperature corrections for C_p
        self.Cp_B = np.matmul(self.Nij, cpbk)
        self.Cp_C = np.matmul(self.Nij, cpck)

        # L_v,stp (latent heat of vaporization at 298 K)
        self.Lv_stp = self.Hv_stp / self.MW  # J/kg

        # -------- Extended per-group tables for ASTM methods ----------------
        # Adds columns not covered by the canonical Constantinou-Gani table:
        # atom counts (heat_of_combustion). Same row-per-property x
        # column-per-group layout as gcmTable.csv so ``Nij @ row`` works
        # unchanged. New properties are appended to this table in later slices.
        self.gcmExtendedFile = os.path.join(GCMTABLE_DIR, "gcmExtendedTable.csv")
        df_ext = pd.read_csv(self.gcmExtendedFile)
        df_ext = df_ext.drop(columns=["Units"])

        def get_ext_row(property_name):
            """
            Get property row from the extended GCM table.

            :param property_name: Name of the property to retrieve.
            :type property_name: str
            :return: Property values for all functional groups.
            :rtype: np.ndarray
            :raises ValueError: If property not found in extended GCM table.
            """
            row = df_ext[df_ext["Property"] == property_name]
            if row.empty:
                raise ValueError(
                    f"Property '{property_name}' not found in extended GCM table."
                )
            return row.iloc[:, 1:].to_numpy().flatten()

        n_C_grp = get_ext_row("n_C")  # C atoms per group
        n_H_grp = get_ext_row("n_H")  # H atoms per group

        # Per-compound atom counts. Kept as float arrays (not int) because they
        # feed into pure numpy helpers used by JAX-portable methods.
        self.n_C = np.matmul(self.Nij, n_C_grp).astype(float)  # (num_compounds,)
        self.n_H = np.matmul(self.Nij, n_H_grp).astype(float)  # (num_compounds,)

        # ---- Experimental Tb/Tm anchoring (docs/ASTM_BRANCH_REVIEW.md 2.3) --
        # CG group counts cannot see molecular symmetry, so CG Tm is
        # unreliable (n-decane: CG 217 K vs NIST 243.5 K) and CG Tb carries
        # family bias (indane: +54 K). Where gcmTableData/property_anchors.csv
        # provides an experimental value (built by
        # tools/build_property_anchors.py; provenance-tagged 'nist' or
        # homologous-'series'), it overrides the CG estimate. The CG values
        # are preserved as ``self.Tb_gcm`` / ``self.Tm_gcm`` /
        # ``self.omega_gcm``.
        #
        # CONSISTENCY: Lee-Kesler psat depends on (Tc, Pc, omega), NOT Tb —
        # anchoring Tb alone would leave every VLE-derived quantity (bubble
        # points, flash, D86) unchanged. For anchored-Tb compounds, omega is
        # therefore re-derived from the Kesler-Lee closure
        #     omega = (-ln(Pc/101325) - f0(Tbr)) / f1(Tbr),  Tbr = Tb/Tc
        # (f0/f1 the Lee-Kesler functions), which makes psat(exp_Tb) =
        # 101325 Pa exact: the compound boils where experiment says it does.
        self.Tb_gcm = self.Tb.copy()
        self.Tm_gcm = self.Tm.copy()
        self.omega_gcm = self.omega.copy()
        self.Tb_source = ["gcm"] * self.num_compounds
        self.Tm_source = ["gcm"] * self.num_compounds

        _anchor_file = os.path.join(GCMTABLE_DIR, "property_anchors.csv")
        df_anchor = pd.read_csv(_anchor_file)

        def _anchor_lookup(col_val, col_src):
            # Duplicate formulas (isomers, e.g. C7H16 = n-heptane AND
            # 2-methylhexane): LAST table occurrence wins, matching the
            # YSI/DCN dict(zip(...)) convention — n-alkane rows come after
            # isoparaffin rows in the bin skeleton, so pure n-alkane fuels
            # (compound keys like "NC7H16") resolve to n-alkane anchors.
            by_bin = {}
            by_formula = {}
            for _, r in df_anchor.iterrows():
                if not pd.isna(r[col_val]):
                    by_bin[r["GCxGC_Bin"]] = (float(r[col_val]), str(r[col_src]))
                    by_formula[r["Formula"]] = (float(r[col_val]), str(r[col_src]))
            return by_bin, by_formula

        _tb_bin, _tb_formula = _anchor_lookup("exp_Tb_K", "Tb_source")
        _tm_bin, _tm_formula = _anchor_lookup("exp_Tm_K", "Tm_source")
        # Formula fallback is for pure-compound fuels whose gcData names are
        # PelePhysics keys (e.g. "NC7H16") unknown to the bin taxonomy. A bin
        # that IS in the anchor table but lacks a value (e.g. deliberately
        # un-anchored Tm of the ATJ archetypes) must NOT fall back — the
        # formula would match a different isomer (ATJ-C12 would inherit
        # n-dodecane's melting point).
        _known_bins = set(df_anchor["GCxGC_Bin"])

        def _formula_key(i):
            return f"C{int(self.n_C[i])}H{int(self.n_H[i])}"

        for i, c in enumerate(self.compounds):
            known = c in _known_bins
            hit = _tb_bin.get(c) or (None if known else _tb_formula.get(_formula_key(i)))
            if hit is not None:
                self.Tb[i], self.Tb_source[i] = hit
            hit = _tm_bin.get(c) or (None if known else _tm_formula.get(_formula_key(i)))
            if hit is not None:
                self.Tm[i], self.Tm_source[i] = hit

        # Kesler-Lee omega closure for anchored-Tb compounds.
        _anchored = np.array([s != "gcm" for s in self.Tb_source])
        if np.any(_anchored):
            Tbr = self.Tb / self.Tc
            f0 = (
                5.92714
                - (6.09648 / Tbr)
                - 1.28862 * np.log(Tbr)
                + 0.169347 * (Tbr**6)
            )
            f1 = (
                15.2518
                - (15.6875 / Tbr)
                - 13.4721 * np.log(Tbr)
                + 0.43577 * (Tbr**6)
            )
            omega_kl = (-np.log(self.Pc / 101325.0) - f0) / f1
            # Validity guard: the closure degenerates as Tbr -> 1 (f1 -> 0)
            # and an omega outside ~[0, 1.2] breaks the Rackett z = 0.29056
            # - 0.08775*omega (z <= 0 -> NaN density). Keep the CG omega for
            # such compounds (heavy bins with extrapolated Tb).
            _valid = (Tbr < 0.90) & (omega_kl > 0.0) & (omega_kl < 1.2)
            self.omega = np.where(_anchored & _valid, omega_kl, self.omega)

        # Ruzicka-Domalski (1993) liquid Cp coefficients, projected onto the
        # CG group set at build time. Per-compound A, B, D via ``Nij @ row``.
        # Used by ``Cl(T)`` — see the ``_cp_liq_rd`` module-level helper.
        rd_A_grp = get_ext_row("rd_A")
        rd_B_grp = get_ext_row("rd_B")
        rd_D_grp = get_ext_row("rd_D")
        self.Cp_L_A = np.matmul(self.Nij, rd_A_grp).astype(float)
        self.Cp_L_B = np.matmul(self.Nij, rd_B_grp).astype(float)
        self.Cp_L_D = np.matmul(self.Nij, rd_D_grp).astype(float)

        # Alibakhshi (2015) flash-point group contributions, projected onto
        # the CG group set at build time. Per-compound phi sum via
        # ``Nij @ row``, then :math:`FP = 12.14 + 0.73 Tb + phi` at runtime.
        alib_phi_grp = get_ext_row("alibakhshi_phi")
        self.alibakhshi_phi = np.matmul(self.Nij, alib_phi_grp).astype(float)

        # Fusion properties for the freeze-point Boehm 2022 eq 21 solve.
        # HISTORY: originally Walden's rule (dSfus = 56.5 J/mol/K for every
        # compound) as a stopgap for Naef 2019 group contributions. Walden is
        # off by 2-3x in BOTH directions for jet-relevant families (n-C12:
        # 140 J/mol/K measured vs 56.5; globular branched: below 40) — see
        # docs/ASTM_BRANCH_REVIEW.md 2.1. The Naef paper remains untranscribed
        # (gcmExtendedTable's naef_* columns are reserved but zero), so per
        # the review's fallback we use FAMILY-RESOLVED linear correlations
        #     dSfus = A + B * (n_C - C_ref)   [J/mol/K]
        # from gcmTableData/fusion_families.csv: the n-alkane series is
        # anchored to NIST dHfus/Tm data (odd-even alternation not modeled);
        # other families are anchored/estimated per the CSV's Source_note.
        # Family classification comes from the DCN/YSI bin taxonomy loaded
        # below; compounds with no family match fall back to Walden 56.5.
        # dHfus = dSfus * Tm inherits the CG Tm error until the Tb/Tm
        # anchoring lands (review item ASTM-3).
        self._dSfus_walden = 56.5  # J/mol/K fallback for unclassified bins
        _fusion_df = pd.read_csv(os.path.join(GCMTABLE_DIR, "fusion_families.csv"))
        self._fusion_families = {
            row["Family"]: (
                float(row["dSfus_A"]),
                float(row["dSfus_B"]),
                float(row["C_ref"]),
            )
            for _, row in _fusion_df.iterrows()
        }
        # self.dSfus / self.dHfus are assigned after the bin-family lookup
        # (needs the DCN table's Family column, loaded later in __init__).

        # ``dCp`` is Cp_solid - Cp_liq at 298 K; used inside eq 21 to correct
        # for the temperature offset from Tm to T_f,mix. Approximated as
        # -0.35 * Cp_liq(298 K) per Naef 2019 typical hydrocarbon ratio.
        # A per-compound Cp,liq is computed here on the fly to avoid a
        # circular reference to self.Cl at runtime.
        _cp_liq_298 = (
            _cp_liq_rd(298.15, self.Cp_L_A, self.Cp_L_B, self.Cp_L_D, self.MW) * self.MW
        )  # J/mol/K
        self.dCp = -0.35 * _cp_liq_298  # J/mol/K

        # Per-compound Unified YSI (Das et al. 2018 Combust. Flame 190:349;
        # values from the McEnally / Pfefferle Yale YSI Database Volume 2,
        # built via tools/build_ysi_table.py). Lookup by GCxGC_Bin against
        # self.compounds first; fall back to molecular-formula match for
        # pure-compound fuels that use PelePhysics keys (e.g. ``NC7H16``)
        # instead of the POSF GCxGC bin names.
        self.dasYsiFile = os.path.join(GCMTABLE_DIR, "das_2018_ysi.csv")
        df_ysi = pd.read_csv(self.dasYsiFile)
        ysi_by_bin = dict(zip(df_ysi["GCxGC_Bin"], df_ysi["YSI"]))
        src_by_bin = dict(zip(df_ysi["GCxGC_Bin"], df_ysi["Source"]))
        err_by_bin = dict(zip(df_ysi["GCxGC_Bin"], df_ysi["YSI_err"]))
        fam_by_bin_ysi = dict(zip(df_ysi["GCxGC_Bin"], df_ysi["Family"]))
        ysi_by_formula = dict(zip(df_ysi["Formula"], df_ysi["YSI"]))
        src_by_formula = dict(zip(df_ysi["Formula"], df_ysi["Source"]))
        err_by_formula = dict(zip(df_ysi["Formula"], df_ysi["YSI_err"]))
        _ysi_known_bins = set(df_ysi["GCxGC_Bin"])

        def _compound_formula(i):
            """Reconstruct a ``C{n}H{m}`` formula from per-compound atom counts."""
            return f"C{int(self.n_C[i])}H{int(self.n_H[i])}"

        # Uncertainty policy (docs/ASTM_BRANCH_REVIEW.md 2.4): the table's
        # YSI_err column is the measurement error for ``measured_*`` rows;
        # for derived rows (extrapolated / holdlargest / crossfill —
        # 54 of 89 bins) the tabulated err understates reality, so it is
        # inflated: max(2x err, 15% of the value).
        self.ysi_pure = np.full(self.num_compounds, np.nan, dtype=float)
        self.ysi_err = np.full(self.num_compounds, np.nan, dtype=float)
        self.ysi_source = ["unknown"] * self.num_compounds
        for i, c in enumerate(self.compounds):
            known = c in _ysi_known_bins
            if known and not np.isnan(ysi_by_bin[c]):
                self.ysi_pure[i] = ysi_by_bin[c]
                self.ysi_err[i] = err_by_bin[c]
                self.ysi_source[i] = src_by_bin[c]
            elif not known:
                # Formula fallback only for bins unknown to the taxonomy
                # (PelePhysics-key pure fuels); a known bin with a NaN value
                # must NOT resolve to a different isomer's value.
                fk = _compound_formula(i)
                if fk in ysi_by_formula and not np.isnan(ysi_by_formula[fk]):
                    self.ysi_pure[i] = ysi_by_formula[fk]
                    self.ysi_err[i] = err_by_formula[fk]
                    self.ysi_source[i] = f"formula_fallback:{src_by_formula[fk]}"
            if not np.isnan(self.ysi_pure[i]) and not str(
                self.ysi_source[i]
            ).startswith(("measured", "formula_fallback:measured")):
                self.ysi_err[i] = max(
                    2.0 * float(np.nan_to_num(self.ysi_err[i])),
                    0.15 * abs(self.ysi_pure[i]),
                )

        # NaN policy: fill unresolved compounds with the family mean of
        # resolved values (fuel-level), tagged + tracked in ysi_filled so
        # ysi() can warn instead of raising mid-optimization.
        self.ysi_filled = np.zeros(self.num_compounds, dtype=bool)
        _nan_idx = np.where(np.isnan(self.ysi_pure))[0]
        if len(_nan_idx) > 0:
            for i in _nan_idx:
                fam = fam_by_bin_ysi.get(self.compounds[i], None)
                pool = [
                    self.ysi_pure[j]
                    for j in range(self.num_compounds)
                    if not np.isnan(self.ysi_pure[j])
                    and fam_by_bin_ysi.get(self.compounds[j], None) == fam
                ]
                fill = (
                    float(np.mean(pool))
                    if pool
                    else float(np.nanmedian(self.ysi_pure))
                )
                self.ysi_pure[i] = fill
                self.ysi_err[i] = max(0.30 * abs(fill), 10.0)
                self.ysi_source[i] = "family_mean_fill"
                self.ysi_filled[i] = True

        # Per-compound Derived Cetane Number (ASTM D6890 IQT scale;
        # n-hexadecane = 100, HMN = 15). Table built by
        # tools/build_dcn_table.py — literature seed anchors + family
        # fits/offset rules, all rows provenance-tagged and pending
        # verification against the NREL Compendium of Experimental Cetane
        # Numbers (NREL/TP-5400-67585). Same bin -> formula lookup strategy
        # as the YSI table above.
        self.dcnFile = os.path.join(GCMTABLE_DIR, "dcn.csv")
        df_dcn = pd.read_csv(self.dcnFile)
        dcn_by_bin = dict(zip(df_dcn["GCxGC_Bin"], df_dcn["DCN"]))
        dcn_src_by_bin = dict(zip(df_dcn["GCxGC_Bin"], df_dcn["Source"]))
        dcn_err_by_bin = dict(zip(df_dcn["GCxGC_Bin"], df_dcn["DCN_err"]))
        fam_by_bin = dict(zip(df_dcn["GCxGC_Bin"], df_dcn["Family"]))
        dcn_by_formula = dict(zip(df_dcn["Formula"], df_dcn["DCN"]))
        dcn_src_by_formula = dict(zip(df_dcn["Formula"], df_dcn["Source"]))
        dcn_err_by_formula = dict(zip(df_dcn["Formula"], df_dcn["DCN_err"]))
        fam_by_formula = dict(zip(df_dcn["Formula"], df_dcn["Family"]))

        _dcn_known_bins = set(df_dcn["GCxGC_Bin"])
        self.dcn_pure = np.full(self.num_compounds, np.nan, dtype=float)
        self.dcn_err = np.full(self.num_compounds, np.nan, dtype=float)
        self.dcn_source = ["unknown"] * self.num_compounds
        # Chemical family per compound (bin taxonomy) — reused by the fusion
        # thermodynamics below and available for family-level diagnostics.
        self.bin_family = ["unknown"] * self.num_compounds
        for i, c in enumerate(self.compounds):
            known = c in _dcn_known_bins
            if known and not np.isnan(dcn_by_bin[c]):
                self.dcn_pure[i] = dcn_by_bin[c]
                self.dcn_err[i] = dcn_err_by_bin[c]
                self.dcn_source[i] = dcn_src_by_bin[c]
                self.bin_family[i] = fam_by_bin[c]
                continue
            if known:
                continue  # known bin, NaN value: never wrong-isomer fallback
            fk = _compound_formula(i)
            if fk in dcn_by_formula and not np.isnan(dcn_by_formula[fk]):
                self.dcn_pure[i] = dcn_by_formula[fk]
                self.dcn_err[i] = dcn_err_by_formula[fk]
                self.dcn_source[i] = f"formula_fallback:{dcn_src_by_formula[fk]}"
                self.bin_family[i] = fam_by_formula[fk]

        # ---- Fusion entropy/enthalpy from family correlations ----------
        # dSfus = A + B*(n_C - C_ref) per gcmTableData/fusion_families.csv
        # (see the fusion-properties comment block above); Walden fallback
        # for compounds whose bin/formula matched no family.
        self.dSfus = np.full(self.num_compounds, self._dSfus_walden)
        for i in range(self.num_compounds):
            fam = self.bin_family[i]
            if fam in self._fusion_families:
                A, B, C_ref = self._fusion_families[fam]
                self.dSfus[i] = max(A + B * (self.n_C[i] - C_ref), 20.0)
        self.dHfus = self.dSfus * self.Tm  # J/mol

        # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
        self.epsilonByKB = (0.7915 + 0.1693 * self.omega) * self.Tc  # K
        Pc_atm = self.Pc / 101325  # atm
        self.sigma = (2.3551 - 0.0874 * self.omega) * (self.Tc / Pc_atm) ** (
            1.0 / 3
        )  # Angstroms
        self.sigma *= 1e-10  # Convert from Angstroms to m

        # ---------------- UNIFAC 2.0 (optional) -----------------------------
        # If a UNIFAC subgroup decomposition file exists for this fuel, load it
        # along with the 113-subgroup R/Q table and the 54x54 a_mn interaction
        # matrix. Used by ``activity()`` and
        # ``mixture_vapor_pressure(..., activity_model='UNIFAC')``.
        self.unifac_file = os.path.join(FUELDATA_UNIFAC_DIR, f"{decompName}.csv")
        self.has_unifac = os.path.isfile(self.unifac_file)
        if self.has_unifac:
            self._load_unifac_tables()

    def _load_unifac_tables(self):
        """
        Load this fuel's UNIFAC subgroup decomposition and the global R/Q and
        a_mn tables. Called from ``__init__`` when ``has_unifac`` is true.

        Populates the following attributes:
        - ``self.Nij_unifac`` (``num_compounds, 113``): subgroup count per compound.
        - ``self.Rk_sub`` (``113,``): subgroup R parameter.
        - ``self.Qk_sub`` (``113,``): subgroup Q parameter.
        - ``self.amn`` (``54, 54``): asymmetric main-group interaction matrix (K).
        - ``self.sub2main_idx`` (``113,``): column index into ``self.amn`` for each subgroup.

        :raises ValueError: If the decomposition row count does not match
                            ``num_compounds`` or contains an unrecognized subgroup.
        :return: None.
        :rtype: NoneType
        """
        sub_df = pd.read_csv(UNIFAC_SUBGROUP_FILE)
        amn_df = pd.read_csv(UNIFAC_AMN_FILE, index_col=0)
        decomp_df = pd.read_csv(self.unifac_file)

        # Decomposition CSVs use the integer Subgroup_No as column headers
        # (always unique; subgroup *names* contain duplicates such as "CHO" that
        # would collide on a pd.read_csv roundtrip).
        subgroup_headers = [str(int(n)) for n in sub_df["Subgroup_No"].tolist()]
        self.Rk_sub = sub_df["R"].to_numpy(dtype=float)
        self.Qk_sub = sub_df["Q"].to_numpy(dtype=float)

        # Build main-group ID → row-index lookup so the non-contiguous main-group
        # IDs (1..51, 55, 84, 85) map onto consecutive 0..53 indices in self.amn.
        mg_ids = [int(c) for c in amn_df.columns]
        mg_to_idx = {mg: i for i, mg in enumerate(mg_ids)}
        sub_main_ids = sub_df["Main_Group_No"].to_numpy(dtype=int)
        try:
            self.sub2main_idx = np.array([mg_to_idx[m] for m in sub_main_ids])
        except KeyError as e:
            raise ValueError(
                f"UNIFAC subgroup table references main group {e!s} not present "
                f"in interaction matrix '{UNIFAC_AMN_FILE}'."
            )
        self.amn = amn_df.to_numpy(dtype=float)

        decomp_cols = [c for c in decomp_df.columns if c != "Compound"]
        if decomp_cols != subgroup_headers:
            raise ValueError(
                f"UNIFAC decomposition columns in {self.unifac_file} do not match "
                f"the subgroup table {UNIFAC_SUBGROUP_FILE}."
            )
        self.Nij_unifac = decomp_df[subgroup_headers].to_numpy(dtype=float)
        if self.Nij_unifac.shape[0] != self.num_compounds:
            raise ValueError(
                f"UNIFAC decomposition row count ({self.Nij_unifac.shape[0]}) "
                f"in {self.unifac_file} does not match num_compounds "
                f"({self.num_compounds})."
            )

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(self, Yi):
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mean molecular weight of the mixture in kg/mol.
        :rtype: float
        """
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / self.MW)  # mean molar weight of the mixture
        else:
            Mbar = 0.0

        return Mbar

    def mass2Y(self, mass):
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Normalize to get group mole fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW)

        return Yi

    def mass2X(self, mass):
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the number of moles for each compound
        num_mole = mass / self.MW

        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = num_mole / total_moles
        else:
            Xi = np.zeros_like(self.MW)

        return Xi

    def X2Y(self, Xi):
        """
        Calculate the mass fractions from the mole fractions of each component.

        :param Xi: Mole fractions of each compound.
        :type Xi: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the mass for each compound
        mass = Xi * self.MW

        # Normalize to get group mass fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW)

        return Yi

    def Y2X(self, Yi):
        """
        Calculate the mole fractions from the mass fractions of each component.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        Mbar = self.mean_molecular_weight(Yi)
        if np.sum(Yi) != 0:
            Xi = Mbar * Yi / self.MW
        else:
            Xi = np.zeros_like(self.MW)

        return Xi

    def density(self, T, comp_idx=None):
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature of the mixture in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound in kg/m^3.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW  # kg/mol
            Vm = self.molar_liquid_vol(T)  # m^3/mol
        else:
            MW = self.MW[comp_idx]  # kg/mol
            Vm = self.molar_liquid_vol(T, comp_idx=comp_idx)  # m^3/mol

        rho = MW / Vm  # kg/m^3
        return rho

    def viscosity_kinematic(self, T, comp_idx=None):
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Viscosity of each component in m^2/s.
        :rtype: np.ndarray
        """

        # Convert temperature to Celsius
        T_cels = K2C(T)
        if comp_idx is None:
            Tb_cels = K2C(self.Tb)
        else:
            Tb_cels = K2C(self.Tb[comp_idx])

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu_i = np.exp(rhs)  # Viscosity in mm^2/s

        # Convert to SI (m^2/s)
        nu_i = nu_i * 1e-6

        return nu_i

    def viscosity_dynamic(self, T, comp_idx=None):
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity in Pa*s.
        :rtype: np.ndarray
        """

        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)  # m^2/s
        rho_i = self.density(T, comp_idx=comp_idx)  # kg/m^3
        mu_i = nu_i * rho_i  # Pa*s
        return mu_i

    def Cp(self, T, comp_idx=None):
        """
        Compute specific heat capacity at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Specific heat capacity in J/mol/K.
        :rtype: np.ndarray
        """

        theta = (T - 298) / 700
        if comp_idx is None:
            Cp_stp = self.Cp_stp
            Cp_B = self.Cp_B
            Cp_C = self.Cp_C
        else:
            Cp_stp = self.Cp_stp[comp_idx]
            Cp_B = self.Cp_B[comp_idx]
            Cp_C = self.Cp_C[comp_idx]

        cp = Cp_stp + Cp_B * theta + Cp_C * theta**2

        return cp

    def Cl(self, T, comp_idx=None):
        """
        Compute liquid specific heat capacity in J/kg/K at a given temperature.

        :meta private: Uses the Ruzicka-Domalski (1993, *J. Phys. Chem. Ref.
        Data* 22, 597) second-order group additivity for the liquid phase, with
        the RD Benson-notation group parameters pre-projected onto the
        Constantinou-Gani group set (see ``tools/build_gcm_extended.py``).
        Model form: :math:`C_{p,L}(T)/R = A + B (T/100) + D (T/100)^2`. The
        method is calibrated from the melting temperature up to the normal
        boiling temperature; extrapolation deteriorates near the critical
        point and above 0.85 T_c.

        :meta private: Hydrocarbon-only. Heteroatom groups (O, N, S, halogens)
        contribute zero to Cp,L, so non-HC compounds silently underpredict —
        FuelLib's 13 fuels are all hydrocarbon.

        :meta private: Superseded the earlier implementation that returned
        ``Cp(T)/MW`` (ideal-gas Cp divided by MW), which was physically wrong
        for the liquid phase.

        :meta private: Validated: n-heptane, n-decane, n-dodecane at 298 K
        agree with NIST to <1%. Mixture Cp,L on POSF 10264 (Jet A) vs the
        Edwards 2020 experimental table shows +2% at -10 C growing to +13%
        at 160 C — the drift stems from RD's iso-alkane and cycloparaffin
        contributions to d(Cp)/dT being larger than experiment. Future work:
        recalibrate the ring-strain corrections or move to the Zabransky-
        Ruzicka 2004 amendment once the coefficient-transcription issue
        (papers/1071_1_online.pdf) is resolved.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Liquid specific heat capacity in J/kg/K.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            A = self.Cp_L_A
            B = self.Cp_L_B
            D = self.Cp_L_D
            MW = self.MW
        else:
            A = self.Cp_L_A[comp_idx]
            B = self.Cp_L_B[comp_idx]
            D = self.Cp_L_D[comp_idx]
            MW = self.MW[comp_idx]
        return _cp_liq_rd(T, A, B, D, MW)

    def psat(self, T, comp_idx=None, correlation="Lee-Kesler"):
        """
        Compute saturated vapor pressure.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure in Pa.
        :rtype: np.ndarray
        """

        if comp_idx is None:
            Tr = T / self.Tc
            Pc = self.Pc
            omega = self.omega
        else:
            Tr = T / self.Tc[comp_idx]
            Pc = self.Pc[comp_idx]
            omega = self.omega[comp_idx]

        if correlation.casefold() == "Ambrose-Walton".casefold():
            # May cause trouble at high temperatures
            tau = 1 - Tr
            f0 = (
                -5.97616 * tau
                + 1.29874 * tau**1.5
                - 0.60394 * tau**2.5
                - 1.06841 * tau**5.0
            )
            f0 /= Tr
            f1 = (
                -5.03365 * tau
                + 1.11505 * tau**1.5
                - 5.41217 * tau**2.5
                - 7.46628 * tau**5.0
            )
            f1 /= Tr
            f2 = (
                -0.64771 * tau
                + 2.41539 * tau**1.5
                - 4.26979 * tau**2.5
                - 3.25259 * tau**5.0
            )
            f2 /= Tr
            rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

        else:  # Default correlation is Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
            rhs = np.exp(f0 + omega * f1)

        psat = Pc * rhs
        return psat

    def psat_antoine_coeffs(self, Tvals=None, units="mks", correlation="Lee-Kesler"):
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: 4 np.ndarrays
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = np.zeros(20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        # Fit Antoine coefficients for each compound
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = np.linspace(273.15, self.Tb[i], 20)
            Pvals = np.zeros_like(T)
            for k in range(len(T)):
                Pvals[k] = 1 / D * self.psat(T[k], correlation=correlation)[i]

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def molar_liquid_vol(self, T, comp_idx=None):
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume in m^3/mol.
        :rtype: np.ndarray
        """

        Tstp = 298.0
        if comp_idx is None:
            Tc = self.Tc
            omega = self.omega
            Vm_stp = self.Vm_stp
        else:
            Tc = np.array([self.Tc[comp_idx]])
            omega = np.array([self.omega[comp_idx]])
            Vm_stp = np.array([self.Vm_stp[comp_idx]])
        phi = np.zeros_like(Tc)
        for i in range(len(Tc)):
            if T > Tc[i]:
                phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
            else:
                phi[i] = ((1 - (T / Tc[i])) ** (2.0 / 7.0)) - (
                    (1 - (Tstp / Tc[i])) ** (2.0 / 7.0)
                )
        z = 0.29056 - 0.08775 * omega
        Vmi = Vm_stp * np.power(z, phi)
        if comp_idx is not None:
            Vmi = Vmi[0]
        return Vmi

    def latent_heat_vaporization(self, T, comp_idx=None):
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization in J/kg.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            Tc = self.Tc
            Tb = self.Tb
            Lv_stp = self.Lv_stp
        else:
            Tc = np.array([self.Tc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            Lv_stp = np.array([self.Lv_stp[comp_idx]])

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        Lvi = np.zeros_like(Tc)
        for i in range(len(Tc)):
            if T > Tc[i]:
                Lvi[i] = 0.0
            else:
                Lvi[i] = Lv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

        if comp_idx is not None:
            Lvi = Lvi[0]
        return Lvi

    def heat_of_combustion(self, Yi=None, basis="mass"):
        """
        Compute the net heat of combustion (lower heating value) of the fuel.

        :meta private: Uses a Hess cycle on the Constantinou-Gani enthalpy of
        formation with hydrocarbon combustion stoichiometry
        :math:`C_aH_b + (a + b/4) O_2 \\to a CO_2 + (b/2) H_2O(g)`. Gaseous
        H2O gives the net (lower) heating value that ASTM D4809/D3338 report.

        :meta private: Hydrocarbon-only. Raises ``NotImplementedError`` if the
        fuel decomposition contains non-zero heteroatom groups.

        :meta private: Mixture rule is mass-fraction linear
        (heat-of-mixing negligible for HC-HC per Boehm & Heyne 2022).

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param basis: "mass" returns MJ/kg; "mol" returns kJ/mol.
        :type basis: str, optional
        :return: Net heat of combustion of the mixture.
        :rtype: float
        :raises NotImplementedError: If a non-hydrocarbon group has non-zero
            occupancy in the fuel decomposition, or if ``basis`` is unknown.
        """
        if Yi is None:
            Yi = self.Y_0

        # Hydrocarbon-only guard. Non-HC first-order groups per gcmTable.csv:
        # indices 15..51 (O/N/S/halogen groups) and 54..77. Indices 0..14 are
        # HC (CH_n, CH_n=CH_m, aromatics), 52..53 are C-triple-C, and 78..120
        # are second-order corrections (no atoms).
        non_hc = list(range(15, 52)) + list(range(54, 78))
        if np.any(self.Nij[:, non_hc] != 0):
            raise NotImplementedError(
                "heat_of_combustion currently supports hydrocarbon-only fuels. "
                "Detected non-zero heteroatom (O/N/S/halogen) group occupancy "
                f"in the decomposition of '{self.name}'."
            )

        # Constantinou-Gani Hf is the ideal-gas enthalpy of formation at 298 K.
        # ASTM D4809 measures combustion of the LIQUID fuel, so shift to the
        # liquid-phase reference by subtracting the enthalpy of vaporization:
        # Hf(liquid) = Hf(gas) - Hv_stp   (Hv_stp is the positive vaporization
        # enthalpy at 298 K, already computed at self.Hv_stp in J/mol).
        Hf_liq = self.Hf - self.Hv_stp
        lhv_i = _lhv_hess(self.n_C, self.n_H, Hf_liq, self.MW)  # MJ/kg per cmpd

        if basis == "mass":
            return float(np.sum(Yi * lhv_i))
        elif basis == "mol":
            Xi = self.Y2X(Yi)
            lhv_i_kJmol = lhv_i * self.MW * 1e3  # MJ/kg * kg/mol -> kJ/mol
            return float(np.sum(Xi * lhv_i_kJmol))
        else:
            raise NotImplementedError(
                f"heat_of_combustion basis '{basis}' not supported "
                "(use 'mass' for MJ/kg or 'mol' for kJ/mol)."
            )

    def freeze_point(self, Yi=None, method="Boehm2022", alpha=1.0):
        """
        Compute the freeze point of the fuel via SLE-consistent modeling.

        :meta private: Uses Boehm et al. 2022 (*Energy & Fuels* 36, 12046,
        ``papers/blend-prediction-model-...pdf``) eq 21 with the classical
        max-over-components outer loop: the freeze point of a fuel is set
        by the compound whose in-mixture freeze temperature (accounting for
        SLE dilution + mixing entropy) is highest. Solved iteratively per
        candidate compound; result is ``max_j T_f,mix,j(x_j)``.

        :meta private: Per-compound entropy of fusion (``self.dSfus``) comes
        from family-resolved linear correlations in
        ``gcmTableData/fusion_families.csv`` (n-alkane series anchored to
        NIST dHfus/Tm data; other families anchored/estimated per that
        file's Source_note), with ``dHfus = dSfus * Tm`` and Walden's rule
        (56.5 J/mol/K) only as a fallback for unclassified compounds.
        Historically this method used Walden for ALL compounds — off by
        2-3x in both directions (n-C12: 140 J/mol/K measured; globular
        branched: < 40) — see docs/ASTM_BRANCH_REVIEW.md 2.1.

        :meta private: ``alpha`` default changed 0.25 -> 1.0 together with
        the fusion-data upgrade (2026-07-14). Reasoning: for a dilute
        crystallizing component, eq 21's mixing-entropy term expands as
        ``alpha * dS_mix ~ -alpha * R * ln(x)``, while the classical ideal
        SLE relation ``ln(x) = -dHfus/R (1/T - 1/Tm)`` requires exactly
        ``-R ln(x)`` — i.e. alpha = 1 for an ideal solution. The historic
        alpha = 0.25 was calibrated IN TANDEM with Walden's too-small
        dSfus (the two errors partially cancelled); keeping 0.25 with
        physical dSfus under-predicts freezing-point depression by ~4x
        (measured on the POSF fuels: +26..+41 K biases). With alpha = 1 the
        remaining bias equals the CG Tm error (pure-compound freeze ==
        CG Tm by construction), which the Tb/Tm experimental anchoring
        (review item ASTM-3) removes.

        :meta private: Bell & Boehm & Heyne (2025) control-curve refinement
        (``papers/freezing-point-of-hydrocarbon-fuels-...pdf``) is left as
        future work — it needs the tabulated per-species (m, b) coefficients
        from that paper's SI Fig 1A, which are not yet transcribed.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param method: Freeze-point model. Currently only ``"Boehm2022"`` is
            implemented.
        :type method: str, optional
        :param alpha: Mixing-entropy scaling (default 1.0 = classical ideal
            SLE; see notes above for why the historic 0.25 is only valid
            together with Walden fusion constants).
        :type alpha: float, optional
        :return: Mixture freeze point in K.
        :rtype: float
        :raises NotImplementedError: If ``method`` is not ``"Boehm2022"``.
        """
        if Yi is None:
            Yi = self.Y_0
        if method.casefold() != "Boehm2022".casefold():
            raise NotImplementedError(
                f"freeze_point method '{method}' not supported "
                "(use 'Boehm2022'; Bell2025 pending SI transcription)."
            )
        Xi = self.Y2X(Yi)
        return _freeze_max_over_j(
            Xi, self.Tm, self.dHfus, self.dSfus, self.dCp, alpha=alpha
        )

    def flash_point(self, Yi=None, method="Alibakhshi", mixing="Liaw"):
        """
        Compute the flash point of the fuel or a per-compound value.

        :meta private: Pure-component flash point via ``method="Alibakhshi"``
        (default, Alibakhshi et al. 2015 IECR 54:11230, AAD 5.83 K on 1533
        organics, ``papers/a-modified-group-contribution-method-...pdf``) or
        ``method="Alqaheem"`` (Alqaheem & Riazi 2017, FP = 0.70 * Tb, AAD
        1.7% on hydrocarbons). Mixture rule via ``mixing="Liaw"`` (Liaw-Chiu
        modified Le Chatelier with activity coefficients set to unity — for
        jet-fuel HC-HC mixtures the ideality assumption is validated by
        Paricaud et al. *Fuel* 263, 116534 (2020) which reports ~1 C AAD
        vs experiment) or ``mixing="linear"`` (simple mole-fraction
        weighted average of pure flash points — a lower-quality fallback).

        :meta private: The Liaw-Chiu solve is a fixed-iteration Newton on
        the modified Le Chatelier residual (see ``_fp_liaw_ideal_iter``).
        Converges in <=8 iterations for typical fuel compositions.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
            If ``mixing="linear"``, the return is a mixture value; per-
            compound values are always computed internally.
        :type Yi: np.ndarray, optional
        :param method: Pure-component model ("Alibakhshi" or "Alqaheem").
        :type method: str, optional
        :param mixing: Mixture rule ("Liaw" or "linear").
        :type mixing: str, optional
        :return: Mixture flash point in K.
        :rtype: float
        :raises NotImplementedError: If ``method`` or ``mixing`` is unknown.
        """
        if Yi is None:
            Yi = self.Y_0
        if method.casefold() == "Alibakhshi".casefold():
            Tf_i = _fp_alibakhshi(self.Tb, self.alibakhshi_phi)
        elif method.casefold() == "Alqaheem".casefold():
            Tf_i = _fp_alqaheem(self.Tb)
        else:
            raise NotImplementedError(
                f"flash_point method '{method}' not supported "
                "(use 'Alibakhshi' or 'Alqaheem')."
            )
        if mixing.casefold() == "linear".casefold():
            Xi = self.Y2X(Yi)
            return float(np.sum(Xi * Tf_i))
        elif mixing.casefold() == "Liaw".casefold():
            Xi = self.Y2X(Yi)
            return _fp_liaw_ideal_iter(Xi, Tf_i, self.Tc, self.Pc, self.omega)
        else:
            raise NotImplementedError(
                f"flash_point mixing rule '{mixing}' not supported "
                "(use 'Liaw' or 'linear')."
            )

    def ysi(self, Yi=None):
        """
        Compute the Unified Yield Sooting Index of the mixture.

        :meta private: Per-compound YSI values come from the McEnally /
        Pfefferle Yale YSI Database Volume 2 (unified scale, ``benzene = 100``,
        ``n-hexane = 30``), which supersedes and extends the ~370-compound
        Das et al. 2018 (*Combust. Flame* 190, 349) tabulation. Values for
        compounds outside the measured database are filled by intra-family
        linear extrapolation (see ``tools/build_ysi_table.py`` and the
        ``Source`` column of ``gcmTableData/das_2018_ysi.csv``).

        :meta private: Mixing rule is mole-fraction linear (Das 2018 Section
        3.2 verified this on binary n-dodecane / iso-butylbenzene blends;
        Olson-Pickens-Gill 1985 for older TSI). No non-linear synergy on the
        unified scale.

        :meta private: NaN policy (changed 2026-07-14, review item ASTM-5):
        compounds with no tabulated YSI are filled AT CONSTRUCTION with the
        fuel-level family mean (``ysi_source = 'family_mean_fill'``,
        ``self.ysi_filled`` mask, inflated ``ysi_err``). This method warns —
        instead of raising — when filled compounds carry weight, so
        optimization loops keep running with the uncertainty made explicit
        via :meth:`ysi_uncertainty`.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :return: Mole-fraction-weighted mixture Unified YSI.
        :rtype: float
        """
        if Yi is None:
            Yi = self.Y_0
        Xi = self.Y2X(Yi)
        contributing = self.ysi_filled & (Xi > 1e-6)
        if np.any(contributing):
            import warnings

            names = [self.compounds[i] for i in np.where(contributing)[0]]
            warnings.warn(
                f"YSI for '{self.name}' uses family-mean fills for {names} "
                "(no tabulated value); see ysi_err / ysi_uncertainty().",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(_ysi_mix(Xi, self.ysi_pure))

    def ysi_uncertainty(self, Yi=None):
        """
        1-sigma uncertainty of :meth:`ysi` from per-compound ``ysi_err``.

        :meta private: Independent-error propagation through the linear
        mole-fraction blend: ``sigma_mix = sqrt(sum (Xi * sigma_i)^2)``.
        Per-compound sigmas: tabulated ``YSI_err`` for measured rows;
        inflated (max(2x tabulated, 15%)) for extrapolated/holdlargest/
        crossfill rows; >= 30% for family-mean fills. See the constructor.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :return: 1-sigma uncertainty on the mixture Unified YSI.
        :rtype: float
        """
        if Yi is None:
            Yi = self.Y_0
        Xi = self.Y2X(Yi)
        err = np.nan_to_num(self.ysi_err, nan=0.3 * np.nanmean(self.ysi_pure))
        return float(np.sqrt(np.sum((Xi * err) ** 2)))

    def dcn_uncertainty(self, Yi=None, T_ref=288.15):
        """
        1-sigma uncertainty of :meth:`dcn` from per-compound ``dcn_err``.

        :meta private: Independent-error propagation through the linear
        volume-fraction blend: ``sigma_mix = sqrt(sum (phi_i * sigma_i)^2)``.
        Note this covers TABLE uncertainty only — the blending-rule error
        (linear-by-volume vs. reality) is not included.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param T_ref: Reference temperature (K) for the volume fractions.
        :type T_ref: float, optional
        :return: 1-sigma uncertainty on the mixture DCN.
        :rtype: float
        """
        if Yi is None:
            Yi = self.Y_0
        Yi = np.asarray(Yi, dtype=float)
        rho_i = self.density(T_ref)
        vol = Yi / rho_i
        vol_sum = np.sum(vol)
        phi = vol / vol_sum if vol_sum > 0 else np.zeros_like(vol)
        err = np.nan_to_num(self.dcn_err, nan=8.0)
        return float(np.sqrt(np.sum((phi * err) ** 2)))

    def dcn(self, Yi=None, T_ref=288.15):
        """
        Compute the Derived Cetane Number (DCN) of the fuel mixture.

        :meta private: Per-compound DCN values come from
        ``gcmTableData/dcn.csv`` (built by ``tools/build_dcn_table.py``):
        literature seed anchors on the ASTM D6890 IQT scale (n-hexadecane =
        100, heptamethylnonane = 15) plus family fits / offset rules, every
        row provenance-tagged. Seed values are pending verification against
        the NREL Compendium of Experimental Cetane Numbers
        (NREL/TP-5400-67585) — see the table's ``Source`` column and
        ``self.dcn_source`` / ``self.dcn_err``.

        :meta private: Mixing rule is linear in LIQUID VOLUME fraction
        (the standard first-order convention for cetane blending), with
        volume fractions computed from mass fractions and per-compound
        liquid densities at ``T_ref``. Known second-order non-linearities
        (aromatic antagonism) are not modeled; a Ghosh-style beta-weighted
        rule is the planned v2 if binary-blend residuals warrant it.

        :meta private: Mixture validation targets (Edwards, AIAA 2017-0146):
        A-1/POSF10264 = 48.8, A-2/POSF10325 = 48.3, A-3/POSF10289 = 39.2,
        C-1/POSF11498 = 17.1. NOTE: C-1 is expected to FAIL badly until the
        posf11498 decomposition maps its iso-C12 bin to a heavily-branched
        isomer (2,2,4,6,6-pentamethylheptane, DCN ~ 17-24) instead of the
        lightly-branched 2-methylundecane archetype — see
        tests/test_dcn.py for the documented expected failure.

        :param Yi: Mass fractions of the compounds. Defaults to ``self.Y_0``.
        :type Yi: np.ndarray, optional
        :param T_ref: Reference temperature (K) for the liquid densities
            used in the mass -> volume fraction conversion (default 15 C).
        :type T_ref: float, optional
        :return: Volume-fraction-weighted mixture DCN.
        :rtype: float
        :raises NotImplementedError: If any compound has NaN DCN and a
            non-zero mass fraction.
        """
        if Yi is None:
            Yi = self.Y_0
        Yi = np.asarray(Yi, dtype=float)
        nan_mask = np.isnan(self.dcn_pure)
        contributing = nan_mask & (Yi > 0.0)
        if np.any(contributing):
            names = [self.compounds[i] for i in np.where(contributing)[0]]
            raise NotImplementedError(
                f"DCN is not tabulated for the following compounds in "
                f"'{self.name}' (mass fraction > 0): {names}. "
                "Add a value in gcmTableData/dcn.csv (see "
                "tools/build_dcn_table.py) and re-run."
            )
        # Mass -> liquid volume fractions at T_ref.
        rho_i = self.density(T_ref)  # (num_compounds,) kg/m^3
        vol = Yi / rho_i
        vol_sum = np.sum(vol)
        phi = vol / vol_sum if vol_sum > 0 else np.zeros_like(vol)
        safe_dcn = np.where(nan_mask, 0.0, self.dcn_pure)
        return float(_dcn_mix(phi, safe_dcn))

    def diffusion_coeff(
        self,
        p,
        T,
        sigma_gas=3.62e-10,
        epsilonByKB_gas=97.0,
        MW_gas=28.97e-3,
        correlation="Tee",
    ):
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure in Pa.
        :type p: float
        :param T: Temperature in Kelvin.
        :type T: float
        :param sigma_gas: Collision diameter in m.
        :type sigma_gas: float, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant, in K.
        :type epsilonByKB_gas: float, optional
        :param MW_gas: Mean molecular weight of ambient gas in kg/mol.
        :type MW_gas: float, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: np.ndarray
        """

        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma * 1e10  # convert from m to Angstroms
            epsilonByKB_i = self.epsilonByKB  # K
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = np.zeros_like(self.Tb)
            for n in range(self.num_compounds):
                Vmb_i[n] = self.molar_liquid_vol(self.Tb[n])[n] * 1e6  # cm^3/mol
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb  # K , Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas = sigma_gas * 1e10  # convert from m to Angstroms
        sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (
            epsilonByKB_gas * epsilonByKB_i
        ) ** 0.5  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        Tstar_i = T / epsilonAB_byKB_i  # [1]
        A = 1.06036
        B = 0.15610
        C = 0.193
        D = 0.47635
        E = 1.03587
        F = 1.52996
        G = 1.76474
        H = 3.89411
        omegaD_i = (
            A / (Tstar_i**B)
            + C / np.exp(D * Tstar_i)
            + E / np.exp(F * Tstar_i)
            + G / np.exp(H * Tstar_i)
        )

        # Convert molecular weights from kg/mol to g/mol then calculate M_AB
        MW_gas = MW_gas * 1e3
        MW_i = self.MW * 1e3
        M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

        # Convert pressure from Pa to bar
        p = p * 1e-5  # bar

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T**1.5)
            / (p * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s
        D_AB_i = D_AB_i * 1e-4  # Convert to m^2/s

        return D_AB_i

    def surface_tension(self, T, comp_idx=None, correlation="Brock-Bird"):
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: str, optional
        :return: Surface tension in N/m.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            Tc = self.Tc
            Pc = self.Pc
            Tb = self.Tb
            omega = self.omega
        else:
            Tc = np.array([self.Tc[comp_idx]])
            Pc = np.array([self.Pc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            omega = np.array([self.omega[comp_idx]])
        Tr = T / Tc
        Pc = Pc * 1e-5  # convert from Pa to bar

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:
            w = omega
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st = Pc ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)

        st = st * 1e-3  # Convert from dyn/cm to N/m
        if comp_idx is not None:
            st = st[0]

        return st

    def thermal_conductivity(self, T, comp_idx=None):
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity in W/m/K.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW
            Tc = self.Tc
            Tb = self.Tb
            fam = self.fam
        else:
            MW = np.array([self.MW[comp_idx]])
            Tc = np.array([self.Tc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            fam = np.array([self.fam[comp_idx]])

        Astar = 0.00350 + np.zeros_like(Tc)
        alpha = 1.2
        beta = 0.5 + np.zeros_like(Tc)
        gamma = 0.167
        MW_beta = MW * 1e3  # convert from kg/mol to g/mol
        Tr = T / Tc

        for i in range(len(Tc)):
            if fam[i] == 1:
                # Aromatics
                Astar[i] = 0.0346
                beta[i] = 1.0
            elif fam[i] == 2:
                # Cycloparaffins
                Astar[i] = 0.0310
                beta[i] = 1.0
            elif fam[i] == 3:
                # Olefins
                Astar[i] = 0.0361
                beta[i] = 1.0
            MW_beta[i] = MW_beta[i] ** beta[i]

        A = Astar * Tb**alpha / (MW_beta * Tc**gamma)
        tc = A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6))

        if comp_idx is not None:
            tc = tc[0]
        return tc

    # --- Mixture functions ---
    def activity(self, Xi, T):
        """
        Compute UNIFAC 2.0 liquid-phase activity coefficients for the mixture.

        Implements the classical UNIFAC equations (Fredenslund 1975) with the
        completed pair-interaction matrix from UNIFAC 2.0 (Hayer et al. 2025,
        DOI: 10.1016/j.cej.2024.158667). The combinatorial part is the
        Staverman-Guggenheim form with z = 10; the residual part uses the
        single-parameter group interaction psi_mn = exp(-a_mn / T) where a_mn
        is shared by all subgroups belonging to the same main group.

        :param Xi: Mole fractions of each compound (shape ``num_compounds``).
                    Components with ``Xi = 0`` are excluded from the mixture
                    contribution and receive gamma = 1.
        :type Xi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Activity coefficients gamma_i for each compound
                 (shape ``num_compounds``).
        :rtype: np.ndarray
        :raises FileNotFoundError: If this fuel has no UNIFAC decomposition file.
        """
        if not self.has_unifac:
            raise FileNotFoundError(
                f"No UNIFAC decomposition for fuel '{self.name}': expected file "
                f"{self.unifac_file}. Add a CSV under fuelData/unifacDecomposition/ "
                "or use a fuel that has one."
            )

        Xi = np.asarray(Xi, dtype=float).flatten()
        v = self.Nij_unifac  # (num_compounds, 113)
        Rk = self.Rk_sub  # (113,)
        Qk = self.Qk_sub  # (113,)

        # --- Combinatorial part (Staverman-Guggenheim, z = 10) ---
        # Per-compound r and q.
        r_j = v @ Rk  # (num_compounds,)
        q_j = v @ Qk  # (num_compounds,)
        sum_xr = np.dot(Xi, r_j)
        sum_xq = np.dot(Xi, q_j)
        with np.errstate(divide="ignore", invalid="ignore"):
            V_j = np.where(sum_xr > 0, r_j / sum_xr, 0.0)
            F_j = np.where(sum_xq > 0, q_j / sum_xq, 0.0)
            VF = np.where(F_j > 0, V_j / F_j, 0.0)
            log_V = np.where(V_j > 0, np.log(V_j), 0.0)
            log_VF = np.where(VF > 0, np.log(VF), 0.0)
        ln_gamma_C = 1.0 - V_j + log_V - 5.0 * q_j * (1.0 - VF + log_VF)

        # --- Residual part ---
        # Expand the 54x54 main-group a_mn into a (113, 113) subgroup interaction
        # matrix by indexing through sub2main_idx, then form psi.
        A_sub = self.amn[self.sub2main_idx][:, self.sub2main_idx]  # (113, 113) K
        Psi = np.exp(-A_sub / T)

        # Group counts in the mixture: weight subgroup counts by mole fraction.
        v_sum = v.sum(axis=1)  # (num_compounds,) total subgroups per compound
        total_groups = float(np.dot(Xi, v_sum))
        if total_groups == 0.0:
            return np.ones(self.num_compounds)

        X_mix = (Xi @ v) / total_groups  # (113,) group mole fractions in mixture
        Theta_mix = self._theta_from_X(X_mix)
        ln_G_mix = self._ln_Gamma(Theta_mix, Psi)  # (113,)

        # Pure-component group activities (vectorized over compounds).
        with np.errstate(divide="ignore", invalid="ignore"):
            X_pure = np.where(
                v_sum[:, None] > 0, v / v_sum[:, None], 0.0
            )  # (num_compounds, 113)
            Q_X_pure_sum = X_pure @ Qk  # (num_compounds,)
            Theta_pure = np.where(
                Q_X_pure_sum[:, None] > 0,
                Qk[None, :] * X_pure / Q_X_pure_sum[:, None],
                0.0,
            )
        ln_G_pure = self._ln_Gamma(Theta_pure, Psi)  # (num_compounds, 113)

        ln_gamma_R = np.sum(v * (ln_G_mix - ln_G_pure), axis=1)  # (num_compounds,)

        gamma = np.exp(ln_gamma_C + ln_gamma_R)
        # Zero-Xi components: gamma is undefined at the limit; return 1 by convention
        # (matches the 2025 prototype). Combinatorial guards above already kept the
        # math finite, but this final mask makes the contract explicit.
        gamma = np.where(Xi > 0, gamma, 1.0)
        return gamma

    def _theta_from_X(self, X):
        """
        Convert group mole fractions to group surface-area fractions.

        :param X: Group mole fractions (shape ``113`` or ``num_compounds, 113``).
        :type X: np.ndarray
        :return: Group surface-area fractions, same shape as ``X``.
        :rtype: np.ndarray
        """
        Qk = self.Qk_sub
        qx = X @ Qk if X.ndim == 1 else X @ Qk
        with np.errstate(divide="ignore", invalid="ignore"):
            if X.ndim == 1:
                return np.where(qx > 0, Qk * X / qx, 0.0)
            return np.where(qx[:, None] > 0, Qk[None, :] * X / qx[:, None], 0.0)

    def _ln_Gamma(self, Theta, Psi):
        """
        Evaluate the UNIFAC residual group activity coefficient expression.

        Supports a single mixture (1D ``Theta``) or batched pure-component
        evaluation (2D ``Theta`` with one row per compound).

        :param Theta: Group surface-area fractions, shape ``(113,)`` or
                      ``(num_compounds, 113)``.
        :type Theta: np.ndarray
        :param Psi: Subgroup interaction matrix exp(-a_mn / T), shape ``(113, 113)``.
        :type Psi: np.ndarray
        :return: ``ln Gamma_k`` for each subgroup, same shape as ``Theta``.
        :rtype: np.ndarray
        """
        Qk = self.Qk_sub
        TP = Theta @ Psi  # sum_m Theta_m Psi_{m,k}
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(TP > 0, Theta / TP, 0.0)
            log_TP = np.where(TP > 0, np.log(TP), 0.0)
        # Third term: sum_m Psi_{k,m} * w_m  ==  (w @ Psi.T)
        Psi_w = w @ Psi.T
        return Qk * (1.0 - log_TP - Psi_w)

    def mixture_density(self, Yi, T):
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Mixture density in kg/m^3.
        :rtype: float
        """
        MW = self.MW  # Molecular weights of each component (kg/mol)
        Vmi = self.molar_liquid_vol(T)  # Molar volume of each component (m^3/mol)

        # Calculate density (kg/m^3)
        rho = Yi @ (MW / Vmi)

        return rho

    def mixture_kinematic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity in m^2/s.
        :rtype: float
        """
        nu_i = self.viscosity_kinematic(T)  # Viscosities of individual components

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** (3.0)

        return nu

    def mixture_dynamic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity in Pa*s.
        :rtype: float
        """

        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return rho * nu

    def mixture_vapor_pressure(
        self, Yi, T, correlation="Lee-Kesler", activity_model="ideal"
    ):
        """
        Calculate vapor pressure of the mixture.

        With ``activity_model='ideal'`` (default) the mixture vapor pressure is
        the Raoult-law sum p = sum_i x_i p_sat,i. With ``activity_model='UNIFAC'``
        each term is corrected by the UNIFAC 2.0 activity coefficient:
        p = sum_i gamma_i x_i p_sat,i. The default preserves backward-compatible
        behavior; switching to ``'UNIFAC'`` requires the fuel to have a UNIFAC
        decomposition file (see ``activity``).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :param activity_model: Liquid-phase model: ``'ideal'`` (Raoult's law,
                                default) or ``'UNIFAC'`` (multiply by gamma_i).
        :type activity_model: str, optional
        :return: Mixture vapor pressure in Pa.
        :rtype: float
        :raises ValueError: If ``activity_model`` is not ``'ideal'`` or ``'UNIFAC'``.
        :raises FileNotFoundError: If ``activity_model='UNIFAC'`` is requested but
                                    the fuel has no UNIFAC decomposition file.
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound (Pa)
        p_sati = self.psat(T, correlation=correlation)

        if activity_model == "ideal":
            return p_sati @ Xi
        if activity_model == "UNIFAC":
            gamma = self.activity(Xi, T)
            return (gamma * p_sati) @ Xi
        raise ValueError(
            f"Unknown activity_model={activity_model!r}; "
            "expected 'ideal' or 'UNIFAC'."
        )

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi,
        Tvals=None,
        units="mks",
        correlation="Lee-Kesler",
        activity_model="ideal",
    ):
        """
        Estimate Antoine coefficients for vapor pressure of the mixture.

        The inner vapor pressure evaluation honors ``activity_model``:
        ``'ideal'`` (Raoult, default) or ``'UNIFAC'`` (multiply by gamma_i, see
        ``mixture_vapor_pressure``).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, min(Tb)]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :param activity_model: Liquid-phase model passed through to
                                ``mixture_vapor_pressure``: ``'ideal'`` (default)
                                or ``'UNIFAC'``.
        :type activity_model: str, optional
        :return: Coefficients A, B, C, D
        :rtype: float
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            # Initialize as zeros for now, calculated for each compound later
            X = self.Y2X(Yi)
            Tb = mixing_rule(self.Tb, X)
            T = np.linspace(273.15, np.min(Tb), 20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """
            Antoine equation for vapor pressure.

            :param T: Temperature.
            :type T: float or np.ndarray
            :param A: Antoine coefficient A.
            :type A: float
            :param B: Antoine coefficient B.
            :type B: float
            :param C: Antoine coefficient C.
            :type C: float
            :return: log10(pressure).
            :rtype: float or np.ndarray
            """
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        Pvals = np.zeros_like(T)
        for k in range(len(T)):
            Pvals[k] = (
                self.mixture_vapor_pressure(
                    Yi,
                    T[k],
                    correlation=correlation,
                    activity_model=activity_model,
                )
                / D
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return A, B, C, D

    def mixture_surface_tension(self, Yi, T, correlation="Brock-Bird"):
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: str, optional
        :return: Mixture surface tension in N/m.
        :rtype: float
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Surface tension for each compound (N/m)
        sti = self.surface_tension(T, correlation=correlation)

        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        st = mixing_rule(sti, Xi, "arithmetic")

        return st

    def mixture_thermal_conductivity(self, Yi, T):
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Thermal conductivity in W/m/K.
        :rtype: float
        """
        tc = self.thermal_conductivity(T)
        return np.sum(Yi * tc ** (-2)) ** (-0.5)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def C2K(T):
    """
    Convert temperature from Celsius to Kelvin.

    :param T: Temperature in Celsius.
    :type T: float or np.ndarray
    :return: Temperature in Kelvin.
    :rtype: float or np.ndarray
    """
    return T + 273.15


def K2C(T):
    """
    Convert temperature from Kelvin to Celsius.

    :param T: Temperature in Kelvin.
    :type T: float or np.ndarray
    :return: Temperature in Celsius.
    :rtype: float or np.ndarray
    """
    return T - 273.15


def mixing_rule(var_n, X, pseudo_prop="arithmetic"):
    """
    Mixing rules for computing mixture properties.

    :param var_n: Individual compound properties.
    :type var_n: np.ndarray
    :param X: Mole fractions of the compounds.
    :type X: np.ndarray
    :param pseudo_prop: Type of mean ("arithmetic" or "geometric").
    :type pseudo_prop: str, optional
    :return: Mixture property value.
    :rtype: float
    """
    num_comps = len(var_n)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            if pseudo_prop.casefold() == "geometric":
                # Use geometric mean definition for the pseudo property
                var_ij = (var_n[i] * var_n[j]) ** (0.5)
            else:
                # Use arithmetic definition for the pseudo property
                var_ij = (var_n[i] + var_n[j]) / 2
            var_mix += X[i] * X[j] * var_ij
    return var_mix


def droplet_volume(r):
    """
    Calculate spherical volume of a droplet given the radius.

    :param r: Radius of the droplet in meters.
    :type r: float
    :return: Spherical volume of droplet in cubic meters.
    :rtype: float
    """
    return 4.0 / 3.0 * np.pi * r**3


def droplet_mass(fuel, r, Yi, T):
    """
    Calculate the mass of each compound in the fuel provided the radius of the droplet.

    :param fuel: An instance of the groupContribution class.
    :type fuel: groupContribution object
    :param r: Radius of the droplet in meters.
    :type r: float
    :param Yi: Mass fractions of each compound.
    :type Yi: np.ndarray
    :param T: Droplet temperature in Kelvin.
    :type T: float
    :return: Mass of each compound in droplet in kg.
    :rtype: np.ndarray
    """
    volume = droplet_volume(r)  # m^3
    if volume > 0:
        return volume / (fuel.molar_liquid_vol(T) @ Yi) * Yi * fuel.MW
    else:
        return np.zeros_like(fuel.MW)
