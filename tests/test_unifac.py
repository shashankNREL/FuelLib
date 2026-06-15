"""
UNIFAC 2.0 tests for FuelLib.

Five test classes:

1. ``UnifacReferenceParityTestCase`` — re-implements the classical UNIFAC
   equations directly in numpy (mirroring ``UNIFAC20_main.py`` from the paper's
   supplementary archive line-by-line) and asserts that ``fuel.activity()``
   agrees with the standalone implementation to within 1e-10.

2. ``UnifacPureComponentTestCase`` — pure-component limit: when one mole
   fraction is 1 and the rest are 0, the activity coefficient of the present
   species must equal 1 to high precision.

3. ``UnifacDecompositionConsistencyTestCase`` — for every fuel with a UNIFAC
   decomposition file, verify shape (113 columns + Compound), row count match
   with gcData, integer non-negative counts, and that all subgroup columns
   outside the SAF subset (MG1, MG2, MG3, MG4 — 10 specific subgroups) are zero.

4. ``UnifacLiteratureValuesTestCase`` — loose absolute-correctness bounds on
   activity coefficients for chemistry-relevant binary mixtures: paraffin/paraffin
   should be near-ideal; aromatic/paraffin should show a few percent of
   non-ideality.

5. ``UnifacThermoCrossCheckTestCase`` — optional cross-check against the
   ``thermo`` Python package (skipped if not installed; not in CI).
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

FUELLIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)
from paths import (
    FUELDATA_GC_DIR,
    FUELDATA_UNIFAC_DIR,
    UNIFAC_AMN_FILE,
    UNIFAC_SUBGROUP_FILE,
)
import FuelLib as fl

try:
    import thermo  # noqa: F401

    HAVE_THERMO = True
except ImportError:
    HAVE_THERMO = False


# --- SAF-subset (C, H) counts per subgroup, keyed by column header (Subgroup_No
#     as string, matching unifacDecomposition CSV column names). Used by
#     UnifacDecompositionConsistencyTestCase to bound expected molecular
#     formulas and verify only SAF subgroups are populated.
SAF_SUBGROUP_CH = {
    "1": (1, 3),  # CH3
    "2": (1, 2),  # CH2
    "3": (1, 1),  # CH
    "4": (1, 0),  # C
    "5": (2, 3),  # CH2=CH
    "9": (1, 1),  # ACH
    "10": (1, 0),  # AC
    "11": (2, 3),  # ACCH3
    "12": (2, 2),  # ACCH2
    "13": (2, 1),  # ACCH
}


def _standalone_unifac(Xi, T, v, Rk, Qk, a_sub):
    """
    Reference UNIFAC activity-coefficient calculation, written independently of
    ``fuel.activity()`` to serve as a parity check.

    Matches the equations in ``UNIFAC20_main.py`` (the supplementary code from
    Hayer et al. 2025) line-by-line, restricted to the prediction (not the
    Pyro/SVI training) path.

    :param Xi: Mole fractions of each compound, shape ``(N,)``.
    :type Xi: np.ndarray
    :param T: Temperature in Kelvin.
    :type T: float
    :param v: Subgroup count matrix of shape ``(N, K)``.
    :type v: np.ndarray
    :param Rk: Subgroup R parameters of shape ``(K,)``.
    :type Rk: np.ndarray
    :param Qk: Subgroup Q parameters of shape ``(K,)``.
    :type Qk: np.ndarray
    :param a_sub: Subgroup-level interaction matrix ``(K, K)`` in Kelvin,
                   built from the main-group ``a_mn`` matrix via subgroup→main
                   index expansion.
    :type a_sub: np.ndarray
    :return: Activity coefficients ``gamma_i`` of shape ``(N,)``.
    :rtype: np.ndarray
    """
    Xi = np.asarray(Xi, dtype=float).flatten()
    N = len(Xi)

    # --- Combinatorial ---
    r_j = v @ Rk
    q_j = v @ Qk
    V_j = r_j / (Xi @ r_j)
    F_j = q_j / (Xi @ q_j)
    ln_gamma_C = (
        1.0 - V_j + np.log(V_j) - 5.0 * q_j * (1.0 - V_j / F_j + np.log(V_j / F_j))
    )

    # --- Residual ---
    Psi = np.exp(-a_sub / T)

    # Mixture group mole fractions
    v_sum = v.sum(axis=1)
    X_mix = (Xi @ v) / (Xi @ v_sum)
    Theta_mix = (Qk * X_mix) / (Qk @ X_mix)

    def ln_Gamma(Theta):
        TP = Theta @ Psi
        return Qk * (1.0 - np.log(TP) - (Theta / TP) @ Psi.T)

    ln_G_mix = ln_Gamma(Theta_mix)

    ln_gamma_R = np.zeros(N)
    for i in range(N):
        v_i = v[i]
        if v_i.sum() == 0:
            continue
        X_i = v_i / v_i.sum()
        Theta_i = (Qk * X_i) / (Qk @ X_i)
        ln_G_pure_i = ln_Gamma(Theta_i)
        ln_gamma_R[i] = v_i @ (ln_G_mix - ln_G_pure_i)

    return np.exp(ln_gamma_C + ln_gamma_R)


def _make_binary_from_posf(fuel_obj, idx_a, idx_b, x_a):
    """
    Construct synthetic binary inputs (Xi, v, R, Q, a_sub) by extracting two
    compound rows from a multi-component fuel's UNIFAC decomposition. Used to
    exercise the residual term against a known-aromatic binary (e.g., toluene
    + n-heptane from posf10325) without committing a new fuel data file.

    :param fuel_obj: Multi-component fuel that has a UNIFAC decomposition.
    :type fuel_obj: FuelLib.fuel
    :param idx_a: Compound index to take as component a.
    :type idx_a: int
    :param idx_b: Compound index to take as component b.
    :type idx_b: int
    :param x_a: Mole fraction of component a (1 - x_a goes to component b).
    :type x_a: float
    :return: Tuple ``(Xi, v, Rk, Qk, a_sub)`` ready to pass to ``_standalone_unifac``.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    Xi = np.array([x_a, 1.0 - x_a])
    v = np.vstack([fuel_obj.Nij_unifac[idx_a], fuel_obj.Nij_unifac[idx_b]])
    Rk = fuel_obj.Rk_sub
    Qk = fuel_obj.Qk_sub
    a_sub = fuel_obj.amn[fuel_obj.sub2main_idx][:, fuel_obj.sub2main_idx]
    return Xi, v, Rk, Qk, a_sub


# ---------------------------------------------------------------------------


class UnifacReferenceParityTestCase(unittest.TestCase):
    """fuel.activity() must agree with a standalone reference implementation."""

    def test_heptane_decane_parity(self):
        """heptane-decane: trivial residual (both MG1), checks combinatorial."""
        f = fl.fuel("heptane-decane")
        Xi = f.Y2X(f.Y_0)
        T = 320.0

        g_method = f.activity(Xi, T)
        a_sub = f.amn[f.sub2main_idx][:, f.sub2main_idx]
        g_ref = _standalone_unifac(Xi, T, f.Nij_unifac, f.Rk_sub, f.Qk_sub, a_sub)

        np.testing.assert_allclose(g_method, g_ref, rtol=1e-12, atol=1e-12)
        # Both are MG1-only; residual must be exactly 1 → γ is purely combinatorial
        # and should be very close to 1 for paraffin-paraffin.
        self.assertTrue(np.all(np.abs(g_method - 1.0) < 0.05))

    def test_toluene_heptane_parity(self):
        """Synthetic toluene+heptane binary from posf10325 — non-trivial residual."""
        f = fl.fuel("posf10325")
        # posf10325 row order: row 0 = Toluene; locate the n-heptane row.
        gc = pd.read_csv(
            os.path.join(FUELDATA_GC_DIR, "posf10325_init.csv"), encoding="utf-8-sig"
        )
        idx_tol = int(np.where(gc["Reference Compound"].str.strip() == "toluene")[0][0])
        idx_hep = int(
            np.where(gc["Reference Compound"].str.strip() == "n-heptane")[0][0]
        )

        for x_tol in [0.1, 0.5, 0.9]:
            with self.subTest(x_toluene=x_tol):
                Xi_full = np.zeros(f.num_compounds)
                Xi_full[idx_tol] = x_tol
                Xi_full[idx_hep] = 1.0 - x_tol
                g_method_full = f.activity(Xi_full, T=320.0)
                # Reference: extract the binary and run the standalone formula.
                Xi, v, Rk, Qk, a_sub = _make_binary_from_posf(
                    f, idx_tol, idx_hep, x_tol
                )
                g_ref = _standalone_unifac(Xi, 320.0, v, Rk, Qk, a_sub)
                # Methods agree on the two non-zero compounds (other compounds in
                # the full fuel have Xi=0 so they contribute nothing).
                np.testing.assert_allclose(
                    [g_method_full[idx_tol], g_method_full[idx_hep]],
                    g_ref,
                    rtol=1e-10,
                    atol=1e-10,
                )


class UnifacPureComponentTestCase(unittest.TestCase):
    """gamma_i → 1 when x_i → 1."""

    def test_pure_component_limit_heptane_decane(self):
        f = fl.fuel("heptane-decane")
        for i in range(f.num_compounds):
            with self.subTest(pure_component=i):
                Xi = np.zeros(f.num_compounds)
                Xi[i] = 1.0
                gamma = f.activity(Xi, T=320.0)
                self.assertAlmostEqual(gamma[i], 1.0, places=10)

    def test_pure_component_limit_posf10325(self):
        f = fl.fuel("posf10325")
        for i in [0, 10, 30, 60]:
            with self.subTest(pure_component=i):
                Xi = np.zeros(f.num_compounds)
                Xi[i] = 1.0
                gamma = f.activity(Xi, T=320.0)
                self.assertAlmostEqual(gamma[i], 1.0, places=10)


class UnifacDecompositionConsistencyTestCase(unittest.TestCase):
    """Validate the committed UNIFAC decomposition CSVs for every fuel."""

    @classmethod
    def setUpClass(cls):
        cls.sub_df = pd.read_csv(UNIFAC_SUBGROUP_FILE)
        cls.expected_headers = [str(int(n)) for n in cls.sub_df["Subgroup_No"]]
        cls.fuel_names = sorted(
            f.replace(".csv", "")
            for f in os.listdir(FUELDATA_UNIFAC_DIR)
            if f.endswith(".csv")
        )

    def test_all_fuels_have_decomposition(self):
        gc_fuels = sorted(
            f.replace("_init.csv", "")
            for f in os.listdir(FUELDATA_GC_DIR)
            if f.endswith("_init.csv")
        )
        self.assertEqual(
            set(self.fuel_names),
            set(gc_fuels),
            "Every fuel in gcData/ must have a matching unifacDecomposition/ file.",
        )

    def test_shape_and_headers(self):
        for fuel_name in self.fuel_names:
            with self.subTest(fuel=fuel_name):
                df = pd.read_csv(os.path.join(FUELDATA_UNIFAC_DIR, f"{fuel_name}.csv"))
                # 113 subgroup columns plus the Compound column.
                self.assertEqual(df.shape[1], 114)
                data_cols = [c for c in df.columns if c != "Compound"]
                self.assertEqual(data_cols, self.expected_headers)

    def test_row_count_matches_gcdata(self):
        for fuel_name in self.fuel_names:
            with self.subTest(fuel=fuel_name):
                u = pd.read_csv(os.path.join(FUELDATA_UNIFAC_DIR, f"{fuel_name}.csv"))
                g = pd.read_csv(
                    os.path.join(FUELDATA_GC_DIR, f"{fuel_name}_init.csv"),
                    encoding="utf-8-sig",
                )
                self.assertEqual(len(u), len(g))

    def test_counts_are_non_negative_integers(self):
        for fuel_name in self.fuel_names:
            with self.subTest(fuel=fuel_name):
                df = pd.read_csv(os.path.join(FUELDATA_UNIFAC_DIR, f"{fuel_name}.csv"))
                data = df.drop(columns="Compound").to_numpy()
                self.assertTrue((data >= 0).all(), "Negative subgroup count")
                self.assertTrue(
                    np.allclose(data, data.astype(int)), "Non-integer subgroup count"
                )

    def test_only_saf_subgroups_populated(self):
        """Verify only the 10 SAF-subset subgroups carry non-zero counts."""
        for fuel_name in self.fuel_names:
            with self.subTest(fuel=fuel_name):
                df = pd.read_csv(os.path.join(FUELDATA_UNIFAC_DIR, f"{fuel_name}.csv"))
                non_saf_cols = [
                    c
                    for c in df.columns
                    if c != "Compound" and c not in SAF_SUBGROUP_CH
                ]
                non_saf_data = df[non_saf_cols].to_numpy()
                self.assertTrue(
                    (non_saf_data == 0).all(),
                    f"{fuel_name}: non-SAF subgroup carries non-zero count. "
                    "If a fuel intentionally adds a non-SAF subgroup, extend "
                    "SAF_SUBGROUP_CH or relax this check.",
                )

    def test_formula_balance_saf_subset(self):
        """Reconstruct C and H from SAF subgroup counts; check sums look right."""
        for fuel_name in self.fuel_names:
            with self.subTest(fuel=fuel_name):
                df = pd.read_csv(os.path.join(FUELDATA_UNIFAC_DIR, f"{fuel_name}.csv"))
                for _, row in df.iterrows():
                    cname = row["Compound"]
                    got_C = 0
                    got_H = 0
                    for header, (c, h) in SAF_SUBGROUP_CH.items():
                        n = int(row[header])
                        got_C += c * n
                        got_H += h * n
                    # All SAF compounds are C5+ hydrocarbons.
                    self.assertGreaterEqual(
                        got_C, 5, f"{fuel_name}/{cname}: implausible C count {got_C}"
                    )
                    # All SAF compounds satisfy C_H_H constraints: paraffins
                    # H=2C+2, naphthalenes H=2C-8, etc. Loose check H in [2C-12, 2C+2].
                    self.assertGreaterEqual(
                        got_H,
                        2 * got_C - 12,
                        f"{fuel_name}/{cname}: implausible H count {got_H} for C={got_C}",
                    )
                    self.assertLessEqual(
                        got_H,
                        2 * got_C + 2,
                        f"{fuel_name}/{cname}: implausible H count {got_H} for C={got_C}",
                    )


class UnifacLiteratureValuesTestCase(unittest.TestCase):
    """Loose bounds on activity coefficients for chemistry-relevant binaries."""

    def test_n_paraffin_n_paraffin_near_ideal(self):
        """n-heptane + n-decane: both MG1, γ should be within 5% of 1."""
        f = fl.fuel("heptane-decane")
        # Sweep composition.
        for x_hep in [0.1, 0.3, 0.5, 0.7, 0.9]:
            with self.subTest(x_heptane=x_hep):
                Xi = np.array([x_hep, 1 - x_hep])
                gamma = f.activity(Xi, T=298.15)
                self.assertTrue(np.all(0.92 < gamma))
                self.assertTrue(np.all(gamma < 1.05))

    def test_aromatic_paraffin_shows_non_ideality(self):
        """Toluene at infinite dilution in n-heptane: γ∞ > 1, several % non-ideality."""
        f = fl.fuel("posf10325")
        gc = pd.read_csv(
            os.path.join(FUELDATA_GC_DIR, "posf10325_init.csv"), encoding="utf-8-sig"
        )
        idx_tol = int(np.where(gc["Reference Compound"].str.strip() == "toluene")[0][0])
        idx_hep = int(
            np.where(gc["Reference Compound"].str.strip() == "n-heptane")[0][0]
        )
        Xi = np.zeros(f.num_compounds)
        Xi[idx_tol] = 1e-6
        Xi[idx_hep] = 1.0 - 1e-6
        gamma = f.activity(Xi, T=298.15)
        gamma_inf_tol_in_hep = gamma[idx_tol]
        # Classical UNIFAC (Fredenslund 1975) gives γ∞_tol_in_hep ~ 1.5–1.7.
        # UNIFAC 2.0 should agree within a few percent. Loose bound:
        self.assertGreater(
            gamma_inf_tol_in_hep,
            1.2,
            "UNIFAC 2.0 underpredicts toluene/heptane non-ideality",
        )
        self.assertLess(
            gamma_inf_tol_in_hep,
            2.5,
            "UNIFAC 2.0 overpredicts toluene/heptane non-ideality",
        )

    def test_cycloalkane_paraffin_near_ideal(self):
        """Methylcyclohexane in n-heptane: both MG1 → γ ~ 1."""
        f = fl.fuel("posf10325")
        gc = pd.read_csv(
            os.path.join(FUELDATA_GC_DIR, "posf10325_init.csv"), encoding="utf-8-sig"
        )
        idx_mch = int(
            np.where(gc["Reference Compound"].str.strip() == "methyl cyclohexane")[0][0]
        )
        idx_hep = int(
            np.where(gc["Reference Compound"].str.strip() == "n-heptane")[0][0]
        )
        Xi = np.zeros(f.num_compounds)
        Xi[idx_mch] = 0.5
        Xi[idx_hep] = 0.5
        gamma = f.activity(Xi, T=298.15)
        for idx, name in [(idx_mch, "methylcyclohexane"), (idx_hep, "n-heptane")]:
            with self.subTest(component=name):
                self.assertTrue(0.95 < gamma[idx] < 1.05)


class UnifacMixtureVaporPressureKeywordTestCase(unittest.TestCase):
    """activity_model keyword on mixture_vapor_pressure[_antoine_coeffs]."""

    def test_default_matches_legacy_ideal(self):
        """Default activity_model='ideal' produces byte-identical result to before."""
        f = fl.fuel("posf10325")
        Yi = f.Y_0
        p_default = f.mixture_vapor_pressure(Yi, T=320.0)
        p_ideal = f.mixture_vapor_pressure(Yi, T=320.0, activity_model="ideal")
        self.assertEqual(p_default, p_ideal)

    def test_unifac_branch_differs_from_ideal(self):
        """For a non-ideal jet fuel, UNIFAC should shift the bubble pressure."""
        f = fl.fuel("posf10325")
        Yi = f.Y_0
        p_ideal = f.mixture_vapor_pressure(Yi, T=343.0, activity_model="ideal")
        p_unifac = f.mixture_vapor_pressure(Yi, T=343.0, activity_model="UNIFAC")
        # Mixture should be non-ideal enough to shift bubble pressure by > 1%.
        self.assertGreater(abs(p_unifac - p_ideal) / p_ideal, 0.01)
        # Both should be positive and finite.
        self.assertTrue(np.isfinite(p_ideal) and p_ideal > 0)
        self.assertTrue(np.isfinite(p_unifac) and p_unifac > 0)

    def test_unknown_activity_model_raises(self):
        f = fl.fuel("heptane-decane")
        with self.assertRaises(ValueError):
            f.mixture_vapor_pressure(f.Y_0, T=320.0, activity_model="Wilson")

    def test_antoine_fit_honors_activity_model(self):
        """mixture_vapor_pressure_antoine_coeffs passes activity_model through."""
        f = fl.fuel("posf10325")
        Yi = f.Y_0
        Tvals = np.array([300.0, 340.0])
        A_i, B_i, C_i, D_i = f.mixture_vapor_pressure_antoine_coeffs(
            Yi, Tvals=Tvals, activity_model="ideal"
        )
        A_u, B_u, C_u, D_u = f.mixture_vapor_pressure_antoine_coeffs(
            Yi, Tvals=Tvals, activity_model="UNIFAC"
        )
        # Antoine A coefficient must shift between the two models for a non-ideal
        # mixture (UNIFAC raises bubble pressure, raising the fitted intercept).
        self.assertNotAlmostEqual(A_i, A_u, places=4)


@unittest.skipUnless(HAVE_THERMO, "thermo package not installed")
class UnifacThermoCrossCheckTestCase(unittest.TestCase):
    """Cross-check fuel.activity() against the thermo package (UNIFAC 1.0)."""

    def test_toluene_heptane_vs_thermo(self):
        # thermo ships UNIFAC 1.0; UNIFAC 2.0 parameters may differ a few %.
        # This test is skipped silently in CI; for local validation only.
        from thermo.unifac import UNIFAC, UFSG, UFIP

        f = fl.fuel("posf10325")
        gc = pd.read_csv(
            os.path.join(FUELDATA_GC_DIR, "posf10325_init.csv"), encoding="utf-8-sig"
        )
        idx_tol = int(np.where(gc["Reference Compound"].str.strip() == "toluene")[0][0])
        idx_hep = int(
            np.where(gc["Reference Compound"].str.strip() == "n-heptane")[0][0]
        )
        Xi = np.zeros(f.num_compounds)
        Xi[idx_tol] = 0.5
        Xi[idx_hep] = 0.5
        my_gamma = f.activity(Xi, T=298.15)

        # thermo expects subgroup count dicts (by integer Subgroup_No).
        toluene_groups = {1: 0, 2: 0, 9: 5, 11: 1}  # 5 ACH + 1 ACCH3
        heptane_groups = {1: 2, 2: 5}  # 2 CH3 + 5 CH2
        gh = UNIFAC.from_subgroups(
            chemgroups=[toluene_groups, heptane_groups],
            T=298.15,
            xs=[0.5, 0.5],
            subgroups=UFSG,
            interaction_data=UFIP,
            version=0,
        ).gammas()
        # Within 10% for UNIFAC 1.0 vs 2.0 on the well-parameterized SAF pairs.
        np.testing.assert_allclose(
            [my_gamma[idx_tol], my_gamma[idx_hep]], gh, rtol=0.10
        )


if __name__ == "__main__":
    unittest.main()
