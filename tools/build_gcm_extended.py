"""
Generate ``gcmTableData/gcmExtendedTable.csv`` — per-CG-group extensions.

Adds the property rows needed by the new ASTM methods:
- ``n_C`` / ``n_H``  — atom counts per group (used by ``heat_of_combustion``).
- ``rd_A``/``rd_B``/``rd_D`` — Ruzicka-Domalski (1993) + Zabransky-Ruzicka
  (2004) group coefficients for liquid ``Cp,L(T)/R = A + B*(T/100) + D*(T/100)^2``.
  Filled in a later slice; placeholder zeros here.
- ``naef_Cp_sol_298`` / ``naef_Cp_liq_298`` / ``naef_dHfus`` / ``naef_dSfus``
  — Naef (2019) contributions for freeze-point Boehm-eq-21. Placeholder zeros.

Column order matches ``gcmTableData/gcmTable.csv`` exactly (121 group columns
after the ``Property`` column) so that ``np.matmul(self.Nij, row)`` works
without a translation layer.

Run:
    conda activate ct-env
    python tools/build_gcm_extended.py
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from paths import GCMTABLE_DIR

GCM_CANONICAL = os.path.join(GCMTABLE_DIR, "gcmTable.csv")
GCM_EXTENDED = os.path.join(GCMTABLE_DIR, "gcmExtendedTable.csv")

# ---- atom counts per CG first-order group (indices 0..77) --------------------
# Reference: Constantinou & Gani (1994) group definitions. Second-order groups
# (indices 78..120) are corrections, not new atoms, so all zero.
N_C_G1 = [
    1,
    1,
    1,
    1,  # 0-3   CH3, CH2, CH, C
    2,
    2,
    2,
    2,
    2,
    3,  # 4-9   CH2=CH, CH=CH, CH2=C, CH=C, C=C, CH2=C=CH
    1,
    1,  # 10-11 ACH, AC
    2,
    2,
    2,  # 12-14 ACCH3, ACCH2, ACCH
    0,
    1,  # 15-16 OH, ACOH  (heteroatom groups follow)
    2,
    2,
    1,  # 17-19 CH3CO, CH2CO, CHO
    2,
    2,
    1,  # 20-22 CH3COO, CH2COO, HCOO
    1,
    1,
    1,
    1,  # 23-26 CH3O, CH2O, CH-O, FCH2O
    1,
    1,
    1,
    1,
    1,
    1,
    1,  # 27-33 CH2NH2, CHNH2, CH3NH, CH2NH, CHNH, CH3N, CH2N
    1,
    5,
    5,
    2,  # 34-37 ACNH2, C5H4N, C5H3N, CH2CN
    1,  # 38    COOH
    1,
    1,
    1,
    1,
    1,
    1,
    1,  # 39-45 CH2CL, CHCL, CCL, CHCL2, CCL2, CCL3, ACCL
    1,
    1,
    1,  # 46-48 CH2NO2, CHNO2, ACNO2
    1,
    0,
    0,  # 49-51 CH2SH, I, Br
    2,
    2,  # 52-53 CH≡C, C≡C
    0,  # 54    CL—(C=C)
    1,
    3,  # 55-56 ACF, HCON(CH2)2
    1,
    1,
    1,
    1,  # 57-60 CF3, CF2, CF, COO
    1,
    1,
    1,
    0,  # 61-64 CCL2F, HCCLF, CCLF2, Fspecial
    1,
    2,
    2,
    3,
    3,
    3,  # 65-70 CONH2, CONHCH3, CONHCH2, CON(CH3)2, CONCH3CH2, CON(CH2)2
    2,
    2,  # 71-72 C2H5O2, C2H4O2
    1,
    1,
    1,  # 73-75 CH3S, CH2S, CHS
    4,
    4,  # 76-77 C4H3S, C4H2S
]
N_H_G1 = [
    3,
    2,
    1,
    0,  # 0-3   CH3, CH2, CH, C
    3,
    2,
    2,
    1,
    0,
    3,  # 4-9   CH2=CH, CH=CH, CH2=C, CH=C, C=C, CH2=C=CH
    1,
    0,  # 10-11 ACH, AC
    3,
    2,
    1,  # 12-14 ACCH3, ACCH2, ACCH
    1,
    1,  # 15-16 OH, ACOH
    3,
    2,
    1,  # 17-19 CH3CO, CH2CO, CHO
    3,
    2,
    1,  # 20-22 CH3COO, CH2COO, HCOO
    3,
    2,
    1,
    2,  # 23-26 CH3O, CH2O, CH-O, FCH2O
    4,
    3,
    4,
    3,
    2,
    3,
    2,  # 27-33 CH2NH2, CHNH2, CH3NH, CH2NH, CHNH, CH3N, CH2N
    2,
    4,
    3,
    2,  # 34-37 ACNH2, C5H4N, C5H3N, CH2CN
    1,  # 38    COOH
    2,
    1,
    0,
    1,
    0,
    0,
    0,  # 39-45 CH2CL, CHCL, CCL, CHCL2, CCL2, CCL3, ACCL
    2,
    1,
    0,  # 46-48 CH2NO2, CHNO2, ACNO2
    3,
    0,
    0,  # 49-51 CH2SH, I, Br
    1,
    0,  # 52-53 CH≡C, C≡C
    0,  # 54    CL—(C=C)
    0,
    5,  # 55-56 ACF, HCON(CH2)2
    0,
    0,
    1,
    0,  # 57-60 CF3, CF2, CF, COO
    0,
    1,
    0,
    0,  # 61-64 CCL2F, HCCLF, CCLF2, Fspecial
    2,
    4,
    3,
    6,
    5,
    4,  # 65-70 CONH2, CONHCH3, CONHCH2, CON(CH3)2, CONCH3CH2, CON(CH2)2
    5,
    4,  # 71-72 C2H5O2, C2H4O2
    3,
    2,
    1,  # 73-75 CH3S, CH2S, CHS
    3,
    2,  # 76-77 C4H3S, C4H2S
]
assert len(N_C_G1) == 78 == len(N_H_G1)

# ---- Ruzicka-Domalski (1993) liquid Cp group coefficients -------------------
# Source: Ruzicka & Domalski, J. Phys. Chem. Ref. Data 22, 597 (1993)
# (papers/1.555923.pdf), Table 5 (hydrocarbon groups) and Table 6 (ring-strain
# structural corrections). The 2004 Zabransky-Ruzicka amendment
# (papers/1071_1_online.pdf) was originally planned as the primary source, but
# transcribed coefficients gave systematic ~27-37% underprediction on n-alkanes
# vs the paper's own claimed 1-2% AAD; the 1993 values reproduce n-heptane
# (224.9 J/mol/K vs NIST 224) and n-hexadecane (492.9 vs 501.5, Delta 1.7%)
# at 298 K correctly, so 1993 is the shipped source. See CLAUDE.md /
# IMPLEMENTATION_LOG_astm.md for the sanity-check details.
#
# Groups are Benson-style (central atom + nearest neighbors). Delta c_i = a + b*(T/100) + d*(T/100)^2.
# Only hydrocarbon-relevant groups included.
RD1993_GROUPS = {
    "C-(H)3(C)": (3.8452, -0.33997, 0.19489),
    "C-(H)2(C)2": (2.7972, -0.054967, 0.10679),
    "C-(H)(C)3": (-0.42867, 0.93805, 0.29948),
    "C-(C)4": (-2.9353, 1.4255, -0.85271),
    "Cd-(H)2": (4.1763, -0.47392, 0.099928),
    "Cd-(H)(C)": (4.0749, -1.0735, 0.21413),
    "Cd-(C)2": (1.9570, -0.31938, 0.11911),
    "Cd-(H)(Cd)": (3.6968, -1.6037, 0.55022),
    "Cd-(C)(Cd)": (1.0679, -0.50952, 0.33607),
    "C-(H)2(C)(Cd)": (2.0268, 0.20137, 0.11624),
    "C-(H)(C)2(Cd)": (-0.87558, 0.82109, 0.18415),
    "C-(C)3(Cd)": (-4.8006, 2.6004, -0.40688),
    "C-(H)2(Cd)2": (1.4973, -0.46017, 0.52861),
    "Ct-(H)": (9.1633, -4.6695, 1.1400),
    "Ct-(C)": (1.4822, 1.0770, -0.19489),
    "Ca": (3.0880, -0.62917, 1.3760),
    "C-(H)2(Ca)": (12.377, -7.5742, 0.25779),
    "CB-(H)": (2.2609, -0.25000, 0.12592),
    "CB-(C)": (1.5070, -0.13366, 0.11799),
    "CB-(Cd)": (-5.7020, 5.8271, -1.2013),
    "CB-(CB)": (5.8685, -0.86054, -0.063611),
    "C-(H)2(C)(CB)": (1.4142, 0.56919, 0.0053465),
    "C-(H)2(CB)2": (-10.495, 1.0141, -0.71918),
    "C-(H)(C)2(CB)": (1.2367, -1.3997, 0.41385),
    # Ring-fusion aromatic groups. RD 1993 Table 1 assigns 2 x CBF-(CBF)(CB)_2
    # to naphthalene, which is the correct assignment for POSF di-aromatic
    # compounds (methylnaphthalenes, ethylnaphthalenes, ...).
    "CBF-(CBF)(CB)_2": (-11.635, 6.4068, -0.78182),
    "CBF-(CBF)_2(CB)": (26.164, -11.353, 1.2756),
    "CBF-(CBF)(CB)": (-3.5572, 2.8308, -0.39125),
}

# Ring-strain corrections (Table 6). Applied as second-order additive contributions
# per ring occurrence. Only hydrocarbon-relevant corrections included.
RD1993_RSC = {
    "cyclopropane": (4.4297, -4.3392, 1.0222),
    "cyclobutane": (1.2213, -2.8988, 0.75099),
    "cyclopentane_unsub": (-0.33642, -2.8663, 0.70123),
    "cyclopentane_sub": (0.21983, -1.5118, 0.23172),
    "cyclohexane": (-2.0097, -0.72656, 0.14758),
    "cycloheptane": (-11.460, 4.9507, -0.74754),
    "cyclooctane": (-4.1696, 0.52991, -0.018423),
    "cyclopentene": (0.21433, -2.5214, 0.63136),
    "cyclohexene": (-1.2086, -1.5041, 0.42863),
    "cyclohexadiene": (-8.9683, 6.4959, -1.5722),
    "indan": (-6.1414, 3.5709, -0.48620),
    "1H-indene": (-3.6501, 2.4707, -0.60531),
    "tetrahydronaphthalene": (-6.3861, 2.6257, -0.19578),
    "decahydronaphthalene": (-6.8984, 0.66846, -0.070012),
}

# CG-group-index -> list of (RD group or RSC key, count) projections.
# Hydrocarbon-only. Any CG group not listed here contributes 0 to Cp,L, which
# means heteroatom-containing compounds will silently give wrong Cp,L predictions.
# The runtime Cl(T) method gates on the same hydrocarbon-only mask that
# heat_of_combustion uses, so this is caught at the boundary.
CG_TO_RD = {
    # First-order groups
    0: [("C-(H)3(C)", 1)],  # CH3
    1: [("C-(H)2(C)2", 1)],  # CH2
    2: [("C-(H)(C)3", 1)],  # CH
    3: [("C-(C)4", 1)],  # C (quaternary)
    4: [("Cd-(H)2", 1), ("Cd-(H)(C)", 1)],  # CH2=CH (vinyl)
    5: [("Cd-(H)(C)", 2)],  # CH=CH (internal alkene)
    6: [("Cd-(H)2", 1), ("Cd-(C)2", 1)],  # CH2=C (isopropenyl)
    7: [("Cd-(H)(C)", 1), ("Cd-(C)2", 1)],  # CH=C
    8: [("Cd-(C)2", 2)],  # C=C (tetrasubstituted)
    9: [("Cd-(H)2", 1), ("Ca", 1), ("Cd-(H)(C)", 1)],  # CH2=C=CH (allene)
    10: [("CB-(H)", 1)],  # ACH
    # CG "AC" (aromatic C without H) in POSF fuels appears in two situations:
    # (a) naphthalene ring-fusion positions -> RD CBF-(CBF)(CB)_2
    # (b) heavy-alkyl-substituted aromatic C -> RD CB-(C)
    # POSF fuels' aromatic content is dominated by (a) [naphthalene family];
    # CB-(C) would need an additional alkyl-attached counterpart, and CG's
    # ACCH2/ACCH already handle the (b) case. So AC -> CBF-(CBF)(CB)_2.
    11: [("CBF-(CBF)(CB)_2", 1)],  # AC (ring-fusion aromatic C)
    12: [("CB-(C)", 1), ("C-(H)3(C)", 1)],  # ACCH3 (methyl on ring)
    13: [("CB-(C)", 1), ("C-(H)2(C)(CB)", 1)],  # ACCH2
    14: [("CB-(C)", 1), ("C-(H)(C)2(CB)", 1)],  # ACCH
    52: [("Ct-(H)", 1), ("Ct-(C)", 1)],  # CH≡C
    53: [("Ct-(C)", 2)],  # C≡C
    # Second-order ring corrections (CG indices 85..89 per gcmTable.csv column order)
    85: [("cyclopropane", 1)],  # 3-membered ring
    86: [("cyclobutane", 1)],  # 4-membered ring
    # Use "cyclopentane_sub" (substituted) as the default for jet fuels — most
    # cyclopentane rings in POSF fuels have alkyl substituents.
    87: [("cyclopentane_sub", 1)],  # 5-membered ring
    88: [("cyclohexane", 1)],  # 6-membered ring
    89: [("cycloheptane", 1)],  # 7-membered ring
}


# ---- Alibakhshi et al. (2015) flash-point group contributions ---------------
# Source: Alibakhshi, Mirshahvalad, Alibakhshi. *Ind. Eng. Chem. Res.* 54,
# 11230-11235 (2015), `papers/a-modified-group-contribution-method-...pdf`.
# Model form: FP = 12.14 + 0.73 * NBP + sum_i n_i * phi_i, in K. AAD 5.83 K,
# AARE 1.61 % on 1533 organics (Table 3, best among 20 published GC methods).
#
# Below are the phi_i values for the hydrocarbon subset of Alibakhshi's 42
# functional groups (Table 1). We project them onto the CG first-order group
# set. Ring corrections are skipped: Alibakhshi distinguishes "-CH2- (ring)"
# from linear "-CH2-", but CG merges them and puts the ring effect into a
# second-order "N-membered ring" correction; for jet-fuel-relevant cyclos the
# per-ring FP shift is only 1-3 K and is absorbed into the overall 5.83 K
# reported AAD.
#
# CG->Alibakhshi projection: sum of the phi_i values of the Alibakhshi groups
# that make up each CG group (linear per CG group, so the whole projection is
# stored as a single row `alibakhshi_phi` in the extended table).
CG_TO_ALIBAKHSHI = {
    # (name, phi_sum). See Table 1 of Alibakhshi 2015 for individual phi_i.
    0: -2.97,  # CH3            = 1 x -CH3
    1: -1.14,  # CH2            = 1 x -CH2-
    2: -1.64,  # CH             = 1 x >CH-
    3: -0.10,  # C (quaternary) = 1 x >C<
    4: -2.08 + -2.21,  # CH2=CH  = 1 x =CH2 + 1 x =CH-
    5: 2 * -2.21,  # CH=CH        = 2 x =CH-
    6: -2.08 + -2.03,  # CH2=C   = 1 x =CH2 + 1 x =C<
    7: -2.21 + -2.03,  # CH=C
    8: 2 * -2.03,  # C=C
    9: -2.08 + -1.01 + -2.21,  # CH2=C=CH  = 1 x =CH2 + 1 x =C= + 1 x =CH-
    10: -1.40,  # ACH           = 1 x =CH- (ring, aromatic)
    11: -0.39,  # AC            = 1 x =C< (ring, aromatic ring-junction)
    12: -0.39 + -2.97,  # ACCH3 = aromatic C + methyl
    13: -0.39 + -1.14,  # ACCH2
    14: -0.39 + -1.64,  # ACCH
    52: -4.36 + -0.23,  # CH#C  = 1 x #CH + 1 x #C-
    53: 2 * -0.23,  # C#C
}


def project_alibakhshi_onto_cg(n_groups):
    """
    Project Alibakhshi (2015) flash-point group contributions onto the CG set.

    :param n_groups: Number of CG groups.
    :type n_groups: int
    :return: A length-``n_groups`` array of phi_i contributions to be summed
        via ``Nij @ row`` at runtime. Values are zero for CG groups without a
        mapping (heteroatom groups and second-order ring corrections).
    :rtype: numpy.ndarray
    """
    import numpy as np

    phi = np.zeros(n_groups)
    for cg_idx, value in CG_TO_ALIBAKHSHI.items():
        phi[cg_idx] = value
    return phi


def project_rd_onto_cg(n_groups):
    """
    Project Ruzicka-Domalski Benson group coefficients onto the CG group set.

    :param n_groups: Number of CG groups (should be 121: 78 first-order +
        43 second-order).
    :type n_groups: int
    :return: Three vectors ``(rd_A, rd_B, rd_D)``, each of length ``n_groups``,
        giving the per-CG-group projection of the RD coefficients. Values are
        zero for CG groups that have no CG->RD mapping (heteroatom groups and
        unused second-order corrections).
    :rtype: tuple of three numpy.ndarray
    """
    import numpy as np  # local to avoid polluting module scope

    A = np.zeros(n_groups)
    B = np.zeros(n_groups)
    D = np.zeros(n_groups)
    for cg_idx, terms in CG_TO_RD.items():
        for name, count in terms:
            if name in RD1993_GROUPS:
                a, b, d = RD1993_GROUPS[name]
            elif name in RD1993_RSC:
                a, b, d = RD1993_RSC[name]
            else:
                raise KeyError(f"Unknown RD group/rsc key: {name!r}")
            A[cg_idx] += count * a
            B[cg_idx] += count * b
            D[cg_idx] += count * d
    return A, B, D


def main():
    df = pd.read_csv(GCM_CANONICAL)
    columns = df.columns.tolist()
    assert (
        columns[0] == "Property" and columns[1] == "Units"
    ), f"gcmTable.csv header layout changed: got {columns[:3]}"
    group_columns = columns[2:]
    n_groups = len(group_columns)
    assert n_groups == 121, f"expected 121 group columns, got {n_groups}"

    n_C = N_C_G1 + [0] * (n_groups - 78)
    n_H = N_H_G1 + [0] * (n_groups - 78)

    rd_A, rd_B, rd_D = project_rd_onto_cg(n_groups)
    alibakhshi_phi = project_alibakhshi_onto_cg(n_groups)

    rows = []
    rows.append(["n_C", "atoms"] + n_C)
    rows.append(["n_H", "atoms"] + n_H)
    rows.append(["rd_A", "dimensionless"] + list(rd_A))
    rows.append(["rd_B", "1/K"] + list(rd_B))
    rows.append(["rd_D", "1/K^2"] + list(rd_D))
    rows.append(["alibakhshi_phi", "K"] + list(alibakhshi_phi))
    # Placeholder rows for the freeze-point slice.
    for name in (
        "naef_Cp_sol_298",
        "naef_Cp_liq_298",
        "naef_dHfus",
        "naef_dSfus",
    ):
        rows.append([name, "TBD"] + [0] * n_groups)

    ext = pd.DataFrame(rows, columns=columns)
    ext.to_csv(GCM_EXTENDED, index=False)
    print(f"Wrote {GCM_EXTENDED}")
    print(f"  shape: {ext.shape}")
    print(f"  properties: {ext['Property'].tolist()}")

    n_C_arr = ext[ext["Property"] == "n_C"].iloc[:, 2:].to_numpy().flatten()
    n_H_arr = ext[ext["Property"] == "n_H"].iloc[:, 2:].to_numpy().flatten()

    def sanity(fuel_name, expected_MW=None, expected_a=None, expected_b=None):
        decomp = pd.read_csv(
            os.path.join(REPO, "fuelData", "groupDecompositionData", f"{fuel_name}.csv")
        )
        Nij = decomp.iloc[:, 1:].to_numpy()
        a = Nij @ n_C_arr
        b = Nij @ n_H_arr
        print(
            f"  {fuel_name}: n_C={a[0]}, n_H={b[0]}" f" ({decomp.iloc[0, 0].strip()})"
        )
        if expected_a is not None:
            assert (
                a[0] == expected_a
            ), f"{fuel_name}[0]: n_C got {a[0]}, expected {expected_a}"
        if expected_b is not None:
            assert (
                b[0] == expected_b
            ), f"{fuel_name}[0]: n_H got {b[0]}, expected {expected_b}"

    print("\nsanity — pure compound single-row decomps:")
    sanity("heptane", expected_a=7, expected_b=16)
    sanity("decane", expected_a=10, expected_b=22)
    sanity("dodecane", expected_a=12, expected_b=26)

    # Sanity check RD Cp,L projection: predict n-heptane and n-dodecane at
    # 298.15 K and compare to NIST recommended liquid heat capacities.
    import numpy as np

    R = 8.31446
    rd_A_arr = ext[ext["Property"] == "rd_A"].iloc[:, 2:].to_numpy().flatten()
    rd_B_arr = ext[ext["Property"] == "rd_B"].iloc[:, 2:].to_numpy().flatten()
    rd_D_arr = ext[ext["Property"] == "rd_D"].iloc[:, 2:].to_numpy().flatten()

    def cp_liq_sanity(fuel_name, T=298.15, nist_ref=None):
        decomp = pd.read_csv(
            os.path.join(REPO, "fuelData", "groupDecompositionData", f"{fuel_name}.csv")
        )
        Nij = decomp.iloc[:, 1:].to_numpy()
        A = Nij @ rd_A_arr
        B = Nij @ rd_B_arr
        D = Nij @ rd_D_arr
        t = T / 100.0
        Cp_mol = R * (A + B * t + D * t * t)
        # Single-compound decomps have one row.
        val = Cp_mol[0]
        msg = f"  {fuel_name} Cp,L(298 K) = {val:6.1f} J/mol/K"
        if nist_ref is not None:
            err = 100 * (val - nist_ref) / nist_ref
            msg += f"   NIST ~ {nist_ref} J/mol/K   ({err:+.1f}%)"
        print(msg)

    print("\nsanity — RD 1993 Cp,L projection:")
    cp_liq_sanity("heptane", nist_ref=224)
    cp_liq_sanity("decane", nist_ref=314)
    cp_liq_sanity("dodecane", nist_ref=376)

    # Sanity: flash point via Alibakhshi 2015. Uses CG-computed Tb.
    phi_arr = ext[ext["Property"] == "alibakhshi_phi"].iloc[:, 2:].to_numpy().flatten()
    # CG Tb constants: Tb = 204.359 * ln(Nij @ Tbk)
    df_gcm = pd.read_csv(GCM_CANONICAL)
    tbk_row = df_gcm[df_gcm["Property"] == "tbk"].iloc[:, 2:].to_numpy().flatten()

    def fp_sanity(fuel_name, nist_ref=None):
        decomp = pd.read_csv(
            os.path.join(REPO, "fuelData", "groupDecompositionData", f"{fuel_name}.csv")
        )
        Nij = decomp.iloc[:, 1:].to_numpy()
        Tb = 204.359 * np.log(Nij @ tbk_row)  # K
        phi_sum = Nij @ phi_arr
        fp_alib = 12.14 + 0.73 * Tb + phi_sum
        fp_alqa = 0.70 * Tb
        val = fp_alib[0]
        msg = (
            f"  {fuel_name} FP: Alibakhshi = {val:5.1f} K, "
            f"Alqaheem = {fp_alqa[0]:5.1f} K"
        )
        if nist_ref is not None:
            msg += f"  NIST ~ {nist_ref} K  (dAlib = {val - nist_ref:+.1f})"
        print(msg)

    print("\nsanity — flash point per compound:")
    fp_sanity("heptane", nist_ref=269)  # -4 C
    fp_sanity("decane", nist_ref=319)  # +46 C
    fp_sanity("dodecane", nist_ref=347)  # +74 C


if __name__ == "__main__":
    main()
