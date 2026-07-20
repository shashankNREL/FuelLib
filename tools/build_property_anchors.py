"""
Build gcmTableData/property_anchors.csv — experimental Tb/Tm anchors per bin.

Motivation (docs/ASTM_BRANCH_REVIEW.md 2.3): CG group contributions cannot
see molecular symmetry, so CG Tm is unreliable (n-decane: CG 217 K vs NIST
243.5 K) and CG Tb carries family-level bias (indane: CG 504 K vs 450 K).
D86/flash/psat consistency and the freeze point are all downstream.

Anchor policy — three provenance classes, tagged per value:
1. ``nist``      — well-established NIST/DIPPR values, hand-transcribed
                   (n-alkanes, common aromatics/cycloalkanes). Verify pass
                   recommended, but these are textbook-stable numbers.
2. ``series``    — homologous-series extension: linear CH2-increment
                   extrapolation anchored on >= 2 measured members of the
                   same series. Reliable for Tb (~ +14..22 K per CH2,
                   flattening with C); NOT used for Tm (symmetry-sensitive).
3. (absent)      — no anchor; FuelLib keeps the CG value. Applies to Tm of
                   most substituted compounds and everything about the
                   tricycloparaffin SMILES placeholders.

Tm anchors additionally pull from ``tests/pureComponentReference.csv``
(NIST FreezePoint_K column, ~30 compounds) so the repo's existing NIST
transcription is reused, not duplicated.

Usage::

    python tools/build_property_anchors.py   # writes gcmTableData/property_anchors.csv
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON = os.path.join(REPO_ROOT, "gcmTableData", "das_2018_ysi.csv")
PURE_REF = os.path.join(REPO_ROOT, "tests", "pureComponentReference.csv")
OUT = os.path.join(REPO_ROOT, "gcmTableData", "property_anchors.csv")

# ---------------------------------------------------------------------------
# Tb anchors (K) by bin — class 'nist' unless noted.
# ---------------------------------------------------------------------------
TB_NIST = {
    # n-alkanes (textbook)
    "n-C07": 371.6, "n-C08": 398.8, "n-C09": 424.0, "n-C10": 447.3,
    "n-C11": 469.1, "n-C12": 489.5, "n-C13": 508.6, "n-C14": 526.7,
    "n-C15": 543.8, "n-C16": 560.0, "n-C17": 575.2, "n-C18": 589.5,
    "n-C19": 602.9, "n-C20": 616.9,
    # 2-methylalkanes (reference compounds of the Isoparaffin bins)
    "C07-Isoparaffin": 363.2,   # 2-methylhexane
    "C08-Isoparaffin": 390.8,   # 2-methylheptane
    "C09-Isoparaffin": 416.4,   # 2-methyloctane
    "C10-Isoparaffin": 440.1,   # 2-methylnonane
    "C11-Isoparaffin": 462.4,   # 2-methyldecane
    "C12-Isoparaffin": 483.2,   # 2-methylundecane
    # alkylbenzenes
    "Toluene": 383.8, "C2-Benzene": 409.3, "C3-Benzene": 432.4,
    "C4-Benzene": 456.5, "C5-Benzene": 478.6, "C6-Benzene": 499.3,
    # alkylcyclohexanes
    "C07-Monocycloparaffin": 374.1,  # methylcyclohexane
    "C08-Monocycloparaffin": 404.9,  # ethylcyclohexane
    "C09-Monocycloparaffin": 429.9,  # propylcyclohexane
    "C10-Monocycloparaffin": 454.1,  # butylcyclohexane
    # di/cycloaromatics and fused rings
    "C10-Dicycloparaffin": 464.5,    # decalin (cis/trans mix)
    "Cycloaromatic-C09": 451.1,      # indane
    "Cycloaromatic-C10": 480.8,      # tetralin
    "Diaromatic-C10": 491.1,         # naphthalene
    "Diaromatic-C11": 517.8,         # 1-methylnaphthalene
    # 1-alkenes
    "C10-Alkene": 443.7, "C12-Alkene": 486.5,
}

# Homologous series for 'series'-class Tb extension: (bin prefix/format,
# anchored members list [(C, Tb)], carbon range to fill).
TB_SERIES = [
    # (bin format, [(C, Tb anchors)], fill range)
    ("n-C{:02d}", [(19, 602.9), (20, 616.9)], range(21, 24)),
    ("C{:02d}-Isoparaffin", [(11, 462.4), (12, 483.2)], range(13, 25)),
    ("C{}-Benzene", [(11, 478.6), (12, 499.3)], range(13, 17)),  # C5/C6-Benzene = C11/C12
    ("C{:02d}-Monocycloparaffin", [(9, 429.9), (10, 454.1)], range(11, 20)),
    ("C{:02d}-Dicycloparaffin", [(10, 464.5)], range(11, 18)),  # +20 K/CH2 assumed
    ("Cycloaromatic-C{:02d}", [(10, 480.8)], range(11, 16)),    # +20 K/CH2 assumed
    ("Diaromatic-C{:02d}", [(10, 491.1), (11, 517.8)], range(12, 15)),
    ("C{:02d}-Alkene", [(10, 443.7), (12, 486.5)], range(14, 17, 2)),
]

# Tm anchors (K) beyond what tests/pureComponentReference.csv provides.
TM_NIST = {
    "n-C07": 182.6, "n-C08": 216.4, "n-C09": 219.7, "n-C10": 243.5,
    "n-C11": 247.6, "n-C12": 263.6, "n-C13": 267.8, "n-C14": 279.0,
    "n-C15": 283.1, "n-C16": 291.3, "n-C17": 295.1, "n-C18": 301.3,
    "n-C19": 305.0, "n-C20": 309.6,
    "Toluene": 178.2, "C2-Benzene": 178.2, "C3-Benzene": 173.6,
    "C4-Benzene": 185.3,
    "C07-Monocycloparaffin": 146.6,  # methylcyclohexane
    "C08-Monocycloparaffin": 161.8,
    "C09-Monocycloparaffin": 178.3,
    "C10-Monocycloparaffin": 198.4,
    "C10-Dicycloparaffin": 230.2,    # cis-decalin (winning polymorph range)
    "Cycloaromatic-C09": 221.7,      # indane
    "Cycloaromatic-C10": 237.4,      # tetralin
    "Diaromatic-C10": 353.4,         # naphthalene
    "Diaromatic-C11": 242.7,         # 1-methylnaphthalene
    "C07-Isoparaffin": 154.9,        # 2-methylhexane
    "C08-Isoparaffin": 164.2,        # 2-methylheptane
    "C10-Alkene": 206.9, "C12-Alkene": 237.9,
}


def _series_value(anchors, C):
    """Linear extension along a homologous series."""
    if len(anchors) == 1:
        (c0, t0) = anchors[0]
        return t0 + 20.0 * (C - c0)  # default CH2 increment
    (c0, t0), (c1, t1) = anchors[-2], anchors[-1]
    slope = (t1 - t0) / (c1 - c0)
    return t1 + slope * (C - c1)


def main():
    skel = pd.read_csv(SKELETON)
    df = skel[["GCxGC_Bin", "Formula", "Reference_Compound", "Family",
               "Carbon_Count"]].copy()
    df["exp_Tb_K"] = np.nan
    df["Tb_source"] = ""
    df["exp_Tm_K"] = np.nan
    df["Tm_source"] = ""

    # Tm from the repo's NIST transcription (pureComponentReference.csv)
    ref = pd.read_csv(PURE_REF)
    tm_from_ref = {
        r["GCxGC_Bin"]: float(r["FreezePoint_K"])
        for _, r in ref.iterrows()
        if not pd.isna(r.get("FreezePoint_K"))
    }

    for i, row in df.iterrows():
        b = row["GCxGC_Bin"]
        if b in TB_NIST:
            df.loc[i, ["exp_Tb_K", "Tb_source"]] = TB_NIST[b], "nist"
        if b in TM_NIST:
            df.loc[i, ["exp_Tm_K", "Tm_source"]] = TM_NIST[b], "nist"
        elif b in tm_from_ref:
            df.loc[i, ["exp_Tm_K", "Tm_source"]] = (
                tm_from_ref[b], "nist_pureComponentReference",
            )

    # series-class Tb extension
    for fmt, anchors, crange in TB_SERIES:
        for C in crange:
            if "Benzene" in fmt:
                b = fmt.format(C - 6)  # Cx-Benzene bins count the side chain
            else:
                b = fmt.format(C)
            hit = df["GCxGC_Bin"] == b
            if hit.any() and np.isnan(df.loc[hit, "exp_Tb_K"]).all():
                df.loc[hit, ["exp_Tb_K", "Tb_source"]] = (
                    _series_value(anchors, C), "series",
                )

    df.to_csv(OUT, index=False)
    n_tb = df["exp_Tb_K"].notna().sum()
    n_tm = df["exp_Tm_K"].notna().sum()
    print(f"wrote {OUT}: Tb anchors {n_tb}/89 "
          f"({(df['Tb_source'] == 'nist').sum()} nist, "
          f"{(df['Tb_source'] == 'series').sum()} series), "
          f"Tm anchors {n_tm}/89")


if __name__ == "__main__":
    main()
