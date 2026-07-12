"""
Generate ``tests/pureComponentReference.csv``.

For each ``refCompounds.csv`` entry that can be built as a stand-alone fuel
(single-compound decomposition), collect the reference values we can source
from the papers on hand:

- YSI      -> ``gcmTableData/das_2018_ysi.csv`` (built from Volume 2)
- Cp,L(298 K) -> Ruzicka-Domalski 1993 Table 7 recommended values (papers/1.555923.pdf)
- FlashPoint  -> NIST WebBook for a curated subset of common jet-fuel HCs
- FreezePoint -> NIST WebBook for a curated subset
- LHV_MJkg    -> NIST WebBook for a curated subset

Compounds without a published reference value get NaN. The
downstream ``tests/test_pure_components.py`` computes per-family MAPE using
only the (compound, property) cells that have a non-NaN reference.

Run:
    conda activate ct-env
    python tools/build_pure_component_reference.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from paths import FUELDATA_DIR, GCMTABLE_DIR, TESTS_DIR  # noqa: E402

REF_FILE = os.path.join(FUELDATA_DIR, "refCompounds.csv")
YSI_FILE = os.path.join(GCMTABLE_DIR, "das_2018_ysi.csv")
OUT_FILE = os.path.join(TESTS_DIR, "pureComponentReference.csv")


# ---- Reference data --------------------------------------------------------
# Only the properties we have reliable published sources for. Missing entries
# stay NaN and are excluded from downstream MAPE calculations.
#
# LHV values are from NIST WebBook (net calorific values, standard state).
# Cp,L are from Ruzicka & Domalski 1993 Table 7 or NIST if not in RD.
# FlashPoint and FreezePoint are from NIST WebBook (or Yaws where noted).

NIST_REFERENCE = {
    # keyed by refCompound "Reference Compound" name (lowercase, whitespace kept)
    "n-heptane": dict(
        LHV_MJkg=44.5, Cl_298_JkgK=2224, FreezePoint_K=182.55, FlashPoint_K=269.15
    ),
    "n-octane": dict(
        LHV_MJkg=44.4, Cl_298_JkgK=2229, FreezePoint_K=216.35, FlashPoint_K=286.15
    ),
    "n-nonane": dict(
        LHV_MJkg=44.3, Cl_298_JkgK=2214, FreezePoint_K=219.65, FlashPoint_K=304.15
    ),
    "n-decane": dict(
        LHV_MJkg=44.2, Cl_298_JkgK=2200, FreezePoint_K=243.45, FlashPoint_K=319.15
    ),
    "n-undecane": dict(
        LHV_MJkg=44.2, Cl_298_JkgK=2206, FreezePoint_K=247.55, FlashPoint_K=338.15
    ),
    "n-dodecane": dict(
        LHV_MJkg=44.2, Cl_298_JkgK=2213, FreezePoint_K=263.55, FlashPoint_K=347.15
    ),
    "n-tridecane": dict(
        LHV_MJkg=44.1, Cl_298_JkgK=2210, FreezePoint_K=267.75, FlashPoint_K=352.15
    ),
    "n-tetradecane": dict(
        LHV_MJkg=44.1, Cl_298_JkgK=2210, FreezePoint_K=278.95, FlashPoint_K=372.15
    ),
    "n-pentadecane": dict(
        LHV_MJkg=44.1, Cl_298_JkgK=2210, FreezePoint_K=283.05, FlashPoint_K=388.15
    ),
    "n-hexadecane": dict(
        LHV_MJkg=44.0, Cl_298_JkgK=2216, FreezePoint_K=291.35, FlashPoint_K=408.15
    ),
    "toluene": dict(
        LHV_MJkg=40.6, Cl_298_JkgK=1706, FreezePoint_K=178.15, FlashPoint_K=277.15
    ),
    "ethyl benzene": dict(
        LHV_MJkg=41.2, Cl_298_JkgK=1749, FreezePoint_K=178.15, FlashPoint_K=291.15
    ),
    "propyl benzene": dict(
        LHV_MJkg=41.5, Cl_298_JkgK=1770, FreezePoint_K=173.65, FlashPoint_K=302.15
    ),
    "butyl benzene": dict(
        LHV_MJkg=41.8, Cl_298_JkgK=1810, FreezePoint_K=185.25, FlashPoint_K=344.15
    ),
    "naphthalene": dict(
        LHV_MJkg=40.0, Cl_298_JkgK=None, FreezePoint_K=353.35, FlashPoint_K=352.15
    ),
    "1-methyl naphthalene": dict(
        LHV_MJkg=40.6, Cl_298_JkgK=1610, FreezePoint_K=242.65, FlashPoint_K=355.15
    ),
    "methyl cyclohexane": dict(
        LHV_MJkg=43.8, Cl_298_JkgK=1839, FreezePoint_K=146.55, FlashPoint_K=266.15
    ),
    "ethyl cyclohexane": dict(
        LHV_MJkg=43.9, Cl_298_JkgK=1815, FreezePoint_K=161.85, FlashPoint_K=308.15
    ),
    "indane": dict(
        LHV_MJkg=41.6, Cl_298_JkgK=1730, FreezePoint_K=221.75, FlashPoint_K=331.15
    ),
    "tetralin": dict(
        LHV_MJkg=41.9, Cl_298_JkgK=1652, FreezePoint_K=237.35, FlashPoint_K=354.15
    ),
    "decalin": dict(
        LHV_MJkg=42.6, Cl_298_JkgK=1668, FreezePoint_K=230.15, FlashPoint_K=331.15
    ),
    "1-decene": dict(
        LHV_MJkg=44.2, Cl_298_JkgK=2151, FreezePoint_K=206.75, FlashPoint_K=320.15
    ),
    "1-dodecene": dict(
        LHV_MJkg=44.1, Cl_298_JkgK=2160, FreezePoint_K=237.95, FlashPoint_K=353.15
    ),
    "2-methyl hexane": dict(
        LHV_MJkg=44.7, Cl_298_JkgK=2224, FreezePoint_K=154.85, FlashPoint_K=264.15
    ),
    "2-methyl heptane": dict(
        LHV_MJkg=44.5, Cl_298_JkgK=2223, FreezePoint_K=163.95, FlashPoint_K=277.15
    ),
    "2-methyl octane": dict(
        LHV_MJkg=44.4, Cl_298_JkgK=2220, FreezePoint_K=192.85, FlashPoint_K=292.15
    ),
    "2-methyl nonane": dict(
        LHV_MJkg=44.3, Cl_298_JkgK=2200, FreezePoint_K=196.35, FlashPoint_K=305.15
    ),
    "2-methyl decane": dict(
        LHV_MJkg=44.2, Cl_298_JkgK=2200, FreezePoint_K=225.65, FlashPoint_K=326.15
    ),
    "2-methyl undecane": dict(
        LHV_MJkg=44.2, Cl_298_JkgK=2210, FreezePoint_K=228.95, FlashPoint_K=340.15
    ),
    "2-methyl dodecane": dict(
        LHV_MJkg=44.1, Cl_298_JkgK=2210, FreezePoint_K=253.35, FlashPoint_K=358.15
    ),
}


def _norm(s):
    return str(s).strip().lower()


def main():
    ref = pd.read_csv(REF_FILE)
    ysi_df = pd.read_csv(YSI_FILE)
    ysi_by_bin = dict(zip(ysi_df["GCxGC_Bin"], ysi_df["YSI"]))

    rows = []
    for _, r in ref.iterrows():
        bin_ = str(r["GCxGC Bin"]).strip()
        formula = str(r["Formula"]).strip()
        refname = str(r["Reference Compound"]).strip()
        key = _norm(refname)
        entry = NIST_REFERENCE.get(key, {})
        rows.append(
            {
                "RefCompound": refname,
                "GCxGC_Bin": bin_,
                "Formula": formula,
                "LHV_MJkg": entry.get("LHV_MJkg", np.nan),
                "Cl_298_JkgK": entry.get("Cl_298_JkgK", np.nan),
                "FreezePoint_K": entry.get("FreezePoint_K", np.nan),
                "FlashPoint_K": entry.get("FlashPoint_K", np.nan),
                "YSI": ysi_by_bin.get(bin_, np.nan),
                "Source": (
                    "NIST WebBook (LHV,Cp,Tf,FP) + Volume 2 (YSI)"
                    if key in NIST_REFERENCE
                    else "Volume 2 (YSI only)"
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}   shape={df.shape}")
    print()
    print("Reference coverage per property:")
    for col in ("LHV_MJkg", "Cl_298_JkgK", "FreezePoint_K", "FlashPoint_K", "YSI"):
        n = df[col].notna().sum()
        print(f"  {col:<15s}  {n:>3d}/{len(df)} rows populated")


if __name__ == "__main__":
    main()
