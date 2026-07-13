"""
Generate ``gcmTableData/das_2018_ysi.csv`` — per-refCompound Unified YSI values.

Primary source: **YSI Database Volume 2** from the McEnally / Pfefferle group at
Yale (``papers/YSI Database Volume 2.xlsx``, distributed via the Yale MAE
combustion database; supersedes the original Das et al. 2018
*Combust. Flame* 190, 349 GitHub CSV). 447 compounds, 6 columns
(Species, Formula, CAS, Unified YSI, Unified YSI Error, SMILES).

Coverage of FuelLib's 89 ``refCompounds.csv`` entries:
- ~40 have direct name/formula matches in Volume 2.
- ~50 (n-alkanes above n-C12, iso-alkanes above C12, longer cycloparaffins,
  di-/tri-cycloparaffins) are outside the measured database. For these we
  fit a linear (or log-linear) trend on the matched members of each family
  and extrapolate. Every extrapolated row is tagged ``source =
  "extrapolated_<family>"`` in the output CSV so downstream users can audit
  and replace with true measurements as they become available.

Run:
    conda activate ct-env
    python tools/build_ysi_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from paths import FUELDATA_DIR, GCMTABLE_DIR

REF_FILE = os.path.join(FUELDATA_DIR, "refCompounds.csv")
VOL2_FILE = os.path.join(REPO, "papers", "YSI Database Volume 2.xlsx")
OUT_FILE = os.path.join(GCMTABLE_DIR, "das_2018_ysi.csv")


def _norm(name):
    """Case-insensitive, whitespace-free, hyphen-free canonical form."""
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "").strip()


def load_vol2():
    """Load YSI Database Volume 2, keyed by both normalized name and formula."""
    df = pd.read_excel(VOL2_FILE, sheet_name="Database")
    df["Species_norm"] = df["Species"].astype(str).map(_norm)
    df["Formula"] = df["Formula"].astype(str).str.strip()
    return df


def match_refcompounds(ref_df, vol2):
    """
    Match each row of refCompounds.csv to a Volume 2 entry.

    Match strategy, tried in order:
      1. Exact match on the normalized Reference Compound name.
      2. Prefix match — Vol2 Species starts with the normalized ref name
         (catches ``heptane`` -> ``nheptane`` after normalization, or
         ``methylcyclohexane`` -> ``methylcyclohexane1``). Requires a
         unique hit and a length delta <= 4 characters to avoid false
         positives like ``cyclohexane`` matching ``pentylcyclohexane``.
      3. Formula match — accepted only when either (a) there is exactly
         one Vol2 entry for that formula, or (b) exactly one entry for
         that formula contains a key isomer keyword taken from the ref
         name (e.g. ``propyl`` when the ref is ``propyl cyclohexane``).

    Substring matching in the reverse direction (Vol2 name contained in
    ref name) is intentionally NOT attempted because it produced
    false-positive matches like ``pentylcyclohexane`` -> ``cyclohexane``
    and ``hydrindane`` -> ``indane`` in an earlier revision of this
    script (indane is a cycloaromatic, hydrindane is a saturated
    bicyclic; they are different families).

    :param ref_df: refCompounds.csv, one row per FuelLib bin.
    :type ref_df: pandas.DataFrame
    :param vol2: YSI Volume 2 database.
    :type vol2: pandas.DataFrame
    :return: List of dicts with keys
        ``bin, formula, ref_name, ysi, ysi_err, source, source_species``.
    :rtype: list
    """
    by_name = dict(
        zip(
            vol2["Species_norm"],
            zip(vol2["Species"], vol2["Unified YSI"], vol2["Unified YSI Error"]),
        )
    )
    by_formula = vol2.groupby("Formula")

    def _isomer_keyword(ref_name):
        """
        Pull the strongest isomer keyword from the ref compound name.

        Uses whole-word matching so that ``methyl`` (which appears as a
        substring of many alkyl-group names) is picked correctly over
        ``ethyl`` for a name like ``2-methyl octane``. Longer / more
        specific keywords are tested first.
        """
        import re

        for kw in (
            "tridecyl",
            "dodecyl",
            "undecyl",
            "decyl",
            "nonyl",
            "octyl",
            "heptyl",
            "hexyl",
            "pentyl",
            "butyl",
            "propyl",
            "methyl",
            "ethyl",
        ):
            if re.search(rf"\b{kw}\b", ref_name.lower()):
                return kw
        return None

    out = []
    for _, r in ref_df.iterrows():
        bin_ = str(r["GCxGC Bin"]).strip()
        formula = str(r["Formula"]).strip()
        ref_name = str(r["Reference Compound"]).strip()
        n_ref = _norm(ref_name)

        ysi = np.nan
        ysi_err = np.nan
        source = "unmatched"
        src_sp = ""

        # Tier 1 — exact normalized name
        if n_ref in by_name:
            sp, y, e = by_name[n_ref]
            ysi, ysi_err, source, src_sp = y, e, "measured_name", sp

        # Tier 2 — Vol2 name starts with ref name, with small length delta
        if source == "unmatched" and n_ref:
            hits = [
                (sp, y, e)
                for norm, (sp, y, e) in by_name.items()
                if norm.startswith(n_ref) and (len(norm) - len(n_ref)) <= 4
            ]
            if len(hits) == 1:
                sp, y, e = hits[0]
                ysi, ysi_err, source, src_sp = y, e, "measured_prefix", sp

        # Tier 3 — unique formula match
        if source == "unmatched" and formula in by_formula.groups:
            grp = by_formula.get_group(formula)
            if len(grp) == 1:
                sp = grp["Species"].iloc[0]
                ysi, ysi_err, source, src_sp = (
                    grp["Unified YSI"].iloc[0],
                    grp["Unified YSI Error"].iloc[0],
                    "measured_formula_unique",
                    sp,
                )

        # Tier 3b — multi-isomer formula match, disambiguate by keyword
        if source == "unmatched" and formula in by_formula.groups:
            grp = by_formula.get_group(formula)
            kw = _isomer_keyword(ref_name)
            if kw is not None:
                matching = grp[grp["Species"].str.lower().str.contains(kw)]
                if len(matching) == 1:
                    sp = matching["Species"].iloc[0]
                    ysi, ysi_err, source, src_sp = (
                        matching["Unified YSI"].iloc[0],
                        matching["Unified YSI Error"].iloc[0],
                        f"measured_formula_kw_{kw}",
                        sp,
                    )

        out.append(
            {
                "bin": bin_,
                "formula": formula,
                "ref_name": ref_name,
                "ysi": ysi,
                "ysi_err": ysi_err,
                "source": source,
                "source_species": src_sp,
            }
        )
    return out


def carbon_count(formula):
    """
    Extract the carbon count from a molecular formula like 'C12H26'.

    :param formula: Molecular formula string.
    :type formula: str
    :return: Number of carbon atoms, or None if unparseable.
    :rtype: int or None
    """
    import re

    m = re.match(r"^C(\d+)", formula.strip())
    return int(m.group(1)) if m else None


def bin_family(bin_name):
    """
    Assign each FuelLib bin to a homologous family for extrapolation.

    :param bin_name: refCompound GCxGC bin identifier.
    :type bin_name: str
    :return: Family tag used to group compounds for linear extrapolation.
    :rtype: str
    """
    b = bin_name.lower()
    if b.startswith("n-c"):
        return "n_alkane"
    if "isoparaffin" in b:
        return "iso_alkane"
    if "monocycloparaffin" in b:
        return "mono_cyclo"
    if "dicycloparaffin" in b:
        return "di_cyclo"
    if "tricycloparaffin" in b:
        return "tri_cyclo"
    if "cycloaromatic" in b:
        return "cyclo_arom"
    if "diaromatic" in b:
        return "di_arom"
    if "alkene" in b:
        return "alkene"
    if "benzene" in b or b == "toluene":
        return "alkyl_benzene"
    return "other"


def extrapolate(rows):
    """
    Fill ``NaN`` YSI entries by linear regression on carbon count within each
    homologous family, using only the measured entries in that family.

    Reports each extrapolation to stdout so the source is auditable.

    :param rows: Output of :func:`match_refcompounds`; modified in place.
    :type rows: list
    :return: Updated rows with extrapolated values where possible.
    :rtype: list
    """
    for row in rows:
        row["family"] = bin_family(row["bin"])
        row["C_count"] = carbon_count(row["formula"])

    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)

    for fam, group in fams.items():
        measured = [r for r in group if not np.isnan(r["ysi"])]
        missing = [r for r in group if np.isnan(r["ysi"])]
        if not missing:
            continue
        if len(measured) == 0:
            # Zero in-family measurements. For known-hydrocarbon families we
            # cross-borrow from a related family: tricycloparaffins from
            # mono-cyclos scaled up (more rings -> more PAH precursor pool).
            crossfill_map = {"tri_cyclo": ("mono_cyclo", 1.3)}
            if fam in crossfill_map:
                donor_fam, factor = crossfill_map[fam]
                donor = [r for r in fams.get(donor_fam, []) if not np.isnan(r["ysi"])]
                if len(donor) >= 3:
                    dx = np.array([r["C_count"] for r in donor], dtype=float)
                    dy = np.array([r["ysi"] for r in donor], dtype=float)
                    m, b = np.polyfit(dx, dy, 1)
                    for r in missing:
                        r["ysi"] = float(factor * (m * r["C_count"] + b))
                        r["ysi_err"] = float(np.std(dy - (m * dx + b), ddof=0))
                        r["source"] = f"crossfill_{fam}_from_{donor_fam}_x{factor}"
                        r["source_species"] = (
                            f"{factor}x {donor_fam} linear fit over "
                            f"C{int(dx.min())}_C{int(dx.max())}"
                        )
                    print(
                        f"  family {fam!r}: cross-filled from {donor_fam} "
                        f"({factor}x fit); {len(missing)} entries"
                    )
                    continue
            # Last resort — leave as NaN with a source explaining why.
            for r in missing:
                r["source"] = f"insufficient_data_{fam}"
                r["source_species"] = "no measured YSI in Volume 2 for this family"
            print(
                f"  family {fam!r}: 0 measured, {len(missing)} left as NaN "
                "(no in-family fill possible)"
            )
            continue
        # Require at least 3 measured points AND a non-negative slope in a
        # simple linear fit; otherwise fall back to holding the largest
        # measured YSI (safer than extrapolating into unphysical negative
        # values, and better than dropping the compound outright).
        xs = np.array([r["C_count"] for r in measured], dtype=float)
        ys = np.array([r["ysi"] for r in measured], dtype=float)
        if len(measured) >= 3:
            m, b = np.polyfit(xs, ys, 1)
            if m > 0:
                for r in missing:
                    r["ysi"] = float(m * r["C_count"] + b)
                    r["ysi_err"] = float(np.std(ys - (m * xs + b), ddof=0))
                    r["source"] = f"extrapolated_{fam}"
                    r["source_species"] = (
                        f"linear_fit_C{int(xs.min())}_C{int(xs.max())}"
                    )
                print(
                    f"  family {fam!r}: fit YSI = {m:.3f}*C + {b:.3f}"
                    f"   (n_measured={len(measured)}, extrapolated={len(missing)})"
                )
                continue
        # Fallback: hold the largest-C measured value
        largest = measured[int(np.argmax(xs))]
        for r in missing:
            r["ysi"] = float(largest["ysi"])
            r["ysi_err"] = (
                float(largest["ysi_err"])
                if not np.isnan(largest["ysi_err"])
                else float(ys.std())
            )
            r["source"] = f"holdlargest_{fam}"
            r["source_species"] = (
                f"held_at_C{int(largest['C_count'])}_{largest['ref_name']}"
            )
        print(
            f"  family {fam!r}: fallback = hold largest-C YSI "
            f"({largest['ysi']:.1f} at C{int(largest['C_count'])});"
            f" n_measured={len(measured)}, filled={len(missing)}"
        )
    return rows


def main():
    ref = pd.read_csv(REF_FILE)
    print(f"Loaded {len(ref)} refCompound bins.")
    vol2 = load_vol2()
    print(f"Loaded {len(vol2)} YSI Volume 2 entries.")

    rows = match_refcompounds(ref, vol2)
    n_matched = sum(1 for r in rows if not np.isnan(r["ysi"]))
    print(f"\nDirect matches: {n_matched}/{len(rows)}")

    print("\nExtrapolating missing entries within homologous families:")
    rows = extrapolate(rows)

    n_final = sum(1 for r in rows if not np.isnan(r["ysi"]))
    print(f"\nFinal coverage: {n_final}/{len(rows)} refCompounds")
    print("Sources:")
    src_counts = {}
    for r in rows:
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:>3d}  {k}")

    out_df = pd.DataFrame(rows)[
        [
            "bin",
            "formula",
            "ref_name",
            "family",
            "C_count",
            "ysi",
            "ysi_err",
            "source",
            "source_species",
        ]
    ]
    out_df = out_df.rename(
        columns={
            "bin": "GCxGC_Bin",
            "formula": "Formula",
            "ref_name": "Reference_Compound",
            "family": "Family",
            "C_count": "Carbon_Count",
            "ysi": "YSI",
            "ysi_err": "YSI_err",
            "source": "Source",
            "source_species": "Source_Species",
        }
    )
    out_df.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}   shape={out_df.shape}")


if __name__ == "__main__":
    main()
