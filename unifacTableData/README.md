# UNIFAC 2.0 Parameter Tables

This directory contains the parameter tables for the UNIFAC 2.0 activity-coefficient model used by `fuel.activity()` and `fuel.mixture_vapor_pressure(..., activity_model="UNIFAC")`.

## Citation

> N. Hayer, T. Wendel, S. Mandt, H. Hasse, F. Jirasek. *Advancing thermodynamic group-contribution methods by machine learning: UNIFAC 2.0.* Chemical Engineering Journal **504** (2025) 158667. DOI: [10.1016/j.cej.2024.158667](https://doi.org/10.1016/j.cej.2024.158667).

License: CC-BY-4.0 (Elsevier open-access).

## Files

### `unifac_subgroups.csv`

113 rows × 6 columns. One row per UNIFAC subgroup.

| Column | Type | Meaning |
|---|---|---|
| `Subgroup_No` | int | Subgroup identifier (not contiguous; max is 179) |
| `Subgroup_Name` | str | Name as used in the literature (e.g., `CH3`, `ACH`, `CY-CH2`) |
| `Main_Group_No` | int | Main-group identifier; subgroups sharing this ID share interaction parameters |
| `Main_Group_Name` | str | Main-group name |
| `R` | float | Van der Waals volume parameter (Bondi 1968) |
| `Q` | float | Van der Waals surface-area parameter (Bondi 1968) |

The R / Q values are identical to those published for classical UNIFAC 1.0 (Magnussen et al. 1981; Hansen et al. 1991). UNIFAC 2.0 does not modify the size parameters; only the pair-interaction matrix changes.

### `unifac_amn.csv`

54 × 54 directed pair-interaction matrix in Kelvin. Header row and index column both list the 54 main-group IDs in order: **1–51, 55, 84, 85** (not contiguous). The matrix is asymmetric (`a_mn ≠ a_nm`) and diagonal elements are zero by construction.

A reader of this file must:
- Use `pd.read_csv(..., index_col="Main_Group_No")` (or equivalent), not positional indexing.
- Build a `main_group_id → matrix_row_index` lookup at load time.

The residual term in classical UNIFAC uses `ψ_mn = exp(−a_mn / T)`. UNIFAC 2.0 supplies this `a_mn` matrix with **no missing entries** — the Bayesian matrix completion of the paper fills every off-diagonal cell.

## Source

Both CSVs are converted (semicolon → comma, header-name normalization to ASCII underscores) from the article's supplementary material:

- `1-s2.0-S1385894724101581-mmc2.csv` → `unifac_amn.csv`
- `1-s2.0-S1385894724101581-mmc3.csv` → `unifac_subgroups.csv`

Retrieved: 2026-06-05 from <https://www.sciencedirect.com/science/article/pii/S1385894724101581>.

For the authoritative prediction-equation forms (combinatorial and residual), see `UNIFAC20_main.py` in the article's supplementary archive `1-s2.0-S1385894724101581-mmc4.zip`. FuelLib's `activity()` matches that script's equations.

## Scope notes

- This is **classical UNIFAC equations** (Fredenslund 1975) with an updated, complete `a_mn` table — not Modified UNIFAC (Dortmund). The combinatorial term uses linear `r` (not `r^(3/4)`); the residual uses a single interaction parameter per directed pair (no `b_mn`, `c_mn`).
- Only the asymmetric variant is shipped here. The symmetric variant mentioned in §2.2 of the paper is not in the supplementary material.
- The 54 main-group set covers all SAF-relevant hydrocarbon main groups (MG1, MG3, MG4, MG42) with complete cross-pair coverage.
