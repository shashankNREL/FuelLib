# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What This Project Does

FuelLib is a Python library for computing thermodynamic and transport properties of fuel mixtures using the Group Contribution Method (GCM) by Constantinou and Gani (1994/1995). It supports single-component fuels and complex jet fuels (e.g., POSF AFRL fuels), and can export property tables for CFD solvers (PeleLM, Converge).

## Environment Setup

```bash
conda create --name fuellib-env matplotlib pandas scipy black
conda activate fuellib-env
```

## Running Tests

Tests must be run from the `tests/` directory (imports are relative):

```bash
cd tests && python test_accuracy.py
```

To regenerate baseline predictions (used as CI thresholds):

```bash
cd tests && python test_baseline.py
```

## Code Formatting

Format all Python files before committing:

```bash
find . -name "*.py" -print0 | xargs -0 black
```

CI will fail if Black formatting check fails.

## Export Scripts

```bash
# Export fuel properties for PeleLM simulations
python source/Export4Pele.py --fuel_name heptane

# Export mixture properties for Converge simulations
python source/Export4Converge.py --fuel_name heptane
```

Both scripts support `--help` for full option lists.

## Architecture

### Core module: `source/FuelLib.py`

The `fuel` class is the main entry point. On construction it:
1. Reads the compound list and initial mass fractions from `fuelData/gcData/<name>_init.csv`
2. Reads per-compound functional group counts from `fuelData/groupDecompositionData/<name>.csv`
3. Reads GCM group contribution table from `gcmTableData/gcmTable.csv`
4. Pre-computes all critical/thermodynamic properties (Tc, Pc, Vc, Tb, Tm, Hf, omega, etc.) as numpy arrays of shape `(num_compounds,)`

All property methods (`density`, `viscosity_kinematic`, `psat`, `thermal_conductivity`, etc.) accept a temperature `T` in Kelvin and optionally a `comp_idx` to return a single compound's value instead of the full array. Mixture-level methods (`mixture_density`, `mixture_kinematic_viscosity`, etc.) additionally accept `Yi` (mass fractions array).

Utility functions at module level: `C2K`, `K2C`, `mixing_rule`, `droplet_volume`, `droplet_mass`.

### Path management: `paths.py`

Defines all directory constants (`FUELDATA_DIR`, `FUELDATA_GC_DIR`, `FUELDATA_DECOMP_DIR`, etc.) relative to the repo root. Every script imports `from paths import *` to resolve file locations.

### Data files

- `fuelData/gcData/<name>_init.csv` — compound names, weight percentages, and optional PelePhysics keys
- `fuelData/groupDecompositionData/<name>.csv` — first- and second-order functional group counts (must have ≥ 78 columns for first-order groups)
- `fuelData/propertiesData/<name>.csv` — experimental data used for accuracy testing
- `gcmTableData/gcmTable.csv` — GCM group contribution table (Constantinou & Gani coefficients)
- `tests/baselinePredictions/<name>.csv` — baseline error snapshots; CI checks that new predictions do not exceed these

### Adding a new fuel

Create two CSV files:
1. `fuelData/gcData/<name>_init.csv` with columns: `Compound`, `Weight %` (and optionally `PelePhysics Key`)
2. `fuelData/groupDecompositionData/<name>.csv` with one row per compound and columns for all 78+ functional groups

Then instantiate with `fl.fuel("<name>")`.

## Docs

```bash
cd docs && sphinx-build -M html . _build/
# Open docs/_build/html/index.html
```

Requires a separate conda environment: `conda create --name sphinx-env sphinx sphinx_rtd_theme sphinxcontrib-bibtex pandas scipy`
