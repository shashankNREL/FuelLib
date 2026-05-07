#!/usr/bin/env python3
"""
Verification test for nasa7_vapor_cp module.

Tests:
  1. NASA7 coefficient extraction — spot-check against mechanism.yaml
  2. Cross-validation against Cantera's species.thermo.cp(T)
  3. Liquid-proxy vs NASA7 comparison at D86-relevant temperatures
"""

import sys
import os
import numpy as np

# Add FuelLib source to path
FUELLIB_SOURCE = os.path.dirname(os.path.abspath(__file__))
FUELLIB_ROOT = os.path.dirname(FUELLIB_SOURCE)
sys.path.insert(0, FUELLIB_SOURCE)
sys.path.insert(0, FUELLIB_ROOT)

from FuelLib import fuel
from nasa7_vapor_cp import NASA7VaporCp, R_UNIVERSAL

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FUEL_NAME = "posf10325"
MECHANISM_YAML = os.path.join(
    os.path.dirname(FUELLIB_ROOT),
    "PelePhysics", "Mechanisms", "fuellib_posf_nonreacting", "mechanism.yaml"
)

print(f"Fuel:      {FUEL_NAME}")
print(f"Mechanism: {MECHANISM_YAML}")
print()

# --------------------------------------------------------------------------
# 1. Initialise fuel and NASA7 evaluator
# --------------------------------------------------------------------------
my_fuel = fuel(FUEL_NAME)
nasa7 = NASA7VaporCp(MECHANISM_YAML, my_fuel)

print(f"FuelLib compounds:      {my_fuel.num_compounds}")
print(f"NASA7 species extracted: {nasa7.n_fuel}")
assert nasa7.n_fuel == my_fuel.num_compounds, "Shape mismatch!"
print("✓ Coefficient array shapes match FuelLib compound count.\n")

# --------------------------------------------------------------------------
# 2. Spot-check: n-decane (NC10H22) coefficients
# --------------------------------------------------------------------------
try:
    idx_nc10 = nasa7.species_names.index("NC10H22")
    print(f"n-Decane (NC10H22) — FuelLib index = {idx_nc10}")
    print(f"  T_mid = {nasa7.T_mid[idx_nc10]:.2f} K")
    print(f"  a_lo  = {nasa7.a_lo[idx_nc10, :5]}")
    print(f"  a_hi  = {nasa7.a_hi[idx_nc10, :5]}")
    print()
except ValueError:
    print("NC10H22 not found in this fuel — skipping spot check.\n")

# --------------------------------------------------------------------------
# 3. Cross-validate against Cantera species.thermo.cp(T)
# --------------------------------------------------------------------------
import cantera as ct

gas = ct.Solution(MECHANISM_YAML)
test_temps = [300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0]

print("Cross-validation: NASA7 polynomial vs Cantera species.thermo.cp(T)")
print(f"{'Species':<20s} {'T (K)':>8s} {'NASA7 (J/mol/K)':>16s} "
      f"{'Cantera (J/mol/K)':>18s} {'Rel Err':>10s}")
print("-" * 78)

max_rel_err = 0.0
for i, key in enumerate(nasa7.species_names):
    sp = gas.species(key)
    for T in test_temps:
        cp_nasa7 = nasa7.cp_per_species(T)[i]        # J/mol/K
        cp_cantera = sp.thermo.cp(T) / 1000.0         # J/kmol/K → J/mol/K
        rel_err = abs(cp_nasa7 - cp_cantera) / cp_cantera if cp_cantera > 0 else 0.0
        max_rel_err = max(max_rel_err, rel_err)

# Print a summary instead of all N×T lines
print(f"  (checked {len(nasa7.species_names)} species × {len(test_temps)} temperatures)")
print(f"  Max relative error: {max_rel_err:.2e}")
if max_rel_err < 1e-10:
    print("  ✓ Agreement to machine precision.\n")
else:
    print(f"  ✗ ERROR: relative error {max_rel_err:.2e} exceeds tolerance!\n")

# Print a few representative values
print("Representative values:")
print(f"{'Species':<20s} {'T (K)':>8s} {'NASA7':>14s} {'Cantera':>14s}")
print("-" * 60)
sample_species = ["C6H5CH3", "NC10H22", "DECALIN", "C12H26-2"]
for key in sample_species:
    if key in nasa7.species_names:
        i = nasa7.species_names.index(key)
        sp = gas.species(key)
        for T in [400.0, 700.0]:
            cp_n = nasa7.cp_per_species(T)[i]
            cp_c = sp.thermo.cp(T) / 1000.0
            print(f"  {key:<18s} {T:8.1f} {cp_n:14.6f} {cp_c:14.6f}")
print()

# --------------------------------------------------------------------------
# 4. Liquid-proxy vs NASA7 comparison
# --------------------------------------------------------------------------
print("=" * 70)
print("Comparison: Liquid-Cp proxy  vs  NASA7 ideal-gas Cp")
print("=" * 70)

# Use the fuel's initial composition as a representative vapour
Xi_init = my_fuel.Y2X(my_fuel.Y_0)
Yi_test = Xi_init / np.sum(Xi_init)  # normalised mole fractions

print(f"\nFuel: {FUEL_NAME} (initial composition as vapour proxy)")
print(f"{'T (K)':>8s} {'Liquid Cp':>14s} {'NASA7 Cp':>14s} {'Diff (%)':>10s}")
print("-" * 50)

for T in [350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]:
    # Liquid-proxy (original method)
    Cp_liq = float(np.dot(Yi_test, my_fuel.Cp(T)))
    # NASA7 ideal-gas
    Cp_ig = nasa7.mixture_cp(T, Yi_test)
    pct_diff = 100.0 * (Cp_liq - Cp_ig) / Cp_ig if Cp_ig != 0 else 0.0
    print(f"  {T:6.1f} {Cp_liq:14.4f} {Cp_ig:14.4f} {pct_diff:+10.2f}%")

print("\nDone.")
