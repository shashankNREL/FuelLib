"""
Tutorial: UNIFAC 2.0 activity coefficients in FuelLib.

Demonstrates ``fuel.activity()`` and the ``activity_model='UNIFAC'`` option on
``mixture_vapor_pressure`` across three cases of increasing chemical complexity:

1. heptane-decane (near-ideal n-paraffin binary, both Main Group 1 only).
2. toluene at infinite dilution in n-heptane (extracted from posf10325) — the
   classic aromatic/paraffin non-ideal test case from Fredenslund 1975.
3. posf10325 jet-fuel mixture — full multi-component activity coefficient
   vector and Raoult vs UNIFAC bubble pressure at 343 K.

Prints results to stdout; no matplotlib required.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
import paths
import FuelLib as fl

# ---------------------------------------------------------------------------
# Case 1: heptane-decane binary at 320 K — near-ideal paraffin/paraffin
# ---------------------------------------------------------------------------
print("=" * 70)
print("Case 1: heptane-decane (paraffin/paraffin, near-ideal)")
print("=" * 70)

hd = fl.fuel("heptane-decane")
Xi = hd.Y2X(hd.Y_0)
gamma = hd.activity(Xi, T=320.0)
print(f"Compounds        : {hd.compounds}")
print(f"Mole fractions   : {Xi.round(4)}")
print(f"gamma at 320 K   : {gamma.round(4)}")
print(f"(near 1.0 because both compounds decompose to Main Group 1 only)")


# ---------------------------------------------------------------------------
# Case 2: toluene + n-heptane infinite-dilution non-ideality, extracted from posf10325
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 2: toluene at infinite dilution in n-heptane (aromatic/paraffin)")
print("=" * 70)

posf = fl.fuel("posf10325")
gc = pd.read_csv(
    os.path.join(paths.FUELDATA_GC_DIR, "posf10325_init.csv"), encoding="utf-8-sig"
)
idx_tol = int(np.where(gc["Reference Compound"].str.strip() == "toluene")[0][0])
idx_hep = int(np.where(gc["Reference Compound"].str.strip() == "n-heptane")[0][0])

# Synthetic infinite-dilution binary: x_toluene → 0, x_heptane → 1.
Xi_inf = np.zeros(posf.num_compounds)
Xi_inf[idx_tol] = 1e-6
Xi_inf[idx_hep] = 1.0 - 1e-6
gamma_inf = posf.activity(Xi_inf, T=298.15)

print(f"gamma_inf(toluene in heptane)   at 298.15 K : {gamma_inf[idx_tol]:.4f}")
print("Reference (Gmehling DDBST, UNIFAC 1.0, ~1.4–1.7)")


# ---------------------------------------------------------------------------
# Case 3: posf10325 jet fuel — full activity vector and bubble pressure shift
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 3: posf10325 multi-component jet fuel")
print("=" * 70)

Xi_full = posf.Y2X(posf.Y_0)
gamma_full = posf.activity(Xi_full, T=320.0)
print(f"num_compounds            : {posf.num_compounds}")
print(f"gamma at 320 K:")
print(
    f"  min, median, max       : {gamma_full.min():.4f}, "
    f"{np.median(gamma_full):.4f}, {gamma_full.max():.4f}"
)

# Bubble pressure comparison.
T_bubble = 343.0
p_ideal = posf.mixture_vapor_pressure(posf.Y_0, T=T_bubble, activity_model="ideal")
p_unifac = posf.mixture_vapor_pressure(posf.Y_0, T=T_bubble, activity_model="UNIFAC")
print(f"\nBubble pressure at {T_bubble} K:")
print(f"  ideal Raoult           : {p_ideal:>12.2f} Pa")
print(f"  UNIFAC-corrected       : {p_unifac:>12.2f} Pa")
print(f"  relative shift         : {100.0*(p_unifac - p_ideal)/p_ideal:>+12.2f} %")
