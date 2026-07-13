"""
Tutorial — freeze-point blending physics via Boehm 2022 eq 21.

Illustrates how the mixture freeze point of a hydrocarbon blend varies as
a small amount of a high-freezing-point species is added to a low-
freezing-point solvent — the practical scenario Boehm et al. 2022
(*Energy & Fuels* **36**, 12046,
``papers/blend-prediction-model-for-the-freeze-point-of-jet-fuel-range-hydrocarbons.pdf``)
target with their eq 21. It mirrors their Figure 4 conceptually: dope
n-heptane (a low-freezing solvent) with increasing mole fraction of
n-decane (a higher-freezing solute) and watch the mixture freeze
point rise from the solvent Tm toward the solute Tm.

Uses the ``heptane-decane`` binary already in FuelLib. Pure endpoints
match their Constantinou-Gani-predicted melting points; the shape of
the curve is set by the SLE dilution + mixing-entropy terms in
Boehm's eq 21.

Known limitation: the Walden fallback for per-compound ΔHfus / ΔSfus
(``dSfus = 56.5 J/mol/K`` constant, ``dHfus = dSfus x Tm``) is
adequate at the low-dopant end but under-constrains the model near
50/50, where both candidate ``j`` compounds have comparable ``x_j``
and the outer ``max_over_j`` fluctuates. A Naef-2019 GC upgrade
(pending paper access) is expected to remove this instability. See
``tools/IMPLEMENTATION_LOG_astm.md`` Slice 6 for detailed deviation
notes.
"""

import os
import sys

import numpy as np

FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
import paths  # noqa: E402,F401 — appends source/ to sys.path
import FuelLib as fl  # noqa: E402


def main():
    fuel = fl.fuel("heptane-decane")
    print(f"Fuel: {fuel.name}")
    print(f"Compounds: {fuel.compounds}")
    print(f"CG-predicted pure Tm: {fuel.Tm} K")
    print("Reference (NIST): n-heptane Tm = 182.55 K, n-decane Tm = 243.45 K")
    print()

    # Sweep n-decane mass fraction from 0.01 to 0.10 — the dilute-dopant
    # regime where Boehm 2022 eq 21 is well-conditioned. This mirrors the
    # published Boehm Fig 4 scenario: small amount of high-Tm dopant in a
    # lower-Tm solvent.
    print("Dilute-dopant regime (0.005 <= w_decane <= 0.10):")
    print(
        f"  {'w_decane':>9s}  {'x_decane':>9s}  {'freeze pt (K)':>14s}  "
        f"{'freeze pt (C)':>14s}"
    )
    for y_decane in np.linspace(0.005, 0.10, 20):
        Yi = np.array([1.0 - y_decane, y_decane])
        Xi = fuel.Y2X(Yi)
        Tf = fuel.freeze_point(Yi=Yi, method="Boehm2022")
        print(f"  {y_decane:>9.3f}  {Xi[1]:>9.3f}  {Tf:>14.2f}  {fl.K2C(Tf):>14.2f}")

    print()
    print("Physics interpretation (Boehm 2022 eq 21 with Walden fallback):")
    print("  * At w_decane = 0.005 the mixture freezes at ~n-heptane Tm.")
    print("    n-decane is too dilute to be the highest-freezing component.")
    print("  * As w_decane rises to ~5 %, n-decane's in-mixture freeze")
    print("    temperature climbs above n-heptane's Tm and takes over as")
    print("    the max-over-j winner. Mixture freeze pt tracks n-decane's")
    print("    concentration-dependent SLE curve.")
    print("  * This is the same physics that lets Jet A carry 1-2 % n-C15")
    print("    or n-C16 (Tm ~ +10 C) yet still meet the -40 C freeze spec.")
    print()
    print("Known limitation: the near-50/50 and pure-dopant regimes are")
    print("intentionally not swept here — the Walden fallback for dHfus")
    print("and dSfus (a constant 56.5 J/mol/K for all hydrocarbons) is")
    print("adequate for dilute dopants but under-constrains the model at")
    print("higher concentrations. Upgrade to Naef-2019 GC parameters is")
    print("on the roadmap; see IMPLEMENTATION_LOG_astm.md Slice 6 for the")
    print("full deviation notes.")


if __name__ == "__main__":
    main()
