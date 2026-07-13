"""
Tutorial — ASTM aviation-fuel qualification properties.

Demonstrates the five new group-contribution property methods added to
``fuel``:

- ``heat_of_combustion``   — net LHV in MJ/kg (Hess cycle on CG H_f, ASTM D4809)
- ``Cl``                    — liquid heat capacity in J/kg/K (Ruzicka-Domalski 1993)
- ``ysi``                   — Unified Yield Sooting Index (Das/McEnally Vol 2)
- ``flash_point``           — closed-cup flash point in K (Alibakhshi 2015 + Liaw)
- ``freeze_point``          — mixture freeze point in K (Boehm 2022 SLE)

Prints a side-by-side comparison of every FuelLib fuel against reference
values from Edwards, T. *Reference Jet Fuels for Combustion Testing*,
AIAA 2017-0146, and NIST for pure components.
"""

import os
import sys

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
import paths  # noqa: E402,F401 — appends source/ to sys.path
import FuelLib as fl  # noqa: E402

# Reference values collected from Edwards 2020 (POSF fuels) and NIST WebBook /
# DIPPR (pure compounds). Populate only what is publicly published; use None
# for values not reported.
REFERENCES = {
    "heptane": {
        "LHV_MJkg": 44.5,
        "Cl_298_JkgK": 2224,
        "FreezePoint_K": 182,
        "FlashPoint_K": 269,
        "YSI": 36.0,
        "source": "NIST WebBook",
    },
    "decane": {
        "LHV_MJkg": 44.2,
        "Cl_298_JkgK": 2200,
        "FreezePoint_K": 243,
        "FlashPoint_K": 319,
        "YSI": 57.2,
        "source": "NIST WebBook",
    },
    "dodecane": {
        "LHV_MJkg": 44.2,
        "Cl_298_JkgK": 2213,
        "FreezePoint_K": 263,
        "FlashPoint_K": 347,
        "YSI": 71.7,
        "source": "NIST WebBook",
    },
    "posf10264": {
        "LHV_MJkg": 43.15,
        "Cl_298_JkgK": 2018,
        "FreezePoint_K": 226,
        "FlashPoint_K": 315,
        "YSI": None,  # not in Edwards 2020
        "source": "Edwards 2020 AIAA 2017-0146 (Jet A)",
    },
    "posf10289": {
        "LHV_MJkg": 43.15,
        "Cl_298_JkgK": None,
        "FreezePoint_K": 219,
        "FlashPoint_K": 322,
        "YSI": None,
        "source": "Edwards 2020 AIAA 2017-0146 (JP-8)",
    },
    "posf10325": {
        "LHV_MJkg": 43.17,
        "Cl_298_JkgK": None,
        "FreezePoint_K": 226,
        "FlashPoint_K": 337,
        "YSI": None,
        "source": "Edwards 2020 AIAA 2017-0146 (JP-5)",
    },
    "posf11498": {
        "LHV_MJkg": 44.11,
        "Cl_298_JkgK": None,
        "FreezePoint_K": 240,
        "FlashPoint_K": 320,
        "YSI": None,
        "source": "NJFCP (HEFA-SPK)",
    },
    "posf4658": {
        "LHV_MJkg": None,
        "Cl_298_JkgK": None,
        "FreezePoint_K": 225,
        "FlashPoint_K": 323,
        "YSI": None,
        "source": "NJFCP (Jet A)",
    },
}


def _fmt(value, unit, precision=1):
    """Format ``value`` in ``unit`` or dashes if None / NaN."""
    if value is None:
        return "-".rjust(8)
    try:
        return f"{value:8.{precision}f}"
    except (TypeError, ValueError):
        return "-".rjust(8)


def _dev(gc, ref):
    """Format signed deviation gc - ref, or dashes."""
    if gc is None or ref is None:
        return "-".rjust(7)
    try:
        return f"{gc - ref:+7.1f}"
    except TypeError:
        return "-".rjust(7)


def report_fuel(name):
    """Compute all 5 ASTM props for ``name`` and print vs reference."""
    print(
        f"\n=== {name} ({REFERENCES.get(name, {}).get('source', 'no reference')}) ==="
    )
    ref = REFERENCES.get(name, {})
    fuel = fl.fuel(name)

    # Bulk mixture properties (single value per fuel)
    lhv = fuel.heat_of_combustion()  # MJ/kg
    fp = fuel.flash_point(method="Alibakhshi", mixing="Liaw")  # K
    fz = fuel.freeze_point(method="Boehm2022")  # K
    ysi = fuel.ysi()  # dimensionless
    cl_298 = float(fuel.mixture_density(fuel.Y_0, 298.15) * 0.0)  # placeholder
    # Mass-fraction weighted mixture Cp,L at 298 K:
    import numpy as np

    cl_298 = float(np.sum(fuel.Y_0 * fuel.Cl(298.15)))  # J/kg/K

    header = (
        f"  {'Property':<20s} {'unit':>10s}    {'GC':>8s}    "
        f"{'reference':>10s}    {'delta':>7s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'Heat of combustion':<20s} {'MJ/kg':>10s}    "
        f"{_fmt(lhv, 'MJ/kg', 2)}    {_fmt(ref.get('LHV_MJkg'), '', 2)}    "
        f"{_dev(lhv, ref.get('LHV_MJkg'))}"
    )
    print(
        f"  {'Liquid Cp @ 298 K':<20s} {'J/kg/K':>10s}    "
        f"{_fmt(cl_298, 'J/kg/K', 0)}    {_fmt(ref.get('Cl_298_JkgK'), '', 0)}    "
        f"{_dev(cl_298, ref.get('Cl_298_JkgK'))}"
    )
    print(
        f"  {'Flash point':<20s} {'K':>10s}    "
        f"{_fmt(fp, 'K', 1)}    {_fmt(ref.get('FlashPoint_K'), '', 1)}    "
        f"{_dev(fp, ref.get('FlashPoint_K'))}"
    )
    print(
        f"  {'Freeze point':<20s} {'K':>10s}    "
        f"{_fmt(fz, 'K', 1)}    {_fmt(ref.get('FreezePoint_K'), '', 1)}    "
        f"{_dev(fz, ref.get('FreezePoint_K'))}"
    )
    print(
        f"  {'YSI (Unified)':<20s} {'-':>10s}    "
        f"{_fmt(ysi, '', 1)}    {_fmt(ref.get('YSI'), '', 1)}    "
        f"{_dev(ysi, ref.get('YSI'))}"
    )


def main():
    print("ASTM aviation-fuel qualification properties via FuelLib")
    print("Reference values from Edwards 2020 (POSF) and NIST WebBook (pure).")

    fuel_list = [
        "heptane",
        "decane",
        "dodecane",
        "posf11498",
        "posf10264",
        "posf10289",
        "posf10325",
        "posf4658",
    ]
    for name in fuel_list:
        report_fuel(name)

    print(
        "\nAll five properties are pure numpy today; the underlying helpers "
        "are structured to be JAX-portable (see tests/test_jax_compat.py) so "
        "they can serve as forward evaluators inside a jax.grad-based "
        "inverse-design loop."
    )


if __name__ == "__main__":
    main()
