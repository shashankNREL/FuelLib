#!/usr/bin/env python3
"""
Regenerate the committed D86 baseline curves for the POSF fuels used in
``tests/test_integration_d86.py``.

Usage
-----
    python scripts/regenerate_d86_baselines.py            # all 3 fuels
    python scripts/regenerate_d86_baselines.py posf10264  # just one

Each baseline file contains the simulated D86 distillation-temperature curve
(°C) interpolated onto the NJFCP experimental volume-percentage grid
(0.5, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99.5).  The integration test in
``tests/test_integration_d86.py`` compares a fresh simulation against both the
NJFCP experimental data (loose tolerance) and the committed baseline (tight
tolerance for regression detection).

The default simulation parameters (dt, Q1, controller gains, glassware
capacitance) are those used by the test harness and must be kept in sync
with ``tests/conftest.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_FUELLIB_DIR = os.path.dirname(_SCRIPT_DIR)
if _FUELLIB_DIR not in sys.path:
    sys.path.insert(0, _FUELLIB_DIR)

from paths import EXP_D86_FILE, TESTS_DIR
from source.FuelLib import fuel
from source.distillation_rk2 import run_d86_simulation_rk2


FUELS = ["posf10264", "posf10325", "posf10289"]
BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions", "d86")


def default_sim_params() -> dict:
    """Default D86 simulation parameters (kept in sync with conftest.py)."""
    return dict(
        dt=2.0,
        min_volume_W_mL=5.0,
        P_atm=101325.0,
        T_room=298.15,
        Q1=60.0,
        initial_moles_air=0.008,
        C_glass=0.42,
        T_bubble_lo=350.0,
        T_bubble_hi=650.0,
        controller_Kp=2.0,
        controller_Ki=0.05,
        target_rate_ml_min=4.5,
        use_srk=False,
        verbose=False,
        record_every_n_steps=1,
    )


def run_d86_for_fuel(fuel_name: str, target_vols: np.ndarray) -> pd.DataFrame:
    """Run the RK2 D86 simulation for *fuel_name* and return a DataFrame
    with columns ``VolumePercentage`` (°C) and ``T_D86_degC`` at *target_vols*."""
    f_obj = fuel(fuel_name)
    Xi = f_obj.Y2X(f_obj.Y_0)
    sp = default_sim_params()
    results = run_d86_simulation_rk2(
        f_obj, Xi, volume_initial_mL=100.0, sim_params=sp,
    )
    vols = np.asarray(results["distillate_vol"], dtype=float)
    T_C  = np.asarray(results["T_D86_degC"],      dtype=float)

    # interp1d requires strictly increasing x; drop duplicates / non-monotonic
    _, uniq = np.unique(vols, return_index=True)
    vols = vols[uniq]
    T_C  = T_C[uniq]

    # np.interp handles out-of-range by clamping to endpoints, which is what
    # we want for volumes slightly outside the simulated range.
    T_at_targets = np.interp(target_vols, vols, T_C)
    return pd.DataFrame(
        {
            "VolumePercentage": target_vols,
            "T_D86_degC":       T_at_targets,
        },
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fuels_to_run = argv if argv else FUELS

    os.makedirs(BASELINE_DIR, exist_ok=True)

    # Use the same volume grid as the experimental data.
    df_exp = pd.read_csv(EXP_D86_FILE)
    target_vols = df_exp["VolumePercentage"].to_numpy(dtype=float)

    for fuel_name in fuels_to_run:
        print(f"[regenerate_d86_baselines] Running {fuel_name} ...", flush=True)
        df = run_d86_for_fuel(fuel_name, target_vols)
        out_path = os.path.join(BASELINE_DIR, f"{fuel_name}.csv")
        df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"[regenerate_d86_baselines]   wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
