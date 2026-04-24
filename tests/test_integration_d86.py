"""
D86 distillation integration test — runs the full RK2 simulation for
posf10264 / posf10325 / posf10289, interpolates the simulated distillate
temperature onto the NJFCP experimental volume grid, and compares against:

1. the committed baseline in ``tests/baselinePredictions/d86/{fuel}.csv``
   with a tight tolerance (regression detection), and
2. the NJFCP experimental data in
   ``fuelData/experimentalData/d86_NJFCP.csv`` with a loose tolerance
   (catches catastrophic model breakage).

The baseline can be regenerated from a fresh simulation by running::

    python scripts/regenerate_d86_baselines.py
    # or
    pytest tests/test_integration_d86.py --update-baselines

The loose tolerance against the NJFCP data is deliberately wide — the
purpose of this test is to guard against catastrophic model breakage
(e.g., NaNs, the simulator terminating early, an accidental factor-of-ten
in a physical property), not to validate the quality of the simulation.
Model-vs-experiment calibration is the job of the Optuna drivers in
``source/optimize_d86_optuna*.py``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from source.distillation_rk2 import run_d86_simulation_rk2


FUELS = ("posf10325", "posf10289", "posf10264")
BASELINE_DIR = os.path.join(
    os.path.dirname(__file__), "baselinePredictions", "d86",
)

# Regression tolerance: simulated curve vs committed baseline.
# The tight tolerance catches accidental changes to the physics or solver
# tuning.  A deliberate model change MUST be accompanied by a baseline
# regeneration (see docstring).
REGRESSION_MAX_ABS_DEV_K = 1.0   # Kelvin / °C (same thing for deltas)
REGRESSION_MAE_K         = 0.5   # Kelvin

# Sanity tolerance: simulated curve vs NJFCP experimental data.
# Wide — see module docstring.
EXP_MAE_K       = 60.0  # Kelvin — catches catastrophic breakage only
EXP_MAX_ABS_K   = 90.0


@pytest.fixture(scope="module")
def d86_simulations(posf_fuels, default_d86_sim_params, experimental_d86_data):
    """Run the D86 simulation once for each fuel and cache the results."""
    target_vols = experimental_d86_data["VolumePercentage"].to_numpy(dtype=float)
    runs = {}
    for name in FUELS:
        f_obj = posf_fuels[name]
        Xi = f_obj.Y2X(f_obj.Y_0)
        sp = dict(default_d86_sim_params)
        results = run_d86_simulation_rk2(
            f_obj, Xi, volume_initial_mL=100.0, sim_params=sp,
        )
        vols = np.asarray(results["distillate_vol"], dtype=float)
        T_C  = np.asarray(results["T_D86_degC"],      dtype=float)

        assert vols.size >= 3, (
            f"{name}: simulation produced fewer than 3 records "
            f"(vols={vols}); check Q1 / controller / min_volume_W_mL."
        )
        assert vols[-1] >= 85.0, (
            f"{name}: simulation stopped at only {vols[-1]:.1f} mL distilled; "
            "it should reach at least 85 mL for the 80 % / 90 % comparison."
        )
        assert np.all(np.isfinite(T_C)), f"{name}: non-finite T_D86 in results."

        # interp1d requires strictly increasing x; drop duplicates.
        _, uniq = np.unique(vols, return_index=True)
        vols = vols[uniq]
        T_C  = T_C[uniq]

        T_at_targets = np.interp(target_vols, vols, T_C)
        runs[name] = {
            "target_vols":  target_vols,
            "T_at_targets": T_at_targets,
            "vols_full":    vols,
            "TC_full":      T_C,
        }
    return runs


@pytest.mark.d86
@pytest.mark.parametrize("fuel_name", FUELS)
def test_d86_matches_committed_baseline(
    fuel_name, d86_simulations, update_baselines,
):
    """Each fuel's D86 curve must match the committed baseline to within
    ``REGRESSION_MAX_ABS_DEV_K``."""
    run = d86_simulations[fuel_name]
    target_vols = run["target_vols"]
    T_sim = run["T_at_targets"]

    baseline_path = os.path.join(BASELINE_DIR, f"{fuel_name}.csv")

    if update_baselines:
        os.makedirs(BASELINE_DIR, exist_ok=True)
        pd.DataFrame(
            {"VolumePercentage": target_vols, "T_D86_degC": T_sim},
        ).to_csv(baseline_path, index=False, float_format="%.4f")
        pytest.skip(f"baseline regenerated for {fuel_name}")

    assert os.path.isfile(baseline_path), (
        f"baseline file missing for {fuel_name}: {baseline_path}.\n"
        "Generate it with `python scripts/regenerate_d86_baselines.py` or "
        "`pytest --update-baselines`."
    )
    base = pd.read_csv(baseline_path)
    # Align on the volume grid (should already match 1:1 but be tolerant).
    T_base = np.interp(target_vols, base["VolumePercentage"].to_numpy(),
                       base["T_D86_degC"].to_numpy())

    dev = T_sim - T_base
    mae = float(np.mean(np.abs(dev)))
    max_abs = float(np.max(np.abs(dev)))

    assert mae <= REGRESSION_MAE_K, (
        f"{fuel_name}: mean abs deviation from baseline {mae:.3f} K "
        f"exceeds {REGRESSION_MAE_K} K.  If this is an intentional model "
        f"change, regenerate with `pytest --update-baselines`.\n"
        f"  target_vols = {target_vols.tolist()}\n"
        f"  sim         = {T_sim.tolist()}\n"
        f"  baseline    = {T_base.tolist()}\n"
        f"  deviation   = {dev.tolist()}"
    )
    assert max_abs <= REGRESSION_MAX_ABS_DEV_K, (
        f"{fuel_name}: max abs deviation from baseline {max_abs:.3f} K "
        f"exceeds {REGRESSION_MAX_ABS_DEV_K} K.\n"
        f"  deviation = {dev.tolist()}"
    )


@pytest.mark.d86
@pytest.mark.parametrize("fuel_name", FUELS)
def test_d86_within_sanity_of_njfcp(
    fuel_name, d86_simulations, experimental_d86_data,
):
    """Simulated D86 curve must be within a loose MAE of the NJFCP
    experimental data.  This is a catastrophic-breakage guard, **not**
    a model-quality check — see module docstring."""
    run = d86_simulations[fuel_name]
    T_sim = run["T_at_targets"]
    T_exp = experimental_d86_data[fuel_name].to_numpy(dtype=float)

    dev = T_sim - T_exp
    mae = float(np.mean(np.abs(dev)))
    max_abs = float(np.max(np.abs(dev)))

    assert np.all(np.isfinite(T_sim)), f"{fuel_name}: T_sim has non-finite values"
    assert mae <= EXP_MAE_K, (
        f"{fuel_name}: mean abs deviation from NJFCP experimental curve "
        f"{mae:.1f} K exceeds sanity tolerance {EXP_MAE_K} K — the model "
        f"has likely broken (a NaN, unit error, or similar).\n"
        f"  sim = {T_sim.tolist()}\n"
        f"  exp = {T_exp.tolist()}"
    )
    assert max_abs <= EXP_MAX_ABS_K, (
        f"{fuel_name}: max abs deviation from NJFCP experimental curve "
        f"{max_abs:.1f} K exceeds sanity tolerance {EXP_MAX_ABS_K} K."
    )


@pytest.mark.d86
@pytest.mark.parametrize("fuel_name", FUELS)
def test_d86_curve_is_monotonic_nondecreasing(fuel_name, d86_simulations):
    """D86 distillation temperature should monotonically (non-strictly)
    increase with volume distilled, modulo small controller wiggles
    (< 2 K down-steps are tolerated)."""
    run = d86_simulations[fuel_name]
    T_full = run["TC_full"]
    # Allow small, bounded non-monotonic dips (< 2 K) from PI-controller
    # response or stage-2 dew-point recalculations; reject anything larger.
    dT = np.diff(T_full)
    worst_drop = float(np.min(dT)) if dT.size else 0.0
    assert worst_drop > -2.0, (
        f"{fuel_name}: D86 temperature drops by {worst_drop:.2f} K "
        f"between consecutive records — the curve should be (nearly) "
        f"non-decreasing.  Largest drop between indices "
        f"{int(np.argmin(dT))} and {int(np.argmin(dT)) + 1}."
    )
