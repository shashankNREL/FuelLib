"""
Shared pytest fixtures and command-line options for the FuelLib test suite.

This module adds the repository root to ``sys.path`` so that ``paths``,
``source.FuelLib`` and the ``source.distillation*`` drivers can be imported
from any test module without extra boilerplate, and provides cached fuel
fixtures for the three POSF jet fuels used by the D86 integration tests.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

# Make the repository root importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Silence noisy Cp / psat under-range warnings from the library — they fire
# frequently during the long D86 runs and are not informative for testing.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: "pytest.Parser") -> None:
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help=(
            "Regenerate committed baseline CSVs (e.g. "
            "tests/baselinePredictions/d86/*.csv) instead of asserting "
            "against them.  Assertion tests marked with "
            "`baseline_can_update` become write-through."
        ),
    )


@pytest.fixture(scope="session")
def update_baselines(request: "pytest.FixtureRequest") -> bool:
    """Whether the user passed ``--update-baselines``."""
    return bool(request.config.getoption("--update-baselines"))


# ---------------------------------------------------------------------------
# Shared fuel fixtures
# ---------------------------------------------------------------------------

POSF_FUELS = ("posf10264", "posf10325", "posf10289")


@pytest.fixture(scope="session")
def fuel_class():
    """Return the :class:`FuelLib.fuel` class."""
    from source.FuelLib import fuel
    return fuel


@pytest.fixture(scope="session")
def posf_fuels(fuel_class) -> dict:
    """Cache of the three POSF fuel objects used by the D86 integration tests."""
    return {name: fuel_class(name) for name in POSF_FUELS}


@pytest.fixture(scope="session")
def default_d86_sim_params() -> dict:
    """
    Default D86 simulation parameters used by the integration tests.

    Kept in sync with ``scripts/regenerate_d86_baselines.py`` — any change
    here must be mirrored there, otherwise the baseline curves in
    ``tests/baselinePredictions/d86/`` will stop matching.
    """
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


@pytest.fixture(scope="session")
def experimental_d86_data() -> "pd.DataFrame":
    """Experimental NJFCP D86 curves as a pandas DataFrame."""
    import pandas as pd
    from paths import EXP_D86_FILE
    return pd.read_csv(EXP_D86_FILE)
