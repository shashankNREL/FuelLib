"""Pytest configuration for ``tests/jax_diff``.

Adds the package root to ``sys.path`` so the ``source.jax_diff`` import
works without a conftest fragment in every test file, and exposes
shared fixtures for the small reference fuels used across W2–W6.
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure the repo root is importable.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Enable float64 *before* any jax.numpy use.
import jax
jax.config.update("jax_enable_x64", True)

from source.FuelLib import fuel  # noqa: E402
import source.jax_diff as jd  # noqa: E402


# Tests are slow because UNIFAC + bisection-Newton inside vmap/jit warms
# up the compiler the first time.  We mark all tests in this directory
# with the ``jax`` marker; CI can keep the heavier ones gated.
@pytest.fixture(scope="session", params=["heptane-decane", "posf10325"])
def fuel_obj(request):
    """Two reference mixtures: a tiny binary and a kerosene surrogate."""
    return fuel(request.param)


@pytest.fixture(scope="session")
def small_fuel():
    """Always the binary heptane-decane (fast tests)."""
    return fuel("heptane-decane")


@pytest.fixture(scope="session")
def kerosene_fuel():
    """Always posf10325 (representative kerosene surrogate)."""
    return fuel("posf10325")


@pytest.fixture(scope="session")
def small_tables(small_fuel):
    return jd.build_fuel_tables(small_fuel)


@pytest.fixture(scope="session")
def kerosene_tables(kerosene_fuel):
    return jd.build_fuel_tables(kerosene_fuel)
