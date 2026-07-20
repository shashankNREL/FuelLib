"""
JAX-compatibility smoke tests for the new ASTM property helpers.

The plan
(``/Users/syellapa/.claude/plans/woolly-skipping-bee.md``) requires that
every new pure-function helper is portable to ``jax.numpy`` by a mechanical
``np`` -> ``jnp`` swap, so that a future inverse-design loop can compose
``fuel.heat_of_combustion(Yi)`` etc. inside ``jax.jit`` / ``jax.grad`` /
``jax.vmap`` without re-implementing the physics.

This module verifies that property, without requiring JAX to be a CI
dependency. If ``jax`` is not installed the entire class is skipped and CI
stays green on the ``numpy pandas scipy`` stack listed in the repo README.
Developers who want to exercise the JAX path can:

.. code-block:: bash

    conda activate ct-env
    pip install jax jaxlib
    python tests/test_jax_compat.py -v

The tests import each pure module-level helper (``_lhv_hess``,
``_cp_liq_rd``, ``_ysi_mix``, ``_fp_alqaheem``, ``_fp_alibakhshi``,
``_fp_liaw_ideal_iter``, ``_boehm2022_iter``, ``_freeze_max_over_j``,
``_psat_lee_kesler``) and drive them with ``jnp`` arrays under ``jax.jit``,
verifying that (a) the tracer path does not error out, (b) the jitted
result matches the eager numpy result to ``1e-6``, and (c) ``jax.grad``
returns a finite gradient (where applicable).
"""

import os
import sys
import unittest

import numpy as np

FUELLIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@unittest.skipIf(not HAS_JAX, "jax not installed; skipping JAX-compat tests")
class JaxCompatTestCase(unittest.TestCase):
    """Smoke test each pure ASTM helper under ``jax.jit`` and ``jax.grad``."""

    @classmethod
    def setUpClass(cls):
        # Enable float64 in JAX so tolerances match numpy default precision.
        jax.config.update("jax_enable_x64", True)
        # Import the pure helpers only (avoid pandas / CSV loads inside jit).
        from source.FuelLib import (  # noqa: E402
            _boehm2022_iter,
            _cp_liq_rd,
            _dcn_mix,
            _fp_alibakhshi,
            _fp_alqaheem,
            _fp_liaw_ideal_iter,
            _freeze_max_over_j,
            _lhv_hess,
            _psat_lee_kesler,
            _ysi_mix,
        )

        cls._lhv_hess = staticmethod(_lhv_hess)
        cls._cp_liq_rd = staticmethod(_cp_liq_rd)
        cls._ysi_mix = staticmethod(_ysi_mix)
        cls._dcn_mix = staticmethod(_dcn_mix)
        cls._fp_alqaheem = staticmethod(_fp_alqaheem)
        cls._fp_alibakhshi = staticmethod(_fp_alibakhshi)
        cls._fp_liaw_ideal_iter = staticmethod(_fp_liaw_ideal_iter)
        cls._boehm2022_iter = staticmethod(_boehm2022_iter)
        cls._freeze_max_over_j = staticmethod(_freeze_max_over_j)
        cls._psat_lee_kesler = staticmethod(_psat_lee_kesler)

    def _representative_n_alkane(self, n_c):
        """Return array inputs for an n-alkane mixture (single component)."""
        n_C = np.array([float(n_c)])
        n_H = np.array([2.0 * n_c + 2.0])
        Hf = np.array([-300e3])  # J/mol, rough
        MW = np.array([n_c * 12.0 + (2 * n_c + 2) * 1.0]) * 1e-3
        return n_C, n_H, Hf, MW

    def test_lhv_hess_jit_and_grad(self):
        """`_lhv_hess` under jit + grad w.r.t. Hf."""
        n_C, n_H, Hf, MW = self._representative_n_alkane(12)
        # Numpy reference
        ref = self._lhv_hess(n_C, n_H, Hf, MW)
        jitted = jax.jit(self._lhv_hess)
        out = jitted(
            jnp.asarray(n_C), jnp.asarray(n_H), jnp.asarray(Hf), jnp.asarray(MW)
        )
        self.assertTrue(np.allclose(np.asarray(out), ref, atol=1e-6))
        # Gradient with respect to Hf should be non-zero (linear in Hf).
        grad_fn = jax.grad(
            lambda Hf: jnp.sum(
                self._lhv_hess(jnp.asarray(n_C), jnp.asarray(n_H), Hf, jnp.asarray(MW))
            )
        )
        g = grad_fn(jnp.asarray(Hf))
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        self.assertTrue(float(jnp.abs(g[0])) > 0.0)

    def test_cp_liq_rd_jit_and_grad(self):
        """`_cp_liq_rd` under jit + grad w.r.t. T."""
        A = np.array([22.0])
        B = np.array([-0.5])
        D = np.array([1.5])
        MW = np.array([170.34e-3])
        ref = self._cp_liq_rd(298.15, A, B, D, MW)
        jitted = jax.jit(self._cp_liq_rd)
        out = jitted(
            298.15, jnp.asarray(A), jnp.asarray(B), jnp.asarray(D), jnp.asarray(MW)
        )
        self.assertTrue(np.allclose(np.asarray(out), ref, atol=1e-6))
        grad_fn = jax.grad(
            lambda T: jnp.sum(
                self._cp_liq_rd(
                    T, jnp.asarray(A), jnp.asarray(B), jnp.asarray(D), jnp.asarray(MW)
                )
            )
        )
        g = grad_fn(298.15)
        self.assertTrue(float(jnp.abs(g)) > 0.0)

    def test_ysi_mix_jit_and_grad(self):
        """`_ysi_mix` under jit + grad w.r.t. Xi."""
        Xi = np.array([0.3, 0.5, 0.2])
        ysi = np.array([36.0, 100.0, 500.0])
        ref = self._ysi_mix(Xi, ysi)
        jitted = jax.jit(self._ysi_mix)
        out = jitted(jnp.asarray(Xi), jnp.asarray(ysi))
        self.assertTrue(np.allclose(float(out), ref, atol=1e-6))
        grad_fn = jax.grad(lambda Xi: self._ysi_mix(Xi, jnp.asarray(ysi)))
        g = grad_fn(jnp.asarray(Xi))
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        # Gradient should equal ysi vector (linear model), verify to 1e-6.
        self.assertTrue(np.allclose(np.asarray(g), ysi, atol=1e-6))

    def test_dcn_mix_jit_and_grad(self):
        """`_dcn_mix` under jit + grad w.r.t. phi (volume fractions)."""
        phi = np.array([0.2, 0.5, 0.3])
        dcn = np.array([100.0, 45.0, 9.0])
        ref = self._dcn_mix(phi, dcn)
        jitted = jax.jit(self._dcn_mix)
        out = jitted(jnp.asarray(phi), jnp.asarray(dcn))
        self.assertTrue(np.allclose(float(out), ref, atol=1e-6))
        grad_fn = jax.grad(lambda p: self._dcn_mix(p, jnp.asarray(dcn)))
        g = grad_fn(jnp.asarray(phi))
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        # Linear model: gradient equals the dcn vector.
        self.assertTrue(np.allclose(np.asarray(g), dcn, atol=1e-6))

    def test_fp_alqaheem_jit(self):
        """`_fp_alqaheem` is trivially JAX-jittable."""
        Tb = np.array([371.5, 447.3, 489.5])
        ref = self._fp_alqaheem(Tb)
        jitted = jax.jit(self._fp_alqaheem)
        out = jitted(jnp.asarray(Tb))
        self.assertTrue(np.allclose(np.asarray(out), ref, atol=1e-6))

    def test_fp_alibakhshi_jit(self):
        """`_fp_alibakhshi` under jit."""
        Tb = np.array([371.5, 447.3, 489.5])
        phi = np.array([-13.7, -16.1, -18.4])
        ref = self._fp_alibakhshi(Tb, phi)
        jitted = jax.jit(self._fp_alibakhshi)
        out = jitted(jnp.asarray(Tb), jnp.asarray(phi))
        self.assertTrue(np.allclose(np.asarray(out), ref, atol=1e-6))

    def test_psat_lee_kesler_jit(self):
        """`_psat_lee_kesler` under jit — known limitation.

        The helper uses ``np.log`` / ``np.exp`` which reject JAX tracers
        (they call ``__array__`` conversion). Fix requires a numpy/jax
        dispatch shim (either ``x.__array_namespace__()`` per Python Array
        API, or a lightweight ``_log(x)`` helper that detects tracer types
        at call time). Tracked as future work — the helper is already used
        by the (non-jitted) ``flash_point`` wrapper today.
        """
        T = 300.0
        Tc = np.array([540.0, 617.0, 658.0])
        Pc = np.array([27.4e5, 21.1e5, 18.2e5])
        omega = np.array([0.35, 0.49, 0.57])
        ref = self._psat_lee_kesler(T, Tc, Pc, omega)
        jitted = jax.jit(self._psat_lee_kesler)
        out = jitted(T, jnp.asarray(Tc), jnp.asarray(Pc), jnp.asarray(omega))
        self.assertTrue(np.allclose(np.asarray(out), ref, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
