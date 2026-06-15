UNIFAC 2.0 Activity Coefficients
--------------------------------

FuelLib's ``fuel.activity()`` method computes liquid-phase activity coefficients
:math:`\gamma_i` for every component of a fuel using **UNIFAC 2.0**
(Hayer, Wendel, Mandt, Hasse, Jirasek, *Chem. Eng. J.* **504** (2025) 158667,
DOI `10.1016/j.cej.2024.158667 <https://doi.org/10.1016/j.cej.2024.158667>`_).
The method uses the classical UNIFAC equations of Fredenslund (1975) with the
**complete** pair-interaction parameter matrix produced by Bayesian matrix
completion in the UNIFAC 2.0 paper.

The same machinery is exposed through an opt-in keyword on
:func:`mixture_vapor_pressure`, switching the bubble-pressure calculation from
ideal Raoult to UNIFAC-corrected.

Required input file
^^^^^^^^^^^^^^^^^^^

For every fuel ``<fuel_name>`` whose activity coefficients should be available,
ship a third per-fuel CSV in addition to the gcData and groupDecompositionData
files:

- ``FuelLib/fuelData/unifacDecomposition/<fuel_name>.csv`` — one row per compound
  (same order as ``gcData/<fuel_name>_init.csv``), 113 columns of integer UNIFAC
  subgroup counts. Column headers are the integer ``Subgroup_No`` values from
  ``unifacTableData/unifac_subgroups.csv`` (always unique; the human-readable
  ``Subgroup_Name`` column contains duplicates).

The classical UNIFAC convention applies: ``ACCH3``, ``ACCH2``, ``ACCH`` are
two-atom subgroups that include both the aromatic ring carbon and its directly
bonded alkyl substituent. ``AC`` represents only an unsubstituted no-H aromatic
ring carbon (e.g., naphthalene ring-fusion positions).

A contributor helper, ``tools/decompose_unifac.py``, generates these
decompositions from SMILES via RDKit (dev-only dependency, not used by CI).

Computing activity coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full tutorial is in ``tutorials/unifac.py``.

Single-binary near-ideal case:

.. code-block:: python

    import FuelLib as fl
    hd = fl.fuel("heptane-decane")
    Xi = hd.Y2X(hd.Y_0)
    gamma = hd.activity(Xi, T=320.0)
    print(gamma)   # both close to 1.0 — paraffin/paraffin is near-ideal

Infinite-dilution non-ideality (aromatic in paraffin) at 298.15 K:

.. code-block:: none

    >> gamma_inf(toluene in heptane) at 298.15 K : 1.4524
    >> (reference: Gmehling DDBST, classical UNIFAC, ~1.4-1.7)

Multi-component jet fuel (posf10325):

.. code-block:: none

    >> num_compounds            : 67
    >> gamma at 320 K:
    >>   min, median, max       : 0.8148, 1.0113, 1.7612

Bubble-pressure correction
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default :func:`mixture_vapor_pressure` and
:func:`mixture_vapor_pressure_antoine_coeffs` use Raoult's law, preserving the
behavior of earlier FuelLib releases. Pass ``activity_model="UNIFAC"`` to
multiply by :math:`\gamma_i`:

.. code-block:: python

    p_ideal  = posf.mixture_vapor_pressure(Yi, T=343.0)                        # Raoult
    p_unifac = posf.mixture_vapor_pressure(Yi, T=343.0, activity_model="UNIFAC")

.. code-block:: none

    >> Bubble pressure at 343.0 K:
    >>   ideal Raoult           :      4261.28 Pa
    >>   UNIFAC-corrected       :      4458.56 Pa
    >>   relative shift         :        +4.63 %

For SAF surrogates the dominant non-ideality comes from aromatic/paraffin
interactions (UNIFAC main groups 3 and 4 vs main group 1); paraffin/paraffin
and paraffin/cycloalkane mixtures stay near-ideal because they share Main
Group 1 in the classical UNIFAC group system.

Scope and limitations
^^^^^^^^^^^^^^^^^^^^^

- Classical UNIFAC equations with linear *r* and single-parameter
  :math:`\psi_{mn}=\exp(-a_{mn}/T)`. **Not** Modified UNIFAC (Dortmund), which
  uses :math:`r^{3/4}` and three-parameter :math:`a, b, c`.
- Classical UNIFAC does not distinguish cyclic-CH\ :sub:`2`\  from paraffin-CH\ :sub:`2`\
  (both Main Group 1). For SAF the resulting error is small because
  paraffin/cycloalkane mixtures are near-ideal anyway.
- The 13 fuels shipped in ``fuelData/unifacDecomposition/`` cover the
  SAF-relevant subset of UNIFAC main groups (MG1, MG2 terminal vinyls,
  MG3/MG4 aromatics). Fuels with oxygenates, halogens, or sulfur species would
  require extending ``tools/decompose_unifac.py``.

References
^^^^^^^^^^

- Hayer, N.; Wendel, T.; Mandt, S.; Hasse, H.; Jirasek, F. *Advancing
  thermodynamic group-contribution methods by machine learning: UNIFAC 2.0.*
  Chem. Eng. J. **504** (2025) 158667.
- Fredenslund, A.; Jones, R.L.; Prausnitz, J.M. *Group-contribution estimation
  of activity coefficients in nonideal liquid mixtures.* AIChE J. **21** (1975) 1086.
- Magnussen, T.; Rasmussen, P.; Fredenslund, A. *UNIFAC parameter table for
  prediction of liquid-liquid equilibria.* Ind. Eng. Chem. Process Des. Dev.
  **20** (1981) 331.
