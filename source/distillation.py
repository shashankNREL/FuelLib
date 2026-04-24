import numpy as np
from scipy.optimize import brentq
from FuelLib import fuel, K2C  # noqa: E402 — FuelLib.py must be on sys.path


# ---------------------------------------------------------------------------
# 1. PHYSICAL PROPERTY ROUTINES (backed by FuelLib.fuel)
# ---------------------------------------------------------------------------

def calculate_K_value(fuel_obj: fuel, i: int, T: float, P: float, Xi: np.ndarray) -> float:
    """
    Vapour-liquid equilibrium K-value for component *i* using modified Raoult's law.

    K_i = γ_i * P_i^sat(T) / P

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object for the mixture.
    :param i: Component index.
    :param T: Temperature in Kelvin.
    :param P: System pressure in Pa.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :returns: K-value for component *i*.
    :rtype: float
    """
    psat_i = fuel_obj.psat(T)[i]          # Pa  (Lee-Kesler correlation)
    gamma_i = fuel_obj.activity(Xi, T)[i] # UNIFAC activity coefficient
    return gamma_i * psat_i / P


def calculate_heat_of_vaporization(fuel_obj: fuel, T: float, Xi: np.ndarray) -> float:
    """
    Mole-fraction-averaged latent heat of vaporization for the liquid mixture.

    Uses :meth:`FuelLib.fuel.latent_heat_vaporization` (Watson correlation) and
    :attr:`FuelLib.fuel.MW` to convert from J/kg to J/mol, then averages over
    components with liquid mole fractions *Xi*.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :returns: Mixture latent heat of vaporization in J/mol.
    :rtype: float
    """
    # latent_heat_vaporization returns J/kg for each component
    Lv_i = fuel_obj.latent_heat_vaporization(T)          # (num_compounds,)  J/kg
    Lv_mol_i = Lv_i * fuel_obj.MW                         # J/mol per component
    return float(np.dot(Xi, Lv_mol_i))                    # mixture-averaged J/mol
    """
    Mole-fraction-averaged latent heat of vaporization for the liquid mixture.

    Uses :meth:`FuelLib.fuel.latent_heat_vaporization` (Watson correlation) and
    :attr:`FuelLib.fuel.MW` to convert from J/kg to J/mol, then averages over
    components with liquid mole fractions *Xi*.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :returns: Mixture latent heat of vaporization in J/mol.
    :rtype: float
    """
    # latent_heat_vaporization returns J/kg for each component
    Lv_i = fuel_obj.latent_heat_vaporization(T)          # (num_compounds,)  J/kg
    Lv_mol_i = Lv_i * fuel_obj.MW                         # J/mol per component
    return float(np.dot(Xi, Lv_mol_i))                    # mixture-averaged J/mol


def calculate_liquid_heat_capacity(fuel_obj: fuel, T: float, Xi: np.ndarray, use_srk: bool = False, P_atm: float = 101325.0) -> float:
    """
    Mole-fraction-averaged liquid heat capacity of the mixture.

    Uses empirical :meth:`FuelLib.fuel.Cp` (J/mol/K) by default, or SRK 
    specific heat departure calculation if use_srk=True.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :param use_srk: Boolean flag to use the SRK EoS.
    :param P_atm: System pressure required for SRK (Pa).
    :returns: Mixture liquid Cp in J/mol/K.
    :rtype: float
    """
    if use_srk:
        # specific_heat_srk returns J/kg/K. Convert to J/mol/K
        Cp_kg = fuel_obj.specific_heat_srk(T, P_atm, Xi)
        MW_mix = float(np.dot(Xi, fuel_obj.MW))
        return Cp_kg * MW_mix
    else:
        Cp_i = fuel_obj.Cp(T)           # (num_compounds,) J/mol/K
        return float(np.dot(Xi, Cp_i))


def calculate_vapor_heat_capacity(fuel_obj: fuel, T: float, Yi: np.ndarray) -> float:
    """
    Mole-fraction-averaged vapour heat capacity of the mixture.

    .. warning::

        This currently re-uses :meth:`FuelLib.fuel.Cp`, which is a liquid-phase
        Cp polynomial (group-contribution), as an **approximation** for
        ideal-gas vapour Cp.  For heavy hydrocarbons near their normal boiling
        point the two agree to within ~15 %, which is adequate for the D86
        thermometer thermal-lag model.  A future revision should add a proper
        ideal-gas Cp routine (e.g., Rihani-Doraiswamy) and dispatch to it here
        when vapour Cp is requested.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T: Temperature in Kelvin.
    :param Yi: Vapour-phase mole fractions (shape: num_compounds).
    :returns: Mixture vapour Cp in J/mol/K (approximated from liquid Cp).
    :rtype: float
    """
    Cp_i = fuel_obj.Cp(T)           # (num_compounds,) J/mol/K  (liquid proxy)
    return float(np.dot(Yi, Cp_i))


def compute_h_coeff(T_wall: float, T_room: float, L: float = 0.13) -> float:
    """
    Compute the natural-convection heat-transfer coefficient *h* (W/m²/K)
    for a vertical cylinder using the Churchill-Chu correlation (Eqs. 32–36).

    Air properties (thermal conductivity *k*, Prandtl number *Pr*, and the
    Grashof group βgρ²/μ²) are polynomial fits evaluated at the film
    temperature  T_film = (T_wall + T_room) / 2.

    **Correlations**

    Eq. 34 — thermal conductivity of air (W/m/K)::

        k = -5.696e-4 + 1.025e-4·T - 4.764e-8·T² + 1.330e-11·T³

    Eq. 35 — Prandtl number::

        Pr = 8.123e-1 - 2.948e-4·T - 9.443e-7·T²
             + 3.584e-9·T³ - 3.827e-12·T⁴ + 1.373e-15·T⁵

    Eq. 36 — Grashof group::

        ln(βgρ²/μ²) = 43.83 - 4.4065·ln(T)   [units: 1/m³/K]

    Grashof / Rayleigh / Nusselt numbers::

        Gr  = (βgρ²/μ²) · L³ · ΔT
        Ra  = Gr · Pr
        Nu  = { Eq. 32  if Ra < 1e9
              { Eq. 33  otherwise
        h   = Nu · k / L

    :param T_wall: Wall (column outer surface) temperature in Kelvin.
    :param T_room: Ambient air temperature in Kelvin.
    :param L: Characteristic length of the vertical cylinder in metres (default 0.13 m).
    :returns: Heat-transfer coefficient *h* in W/m²/K.
    :rtype: float
    """
    T_film = 0.5 * (T_wall + T_room)          # film temperature (K)
    dT     = abs(T_wall - T_room)             # temperature difference (K)

    # --- Eq. 34: thermal conductivity of air (W/m/K) ---
    k_air = (
        -5.696e-4
        + 1.025e-4  * T_film
        - 4.764e-8  * T_film**2
        + 1.330e-11 * T_film**3
    )
    # The polynomial can dip below zero well outside its validated fit
    # interval (~200 K to ~1500 K).  Clamp to a physically reasonable lower
    # bound so a Nusselt · k product never turns negative.
    k_air = max(float(k_air), 1.0e-3)

    # --- Eq. 35: Prandtl number ---
    Pr = (
        8.123e-1
        - 2.948e-4  * T_film
        - 9.443e-7  * T_film**2
        + 3.584e-9  * T_film**3
        - 3.827e-12 * T_film**4
        + 1.373e-15 * T_film**5
    )

    # --- Eq. 36: Grashof group βgρ²/μ² (1/m³/K) ---
    ln_Gr_group = 43.83 - 4.4065 * np.log(T_film)
    Gr_group    = np.exp(ln_Gr_group)          # 1/(m³·K)

    # --- Grashof and Rayleigh numbers ---
    Gr = Gr_group * L**3 * dT                  # dimensionless
    Ra = Gr * Pr                               # Rayleigh number

    # --- Nusselt number (Churchill-Chu) ---
    psi = (0.492 / Pr) ** (9.0 / 16.0)        # inner bracket term
    if Ra < 1.0e9:
        # Eq. 32
        Nu = 0.68 + 0.670 * Ra**(1.0/4.0) / (1.0 + psi)**(4.0/9.0)
    else:
        # Eq. 33
        Nu = (0.825 + 0.387 * Ra**(1.0/6.0) / (1.0 + psi)**(8.0/27.0)) ** 2

    h = Nu * k_air / L
    return float(h)


# ---------------------------------------------------------------------------
# 2. CORE MODELING ROUTINES
# ---------------------------------------------------------------------------

def bubble_point_residual(T, fuel_obj, P, Xi, use_srk=False):
    """
    Residual of the bubble-point condition ``Σ Kᵢ·xᵢ − 1`` at temperature *T*.

    A bubble-point temperature is the root ``bubble_point_residual(T) = 0``
    (for fixed composition *Xi* and pressure *P*).  The residual is negative
    below the true bubble point (liquid is sub-cooled) and positive above
    (mixture is super-heated), so any bracketing root finder (``brentq``,
    ``bisect``) can recover *T_bub*.

    :param T: Temperature in Kelvin.
    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param P: System pressure in Pa.
    :param Xi: Liquid mole fractions (shape: ``num_compounds``).
    :param use_srk: If True, use the SRK EoS for K-values; otherwise modified
                    Raoult's law with UNIFAC activity coefficients.
    :returns: Bubble-point residual ``Σ Kᵢ xᵢ − 1``.
    :rtype: float
    """
    if use_srk:
        # Proper SRK VLE: K_i = φ_i^L(x) / φ_i^V(y) with y inner-iterated.
        K = fuel_obj.K_values_srk(T, P, Xi)
    else:
        # Call activity once per T to avoid O(N^2) expense
        gamma = fuel_obj.activity(Xi, T)
        # Vectorised K-value calculation (K_i = γ_i * P_sat_i / P)
        Psat = fuel_obj.psat(T)
        K = gamma * Psat / P
    return np.sum(K * Xi) - 1.0

def solve_stage1_bubble_point(
    fuel_obj: fuel,
    P: float,
    Xi: np.ndarray,
    T_lo: float = 250.0,
    T_hi: float = 650.0,
    use_srk: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Routine A, Part 1 — Bubble-Point Calculation.

    Finds the bubble-point temperature *T1* of a liquid of composition *Xi*
    at pressure *P* using :func:`scipy.optimize.brentq` on
    :func:`bubble_point_residual`.  If the initial bracket ``[T_lo, T_hi]``
    does not bracket a sign change, the bracket is expanded on the appropriate
    side (upward if the residual is negative at both ends, downward if
    positive at both ends) in 10 K steps for up to 50 iterations.

    :returns: ``(T1, vapor_composition)`` — bubble-point temperature (K) and
              equilibrium vapour mole fractions ``yᵢ = Kᵢ xᵢ / Σ(K x)``.
    """
    f_lo = bubble_point_residual(T_lo, fuel_obj, P, Xi, use_srk)
    f_hi = bubble_point_residual(T_hi, fuel_obj, P, Xi, use_srk)

    if f_lo * f_hi > 0:
        # Bracket expansion: walk the appropriate boundary.
        if f_lo < 0 and f_hi < 0:
            # Need a higher T where f > 0 — walk T_hi upward.
            current_T = T_hi
            for _ in range(50):
                current_T += 10.0
                f_test = bubble_point_residual(current_T, fuel_obj, P, Xi, use_srk)
                if f_test > 0:
                    T_hi = current_T
                    break
            else:
                T_hi = current_T  # fall through with degraded bracket
        elif f_lo > 0 and f_hi > 0:
            # Need a lower T where f < 0 — walk T_lo downward.
            current_T = T_lo
            for _ in range(50):
                current_T -= 10.0
                if current_T <= 150.0:
                    break
                f_test = bubble_point_residual(current_T, fuel_obj, P, Xi, use_srk)
                if f_test < 0:
                    T_lo = current_T
                    break
            else:
                T_lo = max(current_T, 150.0)

    T1 = brentq(
        bubble_point_residual, T_lo, T_hi,
        args=(fuel_obj, P, Xi, use_srk),
        xtol=1e-4, rtol=1e-6,
    )

    # Final compositions at equilibrium T1
    if use_srk:
        # Proper SRK: K = φ_L(x) / φ_V(y) with y converged internally.
        K = fuel_obj.K_values_srk(T1, P, Xi)
    else:
        gamma = fuel_obj.activity(Xi, T1)
        Psat = fuel_obj.psat(T1)
        K = gamma * Psat / P

    vapor_comp = K * Xi
    vs = np.sum(vapor_comp)
    if vs > 0:
        vapor_comp = vapor_comp / vs

    return T1, vapor_comp


def solve_stage1_energy_balance(
    fuel_obj: fuel,
    T1: float,
    Xi: np.ndarray,
    Q1: float,
    R2: float,
    T2: float,
    use_srk: bool = False,
    P_atm: float = 101325.0,
) -> float:
    """
    Routine A, Part 2 — Energy Balance.

    Calculates the vapour-generation rate *D1* (mol/s) from the pot.

    Energy balance:
        Q1 + R2 · Cp_l · (T2 − T1) = D1 · ΔH_vap

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param T1: Bubble-point temperature of the liquid (K).
    :param Xi: Liquid mole fractions (shape: num_compounds).
    :param Q1: Heat input to the pot (W = J/s).
    :param R2: Reflux return rate from Stage 2 (mol/s).
    :param T2: Temperature of the reflux stream (K).
    :param use_srk: Boolean flag to use the SRK EoS.
    :param P_atm: System pressure required for SRK (Pa).
    :returns: Vapour generation rate *D1* in mol/s.
    :rtype: float
    """
    deltaH_vap = calculate_heat_of_vaporization(fuel_obj, T1, Xi)     # J/mol
    Cp_l       = calculate_liquid_heat_capacity(fuel_obj, T1, Xi, use_srk=use_srk, P_atm=P_atm)      # J/mol/K

    if deltaH_vap <= 0:
        return 0.0

    D1 = (Q1 + R2 * Cp_l * (T2 - T1)) / deltaH_vap
    if D1 < 0.0:
        import warnings
        warnings.warn(
            f"solve_stage1_energy_balance: negative D1 ({D1:.3e} mol/s) at "
            f"T1={T1:.2f} K, Q1={Q1:.2f} W — heat input is less than sensible "
            f"cooling of the reflux.  Clamping to 0 (no net vaporisation).",
            RuntimeWarning, stacklevel=2,
        )
        D1 = 0.0
    return D1


def solve_rachford_rice(
    fuel_obj: fuel,
    zi: np.ndarray,
    T: float,
    P: float,
    tol: float = 1e-8,
    use_srk: bool = False,
    max_iter: int = 40,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Solve the Rachford-Rice isothermal flash at known temperature *T* and
    pressure *P* for a feed of composition *zi*.

    Uses successive substitution on the K-values (outer loop) with an inner
    :func:`scipy.optimize.brentq` solve for the vapour fraction *V* over the
    Whitson-Brulé negative-flash interval
    ``(1/(1-K_max), 1/(1-K_min))``.  Trivial one-phase feeds are detected by
    inspecting the Rachford-Rice residual at ``V = 0`` and ``V = 1``:
    no sign change ⇒ single-phase feed.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param zi: Feed mole fractions (shape: ``num_compounds``).
    :param T: Flash temperature (K).
    :param P: Flash pressure (Pa).
    :param tol: Absolute tolerance passed to ``brentq`` for *V* (default 1e-8).
    :param use_srk: If True, use the SRK EoS for K-values.
    :param max_iter: Maximum successive-substitution iterations (default 40).
    :returns: ``(V, xi, yi)`` — vapour fraction, liquid and vapour mole-fraction
              vectors.  For trivial feeds, ``xi = yi = zi``.
    :rtype: tuple[float, np.ndarray, np.ndarray]
    """
    zi = np.asarray(zi, dtype=float)
    Psat = fuel_obj.psat(T)

    # Initial guess for Ki using feed composition
    if use_srk:
        # Proper SRK: K = φ_L(z) / φ_V(z) (inner-loop converges y internally).
        Ki = fuel_obj.K_values_srk(T, P, zi)
    else:
        gamma = fuel_obj.activity(zi, T)
        Ki = gamma * Psat / P

    def _trivial_phase(Ki_local):
        """Return 0.0 (all liquid) / 1.0 (all vapour) / None (two-phase)."""
        # Robust two-phase test via residual at V=0 and V=1.
        # f(V=0) = Σ zᵢ(Kᵢ − 1); f(V=1) = Σ zᵢ(Kᵢ − 1)/Kᵢ
        f0 = float(np.sum(zi * (Ki_local - 1.0)))
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = float(np.sum(np.where(Ki_local > 0, zi * (Ki_local - 1.0) / Ki_local, 0.0)))
        if f0 <= 0.0:
            return 0.0  # sub-cooled / bubble-point not reached
        if f1 >= 0.0:
            return 1.0  # super-heated / dew-point exceeded
        return None

    triv = _trivial_phase(Ki)
    if triv is not None:
        return triv, zi.copy(), zi.copy()

    # Successive substitution for isothermal flash (outer loop)
    V = 0.5
    xi = zi.copy()
    converged = False
    for _iter in range(max_iter):
        # Rachford-Rice residual function for inner bracketing solve.
        # (Captures current Ki via closure — updated each outer iteration.)
        def rr_residual(v_frac):
            return float(np.sum(zi * (Ki - 1.0) / (1.0 + v_frac * (Ki - 1.0))))

        # True Whitson-Brulé negative-flash interval (root always inside).
        K_max = float(np.max(Ki))
        K_min = float(np.min(Ki))
        # Avoid singular bracket at exactly 1/(1−K) (where denominator vanishes).
        eps_b  = 1.0e-10
        V_lo = 1.0 / (1.0 - K_max) + eps_b
        V_hi = 1.0 / (1.0 - K_min) - eps_b

        try:
            V = brentq(rr_residual, V_lo, V_hi, xtol=tol, rtol=1e-10, maxiter=200)
        except ValueError:
            # Residual failed to bracket — fall back to whichever endpoint
            # has the smaller residual.
            V = V_lo if abs(rr_residual(V_lo)) < abs(rr_residual(V_hi)) else V_hi

        # Update liquid composition xi and then K-values
        denom = 1.0 + V * (Ki - 1.0)
        xi = zi / denom
        xi_sum = np.sum(xi)
        if xi_sum > 0:
            xi = xi / xi_sum

        if use_srk:
            # Pass current Yi = Ki * xi estimate to φ_V for consistency.
            yi_guess = Ki * xi
            s = np.sum(yi_guess)
            yi_guess = yi_guess / s if s > 0 else None
            Ki_new = fuel_obj.K_values_srk(T, P, xi, Y_vap=yi_guess)
        else:
            gamma_new = fuel_obj.activity(xi, T)
            Ki_new = gamma_new * Psat / P

        if np.allclose(Ki, Ki_new, rtol=1e-5, atol=1e-8):
            Ki = Ki_new
            converged = True
            break
        Ki = Ki_new

    if not converged:
        import warnings
        warnings.warn(
            f"solve_rachford_rice: successive substitution did not converge "
            f"in {max_iter} iterations at T={T:.2f} K, P={P:.0f} Pa "
            f"(last V={V:.4f}).", RuntimeWarning, stacklevel=2,
        )

    yi = Ki * xi
    yi_sum = np.sum(yi)
    if yi_sum > 0:
        yi = yi / yi_sum

    # Clamp V to [0, 1] for downstream consumers (negative flash used only
    # as an internal bracketing trick).
    V = max(0.0, min(1.0, float(V)))

    return V, xi, yi


def solve_stage2_flash(
    fuel_obj: fuel,
    D1: float,
    T1: float,
    vapor_comp_in: np.ndarray,
    h_coeff: float,
    A_area: float,
    T_room: float,
    P: float = 101325.0,
    use_srk: bool = False,
    T_min_floor: float = 5.0,
) -> tuple:
    """
    Routine B — Partial condensation on the exposed column neck.

    The stage-1 vapour (flow *D1* at temperature *T1*, composition *z_i*)
    enters the exposed neck at its dew point, loses heat *Q_loss* to the
    ambient room, and partially condenses.  The outlet is a two-phase
    stream at temperature *T2*:

        * reflux :  R2  →  returned to the pot  (mol/s)
        * forward : D2  =  D1 − R2              (mol/s, to the thermometer)

    **Energy balance (dew-point-pinned approximation)**

    The incoming vapour is at the dew point, so any heat loss drives
    partial condensation.  Over a D86 column neck (length ~13 cm, ambient
    natural convection) ``Q_loss`` is much smaller than the latent-heat
    flux ``D1 · ΔH_vap``, so sensible-cooling of the vapour is negligible
    compared to the latent-heat sink.  Under this approximation:

        Q_loss     = h_coeff · A_area · (T1 − T_room)          # W
        R2         = Q_loss / ΔH_vap(T1)                        # mol/s
        D2         = max(0, D1 − R2)                            # mol/s
        T2         = T1   (vapour stays on the dew-point curve)

    When the heat loss is large enough to drive ``R2 ≥ D1`` (total
    condensation), ``R2`` is clamped to ``D1`` and ``D2 = 0``.  A linear
    depression is then applied to *T2* so the thermocouple model sees a
    continuously falling column temperature rather than a discontinuity.

    **Compositions**

    An isothermal Rachford-Rice flash at ``(T2, P)`` gives the reflux and
    forward-vapour mole-fraction vectors.  For the dew-point-pinned case
    the flash is trivial (``V_flash → 1``) and both compositions collapse
    to the feed; the energy-balance split ``(R2, D2)`` still governs
    material flow so distillation continues.

    .. note::

        This approximation was verified to be acceptable for D86-scale
        geometry.  A proper simultaneous T2–VLE solve is ~100× more
        expensive because each energy-balance residual evaluation runs a
        full Rachford-Rice flash; the current formulation evaluates
        ``Q_loss`` and ``ΔH_vap`` once per call and the RR flash once for
        compositions.  A more rigorous solve should be added behind a
        ``rigorous_stage2`` flag in a follow-up.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param D1: Vapour input from Stage 1 (mol/s).
    :param T1: Temperature of incoming vapour (K).
    :param vapor_comp_in: Mole fractions of incoming vapour ``z_i``.
    :param h_coeff: Heat-transfer coefficient (W/m²/K).
    :param A_area: Heat-transfer area of column (m²).
    :param T_room: Ambient temperature (K).
    :param P: System pressure in Pa (default 101 325).
    :param use_srk: If True, use SRK EoS for the Rachford-Rice flash.
    :param T_min_floor: Lower-bound offset above T_room for T2 (K).
    :returns: ``(T2, R2, reflux_comp, D2, vapor_comp_out)``
    :rtype: tuple
    """
    z = np.asarray(vapor_comp_in, dtype=float)

    if D1 <= 1e-15:
        return T1, 0.0, z.copy(), 0.0, z.copy()

    # Energy balance — dew-point-pinned, linearised in T2.
    deltaH_vap = calculate_heat_of_vaporization(fuel_obj, T1, z)   # J/mol
    Q_loss     = h_coeff * A_area * (T1 - T_room)                   # W

    if deltaH_vap > 0.0:
        R2 = Q_loss / deltaH_vap                                    # mol/s
    else:
        R2 = 0.0

    # Clamp the condensation rate to the physical bound.
    if R2 >= D1:
        # Total condensation — the column is over-cooled for the current
        # vapour flow.  Pin T2 to a physically plausible value between
        # T_room and T1: use the temperature at which sensible cooling of
        # the reflux matches the remaining heat loss.  For stage-3 this
        # depressed T2 is the signal the thermometer actually sees.
        R2 = D1
        Cp_L = calculate_liquid_heat_capacity(
            fuel_obj, T1, z, use_srk=use_srk, P_atm=P,
        )
        if Cp_L > 0.0:
            dT = Q_loss / (D1 * Cp_L + 1.0e-12)
            T2 = max(T_room + T_min_floor, T1 - dT)
        else:
            T2 = T1
        D2 = 0.0
    else:
        R2 = max(0.0, R2)
        D2 = D1 - R2
        T2 = T1

    # Isothermal Rachford-Rice flash at (T2, P) for the compositions.
    V_flash, reflux_comp, vapor_comp_out = solve_rachford_rice(
        fuel_obj, z, T2, P, use_srk=use_srk,
    )

    # Trivial-phase fallback (expected at T2 ≈ T1 dew point).
    if V_flash >= 1.0 - 1e-6 or V_flash <= 1e-6:
        reflux_comp    = z.copy()
        vapor_comp_out = z.copy()

    return T2, R2, reflux_comp, D2, vapor_comp_out


def solve_stage3_cstr(
    fuel_obj: fuel,
    D2: float,
    T2: float,
    vapor_comp_in: np.ndarray,
    n_air_old: float,
    T_air_old: float,
    C_glass: float,
    dt: float = 1.0,
) -> tuple:
    """
    Routine C — Thermocouple thermal-lag model.

    Models the glass thermometer bulb as a first-order lumped-capacitance
    thermal mass exposed to a vapour stream at temperature *T2* with
    molar flow *D2* and composition *vapor_comp_in*.  The governing ODE is

        C_glass · dT/dt = (D2 · Cp_v) · (T2 − T)

    whose analytical solution over one step ``dt`` is

        T_new = T2 + (T_old − T2) · exp(−D2·Cp_v · dt / C_glass)

    This form is unconditionally stable, physically correct in both the
    fast-response (``D2·Cp_v·dt ≫ C_glass``) and slow-response
    (``D2·Cp_v·dt ≪ C_glass``) limits, and does **not** require the
    previous "moles-of-air in the CSTR" kludge to track sensible-heat
    persistence — persistence is encoded in *C_glass*.

    .. note::

        The ``n_air_old`` parameter is retained for backwards compatibility
        with the legacy signature but is no longer used in the thermal
        balance.  It is returned unchanged so existing call-sites continue
        to work.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object.
    :param D2: Vapour forward flow from Stage 2 (mol/s).
    :param T2: Temperature of the incoming vapour (K).
    :param vapor_comp_in: Mole fractions of incoming vapour (shape: num_compounds).
    :param n_air_old: Legacy parameter (unused, returned as-is).
    :param T_air_old: Current thermocouple temperature (K).
    :param C_glass: Effective heat capacity of the glassware + bulb (J/K).
    :param dt: Simulation time step (s). Default 1.0.
    :returns: ``(T_D86, n_air_new)`` — new thermocouple temperature (K) and
              (for backwards compatibility) ``n_air_old``.
    :rtype: tuple[float, float]
    """
    # Vapour heat capacity of the incoming stream (J/mol/K)
    Cp_v = calculate_vapor_heat_capacity(fuel_obj, T2, vapor_comp_in)

    # Heat-capacity rate of the vapour stream (W/K)
    C_flow = float(D2) * Cp_v

    if C_glass <= 0.0:
        # Massless thermometer — instantaneously tracks T2.
        return float(T2), n_air_old

    if C_flow <= 0.0:
        # No flow — thermometer stays at T_air_old (ignoring ambient losses).
        return float(T_air_old), n_air_old

    # Analytical update of the first-order ODE over dt.
    tau_ratio = C_flow * dt / C_glass
    # exp argument can be large — np.expm1 keeps accuracy near 0.
    decay = np.exp(-tau_ratio)
    T_D86 = T2 + (T_air_old - T2) * decay

    return float(T_D86), n_air_old


# ---------------------------------------------------------------------------
# 3. MAIN SIMULATION DRIVER
# ---------------------------------------------------------------------------

def run_d86_simulation(
    fuel_obj: fuel,
    Xi_initial: np.ndarray,
    volume_initial_mL: float = 100.0,
    sim_params: dict = None,
) -> dict:
    """
    Main function to run the entire D86 distillation simulation.

    :param fuel_obj: Initialised :class:`FuelLib.fuel` object for the mixture.
    :param Xi_initial: Initial liquid mole fractions (shape: num_compounds).
    :param volume_initial_mL: Initial volume of liquid fuel (mL). Standard is 100 mL.
    :param sim_params: Dictionary of simulation parameters:

        * ``dt``               — time step (s)
        * ``min_volume_W_mL``  — stopping criterion (volume of liquid left in pot, mL)
        * ``P_atm``            — system pressure (Pa)
        * ``T_room``           — ambient temperature (K)
        * ``Q1``               — heat input to pot (W)
        * ``initial_moles_air``— initial moles of air in the CSTR
        * ``h_coeff``          — heat-transfer coefficient (W/m²/K)
        * ``A_area``           — heat-transfer area (m²)
        * ``C_glass``          — glassware heat capacity (J/K)
        * ``T_bubble_lo``      — lower bound for the brentq bubble-point search (K)
        * ``T_bubble_hi``      — upper bound for the brentq bubble-point search (K)
        * ``record_every_n_steps`` — record every N time steps (default 10)
        * ``verbose``          — print per-record log line (default True)

    .. note::

        The caller's *sim_params* dictionary is **not mutated**: the driver
        works on a shallow copy internally.

    :returns: Dictionary with keys ``"time"``, ``"distillate_vol"``, ``"T_D86"``, ``"T_D86_degC"``.
    :rtype: dict
    """
    if sim_params is None:
        raise ValueError("sim_params dictionary is required")

    # Work on a shallow copy so the caller's sim_params dict is never mutated
    # (A4 — important because optuna and test harnesses reuse dicts).
    sp = dict(sim_params)

    use_srk = sp.get("use_srk", False)
    verbose = sp.get("verbose", True)
    record_every_n = int(sp.get("record_every_n_steps", 10))

    # --- Initialisation ---
    time   = 0.0
    Xi     = Xi_initial.copy()
    distillate_vol_collected = 0.0

    # Convert initial volume (mL) to moles
    T_init = sp.get("T_room", 298.15)
    Yi_initial = fuel_obj.X2Y(Xi)
    if use_srk:
        rho_init = fuel_obj.density_srk(T_init, sp.get("P_atm", 101325.0), Xi)
    else:
        rho_init = fuel_obj.mixture_density(Yi_initial, T_init) # kg/m^3
    mass_init_kg = volume_initial_mL * 1e-6 * rho_init      # kg
    MW_avg_init = float(np.dot(Xi, fuel_obj.MW))            # kg/mol
    W_moles = mass_init_kg / MW_avg_init                    # mol

    # Stage 2 initial state
    R2 = 0.0
    T2 = sp["T_room"]

    # Stage 3 initial state
    n_air = sp["initial_moles_air"]
    T_D86 = sp["T_room"]

    results = {"time": [], "distillate_vol": [], "T_D86": [], "T_D86_degC": []}

    T_lo = sp.get("T_bubble_lo", 350.0)
    T_hi = sp.get("T_bubble_hi", 650.0)

    # Column geometry (exposed neck)
    D_out = sp.get("D_out", 0.025)
    D_in  = sp.get("D_in",  0.0175)
    L     = sp.get("column_length", 0.13)
    A_area = np.pi * (D_out + D_in) * 0.5 * L

    h_mult = sp.get("h_multiplier", 1.0)
    h_coeff = compute_h_coeff(sp["T_room"], sp["T_room"], L) * h_mult

    # --- PI controller state (velocity form with clamping anti-windup) ---
    # Target distillation rate: 4.5 mL/min (within 4-5 mL/min band).
    _target_rate    = sp.get("target_rate_ml_min", 4.5)
    _Kp             = sp.get("controller_Kp",      2.0)   # W / (mL/min)
    _Ki             = sp.get("controller_Ki",      0.05)  # W / (mL/min · s)
    _Q_min          = sp.get("Q1_min",             0.0)   # W
    _Q_max          = sp.get("Q1_max",          1500.0)   # W

    _prev_error = 0.0
    _Q1         = sp.get("Q1", 15.0)
    # Exponential moving-average coefficient on the per-step distillation
    # rate (A5) so the controller sees a smoother signal than dV/dt directly.
    _rate_ema_alpha = float(sp.get("rate_ema_alpha", 0.3))
    _rate_ema       = 0.0

    if verbose:
        print("Starting D86 simulation...")
    V_pot_mL = volume_initial_mL
    dt = sp["dt"]
    step_counter = 0

    # --- Main Simulation Loop ---
    while V_pot_mL > sp.get("min_volume_W_mL", 1.0) and W_moles > 0:

        # --- Routine A: Bubble Point + Energy Balance ---
        T1, vapor_comp1 = solve_stage1_bubble_point(
            fuel_obj, sp["P_atm"], Xi, T_lo=T_lo, T_hi=T_hi, use_srk=use_srk
        )
        D1 = solve_stage1_energy_balance(
            fuel_obj, T1, Xi, _Q1, R2, T2, use_srk=use_srk, P_atm=sp["P_atm"]
        )

        # Update h_coeff from the current Stage-2 wall temperature (T2)
        # so natural-convection heat loss tracks the column temperature.
        h_coeff = compute_h_coeff(T2, sp["T_room"], L) * h_mult

        # --- Routine B: Flash / Condensation Stage ---
        T2, R2, reflux_comp, D2, vapor_comp2 = solve_stage2_flash(
            fuel_obj,
            D1, T1, vapor_comp1,
            h_coeff,
            A_area,
            sp["T_room"],
            P=sp["P_atm"],
            use_srk=use_srk,
        )

        # --- Routine C: CSTR & D86 Temperature ---
        T_D86, n_air = solve_stage3_cstr(
            fuel_obj,
            D2, T2, vapor_comp2,
            n_air, T_D86,
            sp["C_glass"],
            dt=dt,
        )

        # --- Update System State ---
        moles_distilled_step = D2 * dt  # approximate (condensation in Stage 3)

        dW = (R2 - D1) * dt
        W_moles += dW

        # Component material balance in the pot
        for idx in range(fuel_obj.num_compounds):
            d_moles = (R2 * reflux_comp[idx] - D1 * vapor_comp1[idx]) * dt
            new_moles = Xi[idx] * (W_moles - dW) + d_moles
            Xi[idx] = new_moles / W_moles if W_moles > 0 else 0.0

        # Re-normalise mole fractions
        xi_sum = np.sum(Xi)
        if xi_sum > 0:
            Xi /= xi_sum

        # Update pot volume dynamically
        Yi_pot = fuel_obj.X2Y(Xi)
        if use_srk:
            rho_pot = fuel_obj.density_srk(T1, sp["P_atm"], Xi)
        else:
            rho_pot = fuel_obj.mixture_density(Yi_pot, T1)  # Liquid is at bubble point T1
        MW_avg_pot = float(np.dot(Xi, fuel_obj.MW))     # kg/mol
        V_pot_mL = (W_moles * MW_avg_pot / rho_pot) * 1e6

        # Accumulate distillate volume (approx. mole → volume conversion at T_room)
        MW_avg = float(np.dot(vapor_comp2, fuel_obj.MW))        # kg/mol
        # Use T_room for volume consistency with initial 100 mL charge
        T_ref = sp.get("T_room", 298.15)
        if use_srk:
            rho_ref = fuel_obj.density_srk(T_ref, sp["P_atm"], vapor_comp2)
        else:
            rho_ref = fuel_obj.mixture_density(fuel_obj.X2Y(vapor_comp2), T_ref)  # kg/m^3
        dV_mL = moles_distilled_step * MW_avg / rho_ref * 1e6   # mL
        distillate_vol_collected += dV_mL

        # --- Velocity-form PI rate controller with clamping anti-windup ---
        # Smooth the per-step rate with an EMA so the controller doesn't
        # react to per-step noise from the discrete condensation flux.
        current_rate_raw = (dV_mL / dt) * 60.0
        _rate_ema = _rate_ema_alpha * current_rate_raw + (1.0 - _rate_ema_alpha) * _rate_ema
        error = _target_rate - _rate_ema

        # delta_Q = Kp * (e_k - e_{k-1}) + Ki * e_k * dt
        delta_Q = _Kp * (error - _prev_error) + _Ki * error * dt
        _Q1_unclamped = _Q1 + delta_Q
        _Q1 = max(_Q_min, min(_Q_max, _Q1_unclamped))
        # Clamping anti-windup: if Q1 saturated on this step AND the
        # integral contribution is pushing further into saturation, zero
        # out the integral increment by reverting _prev_error to *this*
        # error (so next step's (e_k - e_{k-1}) difference = 0).
        if _Q1_unclamped != _Q1 and (
            (_Q1_unclamped > _Q_max and error > 0.0) or
            (_Q1_unclamped < _Q_min and error < 0.0)
        ):
            _prev_error = error
        else:
            _prev_error = error

        time += dt
        step_counter += 1

        # --- Store Results ---
        if step_counter % record_every_n == 0:
            results["time"].append(time)
            results["distillate_vol"].append(distillate_vol_collected)
            results["T_D86"].append(T_D86)
            results["T_D86_degC"].append(K2C(T_D86))
            if verbose:
                print(f"Time: {time:.1f} s | Distilled: {distillate_vol_collected:.1f} mL"
                      f" | T_D86: {T_D86:.1f} K | T1: {T1:.1f} K")

    if verbose:
        print("Simulation finished.")
    return results


# ---------------------------------------------------------------------------
# 4. EXAMPLE USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate a fuel object (replace "my_fuel" with your mixture data file name)
    # my_fuel = fuel("my_fuel")

    # --- Example simulation parameters ---
    simulation_parameters = {
        "dt":                1.0,       # s, time step
        "min_volume_W_mL":   1.0,       # mL, end condition (stop at 1 mL left)
        "P_atm":             101325.0,  # Pa
        "T_room":            298.15,    # K
        "Q1":                150.0,     # W, heat input to pot
        "initial_moles_air": 0.008,     # mol (reduced to match D86 neck volume)
        "h_coeff":           10.0,      # W/(m^2·K)
        "A_area":            0.01,      # m^2
        "C_glass":           0.42,      # J/K (reduced to match thermocouple thermal mass)
        "T_bubble_lo":       250.0,     # K  — lower bound for bisect
        "T_bubble_hi":       650.0,     # K  — upper bound for bisect
    }

    # Example: uniform initial composition (all compounds equally present)
    # Xi_init = np.ones(my_fuel.num_compounds) / my_fuel.num_compounds
    #
    # final_results = run_d86_simulation(
    #     my_fuel,
    #     Xi_init,
    #     volume_initial_mL=100.0,
    #     sim_params=simulation_parameters,
    # )
    #
    # import matplotlib.pyplot as plt
    # plt.plot(final_results["distillate_vol"], final_results["T_D86"])
    # plt.xlabel("Distillate collected (mL)")
    # plt.ylabel("T_D86 (K)")
    # plt.title("D86 Distillation Curve")
    # plt.show()
    pass
