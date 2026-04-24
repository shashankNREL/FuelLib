"""
D86 distillation simulation with RK2 time integration and a lumped condenser model.

Model stages:
    Stage 1  : Pot bubble-point + energy balance  → D1, T1, vapor_comp1
    Stage 2  : Column neck heat loss (natural convection) → R2 reflux to pot, D2 forward
    Stage 2b : Condenser tube (lumped ε-NTU)      → T_cond, condensate + residual vapor
    Stage 3  : CSTR thermocouple mixing            → T_D86

Reflux back to the pot occurs ONLY from the exposed column neck (Stage 2).
The condenser condensate flows forward to the receiver.
"""

import numpy as np
import sys
import os

# Allow importing FuelLib and distillation routines from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from FuelLib import fuel, K2C
from distillation import (
    calculate_vapor_heat_capacity,
    compute_h_coeff,
    solve_stage1_bubble_point,
    solve_stage1_energy_balance,
    solve_stage2_flash,
    solve_rachford_rice,
    solve_stage3_cstr,
)


# ---------------------------------------------------------------------------
# CONDENSER MODEL  —  Lumped ε-NTU
# ---------------------------------------------------------------------------

def solve_condenser(
    fuel_obj: fuel,
    D_in: float,
    T_in: float,
    vapor_comp_in: np.ndarray,
    T_bath: float,
    UA_cond: float,
    P: float = 101325.0,
    use_srk: bool = False,
) -> tuple:
    """
    Lumped condenser model using the effectiveness-NTU method.

    The condenser is a single tube (D86 specification: 14 mm OD × 560 mm,
    393 mm submerged in a cooling bath at *T_bath*) modelled as one lumped
    heat-exchange element characterised by its *UA_cond* product (W/K).

    Physics
    -------
    1.  Effectiveness-NTU with the isothermal bath (C_max → ∞)::

            NTU = UA_cond / (D_in · Cp_v)
            ε   = 1 − exp(−NTU)

    2.  The outlet temperature is computed from the effectiveness::

            T_out = T_in − ε · (T_in − T_bath)

    3.  A Rachford-Rice VLE flash at (T_out, P) determines the vapour /
        liquid split and compositions.

    Parameters
    ----------
    fuel_obj : fuel
        Initialised FuelLib fuel object.
    D_in : float
        Molar flow entering the condenser (mol/s).
    T_in : float
        Temperature of the vapour entering the condenser (K).
    vapor_comp_in : ndarray
        Mole-fraction vector of the entering vapour.
    T_bath : float
        Cooling-bath temperature (K).
    UA_cond : float
        Overall heat-transfer coefficient × area product (W/K).
    P : float
        System pressure (Pa).
    use_srk : bool
        If True, use SRK EoS for VLE.

    Returns
    -------
    T_out : float
        Condenser outlet temperature (K).
    D_vapor_out : float
        Uncondensed vapour molar flow leaving the condenser (mol/s).
    D_liquid_out : float
        Condensed liquid molar flow leaving the condenser (mol/s).
    vapor_comp_out : ndarray
        Composition of the uncondensed vapour.
    liquid_comp_out : ndarray
        Composition of the condensate.
    """
    nc = fuel_obj.num_compounds
    zeros = np.zeros(nc)

    if D_in <= 1e-15 or T_in <= T_bath:
        # Nothing to condense
        return T_in, D_in, 0.0, vapor_comp_in.copy(), zeros.copy()

    # --- Vapour heat capacity (J/mol/K) ---
    Cp_v = calculate_vapor_heat_capacity(fuel_obj, T_in, vapor_comp_in)
    Cp_v = max(Cp_v, 1e-6)  # safety

    # --- Effectiveness-NTU (isothermal bath ⇒ C_max = ∞) ---
    C_min = D_in * Cp_v                 # W/K
    NTU   = UA_cond / (C_min + 1e-15)
    epsilon = 1.0 - np.exp(-NTU)

    # --- Condenser outlet temperature ---
    T_out = T_in - epsilon * (T_in - T_bath)
    T_out = max(T_out, T_bath)          # cannot go below bath temperature

    # --- VLE flash at (T_out, P) to determine split ---
    V_frac, liquid_comp, vapor_comp_out = solve_rachford_rice(
        fuel_obj, vapor_comp_in, T_out, P, use_srk=use_srk,
    )

    D_vapor_out  = D_in * V_frac
    D_liquid_out = D_in * (1.0 - V_frac)

    return T_out, D_vapor_out, D_liquid_out, vapor_comp_out, liquid_comp


# ---------------------------------------------------------------------------
# RK2 DERIVATIVE FUNCTION  (with condenser)
# ---------------------------------------------------------------------------

def _compute_derivatives_condenser(
    N: np.ndarray,
    T2_prev: float,
    R2_prev: float,
    Q1: float,
    fuel_obj: fuel,
    sim_params: dict,
    L: float,
) -> tuple:
    """
    Evaluate the right-hand side (dN/dt) of the component-mole ODE,
    including the lumped condenser stage between the column neck and
    the CSTR / thermocouple.

    Returns
    -------
    dN_dt : ndarray
        Rate of change of pot moles per component.
    T1 : float
        Pot bubble-point temperature (K).
    T2 : float
        Column neck temperature (K).
    R2 : float
        Reflux molar flow back to pot from neck (mol/s).
    D2 : float
        Vapour forwarded from neck into condenser (mol/s).
    vapor_comp2 : ndarray
        Composition of the vapour exiting the neck (used for T_D86 / Stage 3).
    T_cond : float
        Condenser outlet temperature (K).
    D_cond_total : float
        Total molar flow exiting the condenser (= D2, mol/s).
    D_cond_liq : float
        Condensed liquid molar flow (mol/s).
    cond_comp : ndarray
        Mole-weighted average composition of the condenser outlet.
    reflux_comp : ndarray
        Composition of the reflux returning to the pot from the neck.
    """
    use_srk = sim_params.get("use_srk", False)
    P       = sim_params["P_atm"]
    T_lo    = sim_params.get("T_bubble_lo", 350.0)
    T_hi    = sim_params.get("T_bubble_hi", 650.0)

    nc = fuel_obj.num_compounds
    W_tot = np.sum(N)
    if W_tot <= 0:
        z = np.zeros(nc)
        return z, sim_params["T_room"], sim_params["T_room"], 0.0, 0.0, \
               z, sim_params["T_room"], 0.0, 0.0, z, z

    Xi = N / W_tot

    # ── Stage 1: Bubble-point + energy balance ──────────────────────────
    T1, vapor_comp1 = solve_stage1_bubble_point(
        fuel_obj, P, Xi, T_lo=T_lo, T_hi=T_hi, use_srk=use_srk,
    )
    D1 = solve_stage1_energy_balance(
        fuel_obj, T1, Xi, Q1, R2_prev, T2_prev,
        use_srk=use_srk, P_atm=P,
    )

    # ── Stage 2: Column neck heat loss (natural convection) ─────────────
    h_mult  = sim_params.get("h_multiplier", 1.0)
    h_coeff = compute_h_coeff(T2_prev, sim_params["T_room"], L) * h_mult

    T2, R2, reflux_comp, D2, vapor_comp2 = solve_stage2_flash(
        fuel_obj,
        D1, T1, vapor_comp1,
        h_coeff, sim_params["A_area"],
        sim_params["T_room"],
        P=P, use_srk=use_srk,
    )

    # ── Stage 2b: Condenser (lumped ε-NTU) ──────────────────────────────
    T_cond, D_cond_vap, D_cond_liq, vcomp_cond, lcomp_cond = solve_condenser(
        fuel_obj, D2, T2, vapor_comp2,
        sim_params["T_bath"],
        sim_params["UA_cond"],
        P=P, use_srk=use_srk,
    )

    # Combined composition of the condenser outlet (mole-weighted average)
    D_cond_total = D_cond_vap + D_cond_liq
    if D_cond_total > 1e-15:
        cond_comp = (D_cond_vap * vcomp_cond + D_cond_liq * lcomp_cond) / D_cond_total
    else:
        cond_comp = vapor_comp2.copy()

    # Normalise
    cond_sum = np.sum(cond_comp)
    if cond_sum > 0:
        cond_comp /= cond_sum

    # ── dN/dt for pot (reflux ONLY from neck — condenser does NOT reflux) ─
    dN_dt = R2 * reflux_comp - D1 * vapor_comp1

    return (dN_dt, T1, T2, R2, D2, vapor_comp2,
            T_cond, D_cond_total, D_cond_liq,
            cond_comp, reflux_comp)


# ---------------------------------------------------------------------------
# MAIN SIMULATION DRIVER  —  RK2 (Heun) with condenser
# ---------------------------------------------------------------------------

def run_d86_simulation_rk2_condenser(
    fuel_obj: fuel,
    Xi_initial: np.ndarray,
    volume_initial_mL: float = 100.0,
    sim_params: dict = None,
) -> dict:
    """
    D86 distillation simulation using Runge-Kutta 2nd-order (Heun's method)
    with a lumped condenser stage.

    Required keys in *sim_params* (in addition to those of the base model):

        * ``UA_cond``  — condenser UA product (W/K)
        * ``T_bath``   — cooling-bath temperature (K)
    """
    if sim_params is None:
        raise ValueError("sim_params dictionary is required")

    use_srk = sim_params.get("use_srk", False)

    # ── Initialisation ──────────────────────────────────────────────────
    time                     = 0.0
    distillate_vol_collected = 0.0

    T_init     = sim_params.get("T_room", 298.15)
    Yi_initial = fuel_obj.X2Y(Xi_initial)

    if use_srk:
        rho_init = fuel_obj.density_srk(
            T_init, sim_params.get("P_atm", 101325.0), Xi_initial,
        )
    else:
        rho_init = fuel_obj.mixture_density(Yi_initial, T_init)

    mass_init_kg = volume_initial_mL * 1e-6 * rho_init
    MW_avg_init  = float(np.dot(Xi_initial, fuel_obj.MW))
    W_moles      = mass_init_kg / MW_avg_init

    # State: component moles vector
    N = Xi_initial.copy() * W_moles

    # Stage internal states
    R2_state  = 0.0
    T2_state  = sim_params["T_room"]
    n_air     = sim_params["initial_moles_air"]
    T_D86     = sim_params["T_room"]

    results = {"time": [], "distillate_vol": [], "T_D86": [], "T_D86_degC": []}

    # Geometry (exposed column neck)
    D_out = 0.025
    D_in  = 0.0175
    L     = 0.13
    sim_params["A_area"] = np.pi * (D_out + D_in) * 0.5 * L
    h_mult = sim_params.get("h_multiplier", 1.0)
    sim_params["h_coeff"] = compute_h_coeff(
        sim_params["T_room"], sim_params["T_room"], L,
    ) * h_mult

    # PI controller state
    _target_rate = sim_params.get("target_rate_ml_min", 4.5)
    _Kp          = sim_params.get("controller_Kp", 2.0)
    _Ki          = sim_params.get("controller_Ki", 0.05)
    _Q_min       = sim_params.get("Q1_min", 0.0)
    _Q_max       = sim_params.get("Q1_max", 1500.0)
    _prev_error  = 0.0
    _Q1          = sim_params.get("Q1", 15.0)

    V_pot_mL = volume_initial_mL
    dt       = sim_params["dt"]

    print("Starting D86 simulation (RK2 + condenser)...")

    while V_pot_mL > sim_params.get("min_volume_W_mL", 1.0) and np.sum(N) > 0:

        # ── k1 evaluation ───────────────────────────────────────────────
        (k1, T1_k1, T2_k1, R2_k1, D2_k1, vcomp2_k1,
         Tc_k1, Dtot_k1, Dliq_k1,
         ccomp_k1, rcomp_k1) = _compute_derivatives_condenser(
            N, T2_state, R2_state, _Q1, fuel_obj, sim_params, L,
        )

        # Predictor
        N_pred = np.maximum(N + dt * k1, 0.0)

        # ── k2 evaluation ───────────────────────────────────────────────
        (k2, T1_k2, T2_k2, R2_k2, D2_k2, vcomp2_k2,
         Tc_k2, Dtot_k2, Dliq_k2,
         ccomp_k2, rcomp_k2) = _compute_derivatives_condenser(
            N_pred, T2_k1, R2_k1, _Q1, fuel_obj, sim_params, L,
        )

        # Corrector
        N_new = np.maximum(N + (dt / 2.0) * (k1 + k2), 0.0)

        # ── Averaged quantities ─────────────────────────────────────────
        D2_avg    = 0.5 * (D2_k1 + D2_k2)
        T2_avg    = 0.5 * (T2_k1 + T2_k2)
        Dtot_avg  = 0.5 * (Dtot_k1 + Dtot_k2)
        Dliq_avg  = 0.5 * (Dliq_k1 + Dliq_k2)
        Tc_avg    = 0.5 * (Tc_k1 + Tc_k2)

        # Neck-exit vapor composition (for thermocouple / Stage 3)
        vcomp2_avg = 0.5 * (vcomp2_k1 + vcomp2_k2)
        vcomp2_norm = vcomp2_avg / (np.sum(vcomp2_avg) + 1e-15)

        # Condenser outlet composition (for distillate volume)
        ccomp_avg = 0.5 * (ccomp_k1 + ccomp_k2)
        ccomp_norm = ccomp_avg / (np.sum(ccomp_avg) + 1e-15)

        # ── Stage 3: CSTR / thermocouple ────────────────────────────────
        # The D86 thermocouple is at the flask neck, UPSTREAM of the
        # condenser.  It measures the vapour temperature leaving the pot.
        # Stage 3 CSTR models the thermal lag of the thermometer bulb.
        T_D86, n_air = solve_stage3_cstr(
            fuel_obj, D2_avg, T2_avg, vcomp2_norm,
            n_air, T_D86, sim_params["C_glass"], dt=dt,
        )

        # ── Accept state ────────────────────────────────────────────────
        N = N_new
        W_moles  = np.sum(N)
        R2_state = 0.5 * (R2_k1 + R2_k2)
        T2_state = 0.5 * (T2_k1 + T2_k2)

        # Pot volume update
        Xi = N / W_moles if W_moles > 0 else np.zeros_like(N)
        Yi_pot = fuel_obj.X2Y(Xi)
        if use_srk:
            rho_pot = fuel_obj.density_srk(T1_k2, sim_params["P_atm"], Xi)
        else:
            rho_pot = fuel_obj.mixture_density(Yi_pot, T1_k2)
        MW_avg_pot = float(np.dot(Xi, fuel_obj.MW))
        V_pot_mL = (W_moles * MW_avg_pot / rho_pot) * 1e6

        # ── Distillate volume (from condenser condensate) ───────────────
        moles_distilled_step = Dliq_avg * dt
        MW_avg_dist = float(np.dot(ccomp_norm, fuel_obj.MW))
        T_ref = sim_params.get("T_room", 298.15)
        if use_srk:
            rho_ref = fuel_obj.density_srk(T_ref, sim_params["P_atm"], ccomp_norm)
        else:
            rho_ref = fuel_obj.mixture_density(
                fuel_obj.X2Y(ccomp_norm), T_ref,
            )
        dV_mL = moles_distilled_step * MW_avg_dist / rho_ref * 1e6
        distillate_vol_collected += dV_mL

        # ── PI Controller ───────────────────────────────────────────────
        current_rate_ml_min = (dV_mL / dt) * 60.0
        error = _target_rate - current_rate_ml_min
        delta_Q = _Kp * (error - _prev_error) + _Ki * error * dt
        _Q1 = max(_Q_min, min(_Q_max, _Q1 + delta_Q))
        sim_params["Q1"] = _Q1
        _prev_error = error

        time += dt

        if int(time) % 10 == 0:
            results["time"].append(time)
            results["distillate_vol"].append(distillate_vol_collected)
            results["T_D86"].append(T_D86)
            results["T_D86_degC"].append(K2C(T_D86))
            print(
                f"Time: {time:.0f} s | Distilled: {distillate_vol_collected:.1f} mL"
                f" | T_D86: {T_D86:.1f} K | T_cond: {Tc_avg:.1f} K"
                f" | Q1: {_Q1:.1f} W"
            )

    print("Simulation finished (RK2 + condenser).")
    return results
