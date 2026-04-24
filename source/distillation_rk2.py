import numpy as np
from scipy.optimize import bisect
import sys
import os

# Allow importing FuelLib and distillation routines from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from FuelLib import fuel
from distillation import (
    calculate_heat_of_vaporization,
    calculate_liquid_heat_capacity,
    calculate_vapor_heat_capacity,
    compute_h_coeff,
    solve_stage1_bubble_point,
    solve_stage1_energy_balance,
    solve_stage2_flash,
    solve_stage3_cstr,
    K2C
)

# ---------------------------------------------------------------------------
# RK2 DERIVATIVE FUNCTION
# ---------------------------------------------------------------------------

def _compute_derivatives(
    N: np.ndarray,
    T2_prev: float,
    R2_prev: float,
    Q1: float,
    fuel_obj: fuel,
    sim_params: dict,
    L: float,
) -> tuple:
    """
    Evaluate the right-hand side (dN/dt) of the component-mole ODE.
    """
    use_srk = sim_params.get("use_srk", False)
    P       = sim_params["P_atm"]
    T_lo    = sim_params.get("T_bubble_lo", 350.0)
    T_hi    = sim_params.get("T_bubble_hi", 650.0)

    W_tot = np.sum(N)
    if W_tot <= 0:
        zeros = np.zeros(fuel_obj.num_compounds)
        return zeros, sim_params["T_room"], T2_prev, 0.0, 0.0, zeros, zeros

    # Mole fractions from current mole vector
    Xi = N / W_tot

    # --- Routine A ---
    T1, vapor_comp1 = solve_stage1_bubble_point(
        fuel_obj, P, Xi, T_lo=T_lo, T_hi=T_hi, use_srk=use_srk
    )
    D1 = solve_stage1_energy_balance(
        fuel_obj, T1, Xi, Q1, R2_prev, T2_prev,
        use_srk=use_srk, P_atm=P
    )

    # Update h_coeff from Stage-2 wall temperature
    h_mult = sim_params.get("h_multiplier", 1.0)
    h_coeff = compute_h_coeff(T2_prev, sim_params["T_room"], L) * h_mult

    # --- Routine B ---
    T2, R2, reflux_comp, D2, vapor_comp2 = solve_stage2_flash(
        fuel_obj,
        D1, T1, vapor_comp1,
        h_coeff, sim_params["A_area"],
        sim_params["T_room"],
        P=P, use_srk=use_srk,
    )

    # dN/dt for each component in the pot
    dN_dt = R2 * reflux_comp - D1 * vapor_comp1

    return dN_dt, T1, T2, R2, D2, vapor_comp2, reflux_comp


# ---------------------------------------------------------------------------
# MAIN SIMULATION DRIVER  —  RK2 (Heun's predictor-corrector)
# ---------------------------------------------------------------------------

def run_d86_simulation_rk2(
    fuel_obj: fuel,
    Xi_initial: np.ndarray,
    volume_initial_mL: float = 100.0,
    sim_params: dict = None,
) -> dict:
    """
    D86 distillation simulation using Runge-Kutta 2nd-order (Heun's method).
    """
    if sim_params is None:
        raise ValueError("sim_params dictionary is required")

    use_srk = sim_params.get("use_srk", False)

    # --- Initialisation ---
    time                     = 0.0
    distillate_vol_collected = 0.0

    T_init     = sim_params.get("T_room", 298.15)
    Yi_initial = fuel_obj.X2Y(Xi_initial)

    if use_srk:
        rho_init = fuel_obj.density_srk(T_init, sim_params.get("P_atm", 101325.0), Xi_initial)
    else:
        # Note: Mixture density might be defined in FuelLib or calculated here
        # Using same logic as distillation.py
        rho_init = fuel_obj.mixture_density(Yi_initial, T_init)

    mass_init_kg = volume_initial_mL * 1e-6 * rho_init
    MW_avg_init  = float(np.dot(Xi_initial, fuel_obj.MW))
    W_moles      = mass_init_kg / MW_avg_init

    # State as component moles vector
    N = Xi_initial.copy() * W_moles

    # Stage internal states
    R2_state  = 0.0
    T2_state  = sim_params["T_room"]
    n_air     = sim_params["initial_moles_air"]
    T_D86     = sim_params["T_room"]

    results = {"time": [], "distillate_vol": [], "T_D86": [], "T_D86_degC": []}

    # Geometry
    D_out = 0.025
    D_in  = 0.0175
    L     = 0.13
    sim_params["A_area"] = np.pi * (D_out + D_in) * 0.5 * L
    h_mult = sim_params.get("h_multiplier", 1.0)
    sim_params["h_coeff"] = compute_h_coeff(sim_params["T_room"], sim_params["T_room"], L) * h_mult

    # PI controller state
    _target_rate    = sim_params.get("target_rate_ml_min", 4.5)
    _Kp             = sim_params.get("controller_Kp", 2.0)
    _Ki             = sim_params.get("controller_Ki", 0.05)
    _Q_min          = sim_params.get("Q1_min", 0.0)
    _Q_max          = sim_params.get("Q1_max", 1500.0)
    _prev_error     = 0.0
    _Q1             = sim_params.get("Q1", 15.0)

    V_pot_mL = volume_initial_mL
    dt       = sim_params["dt"]

    print("Starting D86 simulation (RK2 / Heun's method)...")

    while V_pot_mL > sim_params.get("min_volume_W_mL", 1.0) and np.sum(N) > 0:

        # Stage 1: k1
        k1, T1_k1, T2_k1, R2_k1, D2_k1, vcomp2_k1, rcomp_k1 = _compute_derivatives(
            N, T2_state, R2_state, _Q1, fuel_obj, sim_params, L
        )

        # Predictor
        N_pred = N + dt * k1
        N_pred = np.maximum(N_pred, 0.0)

        # Stage 2: k2
        # Use Stage 1 outputs as starting point for Stage 2 derivative eval
        k2, T1_k2, T2_k2, R2_k2, D2_k2, vcomp2_k2, rcomp_k2 = _compute_derivatives(
            N_pred, T2_k1, R2_k1, _Q1, fuel_obj, sim_params, L
        )

        # Corrector
        N_new = N + (dt / 2.0) * (k1 + k2)
        N_new = np.maximum(N_new, 0.0)

        # Averaged values for CSTR and bookkeeping
        D2_avg = 0.5 * (D2_k1 + D2_k2)
        T2_avg = 0.5 * (T2_k1 + T2_k2)
        vcomp2_avg = 0.5 * (vcomp2_k1 + vcomp2_k2)
        vcomp2_norm = vcomp2_avg / (np.sum(vcomp2_avg) + 1e-15)

        # Update CSTR (Stage 3)
        T_D86, n_air = solve_stage3_cstr(
            fuel_obj, D2_avg, T2_avg, vcomp2_norm,
            n_air, T_D86, sim_params["C_glass"], dt=dt
        )

        # Final Accept State
        N = N_new
        W_moles = np.sum(N)
        R2_state = 0.5 * (R2_k1 + R2_k2)
        T2_state = T2_avg

        # Volume update logic
        Xi = N / W_moles if W_moles > 0 else np.zeros_like(N)
        Yi_pot = fuel_obj.X2Y(Xi)
        if use_srk:
            rho_pot = fuel_obj.density_srk(T1_k2, sim_params["P_atm"], Xi)
        else:
            rho_pot = fuel_obj.mixture_density(Yi_pot, T1_k2)
        MW_avg_pot = float(np.dot(Xi, fuel_obj.MW))
        V_pot_mL = (W_moles * MW_avg_pot / rho_pot) * 1e6

        # Distillate accumulation
        moles_distilled_step = D2_avg * dt
        MW_avg_dist = float(np.dot(vcomp2_norm, fuel_obj.MW))
        T_ref = sim_params.get("T_room", 298.15)
        if use_srk:
            rho_ref = fuel_obj.density_srk(T_ref, sim_params["P_atm"], vcomp2_norm)
        else:
            rho_ref = fuel_obj.mixture_density(fuel_obj.X2Y(vcomp2_norm), T_ref)
        dV_mL = moles_distilled_step * MW_avg_dist / rho_ref * 1e6
        distillate_vol_collected += dV_mL

        # PI Controller
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
            print(f"Time: {time:.0f} s | Distilled: {distillate_vol_collected:.1f} mL"
                  f" | T_D86: {T_D86:.1f} K | Q1: {_Q1:.1f} W")

    print("Simulation finished (RK2).")
    return results
