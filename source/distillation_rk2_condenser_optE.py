"""
D86 distillation simulation — Option E controller.

Option E: Inverse-model feed-forward + small proportional trim.

  Q1_ff = m_dot_target * Lv_mix(T1, Xi)     # physics-based feed-forward
  Q1    = clip(Q1_ff + Kp_trim * (target - rate), Q_min, Q_max)

``m_dot_target`` is derived from the target volumetric rate via the liquid
density at pot temperature:

  m_dot_target = rho_liq(T1, Xi) * target_rate_mL_min / (60 * 1e6)

``Lv_mix`` is the mass-fraction-weighted latent heat of vaporization:

  Lv_mix = sum_i(Y_i * Lv_i(T1))            [J/kg]

Because the feed-forward handles most of the required power, Kp_trim can be
small, the closed-loop trim rarely saturates, and the controller is effectively
a single-line stateless equation — making it friendly to JAX automatic
differentiation without any special treatment.

Controller sim_params keys
--------------------------
  controller_Kp_trim   : float  — proportional gain for rate trim W/(mL/min)
                                  (default 10.0)
  rate_ema_alpha        : float  — EMA smoothing coefficient for the rate
                                  signal fed to the trim (default 0.15);
                                  introduces one carry variable (rate_ema)
                                  which is compatible with jax.lax.scan
  Q1_min               : float  — minimum heater power W (default 0.0)
  Q1_max               : float  — maximum heater power W (default 1500.0)
  target_rate_ml_min   : float  — target distillate rate mL/min (default 4.5)

Condenser sim_params keys (required)
-------------------------------------
  UA_cond  : float  — condenser UA product W/K
  T_bath   : float  — cooling-bath temperature K

The results dict returned includes two extra arrays (recorded every
``record_every_n_steps`` steps):
  rate_ml_min   : instantaneous distillate rate at each recorded step
  Q1_history    : heater power at each recorded step
  Q1_ff_history : feed-forward component at each recorded step
"""

import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from FuelLib import fuel, K2C
from distillation import (
    calculate_vapor_heat_capacity,
    calculate_heat_of_vaporization,
    calculate_liquid_heat_capacity,
    compute_h_coeff,
    solve_stage1_bubble_point,
    solve_stage1_energy_balance,
    solve_stage2_flash,
    solve_rachford_rice,
    solve_stage3_cstr,
)
from distillation_rk2_condenser import solve_condenser, _compute_derivatives_condenser

# ---------------------------------------------------------------------------
# MAIN SIMULATION DRIVER  —  Option E controller
# ---------------------------------------------------------------------------


def run_d86_simulation_rk2_condenser_optE(
    fuel_obj: fuel,
    Xi_initial: np.ndarray,
    volume_initial_mL: float = 100.0,
    sim_params: dict = None,
) -> dict:
    """
    D86 distillation simulation using RK2 (Heun) + lumped condenser, with
    the Option E inverse-model feed-forward controller.

    Controller equation (stateless)::

        Q1_ff = m_dot_target * Lv_mix(T1, Xi)
        Q1    = clip(Q1_ff + Kp_trim*(target_rate - rate), Q_min, Q_max)

    Parameters
    ----------
    fuel_obj : fuel
        Initialised FuelLib fuel object.
    Xi_initial : ndarray
        Initial pot mole fractions.
    volume_initial_mL : float
        Initial pot volume in mL.
    sim_params : dict
        Simulation parameters. Must include ``UA_cond`` and ``T_bath``.

    Returns
    -------
    dict with keys: time, distillate_vol, T_D86, T_D86_degC,
                    rate_ml_min, Q1_history, Q1_ff_history.
    """
    if sim_params is None:
        raise ValueError("sim_params dictionary is required")

    sp = dict(sim_params)

    use_srk = sp.get("use_srk", False)
    verbose = sp.get("verbose", True)
    record_every_n = int(sp.get("record_every_n_steps", 10))

    # ── Initialisation ────────────────────────────────────────────────────
    time = 0.0
    distillate_vol_collected = 0.0

    T_init = sp.get("T_room", 298.15)
    Yi_initial = fuel_obj.X2Y(Xi_initial)

    if use_srk:
        rho_init = fuel_obj.density_srk(T_init, sp.get("P_atm", 101325.0), Xi_initial)
    else:
        rho_init = fuel_obj.mixture_density(Yi_initial, T_init)

    mass_init_kg = volume_initial_mL * 1e-6 * rho_init
    MW_avg_init = float(np.dot(Xi_initial, fuel_obj.MW))
    W_moles = mass_init_kg / MW_avg_init

    N = Xi_initial.copy() * W_moles
    R2_state = 0.0
    T2_state = sp["T_room"]
    n_air = sp["initial_moles_air"]
    T_D86 = sp["T_room"]

    results = {
        "time": [],
        "distillate_vol": [],
        "T_D86": [],
        "T_D86_degC": [],
        "rate_ml_min": [],
        "Q1_history": [],
        "Q1_ff_history": [],
    }

    D_out = sp.get("D_out", 0.025)
    D_in = sp.get("D_in", 0.0175)
    L = sp.get("column_length", 0.13)
    sp["A_area"] = np.pi * (D_out + D_in) * 0.5 * L
    h_mult = sp.get("h_multiplier", 1.0)
    sp["h_coeff"] = compute_h_coeff(sp["T_room"], sp["T_room"], L) * h_mult

    # Option E controller parameters
    _target_rate = sp.get("target_rate_ml_min", 4.5)
    _Kp_trim = float(sp.get("controller_Kp_trim", 10.0))
    _Q_min = sp.get("Q1_min", 0.0)
    _Q_max = sp.get("Q1_max", 1500.0)
    _Q1 = float(sp.get("Q1", 15.0))
    _rate_ema_alpha = float(sp.get("rate_ema_alpha", 0.15))
    _rate_ema = 0.0  # EMA carry state (one variable; compatible with lax.scan)

    # Stall detection (identical to condenser base driver)
    _stall_window_s = float(sp.get("stall_window_s", 300.0))
    _stall_vol_tol_mL = float(sp.get("stall_vol_tol_mL", 1.0e-4))
    _startup_stall_window_s = float(sp.get("startup_stall_window_s", 900.0))
    _min_forward_vapor_mol_s = float(sp.get("min_forward_vapor_mol_s", 1.0e-9))
    _stall_ref_time = 0.0
    _stall_ref_volume = 0.0

    V_pot_mL = volume_initial_mL
    dt = sp["dt"]
    step_counter = 0

    if verbose:
        print("Starting D86 simulation (RK2 + condenser, Option E controller)...")

    while V_pot_mL > sp.get("min_volume_W_mL", 1.0) and np.sum(N) > 0:

        # ── k1 ─────────────────────────────────────────────────────────────
        (
            k1,
            T1_k1,
            T2_k1,
            R2_k1,
            D2_k1,
            vcomp2_k1,
            Tc_k1,
            Dtot_k1,
            Dliq_k1,
            ccomp_k1,
            rcomp_k1,
        ) = _compute_derivatives_condenser(
            N,
            T2_state,
            R2_state,
            _Q1,
            fuel_obj,
            sp,
            L,
        )

        # Predictor
        N_pred = np.maximum(N + dt * k1, 0.0)

        # ── k2 ─────────────────────────────────────────────────────────────
        (
            k2,
            T1_k2,
            T2_k2,
            R2_k2,
            D2_k2,
            vcomp2_k2,
            Tc_k2,
            Dtot_k2,
            Dliq_k2,
            ccomp_k2,
            rcomp_k2,
        ) = _compute_derivatives_condenser(
            N_pred,
            T2_k1,
            R2_k1,
            _Q1,
            fuel_obj,
            sp,
            L,
        )

        # Corrector
        N_new = np.maximum(N + (dt / 2.0) * (k1 + k2), 0.0)

        # ── Averaged quantities ─────────────────────────────────────────────
        D2_avg = 0.5 * (D2_k1 + D2_k2)
        T2_avg = 0.5 * (T2_k1 + T2_k2)
        Dtot_avg = 0.5 * (Dtot_k1 + Dtot_k2)
        Dliq_avg = 0.5 * (Dliq_k1 + Dliq_k2)
        Tc_avg = 0.5 * (Tc_k1 + Tc_k2)

        vcomp2_avg = 0.5 * (vcomp2_k1 + vcomp2_k2)
        vcomp2_norm = vcomp2_avg / (np.sum(vcomp2_avg) + 1e-15)

        ccomp_avg = 0.5 * (ccomp_k1 + ccomp_k2)
        ccomp_norm = ccomp_avg / (np.sum(ccomp_avg) + 1e-15)

        # ── Stage 3: CSTR / thermocouple ────────────────────────────────────
        T_D86, n_air = solve_stage3_cstr(
            fuel_obj,
            D2_avg,
            T2_avg,
            vcomp2_norm,
            n_air,
            T_D86,
            sp["C_glass"],
            dt=dt,
        )

        # ── Accept state ────────────────────────────────────────────────────
        N = N_new
        W_moles = np.sum(N)
        R2_state = 0.5 * (R2_k1 + R2_k2)
        T2_state = 0.5 * (T2_k1 + T2_k2)

        Xi = N / W_moles if W_moles > 0 else np.zeros_like(N)
        Yi_pot = fuel_obj.X2Y(Xi)
        if use_srk:
            rho_pot = fuel_obj.density_srk(T1_k2, sp["P_atm"], Xi)
        else:
            rho_pot = fuel_obj.mixture_density(Yi_pot, T1_k2)
        MW_avg_pot = float(np.dot(Xi, fuel_obj.MW))
        V_pot_mL = (W_moles * MW_avg_pot / rho_pot) * 1e6 if rho_pot > 0 else 0.0

        # ── Distillate volume ────────────────────────────────────────────────
        moles_distilled_step = Dliq_avg * dt
        MW_avg_dist = float(np.dot(ccomp_norm, fuel_obj.MW))
        T_ref = sp.get("T_room", 298.15)
        if use_srk:
            rho_ref = fuel_obj.density_srk(T_ref, sp["P_atm"], ccomp_norm)
        else:
            rho_ref = fuel_obj.mixture_density(fuel_obj.X2Y(ccomp_norm), T_ref)
        dV_mL = (
            moles_distilled_step * MW_avg_dist / rho_ref * 1e6 if rho_ref > 0 else 0.0
        )
        distillate_vol_collected += dV_mL

        # ── Option E controller ──────────────────────────────────────────────
        # Full energy-balance feed-forward, inverting the stage-1 relation:
        #
        #   Q1 = D1 * dHv_molar + R2 * Cp_l * (T1 - T2)
        #
        # where D1 = D_distillate_target + R2_prev (reflux must be re-vaporised).
        # R2_state and T2_state carry the previous step's reflux/neck values.

        T1_ff = T1_k2  # best estimate of next-step pot temperature

        # Target distillate molar flow (mol/s)
        # Distillate is measured at room temperature, so use room-temp density.
        target_vol_rate_m3s = _target_rate / (60.0 * 1e6)
        Yi_ff = Yi_pot  # mass fractions at updated state
        T_ref_ff = sp.get("T_room", 298.15)
        if use_srk:
            rho_ref_ff = fuel_obj.density_srk(T_ref_ff, sp["P_atm"], Xi)
        else:
            rho_ref_ff = fuel_obj.mixture_density(Yi_ff, T_ref_ff)
        m_dot_distillate = rho_ref_ff * target_vol_rate_m3s  # kg/s
        MW_mix_ff = float(np.dot(Xi, fuel_obj.MW))  # kg/mol
        D_distillate_target = m_dot_distillate / max(MW_mix_ff, 1e-15)  # mol/s

        # Total vapor target from pot = distillate target + reflux that must
        # be re-vaporised each step.
        D1_target = D_distillate_target + R2_state

        # Thermodynamic properties at pot temperature
        dHv_molar = calculate_heat_of_vaporization(fuel_obj, T1_ff, Xi)  # J/mol
        Cp_l = calculate_liquid_heat_capacity(
            fuel_obj, T1_ff, Xi, use_srk=use_srk, P_atm=sp["P_atm"]
        )  # J/mol/K
        dHv_molar = max(dHv_molar, 1.0)

        # Feed-forward: inversion of stage-1 energy balance
        Q1_ff = D1_target * dHv_molar + R2_state * Cp_l * (T1_ff - T2_state)
        Q1_ff = max(Q1_ff, 0.0)

        # EMA-smoothed rate for the trim (reduces noise-driven saturation);
        # the EMA alpha is kept low (0.15) for stability across the boil curve.
        current_rate = (dV_mL / dt) * 60.0
        _rate_ema = _rate_ema_alpha * current_rate + (1.0 - _rate_ema_alpha) * _rate_ema
        Q1_trim = _Kp_trim * (_target_rate - _rate_ema)

        _Q1 = float(np.clip(Q1_ff + Q1_trim, _Q_min, _Q_max))

        time += dt
        step_counter += 1

        # ── Stall detection ──────────────────────────────────────────────────
        if (
            distillate_vol_collected <= _stall_vol_tol_mL
            and D2_avg <= _min_forward_vapor_mol_s
        ):
            if (time - _stall_ref_time) >= _startup_stall_window_s:
                if verbose:
                    print(
                        f"Simulation stalled before first drop: no forward vapour "
                        f"(D2≈0) for {_startup_stall_window_s:.0f} s at "
                        f"T_D86={T_D86:.1f} K, Q1={_Q1:.1f} W. Terminating early."
                    )
                break
        elif distillate_vol_collected <= _stall_vol_tol_mL:
            _stall_ref_time = time
        elif distillate_vol_collected - _stall_ref_volume > _stall_vol_tol_mL:
            _stall_ref_volume = distillate_vol_collected
            _stall_ref_time = time
        elif (time - _stall_ref_time) >= _stall_window_s:
            if verbose:
                print(
                    f"Simulation stalled: distillate volume unchanged "
                    f"(< {_stall_vol_tol_mL:g} mL) for {_stall_window_s:.0f} s "
                    f"at V={distillate_vol_collected:.2f} mL, T_D86="
                    f"{T_D86:.1f} K, Q1={_Q1:.1f} W. Terminating early."
                )
            break

        if step_counter % record_every_n == 0:
            results["time"].append(time)
            results["distillate_vol"].append(distillate_vol_collected)
            results["T_D86"].append(T_D86)
            results["T_D86_degC"].append(K2C(T_D86))
            results["rate_ml_min"].append(current_rate)
            results["Q1_history"].append(_Q1)
            results["Q1_ff_history"].append(Q1_ff)
            if verbose:
                print(
                    f"Time: {time:.1f} s | Distilled: {distillate_vol_collected:.1f} mL"
                    f" | T_D86: {T_D86:.1f} K | T_cond: {Tc_avg:.1f} K"
                    f" | Q1: {_Q1:.1f} W | Q1_ff: {Q1_ff:.1f} W"
                )

    if verbose:
        print("Simulation finished (RK2 + condenser, Option E).")
    return results
