"""
Bayesian Optimisation (Optuna) of the D86 distillation simulation
using the RK2 integrator WITH the lumped condenser model.

This driver now supports:
- per-fuel/per-volume diagnostics and residual cut metrics,
- fuel-spec audits for the calibrated POSF fuels,
- fairness-aware objective functions,
- multi-seed optimisation with trial export,
- acceptance-criteria-based model selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


# Add FuelLib source to Python path (repo-root-relative, portable)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FUELLIB_DIR = os.path.dirname(_SCRIPT_DIR)
if _FUELLIB_DIR not in sys.path:
    sys.path.insert(0, _FUELLIB_DIR)

from paths import EXP_D86_FILE, FUELDATA_DECOMP_DIR, FUELDATA_GC_DIR
from source.FuelLib import fuel
from source.distillation_rk2_condenser import run_d86_simulation_rk2_condenser


FUELS = ["posf10325", "posf10289", "posf10264"]
CSV_PATH = EXP_D86_FILE
CUTFRACTIONS = {
    "front": (0.0, 30.0),
    "mid": (30.0, 80.0),
    "tail": (80.0, 100.0),
}


def load_experimental_targets() -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
    """Load NJFCP D86 data and return full target grid in Kelvin."""
    df = pd.read_csv(CSV_PATH)
    target_volumes = df["VolumePercentage"].to_numpy(dtype=float)

    targets = {}
    for f_name in FUELS:
        targets[f_name] = df[f_name].to_numpy(dtype=float) + 273.15

    return target_volumes, targets, df


TARGET_VOLUMES, EXP_TARGETS, EXP_DF = load_experimental_targets()


def parse_seeds(seed_csv: str) -> List[int]:
    return [int(token.strip()) for token in seed_csv.split(",") if token.strip()]


def parse_fuel_weights(weight_csv: str | None) -> Dict[str, float]:
    if weight_csv is None:
        return {f: 1.0 for f in FUELS}
    vals = [float(v.strip()) for v in weight_csv.split(",") if v.strip()]
    if len(vals) != len(FUELS):
        raise ValueError(
            f"--fuel-weights must provide {len(FUELS)} values ordered as {FUELS}; got {len(vals)}"
        )
    return {f: v for f, v in zip(FUELS, vals)}


def _sim_params_from_trial(trial: optuna.Trial) -> dict:
    c_glass = trial.suggest_float("C_glass", 0.1, 5.0)
    h_mult = trial.suggest_float("h_multiplier", 0.5, 5.0)
    q1 = trial.suggest_float("initial_Q1", 10.0, 100.0)
    ua_cond = trial.suggest_float("UA_cond", 2.0, 50.0)
    t_bath = trial.suggest_float("T_bath", 268.0, 283.0)
    t_bub_hi = trial.suggest_float("T_bubble_hi", 650.0, 800.0)

    return {
        "dt": 1,
        "initial_moles_air": 0.008,
        "C_glass": c_glass,
        "h_multiplier": h_mult,
        "T_bubble_lo": 375.0,
        "T_bubble_hi": t_bub_hi,
        "P_atm": 101325.0,
        "T_room": 298.15,
        "Q1": q1,
        "target_rate_ml_min": 4.5,
        "controller_Kp": 5.0,
        "controller_Ki": 0.05,
        "min_volume_W_mL": 0.5,
        "use_srk": True,
        "UA_cond": ua_cond,
        "T_bath": t_bath,
        "verbose": False,
        "record_every_n_steps": 10,
    }


def _sim_params_from_best(best_params: dict) -> dict:
    return {
        "dt": 1.0,
        "initial_moles_air": 0.008,
        "C_glass": best_params["C_glass"],
        "h_multiplier": best_params["h_multiplier"],
        "T_bubble_lo": 375.0,
        "T_bubble_hi": best_params.get("T_bubble_hi", 650.0),
        "P_atm": 101325.0,
        "T_room": 298.15,
        "Q1": best_params["initial_Q1"],
        "target_rate_ml_min": 4.5,
        "controller_Kp": 5.0,
        "controller_Ki": 0.05,
        "min_volume_W_mL": 0.5,
        "use_srk": True,
        "UA_cond": best_params["UA_cond"],
        "T_bath": best_params["T_bath"],
        "verbose": False,
        "record_every_n_steps": 10,
    }


def simulate_fuel(fname: str, sim_params: dict) -> dict:
    f_obj = fuel(fname)
    Xi = f_obj.Y2X(f_obj.Y_0)
    results = run_d86_simulation_rk2_condenser(
        f_obj,
        Xi,
        volume_initial_mL=100.0,
        sim_params=sim_params,
    )

    sim_vol = np.asarray(results["distillate_vol"], dtype=float)
    sim_T = np.asarray(results["T_D86"], dtype=float)
    if sim_vol.size < 3 or sim_vol[-1] < 85.0:
        raise RuntimeError(f"{fname} insufficient distilled volume: {sim_vol[-1] if sim_vol.size else 0.0:.2f} mL")

    _, uniq = np.unique(sim_vol, return_index=True)
    sim_vol = sim_vol[uniq]
    sim_T = sim_T[uniq]

    sim_interp = interp1d(sim_vol, sim_T, kind="linear", fill_value="extrapolate")
    sim_target_T = np.asarray(sim_interp(TARGET_VOLUMES), dtype=float)
    exp_target_T = EXP_TARGETS[fname]
    residuals = sim_target_T - exp_target_T

    mae = float(np.mean(np.abs(residuals)))
    max_abs = float(np.max(np.abs(residuals)))
    signed_mean = float(np.mean(residuals))
    temp_span = float(np.max(exp_target_T) - np.min(exp_target_T))
    norm_mae = mae / max(temp_span, 1e-9)

    return {
        "fuel": fname,
        "sim_vol": sim_vol,
        "sim_T": sim_T,
        "sim_target_T": sim_target_T,
        "exp_target_T": exp_target_T,
        "residuals": residuals,
        "mae": mae,
        "max_abs": max_abs,
        "signed_mean": signed_mean,
        "norm_mae": norm_mae,
    }


def summarize_cut_metrics(vols: np.ndarray, residuals: np.ndarray) -> dict:
    out = {}
    for cut, (lo, hi) in CUTFRACTIONS.items():
        mask = (vols >= lo) & (vols <= hi)
        if not np.any(mask):
            out[f"{cut}_mae"] = np.nan
            out[f"{cut}_bias"] = np.nan
            out[f"{cut}_max_abs"] = np.nan
            continue
        rr = residuals[mask]
        out[f"{cut}_mae"] = float(np.mean(np.abs(rr)))
        out[f"{cut}_bias"] = float(np.mean(rr))
        out[f"{cut}_max_abs"] = float(np.max(np.abs(rr)))
    return out


def run_all_fuels(sim_params: dict) -> Dict[str, dict]:
    return {fname: simulate_fuel(fname, sim_params) for fname in FUELS}


def score_simulations(
    fuel_results: Dict[str, dict],
    objective_mode: str,
    fuel_weights: Dict[str, float],
    worst_fuel_weight: float,
) -> float:
    maes = np.asarray([fuel_results[f]["mae"] for f in FUELS], dtype=float)
    norm_maes = np.asarray([fuel_results[f]["norm_mae"] for f in FUELS], dtype=float)
    weights = np.asarray([fuel_weights[f] for f in FUELS], dtype=float)

    if objective_mode == "legacy":
        return float(np.mean(maes))
    if objective_mode == "weighted":
        return float(np.sum(weights * maes) / np.sum(weights))
    if objective_mode == "fair":
        weighted_norm = float(np.sum(weights * norm_maes) / np.sum(weights))
        return weighted_norm + worst_fuel_weight * float(np.max(norm_maes))
    raise ValueError(f"Unsupported objective mode: {objective_mode}")


def make_objective(
    objective_mode: str,
    fuel_weights: Dict[str, float],
    worst_fuel_weight: float,
):
    def objective(trial: optuna.Trial) -> float:
        sim_params = _sim_params_from_trial(trial)
        try:
            fuel_results = run_all_fuels(sim_params)
        except Exception:
            return 1000.0

        score = score_simulations(
            fuel_results,
            objective_mode=objective_mode,
            fuel_weights=fuel_weights,
            worst_fuel_weight=worst_fuel_weight,
        )

        for fname in FUELS:
            trial.set_user_attr(f"mae_{fname}", fuel_results[fname]["mae"])
            trial.set_user_attr(f"norm_mae_{fname}", fuel_results[fname]["norm_mae"])
            trial.set_user_attr(f"maxabs_{fname}", fuel_results[fname]["max_abs"])
            trial.set_user_attr(f"bias_{fname}", fuel_results[fname]["signed_mean"])
        trial.set_user_attr("objective_mode", objective_mode)
        return float(score)

    return objective


def run_optuna_study(
    n_trials: int,
    seed: int,
    objective_mode: str,
    fuel_weights: Dict[str, float],
    worst_fuel_weight: float,
) -> optuna.study.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_objective(
        objective_mode=objective_mode,
        fuel_weights=fuel_weights,
        worst_fuel_weight=worst_fuel_weight,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def evaluate_params(best_params: dict) -> Dict[str, dict]:
    sim_params = _sim_params_from_best(best_params)
    return run_all_fuels(sim_params)


def build_trial_metrics_row(
    seed: int,
    trial_number: int,
    objective_mode: str,
    score: float,
    best_params: dict,
    fuel_results: Dict[str, dict],
) -> dict:
    row = {
        "seed": seed,
        "trial": trial_number,
        "objective_mode": objective_mode,
        "score": score,
    }
    row.update({f"param_{k}": v for k, v in best_params.items()})

    for fname in FUELS:
        r = fuel_results[fname]
        row[f"mae_{fname}"] = r["mae"]
        row[f"maxabs_{fname}"] = r["max_abs"]
        row[f"bias_{fname}"] = r["signed_mean"]
        row[f"norm_mae_{fname}"] = r["norm_mae"]
        row.update({f"{k}_{fname}": v for k, v in summarize_cut_metrics(TARGET_VOLUMES, r["residuals"]).items()})
    row["mae_mean_all"] = float(np.mean([fuel_results[f]["mae"] for f in FUELS]))
    row["mae_other_mean"] = float(np.mean([fuel_results[f]["mae"] for f in FUELS if f != "posf10289"]))
    row["mae_worst"] = float(np.max([fuel_results[f]["mae"] for f in FUELS]))
    return row


def select_candidate_with_acceptance(
    candidate_df: pd.DataFrame,
    legacy_ref: dict | None,
    max_deg_10325: float,
    max_deg_10264: float,
    min_improve_10289: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    if legacy_ref is None:
        return candidate_df.sort_values("score").iloc[0], candidate_df

    feasible = candidate_df[
        (candidate_df["mae_posf10325"] <= legacy_ref["mae_posf10325"] + max_deg_10325)
        & (candidate_df["mae_posf10264"] <= legacy_ref["mae_posf10264"] + max_deg_10264)
        & (candidate_df["mae_posf10289"] <= legacy_ref["mae_posf10289"] - min_improve_10289)
    ]

    if feasible.empty:
        return candidate_df.sort_values("score").iloc[0], feasible
    return feasible.sort_values("score").iloc[0], feasible


def audit_fuel_specifications() -> pd.DataFrame:
    rows = []
    for fname in FUELS:
        gc_path = os.path.join(FUELDATA_GC_DIR, f"{fname}_init.csv")
        decomp_path = os.path.join(FUELDATA_DECOMP_DIR, f"{fname}.csv")

        df_gc = pd.read_csv(gc_path)
        df_decomp = pd.read_csv(decomp_path)

        f_obj = fuel(fname)
        family_mass = {"sat": 0.0, "aromatic": 0.0, "cyclo": 0.0, "olefin": 0.0}
        for frac, fam_id in zip(f_obj.Y_0, f_obj.fam):
            if fam_id == 0:
                family_mass["sat"] += float(frac)
            elif fam_id == 1:
                family_mass["aromatic"] += float(frac)
            elif fam_id == 2:
                family_mass["cyclo"] += float(frac)
            else:
                family_mass["olefin"] += float(frac)

        comp_gc = [c.strip() for c in df_gc["Compound"].tolist()]
        comp_decomp = [c.strip() for c in df_decomp.iloc[:, 0].tolist()]

        row = {
            "fuel": fname,
            "gc_weight_sum": float(df_gc["Weight %"].astype(float).sum()),
            "gc_compound_count": int(len(comp_gc)),
            "decomp_compound_count": int(len(comp_decomp)),
            "compound_lists_match": bool(comp_gc == comp_decomp),
            "missing_in_decomp": len(set(comp_gc) - set(comp_decomp)),
            "missing_in_gc": len(set(comp_decomp) - set(comp_gc)),
            "tc_nonfinite_count": int(np.sum(~np.isfinite(f_obj.Tc))),
            "pc_nonfinite_count": int(np.sum(~np.isfinite(f_obj.Pc))),
            "vc_nonfinite_count": int(np.sum(~np.isfinite(f_obj.Vc))),
            "tc_low_count_lt200K": int(np.sum(f_obj.Tc < 200.0)),
            "pc_low_count_lt10kPa": int(np.sum(f_obj.Pc < 1.0e4)),
            "vc_low_count_lt1e-7": int(np.sum(f_obj.Vc < 1.0e-7)),
            "family_sat_massfrac": family_mass["sat"],
            "family_aromatic_massfrac": family_mass["aromatic"],
            "family_cyclo_massfrac": family_mass["cyclo"],
            "family_olefin_massfrac": family_mass["olefin"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_plots_and_tables(
    best_params: dict,
    best_eval: Dict[str, dict],
    trial_metrics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    selected_row: pd.Series,
    feasible_df: pd.DataFrame,
    legacy_ref: dict | None,
    fuel_audit_df: pd.DataFrame,
) -> None:
    # Comparison + residual figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")

    df_list = [EXP_DF.copy()]
    V_pct_exp = EXP_DF["VolumePercentage"].to_numpy(dtype=float)

    residual_table_rows = []

    for i, fname in enumerate(FUELS):
        r = best_eval[fname]
        T_exp_K = EXP_TARGETS[fname]

        axes[0, i].plot(V_pct_exp, T_exp_K, "ro-", label="Experimental")
        axes[0, i].plot(r["sim_vol"], r["sim_T"], "b-", linewidth=2, label="Simulation")
        axes[0, i].set_title(f"D86 Curve: {fname}")
        axes[0, i].set_ylabel("T_D86 (K)")
        axes[0, i].grid(True)
        axes[0, i].legend()

        axes[1, i].axhline(0.0, color="k", linewidth=1)
        axes[1, i].plot(TARGET_VOLUMES, r["residuals"], "m-o", linewidth=1.5)
        axes[1, i].set_title(f"Residuals: {fname}")
        axes[1, i].set_xlabel("Volume Distilled (%)")
        axes[1, i].set_ylabel("Sim - Exp (K)")
        axes[1, i].grid(True)

        df_list.append(
            pd.DataFrame(
                {
                    f"{fname}_sim_vol": r["sim_vol"],
                    f"{fname}_sim_T": r["sim_T"],
                }
            )
        )

        cut = summarize_cut_metrics(TARGET_VOLUMES, r["residuals"])
        residual_table_rows.append(
            {
                "fuel": fname,
                "mae_K": r["mae"],
                "max_abs_K": r["max_abs"],
                "mean_bias_K": r["signed_mean"],
                **cut,
            }
        )

    plt.tight_layout()
    plot_path = os.path.join(_SCRIPT_DIR, "d86_condenser_comparison_and_residuals.png")
    plt.savefig(plot_path)

    csv_path = os.path.join(_SCRIPT_DIR, "optimized_distillation_curves_condenser.csv")
    pd.concat(df_list, axis=1).to_csv(csv_path, index=False)

    residual_metrics_path = os.path.join(_SCRIPT_DIR, "d86_condenser_residual_metrics.csv")
    pd.DataFrame(residual_table_rows).to_csv(residual_metrics_path, index=False)

    # tradeoff export / plot
    trial_metrics_path = os.path.join(_SCRIPT_DIR, "d86_condenser_trial_metrics.csv")
    trial_metrics_df.to_csv(trial_metrics_path, index=False)

    candidates_path = os.path.join(_SCRIPT_DIR, "d86_condenser_candidate_metrics.csv")
    candidate_df.to_csv(candidates_path, index=False)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    ax2.scatter(candidate_df["mae_posf10289"], candidate_df["mae_other_mean"], c="gray", alpha=0.6, label="Candidates")
    if not feasible_df.empty:
        ax2.scatter(feasible_df["mae_posf10289"], feasible_df["mae_other_mean"], c="green", alpha=0.8, label="Feasible")
    ax2.scatter([selected_row["mae_posf10289"]], [selected_row["mae_other_mean"]], c="red", s=80, label="Selected")
    ax2.set_xlabel("MAE posf10289 (K)")
    ax2.set_ylabel("Mean MAE other fuels (K)")
    ax2.set_title("Pareto-style tradeoff")
    ax2.grid(True)
    ax2.legend()
    tradeoff_plot = os.path.join(_SCRIPT_DIR, "d86_condenser_tradeoff.png")
    plt.tight_layout()
    plt.savefig(tradeoff_plot)

    audit_path = os.path.join(_SCRIPT_DIR, "d86_fuel_spec_audit.csv")
    fuel_audit_df.to_csv(audit_path, index=False)

    summary = {
        "selected_candidate": selected_row.to_dict(),
        "legacy_reference": legacy_ref,
        "objective_volumes": TARGET_VOLUMES.tolist(),
        "fuels": FUELS,
        "plot": plot_path,
        "tradeoff_plot": tradeoff_plot,
        "optimized_curves_csv": csv_path,
        "residual_metrics_csv": residual_metrics_path,
        "trial_metrics_csv": trial_metrics_path,
        "candidate_metrics_csv": candidates_path,
        "fuel_audit_csv": audit_path,
    }

    summary_path = os.path.join(_SCRIPT_DIR, "d86_condenser_optimization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main(args: argparse.Namespace) -> None:
    print("Starting Bayesian Optimisation via Optuna (RK2 + condenser)...")
    print(f"Target experimental points: {TARGET_VOLUMES.tolist()} over {FUELS}")

    seeds = parse_seeds(args.seeds)
    fuel_weights = parse_fuel_weights(args.fuel_weights)

    # Legacy reference used only for acceptance criteria (single seed for speed).
    legacy_ref = None
    if args.run_legacy_reference:
        legacy_study = run_optuna_study(
            n_trials=args.n_trials,
            seed=seeds[0],
            objective_mode="legacy",
            fuel_weights=fuel_weights,
            worst_fuel_weight=args.worst_fuel_weight,
        )
        legacy_eval = evaluate_params(legacy_study.best_params)
        legacy_ref = {
            f"mae_{fname}": legacy_eval[fname]["mae"] for fname in FUELS
        }
        legacy_ref["seed"] = seeds[0]
        legacy_ref["score"] = float(legacy_study.best_value)
        print(f"Legacy reference best MAEs: {legacy_ref}")

    # Fairness-aware multi-seed optimization.
    candidate_rows = []
    all_trial_rows = []

    for seed in seeds:
        study = run_optuna_study(
            n_trials=args.n_trials,
            seed=seed,
            objective_mode=args.objective_mode,
            fuel_weights=fuel_weights,
            worst_fuel_weight=args.worst_fuel_weight,
        )

        best_eval = evaluate_params(study.best_params)
        candidate_rows.append(
            build_trial_metrics_row(
                seed=seed,
                trial_number=-1,
                objective_mode=args.objective_mode,
                score=float(study.best_value),
                best_params=study.best_params,
                fuel_results=best_eval,
            )
        )

        # Export all completed trials for tradeoff diagnostics.
        for tr in study.trials:
            if tr.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {
                "seed": seed,
                "trial": tr.number,
                "objective_mode": args.objective_mode,
                "score": float(tr.value),
            }
            row.update({f"param_{k}": v for k, v in tr.params.items()})
            for fname in FUELS:
                row[f"mae_{fname}"] = tr.user_attrs.get(f"mae_{fname}", np.nan)
                row[f"norm_mae_{fname}"] = tr.user_attrs.get(f"norm_mae_{fname}", np.nan)
                row[f"maxabs_{fname}"] = tr.user_attrs.get(f"maxabs_{fname}", np.nan)
                row[f"bias_{fname}"] = tr.user_attrs.get(f"bias_{fname}", np.nan)
            row["mae_other_mean"] = float(np.nanmean([row["mae_posf10325"], row["mae_posf10264"]]))
            all_trial_rows.append(row)

    candidate_df = pd.DataFrame(candidate_rows).sort_values("score").reset_index(drop=True)
    trial_metrics_df = pd.DataFrame(all_trial_rows).sort_values(["seed", "trial"]).reset_index(drop=True)

    selected_row, feasible_df = select_candidate_with_acceptance(
        candidate_df=candidate_df,
        legacy_ref=legacy_ref,
        max_deg_10325=args.max_degrade_posf10325,
        max_deg_10264=args.max_degrade_posf10264,
        min_improve_10289=args.min_improve_posf10289,
    )

    best_params = {
        k.replace("param_", ""): selected_row[k]
        for k in selected_row.index
        if k.startswith("param_")
    }

    print("\n--- Optimization Finished ---")
    print("Selected Parameters:")
    for k in sorted(best_params):
        print(f"  {k}: {best_params[k]:.6g}")
    print(f"Selected score ({args.objective_mode}): {selected_row['score']:.6g}")

    best_eval = evaluate_params(best_params)
    print("Per-fuel final MAE (K):")
    for fname in FUELS:
        print(
            f"  {fname}: MAE={best_eval[fname]['mae']:.3f}, "
            f"MaxAbs={best_eval[fname]['max_abs']:.3f}, "
            f"Bias={best_eval[fname]['signed_mean']:.3f}"
        )

    fuel_audit_df = audit_fuel_specifications()
    print("\nFuel specification audit summary:")
    print(
        fuel_audit_df[
            [
                "fuel",
                "gc_weight_sum",
                "compound_lists_match",
                "tc_nonfinite_count",
                "pc_nonfinite_count",
                "family_sat_massfrac",
                "family_aromatic_massfrac",
                "family_cyclo_massfrac",
                "family_olefin_massfrac",
            ]
        ].to_string(index=False)
    )

    export_plots_and_tables(
        best_params=best_params,
        best_eval=best_eval,
        trial_metrics_df=trial_metrics_df,
        candidate_df=candidate_df,
        selected_row=selected_row,
        feasible_df=feasible_df,
        legacy_ref=legacy_ref,
        fuel_audit_df=fuel_audit_df,
    )

    print("\nArtifacts written in:")
    print(f"  {_SCRIPT_DIR}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize D86 condenser model with diagnostics and fairness-aware objectives.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials per seed.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated integer seeds.")
    parser.add_argument(
        "--objective-mode",
        choices=["legacy", "weighted", "fair"],
        default="fair",
        help="Objective scoring mode.",
    )
    parser.add_argument(
        "--fuel-weights",
        type=str,
        default=None,
        help=f"Comma-separated fuel weights ordered as {FUELS}.",
    )
    parser.add_argument(
        "--worst-fuel-weight",
        type=float,
        default=0.75,
        help="Worst-fuel penalty coefficient for fair objective.",
    )
    parser.add_argument(
        "--max-degrade-posf10325",
        type=float,
        default=2.0,
        help="Maximum allowed MAE degradation (K) for posf10325 vs legacy reference.",
    )
    parser.add_argument(
        "--max-degrade-posf10264",
        type=float,
        default=2.0,
        help="Maximum allowed MAE degradation (K) for posf10264 vs legacy reference.",
    )
    parser.add_argument(
        "--min-improve-posf10289",
        type=float,
        default=1.0,
        help="Minimum required MAE improvement (K) for posf10289 vs legacy reference.",
    )
    parser.add_argument(
        "--run-legacy-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a legacy-objective reference study for acceptance filtering.",
    )
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
