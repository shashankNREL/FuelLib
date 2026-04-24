import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import optuna
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add FuelLib source to Python path
sys.path.insert(0, '/Users/syellapa/Documents/Research/2025/SAF/FuelLib/FuelLib')
from source.FuelLib import fuel
from source.distillation_rk2 import run_d86_simulation_rk2

TARGET_VOLUMES = np.array([5.0, 10.0, 20.0, 40.0, 50.0, 70.0, 90.0, 95.0])
FUELS = ["posf10325", "posf10289", "posf10264"]
CSV_PATH = "/Users/syellapa/Documents/Research/2025/SAF/FuelLib/DLR_Distillation/ExperimentalData/distillation_NJFCP.csv"

def load_experimental_targets():
    df = pd.read_csv(CSV_PATH)
    V_pct = df["VolumePercentage"].values
    
    targets = {}
    for f in FUELS:
        T_C = df[f].values
        # Interpolator mapping Volume array -> Temperature array
        interp_func = interp1d(V_pct, T_C, kind='linear', fill_value='extrapolate')
        T_target_C = interp_func(TARGET_VOLUMES)
        T_target_K = T_target_C + 273.15
        targets[f] = T_target_K
    return targets

EXP_TARGETS = load_experimental_targets()

def objective(trial):
    # Parameter domain
    air = trial.suggest_float("initial_moles_air", 0.001, 0.05)
    c_glass = trial.suggest_float("C_glass", 0.1, 5.0)
    h_mult = trial.suggest_float("h_multiplier", 0.5, 5.0)
    q1 = trial.suggest_float("initial_Q1", 10.0, 100.0)
    
    total_error = 0.0
    
    sim_params = {
        "dt": 1,
        "initial_moles_air": air,
        "C_glass": c_glass,
        "h_multiplier": h_mult,
        "T_bubble_lo": 375.0,
        "T_bubble_hi": 650.0,
        "P_atm": 101325.0,
        "T_room": 298.15,
        "Q1": q1,
        "target_rate_ml_min": 4.5,
        "controller_Kp": 5.0,
        "controller_Ki": 0.05,
        "min_volume_W_mL": 0.5,
        "use_srk": True,
    }
    
    for fname in FUELS:
        f_obj = fuel(fname)
        Xi = f_obj.Y2X(f_obj.Y_0)
        
        try:
            results = run_d86_simulation_rk2(f_obj, Xi, volume_initial_mL=100.0, sim_params=sim_params)
            
            sim_vol = np.array(results["distillate_vol"])
            sim_T = np.array(results["T_D86"])
            
            # Penalize if it failed to distil significantly
            if len(sim_vol) < 3 or sim_vol[-1] < 85.0:
                return 1000.0
                
            # Drop trailing identical volume entries if any (can cause interp1d to fail)
            _, unique_idx = np.unique(sim_vol, return_index=True)
            sim_vol = sim_vol[unique_idx]
            sim_T = sim_T[unique_idx]
            
            sim_interp = interp1d(sim_vol, sim_T, kind='linear', fill_value='extrapolate')
            sim_target_T = sim_interp(TARGET_VOLUMES)
            
            exp_target_T = EXP_TARGETS[fname]
            # Mean Absolute Error (MAE) per fuel (in Kelvin)
            mae = np.mean(np.abs(sim_target_T - exp_target_T))
            total_error += mae
            
        except Exception:
            return 1000.0  # Constraint/solver failure penalty
            
    # Return average MAE across all 3 fuels
    return total_error / len(FUELS)

if __name__ == "__main__":
    print("Starting Bayesian Optimization via Optuna for D86 DLR distillation (RK2)...")
    print(f"Target experimental points: {TARGET_VOLUMES}% over {FUELS}")
    
    # We create a simple in-memory study
    study = optuna.create_study(direction="minimize")
    
    # Run 25 trials
    study.optimize(objective, n_trials=35)
    
    print("\n--- Optimization Finished ---")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    print(f"Best Mean Absolute Error (K): {study.best_value:.2f}")

    def evaluate_and_plot(best_params):
        print("\nRe-evaluating best parameters to generate plots and CSV...")
        sim_params = {
            "dt": 1.0,
            "initial_moles_air": best_params["initial_moles_air"],
            "C_glass": best_params["C_glass"],
            "h_multiplier": best_params["h_multiplier"],
            "T_bubble_lo": 375.0,
            "T_bubble_hi": 650.0,
            "P_atm": 101325.0,
            "T_room": 298.15,
            "Q1": best_params["initial_Q1"],
            "target_rate_ml_min": 4.5,
            "controller_Kp": 5.0,
            "controller_Ki": 0.05,
            "min_volume_W_mL": 0.5,
            "use_srk": True,
        }
        
        # Experimental data for all fuels
        df_exp = pd.read_csv(CSV_PATH)
        V_pct_exp = df_exp["VolumePercentage"].values
        
        # Store DataFrames to concatenate later for the CSV export
        df_list = [df_exp.copy()]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, fname in enumerate(FUELS):
            f_obj = fuel(fname)
            Xi = f_obj.Y2X(f_obj.Y_0)
            
            # Experimental curve
            T_exp_C = df_exp[fname].values
            T_exp_K = T_exp_C + 273.15
            axes[i].plot(V_pct_exp, T_exp_K, 'ro-', label='Experimental Data')
            

            
            # Simulated tuned curve
            try:
                # Suppress simulation print output for cleaner console
                sys.stdout = open(os.devnull, 'w')
                results = run_d86_simulation_rk2(f_obj, Xi, volume_initial_mL=100.0, sim_params=sim_params)
                sys.stdout = sys.__stdout__
                
                sim_vol = np.array(results["distillate_vol"])
                sim_T = np.array(results["T_D86"])
                
                axes[i].plot(sim_vol, sim_T, 'b-', linewidth=2, label='Simulation')
                
                # Append raw simulated data to the CSV export list
                df_sim = pd.DataFrame({
                    f"{fname}_sim_vol": sim_vol,
                    f"{fname}_sim_T": sim_T
                })
                df_list.append(df_sim)
                
            except Exception as e:
                sys.stdout = sys.__stdout__
                print(f"Plotting/Simulation failed for {fname}: {e}")
                
            axes[i].set_title(f"D86 Curve: {fname}")
            axes[i].set_xlabel("Volume Distilled (%)")
            axes[i].set_ylabel("T_D86 (K)")
            axes[i].grid(True)
            axes[i].legend()
            
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "d86_tuned_comparison.png")
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        
        # Save results to CSV
        csv_export_path = os.path.join(os.path.dirname(__file__), "optimized_distillation_curves.csv")
        pd.concat(df_list, axis=1).to_csv(csv_export_path, index=False)
        print(f"CSV results saved to: {csv_export_path}")

    evaluate_and_plot(study.best_params)
