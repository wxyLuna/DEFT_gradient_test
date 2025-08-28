import numpy as np
import os

# Force 4-digit scientific notation
np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.3e}"})

all_abs, all_rel, all_ratio = [], [], []

for run_idx in range(10):
    base_dir = f"/home/wxyluna/my_project/dev-DFFT_run/gradient_check/run_{run_idx}"
    ith = 0
    file_path = os.path.join(base_dir, f"last_step_error_log_timestep{ith}.npz")

    if os.path.exists(file_path):
        data = np.load(file_path)
        abs_err = data["absolute_error"]      # (10, 3, 13, 3)
        rel_err = data["relative_error"]      # (10, 3, 13, 3)
        ratio   = data["ratio"]               # shape compatible

        all_abs.append(abs_err)
        all_rel.append(rel_err)
        all_ratio.append(ratio)
    else:
        print(f"[run {run_idx}] Warning: {file_path} does not exist.")

if all_abs:
    # stack across runs and average over axis=0 (runs and samples) → tensor (3,13,3)
    all_abs = np.concatenate(all_abs, axis=0)      # (10*#runs, 3, 13, 3)
    all_rel = np.concatenate(all_rel, axis=0)
    all_ratio = np.concatenate(all_ratio, axis=0)

    mean_abs_error = np.mean(all_abs, axis=0)
    mean_rel_error = np.mean(np.abs(all_rel), axis=0)
    mean_ratio     = np.mean(np.abs(all_ratio), axis=0)

    print("=== Global averages across all runs (ith=0) ===")
    print("Mean absolute error tensor:\n", mean_abs_error)
    print("Mean of |relative error| tensor:\n", mean_rel_error)
    print("Mean ratio:\n", mean_ratio)
else:
    print("No valid files found.")
