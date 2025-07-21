import os
import numpy as np

# Base directory
perturbation_tyoe = "d_positions_e6"
base_dir = f"/home/wxyluna/my_project/DEFT_gradient_test/{perturbation_tyoe}"

# Initialize list to collect data
all_errors = []

# Loop through error_logs1 to error_logs10
for i in range(1, 11):
    subfolder = f"error_logs{i}"  # <-- note the "s"
    file_path = os.path.join(base_dir, subfolder, f"relerr_timer1_step0.npy")

    if os.path.exists(file_path):
        data = np.load(file_path)
        all_errors.append(np.abs(data))
    else:
        print(f"Warning: {file_path} does not exist.")

# Compute and save mean if any data was found
if all_errors:
    mean_error = np.mean(all_errors, axis=0)
    print("Mean of all error files:\n", mean_error)
    np.save(os.path.join(base_dir, "mean_relerr_timer1_step0.npy"), mean_error)
else:
    print("No error data found.")
