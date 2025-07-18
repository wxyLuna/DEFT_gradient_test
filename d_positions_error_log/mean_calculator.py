import os
import numpy as np
# Base directory
base_dir = "/home/wxyluna/my_project/DEFT_gradient_test/d_mass_error_log"

# Initialize list to collect data
all_errors = []
file_type = 'abserr_timer1_step0.npy'

# Loop through error_logs1 to error_logs10
for i in range(1, 11):
    subfolder = f"error_logs{i}"  # <-- note the "s"
    file_path = os.path.join(base_dir, subfolder, file_type)

    if os.path.exists(file_path):
        data = np.load(file_path)
        print(data)
        all_errors.append(np.abs(data))
    else:
        print(f"Warning: {file_path} does not exist.")

# Compute and save mean if any data was found
if all_errors:
    mean_error = np.mean(all_errors, axis=0)
    print("Mean of all error files:\n", mean_error)
    np.save(os.path.join(base_dir, f"mean_{file_type}"), mean_error)
else:
    print("No error data found.")
