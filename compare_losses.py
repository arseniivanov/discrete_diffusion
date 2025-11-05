import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# The base directory where Hydra stores all outputs.
outputs_base_dir = Path('outputs/')

# --- Column definitions for your CSV file ---
# This makes the code easier to read and less prone to errors.
# Your log format is: epoch, step, loss
STEP_COLUMN = 1
LOSS_COLUMN = 2

# Create a plot
plt.figure(figsize=(15, 8))

# Use rglob to recursively find all 'loss_log.csv' files
# This is the key change to handle the nested Hydra structure.
all_log_files = sorted(list(outputs_base_dir.rglob('loss_log.csv')))

if not all_log_files:
    print(f"No 'loss_log.csv' files found in '{outputs_base_dir}'. Did you run any training?")
else:
    print(f"Found {len(all_log_files)} log files to plot.")

for log_file in all_log_files:
    try:
        # The parent of the log file is the specific run directory
        # e.g., .../outputs/shakespeare_diffusion_base/2025-11-05_19-45-24
        run_folder = log_file.parent

        # Create a clean, descriptive label for the legend
        # This will look like: "shakespeare_diffusion_base/2025-11-05_19-45-24"
        label = f"{run_folder.parent.name}/{run_folder.name}"

        # Read the CSV file. Since it has no header, we use header=None.
        df = pd.read_csv(log_file, header=None)
        
        # Check if the dataframe is empty or has too few rows for a rolling window
        if not df.empty and len(df) > 1:
            # Apply a rolling mean to smooth the loss curve for better readability
            # Using .iloc because there are no column names
            smoothed_loss = df.iloc[:, LOSS_COLUMN].rolling(window=50, min_periods=1).mean()
            
            # Plot the step number against the smoothed loss
            plt.plot(df.iloc[:, STEP_COLUMN], smoothed_loss, label=label)
            
    except Exception as e:
        print(f"Could not process file {log_file}: {e}")


plt.title('Training Loss Comparison Across Runs', fontsize=16)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss (Smoothed with 50-step window)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
