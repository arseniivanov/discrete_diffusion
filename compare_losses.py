import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing all your individual run folders
runs_base_dir = 'runs/'

# Get a list of all run directories
run_folders = [f for f in os.listdir(runs_base_dir) if os.path.isdir(os.path.join(runs_base_dir, f))]

plt.figure(figsize=(12, 8))

for run_folder in sorted(run_folders):
    log_file = os.path.join(runs_base_dir, run_folder, 'loss_log.csv')
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        plt.plot(df.iloc[:, 1], df.iloc[:, 2].rolling(window=50).mean(), label=run_folder)

plt.title('Training Loss Comparison Across Runs')
plt.xlabel('Training Step')
plt.ylabel('Loss (Smoothed)')
plt.legend()
plt.grid(True)
plt.show()
