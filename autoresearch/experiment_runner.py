import subprocess
import sys
import os
from datetime import datetime

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def run_experiment(config_name: str):
    """
    Runs the main train.py script as a separate process with a specific config.
    """
    print("\n" + "="*60)
    print(f"🚀 LAUNCHING EXPERIMENT: {config_name}")
    print("="*60 + "\n")
    
    # Construct the command to run train.py with Hydra's config_name override
    command = [
        sys.executable,
        os.path.join(_ROOT, "train.py"),
        f"--config-name={config_name}" # The specific config file to use
    ]
    
    try:
        subprocess.run(command, check=True)
        print("\n" + "="*60)
        print(f"✅ EXPERIMENT SUCCEEDED: {config_name}")
        print("="*60 + "\n")
    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print(f"❌ EXPERIMENT FAILED: {config_name}")
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print("="*60 + "\n")

if __name__ == "__main__":
    experiment_queue = [
        "base_config",
        # "another_experiment", # Add more configs as you create them
    ]
    
    start_time = datetime.now()
    print(f"Starting experiment runner at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Found {len(experiment_queue)} experiments in the queue.")
    
    for config in experiment_queue:
        run_experiment(config)
        
    end_time = datetime.now()
    print(f"Experiment runner finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")
