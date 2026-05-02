
import subprocess
import os
import sys
import glob
import pandas as pd

def run_and_evaluate():
    # 1. Execute the training run
    cmd = [sys.executable, "train.py", "trainer.n_epochs=2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FAILED: Run crashed (likely OOM). Error: {result.stderr[-500:]}")
        return

    # 2. Locate the most recent Hydra output directory
    list_of_dirs = glob.glob('./outputs/shakespeare_diffusion_base/*')
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    
    # 3. Extract Validation Loss
    val_log_path = os.path.join(latest_dir, 'val_loss_log.csv')
    try:
        df = pd.read_csv(val_log_path, header=None, names=['epoch', 'val_loss'])
        final_val_loss = df['val_loss'].iloc[-1]
        print(f"SUCCESS: Final Validation Loss: {final_val_loss}")
        
        # 4. Print the final text output for qualitative assessment
        summary_path = os.path.join(latest_dir, 'summary.txt')
        with open(summary_path, 'r') as f:
            print("\nFinal Text Generation:\n" + f.read())
            
    except Exception as e:
        print(f"FAILED: Could not parse results. {e}")

if __name__ == "__main__":
    run_and_evaluate()
