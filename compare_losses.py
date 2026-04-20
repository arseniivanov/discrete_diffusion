import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml  # You might need to install PyYAML: pip install pyyaml
from datetime import datetime

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Plot training and validation losses from Hydra runs.")
parser.add_argument(
    '--date', type=str, default=None,
    help='Optional: Filter by date folder, e.g., "2025-11-05"'
)
parser.add_argument(
    '--runs', type=str, default=None,
    help='Optional: Comma-separated list of run indices, e.g., "5,7,8,13"'
)
parser.add_argument(
    '--best', action='store_true',
    help='Optional: If set, finds the run with the lowest final loss and prints its full summary.txt.'
)
args = parser.parse_args()


# --- Configuration ---
outputs_base_dir = Path('./')
COLUMN_NAMES = ['epoch', 'step', 'loss']


# --- Helper Function for Parsing ---
def parse_summary_file(summary_path):
    """Parses the summary.txt file to extract key parts and the full content."""
    config_dict = None
    sample_text = ""
    final_loss_val = None
    full_content = ""
    try:
        with open(summary_path, 'r') as f:
            content = f.read()
        full_content = content

        # Extract the YAML config block
        config_start = content.find("--- Configuration ---")
        config_end = content.find("--- Training Summary ---")
        if config_start != -1 and config_end != -1:
            yaml_content = content[config_start + len("--- Configuration ---"):config_end]
            config_dict = yaml.safe_load(yaml_content)

        # Extract the Qualitative Sample block
        sample_start = content.find("--- Final Qualitative Sample ---")
        if sample_start != -1:
            sample_text = content[sample_start + len("--- Final Qualitative Sample ---"):].strip()

        # Extract the final loss value as a float
        loss_line_start = content.find("Final Loss:")
        if loss_line_start != -1:
            # Find the end of that line
            loss_line_end = content.find('\n', loss_line_start)
            if loss_line_end == -1: # Handles case where it's the last line
                loss_line_end = len(content)
            
            loss_line = content[loss_line_start:loss_line_end]
            try:
                loss_str = loss_line.split(':')[1].strip()
                final_loss_val = float(loss_str)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse final loss value from line: '{loss_line}'. Error: {e}")


    except Exception as e:
        print(f"Warning: Could not parse {summary_path}. Error: {e}")

    return config_dict, sample_text, final_loss_val, full_content


def create_legend_from_config(config):
    """Builds a descriptive legend string from the parsed config dictionary."""
    if not config:
        return "Unknown"

    parts = []

    # Safely get nested values using .get() to avoid errors if keys are missing
    noise_config = config.get('noise', {})
    trainer_config = config.get('trainer', {})

    parts.append(noise_config.get('type', 'N/A'))
    parts.append(str(trainer_config.get('lr', 'N/A')))

    if trainer_config.get('use_muon', False):
        parts.append('muon')

    if trainer_config.get('prob_sampling', False):
        parts.append('prob')

    return "-".join(parts)


# --- Main Script ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9))
qualitative_samples = []

# --- Variables to track the best run ---
best_run_info = {
    "loss": float('inf'),
    "summary": None,
    "label": "N/A"
}


# --- File Discovery and Filtering ---
all_log_files = sorted(list(outputs_base_dir.rglob('loss_log.csv')), key=lambda p: str(p))
if args.date:
    dates = args.date.split(',')
    all_log_files = [f for f in all_log_files for d in dates if d in str(f)]
if args.runs:
    try:
        selected_indices = {int(i.strip()) for i in args.runs.split(',') if i.strip()}
        all_log_files = [f for f in all_log_files if int(f.parent.name) in selected_indices]
    except ValueError:
        print(f"Error: Invalid --runs argument. Must be comma-separated integers.")
        exit()
        print("\nCould not automatically determine the best run.")

if not all_log_files:
    print("\nNo log files found after applying filters. Exiting.")
else:
    print(f"\nFound {len(all_log_files)} training runs to process.")

# --- Plotting Loop ---
for log_file in all_log_files:
    try:
        run_folder = log_file.parent
        default_label = "/".join(run_folder.parts[-3:]) # Fallback label

        # --- Generate Label from Summary ---
        summary_file = run_folder / 'summary.txt'
        config, sample, final_loss, full_summary = parse_summary_file(summary_file)

        if config:
            label = create_legend_from_config(config)
        else:
            label = default_label

        # --- Check if this is the new best run ---
        if args.best and final_loss is not None:
            if final_loss < best_run_info["loss"]:
                best_run_info["loss"] = final_loss
                best_run_info["summary"] = full_summary
                best_run_info["label"] = label

        # --- Process and Plot Data ---
        df_train = pd.read_csv(log_file, header=None, names=COLUMN_NAMES)
        if df_train.empty: continue

        steps_per_epoch = df_train['step'].max() + 1
        df_train['global_step'] = df_train['epoch'] * steps_per_epoch + df_train['step']
        smoothed_loss = df_train['loss'].rolling(window=50, min_periods=1).mean()

        line = ax.plot(df_train['global_step'], smoothed_loss, label=label, alpha=0.9)
        run_color = line[0].get_color()

        val_log_file = run_folder / 'val_loss_log.csv'
        if val_log_file.exists():
            df_val = pd.read_csv(val_log_file, header=None, names=COLUMN_NAMES)
            if not df_val.empty:
                df_val['global_step'] = df_val['epoch'] * steps_per_epoch + df_val['step']
                ax.plot(
                    df_val['global_step'], df_val['loss'], color=run_color,
                    linestyle='--', marker='o', markersize=4, label=f"{label} (Val)", alpha=0.9
                )

        if sample:
            qualitative_samples.append({"label": label, "sample": [sample, f"Final Loss: {final_loss}"]})

    except Exception as e:
        print(f"Could not process run in folder {run_folder.name}: {e}")

# --- Finalize and Show Plot ---
if all_log_files and False:
    ax.set_title('Training & Validation Loss Comparison', fontsize=18, pad=20)
    ax.set_xlabel('Global Training Step', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    legend_fontsize = 'medium' if len(all_log_files) < 10 else 'small'
    ax.legend(loc='best', fontsize=legend_fontsize)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date_str = args.date.replace(',', '_') if args.date else current_time
    filename = f"loss_plot_{date_str}.png"


    plt.savefig(filename, dpi=300)
    print(f"\nPlot saved to {filename}")

    plt.show()

# --- Print Filtered Qualitative Samples ---
if qualitative_samples:
    print("\n" + "="*80)
    print(" " * 25 + "FINAL QUALITATIVE SAMPLES")
    print("="*80)
    for item in sorted(qualitative_samples, key=lambda x: x['label']): # Sort for consistency
        print(f"\n--- Run: {item['label']} ---\n")
        print(item['sample'][0])
        print("\n" + "-"*80)
        print(item['sample'][1])


# --- Print Full Summary for the Best Run if requested ---
if args.best:
    print("\n" + "="*80)
    print(" " * 24 + "SUMMARY FOR BEST RUN (LOWEST LOSS)")
    print("="*80)

    if best_run_info["summary"]:
        print(f"\nFound best run: '{best_run_info['label']}'")
        print(f"Lowest Final Loss: {best_run_info['loss']:.6f}\n")
        print("-" * 80)
        print("\nDisplaying full summary:\n")
        print(best_run_info["summary"])
    else:
        print("This can happen if no 'Final Loss:' line was found or parsable in the summary files.")

    print("\n" + "="*80)
