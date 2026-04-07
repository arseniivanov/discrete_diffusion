import os
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Important: Make sure these imports point to your project's files
from train import set_seed, sample_masking, sample_substitution
from dataset import StringHandler, ShakespeareDataset
from model import GPT, GPTConfig, GeometricNoise, LogLinearNoise, MaskingNoise

def reevaluate_run(run_path: str):
    """
    Loads a model and its configuration from a Hydra run directory
    and generates a new, fair sample.
    """
    print(f"\n{'='*80}")
    print(f"Re-evaluating run: {run_path}")
    print(f"{'='*80}")

    # --- 1. Load the Configuration ---
    # The original config is saved by Hydra in this subdirectory
    config_dir = os.path.join(run_path, '.hydra')
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), job_name="reeval"):
        # We compose with the 'base_config' from your conf directory
        cfg = compose(config_name="config")

    # --- 2. Find the Final Model Checkpoint ---
    # Find the model with the highest epoch number
    model_files = [f for f in os.listdir(run_path) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not model_files:
        print(f"SKIPPING: No model checkpoints found in {run_path}")
        return

    latest_epoch = -1
    latest_model_path = ""
    for f in model_files:
        try:
            epoch_num = int(f.split('_')[-1].split('.')[0])
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_model_path = os.path.join(run_path, f)
        except ValueError:
            continue

    if not latest_model_path:
        print(f"SKIPPING: Could not identify latest model in {run_path}")
        return

    print(f"Found latest model: {os.path.basename(latest_model_path)}")

    # --- 3. Set Up Model, Noise, and Data ---
    set_seed(cfg.seed)
    device = cfg.device
    sh = StringHandler()
    vocab_size = sh.get_vocab_size()

    # We only need a dataset object for its properties (like distribution),
    # so we can use a small batch size.
    dataset = ShakespeareDataset(cfg.data.dir, vocab_size, 'train', cfg.data.context_length)

    # Setup Noise
    if cfg.noise.type == 'geometric':
        noise = GeometricNoise(sigma_min=cfg.noise.sigma_min, sigma_max=cfg.noise.sigma_max)
    elif cfg.noise.type == 'masking':
        noise = MaskingNoise(schedule=cfg.noise.schedule)
    else:
        noise = LogLinearNoise()

    # Setup Model
    model_args = dict(n_layer=cfg.model.n_layer, n_head=cfg.model.n_head, n_embd=cfg.model.n_embd,
                      cond_dim=cfg.model.cond_dim, bias=cfg.model.bias, vocab_size=vocab_size,
                      block_size=cfg.data.context_length, dropout=cfg.model.dropout)
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)

    # --- 4. Load Weights and Generate Sample ---
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.eval()

    final_x = None
    if cfg.noise.type == 'masking':
        final_x = sample_masking(model, noise, sh, cfg, device)
    else:
        final_x = sample_substitution(model, noise, sh, cfg, device, dataset)

    final_text = decode(final_x[0], sh)

    # --- 5. Print the New Fair Sample ---
    print("\n--- Newly Generated Fair Sample ---")
    print_wrapped(final_text, end='\n\n')

    # Optionally, save the new sample to a file in the run directory
    with open(os.path.join(run_path, "fair_sample.txt"), "w") as f:
        f.write(final_text)
    print(f"Saved new sample to {os.path.join(run_path, 'fair_sample.txt')}")


if __name__ == '__main__':
    # The path to your multirun directory
    multirun_path = 'multirun/2025-11-05/20-00-43/'

    if not os.path.isdir(multirun_path):
        print(f"Error: Directory not found at '{multirun_path}'")
        exit()

    # Find all the individual run directories (they are usually numbers)
    run_dirs = [d for d in os.listdir(multirun_path) if os.path.isdir(os.path.join(multirun_path, d))]

    for run_dir_name in sorted(run_dirs, key=int):
        run_path = os.path.join(multirun_path, run_dir_name)
        try:
            reevaluate_run(run_path)
        except Exception as e:
            print(f"!!! FAILED to process run {run_dir_name}: {e}")
