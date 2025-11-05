import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch
from torch.distributions import Categorical
from dataset import get_data_loader, StringHandler, print_wrapped, decode
from model import GPT, GeometricNoise, GPTConfig, LogLinearNoise
import torch.optim as optim
from losses import loss_function
import os
from inference_helpers import staggered_score, transition, sample_categorical
from tqdm import tqdm
import time
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """
    Main function to run either training or inference based on the provided configuration.
    """
    set_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    # --- Hydra sets the CWD, print it for clarity ---
    print(f"Starting run: {cfg.run_name}")
    print(f"Output directory: {output_dir}")
    
    # --- Initialise ---
    sh = StringHandler()
    device = cfg.device
    
    # Use hydra.utils.to_absolute_path to resolve relative data directory
    data_dir = hydra.utils.to_absolute_path(cfg.data.dir)
    train_dataloader, dataset = get_data_loader(data_dir, sh, 'train', cfg.data.batch_size, cfg.data.context_length)
    val_dataset, _ = get_data_loader(data_dir, sh, 'val', cfg.data.batch_size, cfg.data.context_length)

    # --- Model and Noise Setup ---
    vocab_size = sh.get_vocab_size()
    if cfg.noise.type == 'geometric':
        noise = GeometricNoise(sigma_min=cfg.noise.sigma_min, sigma_max=cfg.noise.sigma_max)
    else:
        noise = LogLinearNoise()

    model_args = dict(n_layer=cfg.model.n_layer, n_head=cfg.model.n_head, n_embd=cfg.model.n_embd,
                      cond_dim=cfg.model.cond_dim, bias=cfg.model.bias, vocab_size=vocab_size,
                      block_size=cfg.data.context_length, dropout=cfg.model.dropout)

    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    
    # --- Decide to Train or Run Inference ---
    # Resolve relative model path
    model_path = hydra.utils.to_absolute_path(cfg.inference.model_path)
    if os.path.exists(model_path) and cfg.inference.run_inference is True:
        print(f"Found existing model at {model_path}. Running inference.")
        run_inference(cfg, model, noise, sh, device, vocab_size)
    else:
        print("No existing model found. Starting training.")
        run_training(cfg, model, noise, sh, device, train_dataloader, dataset, output_dir, val_dataset)

def run_training(cfg: DictConfig, model, noise, sh, device, train_dataloader, dataset, output_dir, val_dataloader):
    """Contains the main training loop logic."""
    # --- Logging ---
    run_dir = output_dir
    loss_log_path = os.path.join(run_dir, 'loss_log.csv')
    val_loss_log_path = os.path.join(run_dir, 'val_loss_log.csv')
    summary_log_path = os.path.join(run_dir, 'summary.txt')

    # --- Optimizer ---
    if cfg.trainer.use_muon:
        matrix_params = [p for n, p in model.named_parameters() if p.dim() == 2]
        other_params = [p for n, p in model.named_parameters() if p.dim() != 2]
        optimizer_matrices = optim.Muon(matrix_params, lr=cfg.trainer.lr)
        optimizer = optim.AdamW(other_params, lr=cfg.trainer.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.lr)

    sampler = None
    if cfg.trainer.prob_sampling:
        distribution = dataset.distribution.to(device)
        sampler = Categorical(distribution)

    model.train()
    start_time = time.time()
    final_loss = 0

    for epoch in range(cfg.trainer.n_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.trainer.n_epochs}")
        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)
            loss = loss_function(model, batch, noise, sh, sampling_eps=cfg.noise.sigma_min, sampler=sampler)
            final_loss = loss.item()

            if cfg.log_run:
                with open(loss_log_path, 'a') as f:
                    f.write(f'{epoch},{i},{final_loss}\n')
            
            optimizer.zero_grad(set_to_none=True)
            if cfg.trainer.use_muon:
                optimizer_matrices.zero_grad(set_to_none=True)
            

            loss.backward()
            optimizer.step()
            if cfg.trainer.use_muon:
                optimizer_matrices.step()

            if i % 1000 == 0 and i > 0 and cfg.log_run:
                model.eval()  # Set the model to evaluation mode
                total_val_loss = 0
                with torch.no_grad(): # Ensure no gradients are calculated
                    for batch in val_dataloader:
                        batch = batch.to(device)
                        # The same loss function you use for training
                        loss = loss_function(model, batch, noise, sh, sampler=None) 
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)

                with open(val_loss_log_path, 'a') as f:
                    f.write(f'{epoch},{i},{avg_val_loss}\n')

                model.train() # Set the model back to training mode

        if cfg.save_model:
            torch.save(model.state_dict(), os.path.join(run_dir, f'model_epoch_{epoch+1}.pth'))

    # --- Final Logging ---
    end_time = time.time()
    duration_sec = end_time - start_time
    duration_str = f"{(duration_sec % 3600) // 60:02.0f}m {duration_sec % 60:02.0f}s"
    if cfg.log_run:
        with open(summary_log_path, 'w') as f:
            f.write(f"--- Configuration ---\n{OmegaConf.to_yaml(cfg)}\n")
            f.write(f"\n--- Training Summary ---\n")
            f.write(f"Total Epochs: {cfg.trainer.n_epochs}\n")
            f.write(f"Final Loss: {final_loss:.4f}\n")
            f.write(f"Total Runtime: {duration_str}\n")
            f.write(f"Total Runtime (seconds): {duration_sec:.2f}\n")

            f.write("\n\n--- Final Qualitative Sample ---\n")
            model.eval() # Set model to evaluation mode for inference
            
            vocab_size = sh.get_vocab_size()
            steps = cfg.inference.steps
            eps = cfg.inference.eps
            timesteps = torch.linspace(1, eps, steps + 1, device=device)
            step_size = (1 - eps) / steps
            x = torch.randint(0, vocab_size, (1, cfg.data.context_length), device=device)

            with torch.no_grad():
                for i in tqdm(range(steps + 1), desc="Generating final sample"):
                    t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                    curr_sigma_bar = noise(t)[0]
                    
                    if i < steps:
                        next_sigma_bar = noise(t - step_size)[0]
                        delta_sigma = curr_sigma_bar - next_sigma_bar
                    else: # Last denoising step
                        delta_sigma = curr_sigma_bar

                    log_score = model(x, curr_sigma_bar)
                    score = torch.exp(log_score)

                    stag_score = staggered_score(score, delta_sigma)
                    probs = stag_score * transition(x, delta_sigma, sh)
                    x = sample_categorical(probs)
            
            final_text = decode(x[0], sh)
            
            f.write(final_text)
            
            print("\n--- Final Generated Text ---")
            print_wrapped(final_text, end='\n\n')

    print(f"Training finished. Final loss: {final_loss:.4f}. Duration: {duration_str}")


def run_inference(cfg: DictConfig, model, noise, sh, device, vocab_size):
    """Contains the inference (sampling) logic."""
    model.load_state_dict(torch.load(hydra.utils.to_absolute_path(cfg.inference.model_path), weights_only=True))
    model.eval()

    steps = cfg.inference.steps
    eps = cfg.inference.eps
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    step_size = (1 - eps) / steps
    x = torch.randint(0, vocab_size, (1, cfg.data.context_length), device=device)

    with torch.no_grad():
        for i in tqdm(range(steps + 1), desc="Inference Step"):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            curr_sigma_bar = noise(t)[0]
            
            if i < steps:
                next_sigma_bar = noise(t - step_size)[0]
                delta_sigma = curr_sigma_bar - next_sigma_bar
            else: # Last denoising step
                delta_sigma = curr_sigma_bar

            log_score = model(x, curr_sigma_bar)
            score = torch.exp(log_score)

            stag_score = staggered_score(score, delta_sigma)
            probs = stag_score * transition(x, delta_sigma, sh)
            x = sample_categorical(probs)

        print(f'\n--- Final Decoded Text ---')
        print_wrapped(decode(x[0], sh), end='\n\n', flush=True)


if __name__ == "__main__":
    main()
