import torch
torch.cuda.memory._set_allocator_settings('expandable_segments:True')  # prevent fragmentation OOM

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.distributions import Categorical
from dataset import get_data_loader, StringHandler, print_wrapped, decode
from model import GPT, GeometricNoise, GPTConfig, LogLinearNoise, MaskingNoise
import torch.optim as optim
from losses import loss_function, flow_loss
import os
from inference_helpers import sample_masking, sample_substitution, sample_discrete_flow
from tqdm import tqdm
import time
import random
import numpy as np
import ast
import gc

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
    elif cfg.noise.type == 'masking':
        noise = MaskingNoise(schedule=cfg.noise.schedule)
    else:
        noise = LogLinearNoise()

    model_args = dict(n_layer=cfg.model.n_layer, n_head=cfg.model.n_head, n_embd=cfg.model.n_embd,
                        cond_dim=cfg.model.cond_dim, bias=cfg.model.bias, vocab_size=vocab_size,
                        block_size=cfg.data.context_length, dropout=cfg.model.dropout, timestep_embedding=cfg.model.timestep_embedding,
                        use_gated_delta=getattr(cfg.model, 'use_gated_delta', False),
                        gated_delta_layers=getattr(cfg.model, 'gated_delta_layers', None),
                        attn_mode=getattr(cfg.model, 'attn_mode', 'chunk'),
                        gated_delta_expand_v=getattr(cfg.model, 'gated_delta_expand_v', 1.0),
                        gated_delta_use_gate=getattr(cfg.model, 'gated_delta_use_gate', True),
                        gated_delta_use_short_conv=False,  # Force disable
                        gated_delta_allow_neg_eigval=getattr(cfg.model, 'gated_delta_allow_neg_eigval', True),
                        gated_delta_conv_size=getattr(cfg.model, 'gated_delta_conv_size', 2),
                        gated_delta_norm_eps=getattr(cfg.model, 'gated_delta_norm_eps', 1e-5),
                    )

    if cfg.model.use_gated_delta:
        model_args['gated_delta_layers'] = ast.literal_eval(model_args['gated_delta_layers'])
        
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    model = torch.compile(model)
    
    # --- Decide to Train or Run Inference ---
    # Resolve relative model path
    model_path = hydra.utils.to_absolute_path(cfg.inference.model_path)
    if os.path.exists(model_path) and cfg.inference.run_inference is True:
        print(f"Found existing model at {model_path}. Running inference.")
        run_inference(cfg, model, noise, sh, device, val_dataset)
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
        _emb = {'transformer.wte.weight', 'transformer.wpe.weight'}
        matrix_params = [p for n, p in model.named_parameters() if p.dim() == 2 and n not in _emb]
        other_params = [p for n, p in model.named_parameters() if p.dim() != 2 or n in _emb]
        optimizer_matrices = optim.Muon(matrix_params, lr=cfg.trainer.lr * 2)
        optimizer = optim.AdamW(other_params, lr=cfg.trainer.lr * 1.5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.lr)

    sampler = None
    if cfg.trainer.prob_sampling:
        distribution = dataset.distribution.to(device)
        sampler = Categorical(distribution)

    model.train()
    start_time = time.time()
    final_loss = 0

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    for epoch in range(cfg.trainer.n_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.trainer.n_epochs}")
        for i, batch in enumerate(progress_bar):
            if i == 0 and epoch == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (i + 1) % 5000 == 0:
                gc.collect()

            batch = batch.to(device)
            with autocast_ctx:
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
            f.write(f'{epoch},{avg_val_loss}\n')

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
            
            final_x = None
            if cfg.noise.type == 'masking':
                final_x = sample_masking(model, noise, sh, cfg, device)
            else: # Covers 'geometric' and 'loglinear'
                final_x = sample_substitution(model, noise, sh, cfg, device, dataset)
            
            final_text = decode(final_x[0], sh)
            
            f.write(final_text)

            print("\n--- Final Generated Text ---")
            print_wrapped(final_text, end='\n\n')

    print(f"Training finished. Final loss: {final_loss:.4f}. Duration: {duration_str}")

def run_inference(cfg: DictConfig, model, noise, sh, device, dataset):
    """
    Contains the fair inference (sampling) logic that dispatches to the correct method.
    """
    model.load_state_dict(torch.load(hydra.utils.to_absolute_path(cfg.inference.model_path), weights_only=True))
    model.eval()

    final_x = None
    if cfg.noise.type == 'masking':
        print("Using Masking-based sampling (iterative unmasking)...")
        final_x = sample_masking(model, noise, sh, cfg, device)
    else: # Covers 'geometric' and 'loglinear' which use substitution
        print("Using Substitution-based sampling...")
        final_x = sample_substitution(model, noise, sh, cfg, device, dataset)
    
    final_text = decode(final_x[0], sh)

    print(f'\n--- Final Decoded Text ---')
    print_wrapped(final_text, end='\n\n', flush=True)

if __name__ == "__main__":
    main()
