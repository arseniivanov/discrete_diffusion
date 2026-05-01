import torch
import triton
from model import GPT, GPTConfig
from dataset import StringHandler
import hydra

DEVICE = 'cuda'

def execute_full_model_profile_pass(config, idx, sigma, target="compile"):
    print(f"\n--- Executing NVTX Profile Pass: Full Model ({target}) ---")
    
    model = GPT(config).to(DEVICE).eval()
    if target == "compile":
        model = torch.compile(model)
        
    with torch.no_grad():
        # JIT and hardware warmup
        for _ in range(3):
            _ = model(idx, sigma)
            
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("FullModelForward")
        
        _ = model(idx, sigma)
        
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def run_profile(cfg):
    sh = StringHandler()
    vocab_size = sh.get_vocab_size()
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        config = GPTConfig(
            block_size=cfg.data.context_length, 
            vocab_size=vocab_size, 
            n_layer=cfg.model.n_layer, 
            n_head=cfg.model.n_head, 
            n_embd=cfg.model.n_embd, 
            cond_dim=cfg.model.cond_dim,
            use_gated_delta=getattr(cfg.model, 'use_gated_delta', False)
        )
        
        idx = torch.randint(0, config.vocab_size, (cfg.data.batch_size, cfg.data.context_length), device=DEVICE)
        sigma = torch.rand(cfg.data.batch_size, device=DEVICE)
        
        # Target can be "eager" or "compile"
        execute_full_model_profile_pass(config, idx, sigma, target="compile")

if __name__ == "__main__":
    run_profile()
