# benchmark.py
import torch
import triton
from model import GPT, GPTConfig, DDiTBlock, precompute_freqs_cis
from triton_model import TritonDDiTBlock
from dataset import StringHandler
import hydra
from omegaconf import DictConfig, OmegaConf

# Set your baseline dimensions. 
BATCH_SIZE = 1
SEQ_LEN = 256
DEVICE = 'cuda'

def benchmark_component(config, x, c, freqs_cis):
    print("\n--- Benchmarking Single DDiTBlock ---")
    # Eager PyTorch
    block_pt = DDiTBlock(config, layer_idx=0).to(DEVICE).eval()
    
    #Needed despite original not having this, otherwise we might skip code
    block_pt.adaLN_modulation.weight.data.normal_(mean=0.0, std=0.02)
    block_pt.adaLN_modulation.bias.data.normal_(mean=0.0, std=0.02)

    # torch.compile baseline
    block_compiled = torch.compile(block_pt)
    
    block_triton = TritonDDiTBlock(config).to(DEVICE).eval()
    block_triton.load_state_dict(block_pt.state_dict(), strict=False)

    #warmup
    with torch.no_grad():
        for _ in range(5):
            _ = block_pt(x, c, freqs_cis)
            _ = block_compiled(x, c, freqs_cis)
            _ = block_triton(x, c, freqs_cis)

        out_pt = block_pt(x, c, freqs_cis).to(torch.bfloat16)
        out_comp = block_compiled(x, c, freqs_cis).to(torch.bfloat16)
        out_triton = block_triton(x, c, freqs_cis)
        
        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print(f"Max Abs Error (Compile vs Eager): {err_comp:.6f}")
        print(f"Max Abs Error (Triton vs Eager):  {err_triton:.6f}")
        
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2, msg="torch.compile output drift")
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2, msg="Triton kernel is spitting garbage")
        print("✓ Correctness verified.")

    # do_bench handles CUDA sync and percentiles automatically
    ms_pt = triton.testing.do_bench(lambda: block_pt(x, c, freqs_cis))
    ms_comp = triton.testing.do_bench(lambda: block_compiled(x, c, freqs_cis))
    ms_triton = triton.testing.do_bench(lambda: block_triton(x, c, freqs_cis))
    
    print(f"Eager Component:    {ms_pt:.4f} ms")
    print(f"Compiled Component: {ms_comp:.4f} ms")
    print(f"Triton Component:   {ms_triton:.4f} ms")

def benchmark_full_model(config, idx, sigma):
    print("\n--- Benchmarking Full GPT Forward Pass ---")
    model_pt = GPT(config).to(DEVICE).eval()
    model_compiled = torch.compile(model_pt)

    with torch.no_grad():
        for _ in range(3):
            _ = model_pt(idx, sigma)
            _ = model_compiled(idx, sigma)

    ms_full_pt = triton.testing.do_bench(lambda: model_pt(idx, sigma))
    ms_full_comp = triton.testing.do_bench(lambda: model_compiled(idx, sigma))

    print(f"Eager Model:    {ms_full_pt:.4f} ms")
    print(f"Compiled Model: {ms_full_comp:.4f} ms")

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def run_benchmark(cfg: DictConfig):
    device = DEVICE
    batch_size = cfg.data.batch_size
    seq_len = cfg.data.context_length

    sh = StringHandler()
    vocab_size = sh.get_vocab_size()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model_dict = OmegaConf.to_container(cfg.model, resolve=True)
        config = GPTConfig(
            block_size=seq_len, 
            vocab_size=vocab_size, 
            **model_dict
        )
        
        x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
        c = torch.randn(batch_size, config.cond_dim, device=device)
        freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, seq_len).to(device)
        
        benchmark_component(config, x, c, freqs_cis)
        
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        sigma = torch.rand(batch_size, device=device)
        
        #benchmark_full_model(config, idx, sigma)

if __name__ == "__main__":
    run_benchmark()
