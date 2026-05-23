# benchmark_pytorch_triton.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import model
from model import GPT, GPTConfig, DDiTBlock, precompute_freqs_cis, modulate, get_norm, DynTanh
from triton_model import (
    TritonDDiTBlock, TritonDDiTDynTanh,
    TritonDDiTBlockFusedMLP, TritonDDiTDynTanhFusedMLP,
    TritonTimestepEmbedder, TritonDDitFinalLayer, TritonGPT,
    fused_modulate_layernorm, fused_gate_modulate_layernorm,
    fused_modulate_dyntanh, fused_gate_modulate_dyntanh,
    fused_linear_gelu, fused_mlp_proj_epilogue, fused_mlp_full,
    triton_layernorm, triton_dyntanh,
)
from cutedsl_model import (
    cute_fused_modulate_dyntanh,
    cute_fused_depthwise_convs,
)
from dataset import StringHandler
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import os

DEVICE = 'cuda'
WORKLOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triton_translate_worklog.md')

def append_to_worklog(note, benchmark_text):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"\n[{timestamp}] - {note}\n--- Benchmarking Single DDiTBlock ---\n{benchmark_text}\n"
    with open(WORKLOG_PATH, 'a') as f:
        f.write(entry)

def print_and_collect(fmt, *args, results=None):
    line = fmt % args
    print(line)
    if results is not None:
        results.append(line)
    return line

# ---------------------------------------------------------------------------
# Sub-component benchmark: Pre-Attention Norm + Modulate
# ---------------------------------------------------------------------------
def benchmark_pre_attn_norm(config, x, shift_msa, scale_msa, block_pt, results):
    print_and_collect("\n=== Pre-Attention Norm + Modulate ===", results=results)
    cute_fn = None
    if config.norm == 'dyntanh':
        eager_fn = lambda x: modulate(block_pt.ln_1(x), shift_msa, scale_msa)
        triton_fn = lambda x: fused_modulate_dyntanh(x, shift_msa, scale_msa,
                                                     block_pt.ln_1.alpha,
                                                     block_pt.ln_1.gamma,
                                                     block_pt.ln_1.beta)
        cute_fn = lambda x: cute_fused_modulate_dyntanh(x, shift_msa, scale_msa,
                                                        block_pt.ln_1.alpha,
                                                        block_pt.ln_1.gamma,
                                                        block_pt.ln_1.beta)
    else:
        eager_fn = lambda x: modulate(block_pt.ln_1(x), shift_msa, scale_msa)
        triton_fn = lambda x: fused_modulate_layernorm(x, shift_msa, scale_msa)

    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(x)
            _ = compiled_fn(x)
            _ = triton_fn(x)
            if cute_fn is not None:
                _ = cute_fn(x)

        out_pt = eager_fn(x).to(torch.bfloat16)
        out_comp = compiled_fn(x).to(torch.bfloat16)
        out_triton = triton_fn(x).to(torch.bfloat16)
        if cute_fn is not None:
            out_cute = cute_fn(x).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        if cute_fn is not None:
            err_cute = (out_cute - out_pt).abs().max().item()
            print_and_collect("Max Abs Error (Cute vs Eager):    %.6f", err_cute, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        if cute_fn is not None:
            torch.testing.assert_close(out_cute, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(x))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(x))
        ms_triton = triton.testing.do_bench(lambda: triton_fn(x))
        if cute_fn is not None:
            ms_cute = triton.testing.do_bench(lambda: cute_fn(x))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    if cute_fn is not None:
        print_and_collect("Cute Component:     %.4f ms", ms_cute, results=results)
    return out_pt

# ---------------------------------------------------------------------------
# Sub-component benchmark: Self-Attention
# ---------------------------------------------------------------------------
def benchmark_attention(config, x_mod, freqs_cis, block_pt, results):
    print_and_collect("\n=== Self-Attention ===", results=results)
    eager_fn = lambda x: block_pt.attn(x, freqs_cis)
    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(x_mod)
            _ = compiled_fn(x_mod)

        out_pt = eager_fn(x_mod).to(torch.bfloat16)
        out_comp = compiled_fn(x_mod).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(x_mod))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(x_mod))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   N/A (no custom Triton attn kernel)", results=results)
    return out_pt

# ---------------------------------------------------------------------------
# Sub-component benchmark: Gate + Post-Attention Norm + Modulate
# ---------------------------------------------------------------------------
def benchmark_post_attn_gate_norm(config, x_attn, gate_msa, x_skip, shift_mlp, scale_mlp, block_pt, results):
    print_and_collect("\n=== Gate + Post-Attention Norm + Modulate ===", results=results)
    if config.norm == 'dyntanh':
        eager_fn = lambda attn, skip: (
            modulate(block_pt.ln_2(gate_msa * attn + skip), shift_mlp, scale_mlp),
            gate_msa * attn + skip
        )
        triton_fn = lambda attn, skip: fused_gate_modulate_dyntanh(
            attn, gate_msa, skip, shift_mlp, scale_mlp,
            block_pt.ln_2.alpha, block_pt.ln_2.gamma, block_pt.ln_2.beta
        )
    else:
        eager_fn = lambda attn, skip: (
            modulate(block_pt.ln_2(gate_msa * attn + skip), shift_mlp, scale_mlp),
            gate_msa * attn + skip
        )
        triton_fn = lambda attn, skip: fused_gate_modulate_layernorm(attn, gate_msa, skip, shift_mlp, scale_mlp)

    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(x_attn, x_skip)
            _ = compiled_fn(x_attn, x_skip)
            _ = triton_fn(x_attn, x_skip)

        out_pt, skip_pt = eager_fn(x_attn, x_skip)
        out_pt = out_pt.to(torch.bfloat16)
        out_comp, skip_comp = compiled_fn(x_attn, x_skip)
        out_comp = out_comp.to(torch.bfloat16)
        out_triton = triton_fn(x_attn, x_skip)[0].to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(x_attn, x_skip))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(x_attn, x_skip))
        ms_triton = triton.testing.do_bench(lambda: triton_fn(x_attn, x_skip))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    return out_pt, skip_pt

# ---------------------------------------------------------------------------
# Sub-component benchmark: MLP Up-proj + GELU
# ---------------------------------------------------------------------------
def benchmark_mlp_up(config, x_mod_2, block_pt, results):
    print_and_collect("\n=== MLP Up-proj + GELU ===", results=results)
    eager_fn = lambda x: block_pt.mlp.gelu(block_pt.mlp.c_fc(x))
    triton_fn = lambda x: fused_linear_gelu(x, block_pt.mlp.c_fc.weight, block_pt.mlp.c_fc.bias)
    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(x_mod_2)
            _ = compiled_fn(x_mod_2)
            _ = triton_fn(x_mod_2)

        out_pt = eager_fn(x_mod_2).to(torch.bfloat16)
        out_comp = compiled_fn(x_mod_2).to(torch.bfloat16)
        out_triton = triton_fn(x_mod_2).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(x_mod_2))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(x_mod_2))
        ms_triton = triton.testing.do_bench(lambda: triton_fn(x_mod_2))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    return out_pt

# ---------------------------------------------------------------------------
# Sub-component benchmark: MLP Down-proj + Gate + Residual
# ---------------------------------------------------------------------------
def benchmark_mlp_down(config, x_mlp_hidden, gate_mlp, x_skip, block_pt, results):
    print_and_collect("\n=== MLP Down-proj + Gate + Residual ===", results=results)
    eager_fn = lambda h, skip: gate_mlp * block_pt.mlp.c_proj(h) + skip
    triton_fn = lambda h, skip: fused_mlp_proj_epilogue(h, block_pt.mlp.c_proj.weight, gate_mlp, skip)
    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(x_mlp_hidden, x_skip)
            _ = compiled_fn(x_mlp_hidden, x_skip)
            _ = triton_fn(x_mlp_hidden, x_skip)

        out_pt = eager_fn(x_mlp_hidden, x_skip).to(torch.bfloat16)
        out_comp = compiled_fn(x_mlp_hidden, x_skip).to(torch.bfloat16)
        out_triton = triton_fn(x_mlp_hidden, x_skip).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(x_mlp_hidden, x_skip))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(x_mlp_hidden, x_skip))
        ms_triton = triton.testing.do_bench(lambda: triton_fn(x_mlp_hidden, x_skip))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    return out_pt

# ---------------------------------------------------------------------------
# Sub-component benchmark: Full MLP fused across logic divides
# ---------------------------------------------------------------------------
def benchmark_full_mlp(config, x_mod_2, gate_mlp, x_skip, block_pt, results):
    print_and_collect("\n=== Full MLP (c_fc + GELU + c_proj + gate + skip) ===", results=results)
    eager_fn = lambda x, skip: gate_mlp * block_pt.mlp.c_proj(block_pt.mlp.gelu(block_pt.mlp.c_fc(x))) + skip
    triton_fn = lambda x, skip: fused_mlp_full(
        x, block_pt.mlp.c_fc.weight, block_pt.mlp.c_proj.weight,
        gate_mlp, skip, T=x.shape[1]
    )
    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(x_mod_2, x_skip)
            _ = compiled_fn(x_mod_2, x_skip)
            _ = triton_fn(x_mod_2, x_skip)

        out_pt = eager_fn(x_mod_2, x_skip).to(torch.bfloat16)
        out_comp = compiled_fn(x_mod_2, x_skip).to(torch.bfloat16)
        out_triton = triton_fn(x_mod_2, x_skip).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(x_mod_2, x_skip))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(x_mod_2, x_skip))
        ms_triton = triton.testing.do_bench(lambda: triton_fn(x_mod_2, x_skip))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    return out_pt

# ---------------------------------------------------------------------------
# Full block benchmark
# ---------------------------------------------------------------------------
def benchmark_full_block(config, x, c, freqs_cis, results):
    print_and_collect("\n--- Benchmarking Full DDiTBlock ---", results=results)
    block_pt = DDiTBlock(config, layer_idx=0).to(DEVICE).eval()
    block_pt.adaLN_modulation.weight.data.normal_(mean=0.0, std=0.02)
    block_pt.adaLN_modulation.bias.data.normal_(mean=0.0, std=0.02)

    block_compiled = torch.compile(block_pt)
    if config.norm == 'dyntanh':
        block_triton = TritonDDiTDynTanh(config).to(DEVICE).eval()
        block_triton_fused = TritonDDiTDynTanhFusedMLP(config).to(DEVICE).eval()
    else:
        block_triton = TritonDDiTBlock(config).to(DEVICE).eval()
        block_triton_fused = TritonDDiTBlockFusedMLP(config).to(DEVICE).eval()

    block_triton.load_state_dict(block_pt.state_dict(), strict=False)
    block_triton_fused.load_state_dict(block_pt.state_dict(), strict=False)

    with torch.no_grad():
        for _ in range(5):
            _ = block_pt(x, c, freqs_cis)
            _ = block_compiled(x, c, freqs_cis)
            _ = block_triton(x, c, freqs_cis)
            _ = block_triton_fused(x, c, freqs_cis)

        out_pt = block_pt(x, c, freqs_cis).to(torch.bfloat16)
        out_comp = block_compiled(x, c, freqs_cis).to(torch.bfloat16)
        out_triton = block_triton(x, c, freqs_cis)
        out_triton_fused = block_triton_fused(x, c, freqs_cis)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        err_triton_fused = (out_triton_fused - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        print_and_collect("Max Abs Error (Triton FusedMLP vs Eager): %.6f", err_triton_fused, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton_fused, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: block_pt(x, c, freqs_cis))
        ms_comp = triton.testing.do_bench(lambda: block_compiled(x, c, freqs_cis))
        ms_triton = triton.testing.do_bench(lambda: block_triton(x, c, freqs_cis))
        ms_triton_fused = triton.testing.do_bench(lambda: block_triton_fused(x, c, freqs_cis))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    print_and_collect("Triton FusedMLP:    %.4f ms", ms_triton_fused, results=results)
    print_and_collect("Roofline 3080Ti:    3.0500 ms", results=results)
    return block_pt

# ---------------------------------------------------------------------------
# Sub-component benchmark: Input Depthwise Convs
# ---------------------------------------------------------------------------
def benchmark_input_convs(config, tok_emb, results):
    print_and_collect("\n=== Input Depthwise Convs ===", results=results)
    from triton_model import fused_input_convs
    pt_model = GPT(config).to(DEVICE).eval()
    w1, b1 = pt_model.local_conv.weight, pt_model.local_conv.bias
    w2, b2 = pt_model.local_conv2.weight, pt_model.local_conv2.bias

    def eager_fn(x):
        x = x + pt_model.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + pt_model.local_conv2(F.silu(x).transpose(1, 2)).transpose(1, 2)
        return x

    def triton_fn(x):
        return fused_input_convs(x, w1, b1, w2, b2)

    cute_fn = lambda x: cute_fused_depthwise_convs(x, w1, b1, w2, b2)

    compiled_fn = torch.compile(eager_fn)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_fn(tok_emb)
            _ = compiled_fn(tok_emb)
            _ = triton_fn(tok_emb)
            _ = cute_fn(tok_emb)

        out_pt = eager_fn(tok_emb).to(torch.bfloat16)
        out_comp = compiled_fn(tok_emb).to(torch.bfloat16)
        out_triton = triton_fn(tok_emb).to(torch.bfloat16)
        out_cute = cute_fn(tok_emb).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        err_cute = (out_cute - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        print_and_collect("Max Abs Error (Cute vs Eager):    %.6f", err_cute, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_cute, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_fn(tok_emb))
        ms_comp = triton.testing.do_bench(lambda: compiled_fn(tok_emb))
        ms_triton = triton.testing.do_bench(lambda: triton_fn(tok_emb))
        ms_cute = triton.testing.do_bench(lambda: cute_fn(tok_emb))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)
    print_and_collect("Cute Component:     %.4f ms", ms_cute, results=results)


# ---------------------------------------------------------------------------
# Sub-component benchmark: TimestepEmbedder
# ---------------------------------------------------------------------------
def benchmark_timestep_embedder(config, sigma, results):
    print_and_collect("\n=== TimestepEmbedder ===", results=results)
    pt_emb = model.TimestepEmbedder(config.cond_dim).to(DEVICE).eval()
    triton_emb = TritonTimestepEmbedder(config.cond_dim).to(DEVICE).eval()
    triton_emb.load_state_dict(pt_emb.state_dict(), strict=False)
    compiled_emb = torch.compile(pt_emb)

    with torch.no_grad():
        for _ in range(5):
            _ = pt_emb(sigma)
            _ = compiled_emb(sigma)
            _ = triton_emb(sigma)

        out_pt = pt_emb(sigma).to(torch.bfloat16)
        out_comp = compiled_emb(sigma).to(torch.bfloat16)
        out_triton = triton_emb(sigma).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: pt_emb(sigma))
        ms_comp = triton.testing.do_bench(lambda: compiled_emb(sigma))
        ms_triton = triton.testing.do_bench(lambda: triton_emb(sigma))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)


# ---------------------------------------------------------------------------
# Sub-component benchmark: Final Layer (ln_f + DDitFinalLayer fused)
# ---------------------------------------------------------------------------
class _EagerFinalLayer(nn.Module):
    def __init__(self, ln_f, final_layer):
        super().__init__()
        self.ln_f = ln_f
        self.final_layer = final_layer
    def forward(self, x, c):
        return self.final_layer(self.ln_f(x), c)

def benchmark_final_layer(config, x, c, results):
    print_and_collect("\n=== Final Layer (ln_f + DDitFinalLayer) ===", results=results)
    pt_model = GPT(config).to(DEVICE).eval()
    ln_f = pt_model.transformer.ln_f
    pt_final = model.DDitFinalLayer(config).to(DEVICE).eval()
    triton_final = TritonDDitFinalLayer(config).to(DEVICE).eval()
    triton_final.load_state_dict(pt_final.state_dict(), strict=False)

    eager_wrapper = _EagerFinalLayer(ln_f, pt_final).to(DEVICE).eval()
    compiled_wrapper = torch.compile(eager_wrapper)

    with torch.no_grad():
        for _ in range(5):
            _ = eager_wrapper(x, c)
            _ = compiled_wrapper(x, c)
            _ = triton_final(x, c, ln_f)

        out_pt = eager_wrapper(x, c).to(torch.bfloat16)
        out_comp = compiled_wrapper(x, c).to(torch.bfloat16)
        out_triton = triton_final(x, c, ln_f).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

    with torch.no_grad():
        ms_pt = triton.testing.do_bench(lambda: eager_wrapper(x, c))
        ms_comp = triton.testing.do_bench(lambda: compiled_wrapper(x, c))
        ms_triton = triton.testing.do_bench(lambda: triton_final(x, c, ln_f))

    print_and_collect("Eager Component:    %.4f ms", ms_pt, results=results)
    print_and_collect("Compiled Component: %.4f ms", ms_comp, results=results)
    print_and_collect("Triton Component:   %.4f ms", ms_triton, results=results)


# ---------------------------------------------------------------------------
# Full model benchmark
# ---------------------------------------------------------------------------
def benchmark_full_model(config, idx, sigma, results):
    print_and_collect("\n--- Benchmarking Full GPT Forward Pass ---", results=results)
    model_pt = GPT(config).to(DEVICE).eval()
    model_compiled = torch.compile(model_pt)
    model_triton = TritonGPT(config).to(DEVICE).eval()
    model_triton.load_state_dict(model_pt.state_dict(), strict=False)

    with torch.no_grad():
        for _ in range(5):
            _ = model_pt(idx, sigma)
            _ = model_compiled(idx, sigma)
            _ = model_triton(idx, sigma)

        out_pt = model_pt(idx, sigma).to(torch.bfloat16)
        out_comp = model_compiled(idx, sigma).to(torch.bfloat16)
        out_triton = model_triton(idx, sigma).to(torch.bfloat16)

        err_comp = (out_comp - out_pt).abs().max().item()
        err_triton = (out_triton - out_pt).abs().max().item()
        print_and_collect("Max Abs Error (Compile vs Eager): %.6f", err_comp, results=results)
        print_and_collect("Max Abs Error (Triton vs Eager):  %.6f", err_triton, results=results)
        torch.testing.assert_close(out_comp, out_pt, atol=5e-2, rtol=2e-2)
        torch.testing.assert_close(out_triton, out_pt, atol=5e-2, rtol=2e-2)
        print_and_collect("Correctness verified.", results=results)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        ms_full_pt = triton.testing.do_bench(lambda: model_pt(idx, sigma))
        ms_full_comp = triton.testing.do_bench(lambda: model_compiled(idx, sigma))
        ms_full_triton = triton.testing.do_bench(lambda: model_triton(idx, sigma))

    print_and_collect("Eager Model:    %.4f ms", ms_full_pt, results=results)
    print_and_collect("Compiled Model: %.4f ms", ms_full_comp, results=results)
    print_and_collect("Triton Model:   %.4f ms", ms_full_triton, results=results)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_component_benchmarks(config, x, c, freqs_cis, idx, sigma):
    results = []
    block_pt = benchmark_full_block(config, x, c, freqs_cis, results)

    # Generate intermediate tensors by running eager block once
    with torch.no_grad():
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block_pt.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x_skip1 = x
        x_mod = modulate(block_pt.ln_1(x), shift_msa, scale_msa)
        x_attn = block_pt.attn(x_mod, freqs_cis)
        x_post_attn = gate_msa * x_attn + x_skip1
        x_skip2 = x_post_attn
        x_mod_2 = modulate(block_pt.ln_2(x_post_attn), shift_mlp, scale_mlp)
        x_mlp_hidden = block_pt.mlp.gelu(block_pt.mlp.c_fc(x_mod_2))

    benchmark_pre_attn_norm(config, x, shift_msa, scale_msa, block_pt, results)
    benchmark_attention(config, x_mod, freqs_cis, block_pt, results)
    benchmark_post_attn_gate_norm(config, x_attn, gate_msa, x_skip1, shift_mlp, scale_mlp, block_pt, results)
    benchmark_mlp_up(config, x_mod_2, block_pt, results)
    benchmark_mlp_down(config, x_mlp_hidden, gate_mlp, x_skip2, block_pt, results)
    benchmark_full_mlp(config, x_mod_2, gate_mlp, x_skip2, block_pt, results)

    # Generate tok_emb for input-conv benchmark
    pt_model_tmp = GPT(config).to(DEVICE).eval()
    with torch.no_grad():
        tok_emb = pt_model_tmp.transformer.wte(idx)
    benchmark_input_convs(config, tok_emb, results)

    benchmark_timestep_embedder(config, sigma, results)
    benchmark_final_layer(config, x, c, results)
    benchmark_full_model(config, idx, sigma, results)

    return "\n".join(results)

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
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        sigma = torch.rand(batch_size, device=device)

        benchmark_text = run_component_benchmarks(config, x, c, freqs_cis, idx, sigma)

    # Log to worklog
    note = f"Full-stack benchmark for norm={config.norm}, n_embd={config.n_embd}, batch={batch_size}, seq={seq_len}"
    append_to_worklog(note, benchmark_text)
    print(f"\nBenchmark results appended to {WORKLOG_PATH}")

if __name__ == "__main__":
    run_benchmark()
