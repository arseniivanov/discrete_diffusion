import torch
import triton
import triton.language as tl

def cdiv(x, y):
    return -(-x // y)

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _fused_mlp_full_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr, gate_ptr, skip_ptr,
    M, N, K_in, K_mid, T,
    stride_xm, stride_xk,
    stride_w1k, stride_w1m, # w1 is [K_in, K_mid]
    stride_w2m, stride_w2n, # w2 is [K_mid, N]
    stride_om, stride_on,
    stride_gm, stride_gn,
    stride_sm, stride_sn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K_IN: tl.constexpr, BLOCK_SIZE_K_MID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Standard tiling logic...
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator for the FINAL output
    final_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the "upscale" intermediate dimension (K_mid)
    for k_mid_idx in range(0, tl.cdiv(K_mid, BLOCK_SIZE_K_MID)):
        offs_k_mid = k_mid_idx * BLOCK_SIZE_K_MID + tl.arange(0, BLOCK_SIZE_K_MID)
        
        # Calculate the intermediate GELU(x @ w1) block for this k_mid chunk
        # This is essentially the "Q" block from image_9415c8.png
        intermediate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K_MID), dtype=tl.float32)
        
        for k_in_idx in range(0, tl.cdiv(K_in, BLOCK_SIZE_K_IN)):
            offs_k_in = k_in_idx * BLOCK_SIZE_K_IN + tl.arange(0, BLOCK_SIZE_K_IN)
            
            x_tile = tl.load(x_ptr + (offs_m[:, None] * stride_xm + offs_k_in[None, :] * stride_xk))
            w1_tile = tl.load(w1_ptr + (offs_k_in[:, None] * stride_w1k + offs_k_mid[None, :] * stride_w1m)).to(tl.bfloat16)
            intermediate_acc += tl.dot(x_tile, w1_tile)

        # Apply GELU to the intermediate tile
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        i_x = intermediate_acc
        intermediate_gelu = 0.5 * i_x * (1.0 + tanh(0.79788456 * (i_x + 0.044715 * i_x * i_x * i_x)))
        intermediate_gelu = intermediate_gelu.to(tl.bfloat16)

        # Now multiply this intermediate tile by the corresponding block of w2
        w2_tile = tl.load(w2_ptr + (offs_k_mid[:, None] * stride_w2m + offs_n[None, :] * stride_w2n)).to(tl.bfloat16)
        final_accumulator += tl.dot(intermediate_gelu, w2_tile)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Map global M index back to Batch index for the gate [B, N]
    batch_idx = offs_m // T
    
    gate_ptrs = gate_ptr + batch_idx[:, None] * stride_gm + offs_n[None, :] * stride_gn
    skip_ptrs = skip_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    skip = tl.load(skip_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Final Epilogue: gate * (xW1W2) + skip
    res = (gate * final_accumulator) + skip
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, res.to(tl.bfloat16), mask=mask)

def fused_mlp_full(x, w1, w2, gate, skip, T):
    B, _, C_in = x.shape
    C_mid = w1.shape[0] # w1 is [4C, C]
    C_out = w2.shape[0] # w2 is [C, 4C]
    M, N = B * T, C_out

    x_flat = x.view(M, C_in).contiguous()
    skip_flat = skip.view(M, N).contiguous()
    gate_flat = gate.squeeze(1).contiguous() # [B, N]
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (cdiv(M, META['BLOCK_SIZE_M']) * cdiv(N, META['BLOCK_SIZE_N']), )
    _fused_mlp_full_kernel[grid](
        x_flat, w1, w2, out_flat, gate_flat, skip_flat,
        M, N, C_in, C_mid, T,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(1), w1.stride(0), # Pass strides to treat as W1.T
        w2.stride(1), w2.stride(0), # Pass strides to treat as W2.T
        out_flat.stride(0), out_flat.stride(1),
        gate_flat.stride(0), gate_flat.stride(1),
        skip_flat.stride(0), skip_flat.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, 
        BLOCK_SIZE_K_IN=32, BLOCK_SIZE_K_MID=32,
    )
    return out_flat.view(B, T, N)

"""
class TritonDDiTBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    @torch.compile
    def forward(self, x, c, freqs_cis: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x_mod = fused_modulate_layernorm(x, shift_msa, scale_msa)
        x_attn = self.attn(x_mod, freqs_cis)

        x_mod_2, x_skip = fused_gate_modulate_layernorm(x_attn, gate_msa, x_skip, shift_mlp, scale_mlp)

        x_out = fused_mlp_full(
            x_mod_2, 
            self.mlp.c_fc.weight, 
            self.mlp.c_proj.weight, 
            gate_mlp, 
            x_skip,
            T=x.shape[1]
        )
        return x_out
"""
