import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from model import SelfAttention, MLP, get_norm, precompute_freqs_cis, DynTanh

def cdiv(x, y):
    return -(-x // y)

@triton.jit
def _fused_modulate_layernorm_kernel(
    x_ptr, shift_ptr, scale_ptr, out_ptr,
    n_cols, T, eps,
    stride_x_row, stride_x_col,
    stride_shift_row, stride_shift_col,
    stride_scale_row, stride_scale_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // T  # Map flat index to batch index
    
    x_row_ptr = x_ptr + row_idx * stride_x_row
    shift_row_ptr = shift_ptr + batch_idx * stride_shift_row
    scale_row_ptr = scale_ptr + batch_idx * stride_scale_row
    out_row_ptr = out_ptr + row_idx * stride_out_row

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _sq_sum += x * x
        
    mean = tl.sum(_sum, axis=0) / n_cols
    sq_mean = tl.sum(_sq_sum, axis=0) / n_cols
    var = sq_mean - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)

    # --- PASS 3: Normalize, Modulate, and Store ---
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # Load the raw data and our dynamic AdaLN conditioning
        x = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_row_ptr + cols * stride_scale_col, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_row_ptr + cols * stride_shift_col, mask=mask, other=0.0).to(tl.float32)
        
        out = (x - mean) * rstd * (1.0 + scale) + shift
        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def fused_modulate_layernorm(x, shift, scale, eps=1e-5):
    B, T, C = x.shape
    x_flat = x.view(-1, C)
    shift_flat = shift.squeeze(1) # Yields [B, C]
    scale_flat = scale.squeeze(1) # Yields [B, C]
    
    out = torch.empty_like(x_flat)
    
    n_rows = x_flat.shape[0]
    n_cols = x_flat.shape[1]
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    grid = (n_rows,)

    _fused_modulate_layernorm_kernel[grid](
        x_flat, shift_flat, scale_flat, out,
        n_cols, T, eps,
        x_flat.stride(0), x_flat.stride(1),
        shift_flat.stride(0), shift_flat.stride(1),
        scale_flat.stride(0), scale_flat.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(B, T, C)

@triton.jit
def _fused_gate_modulate_layernorm_kernel(
    attn_ptr, gate_ptr, skip_ptr, shift_ptr, scale_ptr,
    out_ptr, new_skip_ptr,  # We MUST output the updated skip connection
    n_cols, T, eps,
    stride_attn_row, stride_attn_col,
    stride_gate_row, stride_gate_col,
    stride_skip_row, stride_skip_col,
    stride_shift_row, stride_shift_col,
    stride_scale_row, stride_scale_col,
    stride_out_row, stride_out_col,
    stride_new_skip_row, stride_new_skip_col,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // T

    attn_row_ptr = attn_ptr + row_idx * stride_attn_row
    skip_row_ptr = skip_ptr + row_idx * stride_skip_row
    new_skip_row_ptr = new_skip_ptr + row_idx * stride_new_skip_row
    
    gate_row_ptr = gate_ptr + batch_idx * stride_gate_row
    shift_row_ptr = shift_ptr + batch_idx * stride_shift_row
    scale_row_ptr = scale_ptr + batch_idx * stride_scale_row
    out_row_ptr = out_ptr + row_idx * stride_out_row

    # --- PASS 1: Compute New Skip, Store it, and Accumulate Mean ---
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        attn = tl.load(attn_row_ptr + cols * stride_attn_col, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_row_ptr + cols * stride_gate_col, mask=mask, other=0.0).to(tl.float32)
        skip = tl.load(skip_row_ptr + cols * stride_skip_col, mask=mask, other=0.0).to(tl.float32)

        # Gate the attention output and add the residual
        new_skip = attn * gate + skip
        
        # We must store this to HBM because the MLP block needs it later
        tl.store(new_skip_row_ptr + cols * stride_new_skip_col, new_skip, mask=mask)

        _mean += new_skip

    mean = tl.sum(_mean, axis=0) / n_cols

    # --- PASS 2: Compute Variance from the stored New Skip ---
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        new_skip = tl.load(new_skip_row_ptr + cols * stride_new_skip_col, mask=mask, other=0.0).to(tl.float32)
        centered = tl.where(mask, new_skip - mean, 0.0)
        _var += centered * centered

    var = tl.sum(_var, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    # --- PASS 3: Normalize, Modulate, and Store ---

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        new_skip = tl.load(new_skip_row_ptr + cols * stride_new_skip_col, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_row_ptr + cols * stride_scale_col, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_row_ptr + cols * stride_shift_col, mask=mask, other=0.0).to(tl.float32)

        out = (new_skip - mean) * rstd * (1.0 + scale) + shift
        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def fused_gate_modulate_layernorm(x_attn, gate_msa, x_skip, shift_mlp, scale_mlp, eps=1e-5):
    # Enforce contiguity. Do not skip this.
    x_attn = x_attn.contiguous()
    gate_msa = gate_msa.contiguous()
    x_skip = x_skip.contiguous()
    shift_mlp = shift_mlp.contiguous()
    scale_mlp = scale_mlp.contiguous()

    B, T, C = x_attn.shape
    attn_flat = x_attn.view(-1, C)
    skip_flat = x_skip.view(-1, C)
    
    gate_flat = gate_msa.squeeze(1)
    shift_flat = shift_mlp.squeeze(1)
    scale_flat = scale_mlp.squeeze(1)
    
    out = torch.empty_like(attn_flat)
    new_skip = torch.empty_like(skip_flat)
    
    n_rows = attn_flat.shape[0]
    n_cols = attn_flat.shape[1]
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    _fused_gate_modulate_layernorm_kernel[(n_rows,)](
        attn_flat, gate_flat, skip_flat, shift_flat, scale_flat,
        out, new_skip,
        n_cols, T, eps,
        attn_flat.stride(0), attn_flat.stride(1),
        gate_flat.stride(0), gate_flat.stride(1),
        skip_flat.stride(0), skip_flat.stride(1),
        shift_flat.stride(0), shift_flat.stride(1),
        scale_flat.stride(0), scale_flat.stride(1),
        out.stride(0), out.stride(1),
        new_skip.stride(0), new_skip.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(B, T, C), new_skip.view(B, T, C)

@triton.jit
def _fused_mlp_proj_epilogue_kernel(
    a_ptr, b_ptr, c_ptr, gate_ptr, skip_ptr,
    M, N, K, T,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_gm, stride_gn,
    stride_sm, stride_sn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.bfloat16)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    batch_idx = offs_m // T
    
    gate_ptrs = gate_ptr + batch_idx[:, None] * stride_gm + offs_n[None, :] * stride_gn
    skip_ptrs = skip_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.bfloat16)
    skip = tl.load(skip_ptrs, mask=mask, other=0.0).to(tl.bfloat16)
    
    c = (gate * c) + skip
    
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c, mask=mask)

def fused_mlp_proj_epilogue(x, weight, gate, skip):
    # Flatten everything to 2D for the blocked M/N matmul
    x = x.contiguous()
    gate = gate.contiguous()
    skip = skip.contiguous()
    
    B, T, C_in = x.shape
    C_out = weight.shape[0]
    M, K = B * T, C_in
    N = C_out

    x_flat = x.view(M, K)
    skip_flat = skip.view(M, N)
    gate_flat = gate.squeeze(1) # Yields [B, N]
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (cdiv(M, META['BLOCK_SIZE_M']) * cdiv(N, META['BLOCK_SIZE_N']), )
    _fused_mlp_proj_epilogue_kernel[grid](
        x_flat, weight.t(), out_flat, gate_flat, skip_flat,
        M, N, K, T,
        x_flat.stride(0), x_flat.stride(1),
        weight.t().stride(0), weight.t().stride(1),
        out_flat.stride(0), out_flat.stride(1),
        gate_flat.stride(0), gate_flat.stride(1),
        skip_flat.stride(0), skip_flat.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    return out_flat.view(B, T, N)


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

        x_mlp_hidden = fused_linear_gelu(x_mod_2, self.mlp.c_fc.weight, self.mlp.c_fc.bias)
        x_out = fused_mlp_proj_epilogue(x_mlp_hidden, self.mlp.c_proj.weight, gate_mlp, x_skip)
        return x_out

@triton.jit
def _fused_modulate_dyntanh_kernel(
    x_ptr, shift_ptr, scale_ptr,
    alpha_ptr, gamma_ptr, beta_ptr,
    out_ptr,
    n_cols, T,
    stride_x_row, stride_x_col,
    stride_shift_row, stride_shift_col,
    stride_scale_row, stride_scale_col,
    stride_out_row, stride_out_col,
    HAS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // T

    x_row_ptr = x_ptr + row_idx * stride_x_row
    shift_row_ptr = shift_ptr + batch_idx * stride_shift_row
    scale_row_ptr = scale_ptr + batch_idx * stride_scale_row
    out_row_ptr = out_ptr + row_idx * stride_out_row

    alpha = tl.load(alpha_ptr)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            beta = 0.0

        scale = tl.load(scale_row_ptr + cols * stride_scale_col, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_row_ptr + cols * stride_shift_col, mask=mask, other=0.0).to(tl.float32)

        dt = tanh(x * alpha) * gamma + beta
        out = dt * (1.0 + scale) + shift

        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def fused_modulate_dyntanh(x, shift, scale, alpha, gamma, beta=None):
    B, T, C = x.shape
    x_flat = x.view(-1, C)
    shift_flat = shift.squeeze(1)
    scale_flat = scale.squeeze(1)

    out = torch.empty_like(x_flat)

    n_rows = x_flat.shape[0]
    n_cols = x_flat.shape[1]
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    grid = (n_rows,)

    _fused_modulate_dyntanh_kernel[grid](
        x_flat, shift_flat, scale_flat,
        alpha, gamma, beta,
        out,
        n_cols, T,
        x_flat.stride(0), x_flat.stride(1),
        shift_flat.stride(0), shift_flat.stride(1),
        scale_flat.stride(0), scale_flat.stride(1),
        out.stride(0), out.stride(1),
        HAS_BETA=(beta is not None),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(B, T, C)

@triton.jit
def _fused_gate_modulate_dyntanh_kernel(
    attn_ptr, gate_ptr, skip_ptr, shift_ptr, scale_ptr,
    alpha_ptr, gamma_ptr, beta_ptr,
    out_ptr, new_skip_ptr,
    n_cols, T,
    stride_attn_row, stride_attn_col,
    stride_gate_row, stride_gate_col,
    stride_skip_row, stride_skip_col,
    stride_shift_row, stride_shift_col,
    stride_scale_row, stride_scale_col,
    stride_out_row, stride_out_col,
    stride_new_skip_row, stride_new_skip_col,
    HAS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // T

    attn_row_ptr = attn_ptr + row_idx * stride_attn_row
    skip_row_ptr = skip_ptr + row_idx * stride_skip_row
    new_skip_row_ptr = new_skip_ptr + row_idx * stride_new_skip_row

    gate_row_ptr = gate_ptr + batch_idx * stride_gate_row
    shift_row_ptr = shift_ptr + batch_idx * stride_shift_row
    scale_row_ptr = scale_ptr + batch_idx * stride_scale_row
    out_row_ptr = out_ptr + row_idx * stride_out_row

    alpha = tl.load(alpha_ptr)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        attn = tl.load(attn_row_ptr + cols * stride_attn_col, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_row_ptr + cols * stride_gate_col, mask=mask, other=0.0).to(tl.float32)
        skip = tl.load(skip_row_ptr + cols * stride_skip_col, mask=mask, other=0.0).to(tl.float32)

        new_skip = attn * gate + skip
        tl.store(new_skip_row_ptr + cols * stride_new_skip_col, new_skip, mask=mask)

        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            beta = 0.0

        scale = tl.load(scale_row_ptr + cols * stride_scale_col, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_row_ptr + cols * stride_shift_col, mask=mask, other=0.0).to(tl.float32)

        dt = tanh(new_skip * alpha) * gamma + beta
        out = dt * (1.0 + scale) + shift

        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def fused_gate_modulate_dyntanh(x_attn, gate_msa, x_skip, shift_mlp, scale_mlp, alpha, gamma, beta=None):
    x_attn = x_attn.contiguous()
    gate_msa = gate_msa.contiguous()
    x_skip = x_skip.contiguous()
    shift_mlp = shift_mlp.contiguous()
    scale_mlp = scale_mlp.contiguous()

    B, T, C = x_attn.shape
    attn_flat = x_attn.view(-1, C)
    skip_flat = x_skip.view(-1, C)

    gate_flat = gate_msa.squeeze(1)
    shift_flat = shift_mlp.squeeze(1)
    scale_flat = scale_mlp.squeeze(1)

    out = torch.empty_like(attn_flat)
    new_skip = torch.empty_like(skip_flat)

    n_rows = attn_flat.shape[0]
    n_cols = attn_flat.shape[1]

    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    _fused_gate_modulate_dyntanh_kernel[(n_rows,)](
        attn_flat, gate_flat, skip_flat, shift_flat, scale_flat,
        alpha, gamma, beta,
        out, new_skip,
        n_cols, T,
        attn_flat.stride(0), attn_flat.stride(1),
        gate_flat.stride(0), gate_flat.stride(1),
        skip_flat.stride(0), skip_flat.stride(1),
        shift_flat.stride(0), shift_flat.stride(1),
        scale_flat.stride(0), scale_flat.stride(1),
        out.stride(0), out.stride(1),
        new_skip.stride(0), new_skip.stride(1),
        HAS_BETA=(beta is not None),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(B, T, C), new_skip.view(B, T, C)

@triton.jit
def _fused_linear_gelu_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.bfloat16)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        c += bias[None, :]

    # In-register exact GELU (Matches nn.GELU default)
    c = c * 0.5 * (1.0 + tl.math.erf(c * 0.707106781))
    
    c = c.to(tl.bfloat16)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c, mask=mask)

def fused_linear_gelu(x, weight, bias=None):
    x = x.contiguous()
    B, T, C_in = x.shape
    C_out = weight.shape[0]
    M, K = B * T, C_in
    N = C_out

    x_flat = x.view(M, K)
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (cdiv(M, META['BLOCK_SIZE_M']) * cdiv(N, META['BLOCK_SIZE_N']), )
    _fused_linear_gelu_kernel[grid](
        x_flat, weight.t(), out_flat, bias,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        weight.t().stride(0), weight.t().stride(1),
        out_flat.stride(0), out_flat.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    return out_flat.view(B, T, N)

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

class TritonDDiTDynTanh(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

        self.dt1_alpha = nn.Parameter(torch.ones(1))
        self.dt1_gamma = nn.Parameter(torch.ones(config.n_embd))
        if config.bias:
            self.dt1_beta = nn.Parameter(torch.zeros(config.n_embd))
        else:
            self.register_parameter('dt1_beta', None)

        self.dt2_alpha = nn.Parameter(torch.ones(1))
        self.dt2_gamma = nn.Parameter(torch.ones(config.n_embd))
        if config.bias:
            self.dt2_beta = nn.Parameter(torch.zeros(config.n_embd))
        else:
            self.register_parameter('dt2_beta', None)

    @torch.compile
    def forward(self, x, c, freqs_cis: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x_mod = fused_modulate_dyntanh(
            x, shift_msa, scale_msa,
            self.dt1_alpha, self.dt1_gamma, self.dt1_beta
        )
        x_attn = self.attn(x_mod, freqs_cis)

        x_mod_2, x_skip = fused_gate_modulate_dyntanh(
            x_attn, gate_msa, x_skip, shift_mlp, scale_mlp,
            self.dt2_alpha, self.dt2_gamma, self.dt2_beta
        )

        x_mlp_hidden = fused_linear_gelu(x_mod_2, self.mlp.c_fc.weight, self.mlp.c_fc.bias)
        x_out = fused_mlp_proj_epilogue(x_mlp_hidden, self.mlp.c_proj.weight, gate_mlp, x_skip)
        return x_out


# ---------------------------------------------------------------------------
# Standalone LayerNorm (no modulation)
# ---------------------------------------------------------------------------
@triton.jit
def _triton_layernorm_kernel(
    x_ptr, out_ptr,
    n_cols,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_out_row

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _sq_sum += x * x

    mean = tl.sum(_sum, axis=0) / n_cols
    sq_mean = tl.sum(_sq_sum, axis=0) / n_cols
    var = sq_mean - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd
        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def triton_layernorm(x, eps=1e-5):
    B, T, C = x.shape
    x_flat = x.view(-1, C)
    out = torch.empty_like(x_flat)
    n_rows = x_flat.shape[0]
    n_cols = x_flat.shape[1]
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    grid = (n_rows,)
    _triton_layernorm_kernel[grid](
        x_flat, out,
        n_cols,
        x_flat.stride(0), x_flat.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(B, T, C)


# ---------------------------------------------------------------------------
# Standalone DynTanh (no modulation)
# ---------------------------------------------------------------------------
@triton.jit
def _triton_dyntanh_kernel(
    x_ptr, out_ptr,
    alpha_ptr, gamma_ptr, beta_ptr,
    n_cols,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    HAS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_out_row
    alpha = tl.load(alpha_ptr)

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            beta = 0.0
        out = tanh(x * alpha) * gamma + beta
        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def triton_dyntanh(x, alpha, gamma, beta=None):
    B, T, C = x.shape
    x_flat = x.view(-1, C)
    out = torch.empty_like(x_flat)
    n_rows = x_flat.shape[0]
    n_cols = x_flat.shape[1]
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    grid = (n_rows,)
    _triton_dyntanh_kernel[grid](
        x_flat, out,
        alpha, gamma, beta,
        n_cols,
        x_flat.stride(0), x_flat.stride(1),
        out.stride(0), out.stride(1),
        HAS_BETA=(beta is not None),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(B, T, C)


# ---------------------------------------------------------------------------
# Fully fused MLP: c_fc + GELU + c_proj + gate + skip
# ---------------------------------------------------------------------------
@triton.jit
def _fused_mlp_full_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr, gate_ptr, skip_ptr,
    M, N, K_in, K_mid, T,
    stride_xm, stride_xk,
    stride_w1k, stride_w1m,
    stride_w2m, stride_w2n,
    stride_om, stride_on,
    stride_gm, stride_gn,
    stride_sm, stride_sn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_IN: tl.constexpr, BLOCK_SIZE_K_MID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    final_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_mid_idx in range(0, tl.cdiv(K_mid, BLOCK_SIZE_K_MID)):
        offs_k_mid = k_mid_idx * BLOCK_SIZE_K_MID + tl.arange(0, BLOCK_SIZE_K_MID)
        intermediate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K_MID), dtype=tl.float32)

        for k_in_idx in range(0, tl.cdiv(K_in, BLOCK_SIZE_K_IN)):
            offs_k_in = k_in_idx * BLOCK_SIZE_K_IN + tl.arange(0, BLOCK_SIZE_K_IN)
            x_tile = tl.load(x_ptr + (offs_m[:, None] * stride_xm + offs_k_in[None, :] * stride_xk),
                             mask=(offs_m[:, None] < M) & (offs_k_in[None, :] < K_in), other=0.0).to(tl.bfloat16)
            w1_tile = tl.load(w1_ptr + (offs_k_in[:, None] * stride_w1k + offs_k_mid[None, :] * stride_w1m),
                              mask=(offs_k_in[:, None] < K_in) & (offs_k_mid[None, :] < K_mid), other=0.0).to(tl.bfloat16)
            intermediate_acc += tl.dot(x_tile, w1_tile)

        i_x = intermediate_acc
        intermediate_gelu = 0.5 * i_x * (1.0 + tanh(0.79788456 * (i_x + 0.044715 * i_x * i_x * i_x)))
        intermediate_gelu = intermediate_gelu.to(tl.bfloat16)

        w2_tile = tl.load(w2_ptr + (offs_k_mid[:, None] * stride_w2m + offs_n[None, :] * stride_w2n),
                          mask=(offs_k_mid[:, None] < K_mid) & (offs_n[None, :] < N), other=0.0).to(tl.bfloat16)
        final_accumulator += tl.dot(intermediate_gelu, w2_tile)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    batch_idx = offs_m // T

    gate_ptrs = gate_ptr + batch_idx[:, None] * stride_gm + offs_n[None, :] * stride_gn
    skip_ptrs = skip_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    skip = tl.load(skip_ptrs, mask=mask, other=0.0).to(tl.float32)
    res = (gate * final_accumulator) + skip

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, res.to(tl.bfloat16), mask=mask)

def fused_mlp_full(x, w1, w2, gate, skip, T):
    B, _, C_in = x.shape
    C_mid = w1.shape[0]
    C_out = w2.shape[0]
    M, N = B * T, C_out

    x_flat = x.view(M, C_in).contiguous()
    skip_flat = skip.view(M, N).contiguous()
    gate_flat = gate.squeeze(1).contiguous()
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (cdiv(M, META['BLOCK_SIZE_M']) * cdiv(N, META['BLOCK_SIZE_N']), )
    _fused_mlp_full_kernel[grid](
        x_flat, w1, w2, out_flat, gate_flat, skip_flat,
        M, N, C_in, C_mid, T,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(1), w1.stride(0),
        w2.stride(1), w2.stride(0),
        out_flat.stride(0), out_flat.stride(1),
        gate_flat.stride(0), gate_flat.stride(1),
        skip_flat.stride(0), skip_flat.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128,
        BLOCK_SIZE_K_IN=32, BLOCK_SIZE_K_MID=32,
    )
    return out_flat.view(B, T, N)


# ---------------------------------------------------------------------------
# Fused MLP variant wrappers
# ---------------------------------------------------------------------------
class TritonDDiTBlockFusedMLP(nn.Module):
    """Variant that fuses the entire MLP (c_fc + gelu + c_proj + gate + skip)."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

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
            gate_mlp, x_skip,
            T=x.shape[1]
        )
        return x_out

class TritonDDiTDynTanhFusedMLP(nn.Module):
    """DynTanh variant that fuses the entire MLP."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

        self.dt1_alpha = nn.Parameter(torch.ones(1))
        self.dt1_gamma = nn.Parameter(torch.ones(config.n_embd))
        if config.bias:
            self.dt1_beta = nn.Parameter(torch.zeros(config.n_embd))
        else:
            self.register_parameter('dt1_beta', None)

        self.dt2_alpha = nn.Parameter(torch.ones(1))
        self.dt2_gamma = nn.Parameter(torch.ones(config.n_embd))
        if config.bias:
            self.dt2_beta = nn.Parameter(torch.zeros(config.n_embd))
        else:
            self.register_parameter('dt2_beta', None)

    def forward(self, x, c, freqs_cis: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x_mod = fused_modulate_dyntanh(
            x, shift_msa, scale_msa,
            self.dt1_alpha, self.dt1_gamma, self.dt1_beta
        )
        x_attn = self.attn(x_mod, freqs_cis)

        x_mod_2, x_skip = fused_gate_modulate_dyntanh(
            x_attn, gate_msa, x_skip, shift_mlp, scale_mlp,
            self.dt2_alpha, self.dt2_gamma, self.dt2_beta
        )

        x_out = fused_mlp_full(
            x_mod_2,
            self.mlp.c_fc.weight,
            self.mlp.c_proj.weight,
            gate_mlp, x_skip,
            T=x.shape[1]
        )
        return x_out


# ---------------------------------------------------------------------------
# Fused Depthwise Input Convs (local_conv + SiLU + local_conv2 + residuals)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_depthwise_convs_kernel(
    x_ptr, out_ptr,
    w1_ptr, b1_ptr,
    w2_ptr, b2_ptr,
    B, T, C,
    stride_xb, stride_xt, stride_xc,
    stride_outb, stride_outt, stride_outc,
    BLOCK_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    c_block = tl.program_id(2)

    c_start = c_block * BLOCK_C
    cs = c_start + tl.arange(0, BLOCK_C)
    mask_c = cs < C

    x_batch_ptr = x_ptr + batch_idx * stride_xb
    out_batch_ptr = out_ptr + batch_idx * stride_outb

    # Load x for 5 positions (t-2 .. t+2)
    t_m2 = t_idx - 2
    t_m1 = t_idx - 1
    t_p1 = t_idx + 1
    t_p2 = t_idx + 2

    x_m2 = tl.load(x_batch_ptr + t_m2 * stride_xt + cs * stride_xc,
                   mask=mask_c & (t_m2 >= 0) & (t_m2 < T), other=0.0).to(tl.float32)
    x_m1 = tl.load(x_batch_ptr + t_m1 * stride_xt + cs * stride_xc,
                   mask=mask_c & (t_m1 >= 0) & (t_m1 < T), other=0.0).to(tl.float32)
    x_0  = tl.load(x_batch_ptr + t_idx  * stride_xt + cs * stride_xc,
                   mask=mask_c, other=0.0).to(tl.float32)
    x_p1 = tl.load(x_batch_ptr + t_p1 * stride_xt + cs * stride_xc,
                   mask=mask_c & (t_p1 >= 0) & (t_p1 < T), other=0.0).to(tl.float32)
    x_p2 = tl.load(x_batch_ptr + t_p2 * stride_xt + cs * stride_xc,
                   mask=mask_c & (t_p2 >= 0) & (t_p2 < T), other=0.0).to(tl.float32)

    # Load conv1 weights [C, 3] and bias
    w1_0 = tl.load(w1_ptr + cs * 3 + 0, mask=mask_c, other=0.0).to(tl.float32)
    w1_1 = tl.load(w1_ptr + cs * 3 + 1, mask=mask_c, other=0.0).to(tl.float32)
    w1_2 = tl.load(w1_ptr + cs * 3 + 2, mask=mask_c, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + cs, mask=mask_c, other=0.0).to(tl.float32)

    # Conv1 at t-1, t, t+1
    conv1_m1 = x_m2 * w1_0 + x_m1 * w1_1 + x_0  * w1_2 + b1
    conv1_0  = x_m1 * w1_0 + x_0  * w1_1 + x_p1 * w1_2 + b1
    conv1_p1 = x_0  * w1_0 + x_p1 * w1_1 + x_p2 * w1_2 + b1

    # Residual after conv1
    inter_m1 = x_m1 + conv1_m1
    inter_0  = x_0  + conv1_0
    inter_p1 = x_p1 + conv1_p1

    # SiLU
    silu_m1 = inter_m1 * tl.sigmoid(inter_m1)
    silu_0  = inter_0  * tl.sigmoid(inter_0)
    silu_p1 = inter_p1 * tl.sigmoid(inter_p1)

    # Load conv2 weights and bias
    w2_0 = tl.load(w2_ptr + cs * 3 + 0, mask=mask_c, other=0.0).to(tl.float32)
    w2_1 = tl.load(w2_ptr + cs * 3 + 1, mask=mask_c, other=0.0).to(tl.float32)
    w2_2 = tl.load(w2_ptr + cs * 3 + 2, mask=mask_c, other=0.0).to(tl.float32)
    b2 = tl.load(b2_ptr + cs, mask=mask_c, other=0.0).to(tl.float32)

    # Conv2 at t
    conv2_0 = silu_m1 * w2_0 + silu_0 * w2_1 + silu_p1 * w2_2 + b2

    # Final residual
    out_0 = inter_0 + conv2_0

    # Store
    out_ptr_loc = out_batch_ptr + t_idx * stride_outt + cs * stride_outc
    tl.store(out_ptr_loc, out_0, mask=mask_c)

def fused_input_convs(x, w1, b1, w2, b2):
    """
    Fuses:
        inter = x + depthwise_conv1(x)
        out   = inter + depthwise_conv2(SiLU(inter))
    w1, w2: [C, 1, 3] Conv1d weights
    b1, b2: [C] or None
    """
    B, T, C = x.shape
    out = torch.empty_like(x)

    w1_s = w1.squeeze(1)  # [C, 3]
    w2_s = w2.squeeze(1)  # [C, 3]

    if b1 is None:
        b1 = torch.zeros(C, device=x.device, dtype=torch.float32)
    if b2 is None:
        b2 = torch.zeros(C, device=x.device, dtype=torch.float32)

    BLOCK_C = 128
    grid = (B, T, cdiv(C, BLOCK_C))
    _fused_depthwise_convs_kernel[grid](
        x, out,
        w1_s, b1, w2_s, b2,
        B, T, C,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_C=BLOCK_C,
    )
    return out


# ---------------------------------------------------------------------------
# Triton Timestep Embedder
# ---------------------------------------------------------------------------
@triton.jit
def _timestep_embedding_kernel(
    t_ptr, out_ptr, freqs_ptr,
    batch_size, half_dim,
    stride_t, stride_out_b, stride_out_d,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    t_val = tl.load(t_ptr + batch_idx * stride_t).to(tl.float32)

    for off in range(0, half_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < half_dim
        freqs = tl.load(freqs_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        args = t_val * freqs
        cos_vals = tl.math.cos(args)
        sin_vals = tl.math.sin(args)
        tl.store(out_ptr + batch_idx * stride_out_b + cols * stride_out_d, cos_vals, mask=mask)
        tl.store(out_ptr + batch_idx * stride_out_b + (cols + half_dim) * stride_out_d, sin_vals, mask=mask)


def triton_timestep_embedding(t, freqs, dim):
    """
    t: [B] float tensor (already scaled)
    freqs: [half] float tensor
    dim: int, output dimension (must be even for the fast path)
    """
    B = t.shape[0]
    half = freqs.shape[0]
    out = torch.empty((B, dim), device=t.device, dtype=torch.float32)

    BLOCK_SIZE = min(triton.next_power_of_2(half), 1024)
    grid = (B,)
    _timestep_embedding_kernel[grid](
        t, out, freqs,
        B, half,
        t.stride(0), out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if dim % 2:
        out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1)
    return out


class TritonTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        max_period = 10000
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        t_freq = triton_timestep_embedding(t * 500.0, self.freqs, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ---------------------------------------------------------------------------
# Fully fused final layer kernels (ln_f + norm_final + modulate + linear)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_final_layer_dyntanh_kernel(
    x_ptr, out_ptr,
    ln_f_alpha_ptr, ln_f_gamma_ptr, ln_f_beta_ptr,
    nf_alpha_ptr, nf_gamma_ptr, nf_beta_ptr,
    shift_ptr, scale_ptr,
    w_ptr, b_ptr,
    M, N, K, T,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_shift_b, stride_shift_k,
    stride_scale_b, stride_scale_k,
    HAS_LN_F_BETA: tl.constexpr,
    HAS_NF_BETA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = x_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = w_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.bfloat16)

        batch_idx = offs_am // T

        # ln_f DynTanh
        ln_f_alpha = tl.load(ln_f_alpha_ptr)
        ln_f_gamma = tl.load(ln_f_gamma_ptr + offs_k, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        if HAS_LN_F_BETA:
            ln_f_beta = tl.load(ln_f_beta_ptr + offs_k, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        else:
            ln_f_beta = 0.0
        a = tanh(a * ln_f_alpha) * ln_f_gamma + ln_f_beta

        # norm_final DynTanh
        nf_alpha = tl.load(nf_alpha_ptr)
        nf_gamma = tl.load(nf_gamma_ptr + offs_k, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        if HAS_NF_BETA:
            nf_beta = tl.load(nf_beta_ptr + offs_k, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        else:
            nf_beta = 0.0
        a = tanh(a * nf_alpha) * nf_gamma + nf_beta

        # modulate
        shift = tl.load(shift_ptr + batch_idx[:, None] * stride_shift_b + offs_k[None, :] * stride_shift_k, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + batch_idx[:, None] * stride_scale_b + offs_k[None, :] * stride_scale_k, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        a = a * (1.0 + scale) + shift

        a = a.to(tl.bfloat16)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        c += bias[None, :]

    c = c.to(tl.bfloat16)
    c_ptrs = out_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _fused_final_layer_ln_kernel(
    x_ptr, out_ptr,
    ln_f_w_ptr, ln_f_b_ptr, ln_f_eps,
    nf_w_ptr, nf_b_ptr, nf_eps,
    shift_ptr, scale_ptr,
    w_ptr, b_ptr,
    B, T, C, V,
    stride_xb, stride_xt, stride_xc,
    stride_outb, stride_outt, stride_outv,
    stride_shift_b, stride_shift_c,
    stride_scale_b, stride_scale_c,
    HAS_LN_F_W: tl.constexpr,
    HAS_LN_F_B: tl.constexpr,
    HAS_NF_W: tl.constexpr,
    HAS_NF_B: tl.constexpr,
    HAS_LINEAR_BIAS: tl.constexpr,
    IS_RMS_LN_F: tl.constexpr,
    IS_RMS_NF: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    m_idx = tl.program_id(0)
    batch_idx = m_idx // T
    t_idx = m_idx % T

    # --- Pass 1: ln_f stats ---
    _sum1 = 0.0
    _sq1 = 0.0
    for c_off in range(0, C, BLOCK_C):
        cs = c_off + tl.arange(0, BLOCK_C)
        mask = cs < C
        x_c = tl.load(x_ptr + batch_idx * stride_xb + t_idx * stride_xt + cs * stride_xc,
                      mask=mask, other=0.0).to(tl.float32)
        _sum1 += tl.sum(x_c)
        _sq1 += tl.sum(x_c * x_c)

    if IS_RMS_LN_F:
        mean1 = 0.0
        rstd1 = 1.0 / tl.sqrt(_sq1 / C + ln_f_eps)
    else:
        mean1 = _sum1 / C
        var1 = _sq1 / C - mean1 * mean1
        rstd1 = 1.0 / tl.sqrt(var1 + ln_f_eps)

    # --- Pass 2: norm_final stats ---
    _sum2 = 0.0
    _sq2 = 0.0
    for c_off in range(0, C, BLOCK_C):
        cs = c_off + tl.arange(0, BLOCK_C)
        mask = cs < C
        x_c = tl.load(x_ptr + batch_idx * stride_xb + t_idx * stride_xt + cs * stride_xc,
                      mask=mask, other=0.0).to(tl.float32)

        if IS_RMS_LN_F:
            x_c = x_c * rstd1
        else:
            x_c = (x_c - mean1) * rstd1
        if HAS_LN_F_W:
            w = tl.load(ln_f_w_ptr + cs, mask=mask, other=0.0).to(tl.float32)
            x_c = x_c * w
        if HAS_LN_F_B:
            b = tl.load(ln_f_b_ptr + cs, mask=mask, other=0.0).to(tl.float32)
            x_c = x_c + b

        _sum2 += tl.sum(x_c)
        _sq2 += tl.sum(x_c * x_c)

    if IS_RMS_NF:
        mean2 = 0.0
        rstd2 = 1.0 / tl.sqrt(_sq2 / C + nf_eps)
    else:
        mean2 = _sum2 / C
        var2 = _sq2 / C - mean2 * mean2
        rstd2 = 1.0 / tl.sqrt(var2 + nf_eps)

    # --- Pass 3: compute linear ---
    for v_off in range(0, V, BLOCK_V):
        vs = v_off + tl.arange(0, BLOCK_V)
        v_mask = vs < V
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for c_off in range(0, C, BLOCK_C):
            cs = c_off + tl.arange(0, BLOCK_C)
            mask = cs < C

            x_c = tl.load(x_ptr + batch_idx * stride_xb + t_idx * stride_xt + cs * stride_xc,
                          mask=mask, other=0.0).to(tl.float32)

            # ln_f
            if IS_RMS_LN_F:
                x_c = x_c * rstd1
            else:
                x_c = (x_c - mean1) * rstd1
            if HAS_LN_F_W:
                w = tl.load(ln_f_w_ptr + cs, mask=mask, other=0.0).to(tl.float32)
                x_c = x_c * w
            if HAS_LN_F_B:
                b = tl.load(ln_f_b_ptr + cs, mask=mask, other=0.0).to(tl.float32)
                x_c = x_c + b

            # norm_final
            if IS_RMS_NF:
                x_c = x_c * rstd2
            else:
                x_c = (x_c - mean2) * rstd2
            if HAS_NF_W:
                w = tl.load(nf_w_ptr + cs, mask=mask, other=0.0).to(tl.float32)
                x_c = x_c * w
            if HAS_NF_B:
                b = tl.load(nf_b_ptr + cs, mask=mask, other=0.0).to(tl.float32)
                x_c = x_c + b

            # modulate
            shift = tl.load(shift_ptr + batch_idx * stride_shift_b + cs * stride_shift_c,
                            mask=mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + batch_idx * stride_scale_b + cs * stride_scale_c,
                            mask=mask, other=0.0).to(tl.float32)
            x_c = x_c * (1.0 + scale) + shift

            # linear
            w_c = tl.load(w_ptr + vs[:, None] * C + cs[None, :],
                          mask=v_mask[:, None] & mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(x_c[None, :] * w_c, axis=1)

        if HAS_LINEAR_BIAS:
            b_v = tl.load(b_ptr + vs, mask=v_mask, other=0.0).to(tl.float32)
            acc += b_v

        tl.store(out_ptr + batch_idx * stride_outb + t_idx * stride_outt + vs * stride_outv,
                 acc, mask=v_mask)


def fused_final_layer(x, shift, scale, ln_f, norm_final, linear):
    """Fused ln_f + norm_final + modulate + linear in a single Triton kernel."""
    B, T, C = x.shape
    V = linear.weight.shape[0]
    out = torch.empty((B, T, V), device=x.device, dtype=torch.bfloat16)

    shift_flat = shift.squeeze(1)
    scale_flat = scale.squeeze(1)

    BLOCK_C = min(triton.next_power_of_2(C), 1024)
    BLOCK_V = min(triton.next_power_of_2(V), 1024)
    grid = (B * T,)

    if isinstance(ln_f, DynTanh):
        M, K = B * T, C
        N = V
        x_flat = x.view(M, K)
        out_flat = out.view(M, N)
        grid = lambda META: (cdiv(M, META['BLOCK_SIZE_M']) * cdiv(N, META['BLOCK_SIZE_N']), )
        _fused_final_layer_dyntanh_kernel[grid](
            x_flat, out_flat,
            ln_f.alpha, ln_f.gamma, ln_f.beta,
            norm_final.alpha, norm_final.gamma, norm_final.beta,
            shift_flat, scale_flat,
            linear.weight, linear.bias,
            M, N, K, T,
            x_flat.stride(0), x_flat.stride(1),
            linear.weight.stride(1), linear.weight.stride(0),
            out_flat.stride(0), out_flat.stride(1),
            shift_flat.stride(0), shift_flat.stride(1),
            scale_flat.stride(0), scale_flat.stride(1),
            HAS_LN_F_BETA=(ln_f.beta is not None),
            HAS_NF_BETA=(norm_final.beta is not None),
            HAS_BIAS=(linear.bias is not None),
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8, num_stages=2,
        )
    elif isinstance(ln_f, nn.LayerNorm):
        _fused_final_layer_ln_kernel[grid](
            x, out,
            ln_f.weight, ln_f.bias, ln_f.eps,
            norm_final.weight, norm_final.bias, norm_final.eps,
            shift_flat, scale_flat,
            linear.weight, linear.bias,
            B, T, C, V,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            shift_flat.stride(0), shift_flat.stride(1),
            scale_flat.stride(0), scale_flat.stride(1),
            HAS_LN_F_W=(ln_f.weight is not None),
            HAS_LN_F_B=(ln_f.bias is not None),
            HAS_NF_W=(norm_final.weight is not None),
            HAS_NF_B=(norm_final.bias is not None),
            HAS_LINEAR_BIAS=(linear.bias is not None),
            IS_RMS_LN_F=False,
            IS_RMS_NF=False,
            BLOCK_C=BLOCK_C, BLOCK_V=BLOCK_V,
        )
    elif isinstance(ln_f, nn.RMSNorm):
        _fused_final_layer_ln_kernel[grid](
            x, out,
            ln_f.weight, None, ln_f.eps,
            norm_final.weight, None, norm_final.eps,
            shift_flat, scale_flat,
            linear.weight, linear.bias,
            B, T, C, V,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            shift_flat.stride(0), shift_flat.stride(1),
            scale_flat.stride(0), scale_flat.stride(1),
            HAS_LN_F_W=(ln_f.weight is not None),
            HAS_LN_F_B=False,
            HAS_NF_W=(norm_final.weight is not None),
            HAS_NF_B=False,
            HAS_LINEAR_BIAS=(linear.bias is not None),
            IS_RMS_LN_F=True,
            IS_RMS_NF=True,
            BLOCK_C=BLOCK_C, BLOCK_V=BLOCK_V,
        )
    else:
        raise ValueError(f"Unsupported ln_f type: {type(ln_f)}")

    return out


# ---------------------------------------------------------------------------
# Triton Final Layer
# ---------------------------------------------------------------------------
class TritonDDitFinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(config.cond_dim, 2 * config.n_embd, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # norm_final (must match PyTorch DDitFinalLayer naming for state dict compat)
        self.norm_final = get_norm(config, config.n_embd, bias=config.bias)

    def forward(self, x, c, ln_f):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = fused_final_layer(x, shift, scale, ln_f, self.norm_final, self.linear)
        return x


# ---------------------------------------------------------------------------
# Full Triton GPT
# ---------------------------------------------------------------------------
class TritonGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.sigma_map = TritonTimestepEmbedder(config.cond_dim)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                (TritonDDiTDynTanh if config.norm == 'dyntanh' else TritonDDiTBlock)(config, i)
                for i in range(config.n_layer)
            ]),
            ln_f=get_norm(config, config.n_embd, bias=config.bias),
        ))
        self.local_conv = nn.Conv1d(
            config.n_embd, config.n_embd, kernel_size=3, padding=1,
            groups=config.n_embd, bias=config.bias
        )
        self.local_conv2 = nn.Conv1d(
            config.n_embd, config.n_embd, kernel_size=3, padding=1,
            groups=config.n_embd, bias=config.bias
        )
        self.n_registers = 8
        self.register_tokens = nn.Parameter(torch.zeros(1, self.n_registers, config.n_embd))
        self.sigma_in = nn.Linear(config.cond_dim, config.n_embd, bias=False)
        self.sigma_out = nn.Linear(config.cond_dim, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, config.block_size + self.n_registers)
        )
        self.lm_head = TritonDDitFinalLayer(config)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        torch.nn.init.zeros_(self.sigma_in.weight)
        torch.nn.init.zeros_(self.sigma_out.weight)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, sigma):
        sigma = sigma.reshape(-1)
        b, t = idx.size()
        c = self.sigma_map(sigma)
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        tok_emb = self.transformer.wte(idx)
        tok_emb = fused_input_convs(
            tok_emb,
            self.local_conv.weight, self.local_conv.bias,
            self.local_conv2.weight, self.local_conv2.bias
        )
        x = tok_emb + self.sigma_in(c).unsqueeze(1)
        x = self.transformer.drop(x)
        freqs_cis = self.freqs_cis[:t]
        for block in self.transformer.h:
            x = block(x, c, freqs_cis)

        x = self.lm_head(x, c, self.transformer.ln_f)
        x = x + self.sigma_out(c).unsqueeze(1)
        x = torch.scatter(x, -1, idx[..., None], torch.zeros_like(x[..., :1]))
        return x
