import torch
import torch.nn as nn
import triton
import triton.language as tl
from model import SelfAttention, MLP

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
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
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
        BLOCK_SIZE=256
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
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
    out_flat = torch.empty((M, N), device=x.device, dtype=x.dtype)

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

        x_mlp_hidden = self.mlp.gelu(self.mlp.c_fc(x_mod_2))
        x_out = fused_mlp_proj_epilogue(x_mlp_hidden, self.mlp.c_proj.weight, gate_mlp, x_skip)
        return x_out

@triton.jit
def _fused_modulate_dyntanh_kernel(
    x_ptr, cond_ptr,
    alpha_ptr, gamma_ptr, beta_ptr,
    out_ptr,
    n_cols, T, C,
    stride_x_row, stride_x_col,
    stride_cond_row, stride_cond_col,
    stride_out_row, stride_out_col,
    HAS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // T

    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_out_row

    shift_row_ptr = cond_ptr + batch_idx * stride_cond_row + 0 * C
    scale_row_ptr = cond_ptr + batch_idx * stride_cond_row + 1 * C

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

        scale = tl.load(scale_row_ptr + cols * stride_cond_col, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_row_ptr + cols * stride_cond_col, mask=mask, other=0.0).to(tl.float32)


        dt = tanh(x * alpha) * gamma + beta
        out = dt * (1.0 + scale) + shift

        tl.store(out_row_ptr + cols * stride_out_col, out, mask=mask)

def fused_modulate_dyntanh(x, cond, alpha, gamma, beta=None):
    B, T, C = x.shape
    x_flat = x.view(-1, C)

    out = torch.empty_like(x_flat)

    n_rows = x_flat.shape[0]
    n_cols = x_flat.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    _fused_modulate_dyntanh_kernel[grid](
        x_flat, cond,
        alpha, gamma, beta,
        out,
        n_cols, T, C,
        x_flat.stride(0), x_flat.stride(1),
        cond.stride(0), cond.stride(1),
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
        BLOCK_SIZE=256
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
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
    out_flat = torch.empty((M, N), device=x.device, dtype=x.dtype)

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
        cond = self.adaLN_modulation(c)
        _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = cond[:, None].chunk(6, dim=2)
        x_skip = x
        x_mod = fused_modulate_dyntanh(
            x, cond,
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
