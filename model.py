import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from dataclasses import dataclass
import abc
from fla.layers.gated_deltanet import GatedDeltaNet


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    # q, k: (B, n_head, T, head_dim); freqs_cis: (T, head_dim//2) complex
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(x_c * freqs_cis).flatten(-2).type_as(x)
    return rotate(q), rotate(k)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    cond_dim: int = 64
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    timestep_embedding: bool = True

    # GatedDeltaNet settings (replaces KDA)
    use_gated_delta: bool = False
    gated_delta_layers: Optional[list] = None
    attn_mode: str = 'chunk'  # 'chunk' or 'fused_recurrent'
    gated_delta_expand_v: float = 1.0
    gated_delta_use_gate: bool = True
    gated_delta_use_short_conv: bool = False  # Must be False to avoid OOM
    gated_delta_allow_neg_eigval: bool = True
    gated_delta_conv_size: int = 2  # Ignored when use_short_conv=False
    gated_delta_norm_eps: float = 1e-5

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GatedDeltaAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        
        # FORCE disable short conv (the OOM culprit)
        use_short_conv = False
        
        self.attn = GatedDeltaNet(
            hidden_size=config.n_embd,
            expand_v=getattr(config, 'gated_delta_expand_v', 1.0),
            head_dim=config.n_embd // config.n_head,
            num_heads=config.n_head,
            use_gate=getattr(config, 'gated_delta_use_gate', True),
            use_short_conv=use_short_conv,
            mode=getattr(config, 'attn_mode', 'chunk'),
            allow_neg_eigval=getattr(config, 'gated_delta_allow_neg_eigval', True),
            conv_size=getattr(config, 'gated_delta_conv_size', 2),
            norm_eps=getattr(config, 'gated_delta_norm_eps', 1e-5),
            layer_idx=layer_idx,
        )
        self.resid_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        out, _, _ = self.attn(hidden_states=x)
        return self.resid_dropout(out)

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, freqs_cis: torch.Tensor):
        B, T, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

def bias_add_scale(
    x: torch.Tensor, bias: Optional[torch.Tensor], scale: torch.Tensor, residual: Optional[torch.Tensor]) -> torch.Tensor:
    if bias is not None:
        out = scale * (x + bias)
    else:
        out = scale * x

    if residual is not None:
        out = residual + out
    return out

class DDiTBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Select attention type
        use_gated_delta = getattr(config, 'use_gated_delta', False)
        gated_delta_layers = getattr(config, 'gated_delta_layers', None) 
        
        if use_gated_delta and (gated_delta_layers is None or layer_idx in gated_delta_layers):
            self.attn = GatedDeltaAttention(config, layer_idx=layer_idx)
        else:
            self.attn = SelfAttention(config)
            
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c, freqs_cis: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate(self.ln_1(x), shift_msa, scale_msa)
        if isinstance(self.attn, SelfAttention):
            x = self.attn(x, freqs_cis)
        else:
            x = self.attn(x)
        x = gate_msa * x + x_skip

        x_skip = x
        x = modulate(self.ln_2(x), shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = gate_mlp * x + x_skip

        return x

class DDitFinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(config.cond_dim, 2 * config.n_embd)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        t_emb = self.mlp(t.unsqueeze(-1))
        return t_emb

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if config.timestep_embedding:
            self.sigma_map = TimestepEmbedder(config.cond_dim)
        else:
            self.sigma_map = MLPEmbedder(config.cond_dim)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DDiTBlock(config, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Depthwise conv over token embeddings to capture local char patterns before attention
        self.local_conv = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1,
                                    groups=config.n_embd, bias=config.bias)
        # Register tokens: prepended to the sequence, stripped before output
        self.n_registers = 8
        self.register_tokens = nn.Parameter(torch.zeros(1, self.n_registers, config.n_embd))
        # RoPE frequencies: extra slots for register token positions
        head_dim = config.n_embd // config.n_head
        self.register_buffer('freqs_cis', precompute_freqs_cis(head_dim, config.block_size + self.n_registers))
        self.lm_head = DDitFinalLayer(config)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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
        c = F.silu(self.sigma_map(sigma))
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        tok_emb = tok_emb + self.local_conv(tok_emb.transpose(1, 2)).transpose(1, 2)
        reg = self.register_tokens.expand(b, -1, -1)  # (b, n_reg, n_embd)
        x = torch.cat([reg, tok_emb], dim=1)           # (b, n_reg+t, n_embd)
        x = self.transformer.drop(x)
        n_reg = self.n_registers
        freqs_cis = self.freqs_cis[:n_reg + t]
        for block in self.transformer.h:
            x = block(x, c, freqs_cis)
        x = x[:, n_reg:]  # strip registers before output
        x = self.transformer.ln_f(x)

        x = self.lm_head(x, c)
        x = torch.scatter(x, -1, idx[..., None], torch.zeros_like(x[..., :1]))

        return x

class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass

class GeometricNoise(Noise):
    def __init__(self, sigma_min=1e-4, sigma_max=20):
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

    def __call__(self, t):
        return self.total_noise(t), self.rate_noise(t)

class LogLinearNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def __call__(self, t):
        return self.total_noise(t), self.rate_noise(t)

class MaskingNoise(Noise):
    def __init__(self, schedule='cosine'):
        """
        Args:
            schedule (str): The type of schedule to use for alpha_t. 
                            Options: 'linear', 'cosine'.
        """
        super().__init__()
        if schedule not in ['linear', 'cosine']:
            raise ValueError("Schedule must be 'linear' or 'cosine'")
        self.schedule = schedule

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates alpha_t, the probability of a token remaining original.
        Decreases from ~1 at t=0 to ~0 at t=1.
        """
        if self.schedule == 'linear':
            return 1.0 - t
        elif self.schedule == 'cosine':
            # Cosine schedule, often found to be effective
            return torch.cos(t * math.pi / 2.0)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        This is our sigma_bar. It represents the probability of a token being MASKED.
        sigma_bar = 1 - alpha_t
        """
        return 1.0 - self.alpha_t(t)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        This is our sigma. It is the derivative of total_noise w.r.t. t.
        d/dt (1 - alpha_t) = -alpha'_t
        """
        if self.schedule == 'linear':
            # d/dt(t) = 1
            return torch.ones_like(t)
        elif self.schedule == 'cosine':
            # d/dt (1 - cos(t*pi/2)) = (pi/2) * sin(t*pi/2)
            return (math.pi / 2.0) * torch.sin(t * math.pi / 2.0)

    def __call__(self, t):
        return self.total_noise(t), self.rate_noise(t)
