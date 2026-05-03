# Discrete Diffusion for Character-Level Text Generation

This document describes the architecture, data pipeline, training procedure, and generation strategies for the discrete diffusion model implemented in this repository. The model is trained on a character-level Shakespeare corpus and uses a DiT-style transformer backbone with adaptive layer normalization (adaLN) conditioned on a noise-level embedding.

---

## 1. Data

### 1.1 Source
The dataset is **Tiny Shakespeare** (from Karpathy's `char-rnn` repository). It contains ~1.1 M characters of Shakespearean prose and dialogue.

### 1.2 Preprocessing (`shakespeare_char/prepare.py`)
1. **Character-level tokenization**: Each unique character is mapped to an integer ID.
2. **Vocabulary**: 65 tokens (newline, space, punctuation, digits, upper/lower-case letters). A special `[MASK]` token is appended at runtime by `StringHandler`, bringing the effective vocab size to **66**.
3. **Split**: 90 % train (~1.0 M tokens), 10 % validation (~111 k tokens).
4. **Format**: The encoded IDs are saved as flat binary files (`train.bin`, `val.bin`) in `np.uint16` format. A `meta.pkl` file stores the `stoi` / `itos` mappings.

### 1.3 Loading (`dataset.py`)
- **`ShakespeareDataset`** memory-maps the `.bin` files so the full corpus never sits in RAM.
- Each sample is a contiguous slice of length `context_length` (default **512** characters).
- **`StringHandler`** wraps the vocabulary. It dynamically injects a `[MASK]` token (index 65) used by the masking noise schedule.
- **Batching**: batch size **96**, shuffled, single-process loader (`num_workers=0`).

### 1.4 On-the-fly Noising
The dataset returns *clean* sequences. Corruption is applied inside the training loop by one of three perturbation functions:
- **`perturb_batch`** – uniform random substitution (geometric / log-linear noise).
- **`perturb_batch_with_distribution`** – substitution sampled from the empirical unigram distribution.
- **`perturb_batch_with_masking`** – token masking with a learned `[MASK]` token (masking noise).

---

## 2. Model Architecture (`model.py`)

The model is a **GPT-style transformer** with DiT (Diffusion Transformer) conditioning. It predicts a score over the vocabulary for every sequence position, conditioned on the current noised tokens and a scalar noise level `sigma`.

### 2.1 High-Level Configuration (`GPTConfig`)
```yaml
block_size:      1024   # max sequence length (not all used in current config)
vocab_size:      50304  # upper bound for GPT-2 compat; actual data vocab = 66
n_layer:         3
n_head:          2
n_embd:          384
cond_dim:        128    # dimension of the noise-level embedding
dropout:         0.0
bias:            True
timestep_embedding: True
```
Total parameters: **< 6.5 M** (hard cap enforced at init).

### 2.2 Normalization: `DynTanh`
All `nn.LayerNorm` layers have been replaced by a custom **Dynamic Tanh** normalization:
```
DynTanh(x) = gamma * tanh(alpha * x) + beta
```
- `alpha` – learnable scalar that controls the squashing steepness.
- `gamma`, `beta` – per-channel scale and shift (analogous to LayerNorm affine parameters).

**Why DynTanh?** It removes the expensive mean/variance reduction of LayerNorm, is compiler-friendly, and in practice gave **~18 % training speedup** with comparable or better validation loss.

A variant `PolyDynTanh` (7th-order Taylor polynomial of `tanh` in Horner form) exists in the code for experimentation but is not currently active.

### 2.3 Forward Pass Overview (`GPT.forward`)
```
idx:      (B, T)  integer token IDs
sigma:    (B,)    scalar noise levels
```

1. **Noise embedding** (`sigma_map`)  
   `sigma` is mapped to a `cond_dim` vector `c` via sinusoidal timestep embeddings + 2-layer MLP.

2. **Token embedding** (`wte`)  
   `(B, T) -> (B, T, n_embd)`.

3. **Local depthwise convolutions**  
   Two `Conv1d(k=3, groups=n_embd)` layers are applied to the transposed embeddings:
   ```
   tok_emb += local_conv(tok_emb)
   tok_emb += local_conv2(SiLU(tok_emb))
   ```
   These provide a small receptive field (RF=5) before the transformer stack.

4. **Register tokens**  
   8 learned register tokens are prepended to the sequence, increasing length to `T + 8`. They are stripped before the output head.

5. **Sigma-conditioned input bias** (`sigma_in`)  
   A zero-initialized linear projection adds a global bias to every token based on the noise level.

6. **Transformer blocks** (`DDiTBlock`)  
   A stack of `n_layer` blocks. Each block contains:
   - `DynTanh` pre-normalization.
   - **Self-attention** (see §2.4) or optional `GatedDeltaNet` attention.
   - **MLP** (`c_fc -> GELU -> c_proj`).
   - **adaLN modulation**: a linear layer maps `c` to 6 scalars (`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`) that scale/shift the residual branches DiT-style.

7. **Final normalization** (`ln_f`)  
   Another `DynTanh` before the output head.

8. **Output head** (`DDitFinalLayer`)  
   A final adaLN-modulated `DynTanh` + linear projection to `vocab_size`. The projection weights are zero-initialized.

9. **Sigma-conditioned output bias** (`sigma_out`)  
   A zero-initialized linear layer bypasses the transformer and adds a direct `sigma -> vocab` logit shift.

10. **Self-conditioning mask**  
    The model is forced to predict zero logit for the *input* token at every position (scatter-to-zero), preventing trivial identity solutions.

### 2.4 Self-Attention (`SelfAttention`)
```
x -> c_attn -> split(Q,K,V)
Q, K -> per-head DynTanh (q_norm, k_norm)
Q, K -> RoPE (rotary position embeddings)
Q, K, V -> Flash Attention (or manual softmax) with ALiBi bias
```

- **RoPE**: Rotary positional embeddings are pre-computed up to `block_size + n_registers` and sliced per forward pass. Base frequency `theta = 500`.
- **ALiBi**: Two learnable slopes (`[0.1, 0.05]`) generate a per-head distance penalty added to the attention logits. This encourages local attention.
- **QK-Norm**: `DynTanh` applied per-head to queries and keys before the dot product, stabilizing the attention temperature.
- **Flash Attention**: Used automatically when available (`torch.nn.functional.scaled_dot_product_attention`).

### 2.5 Noise Schedules (`model.py`)
Three noise processes are implemented. The active schedule is chosen in the config.

| Schedule | `total_noise(t)` (sigma_bar) | `rate_noise(t)` (sigma) | Usage |
|---|---|---|---|
| **Masking** (cosine) | `1 - cos(t * pi / 2)` | `(pi/2) * sin(t * pi / 2)` | Mask tokens with `[MASK]` |
| **Masking** (linear) | `t` | `1` | Simple linear masking |
| **Geometric** | `sigma_min^(1-t) * sigma_max^t` | `total * log(sigma_max/sigma_min)` | Substitution noise |
| **Log-Linear** | `-log1p(-(1-eps)*t)` | `(1-eps) / (1-(1-eps)*t)` | Substitution noise |

The default config uses **MaskingNoise with cosine schedule**.

---

## 3. Training (`train.py`)

### 3.1 Loss Function (`losses.py`)
The training loss depends on the noise type:

#### Score-Entropy Loss (for substitution noise)
Used with `GeometricNoise` and `LogLinearNoise`. It implements the discrete score-matching objective:
```
loss = sigma(t) * score_entropy(log_score, sigma_bar, x_t, x0)
```
`score_entropy` analytically computes the cross-entropy between the model's predicted score and the true transition kernel, split into move / no-move cases.

#### Flow Matching Loss (for masking noise)
Used with `MaskingNoise`. A simplified discrete flow-matching objective:
```
x_t = mask(x0) with probability sigma_bar
loss = CrossEntropy(model(x_t, sigma_bar), x0)
```

### 3.2 Optimization
- **Base optimizer**: `AdamW` with `lr = 4e-3`.
- **Muon optimizer** (optional, enabled by default): 2D weight matrices (excluding embeddings) are optimized with the Muon optimizer at `5x` the base LR; all other parameters stay on AdamW.
- **Scheduler**: `CosineAnnealingWarmRestarts` with `T_0 = steps_per_epoch`, `T_mult = 1`, and `eta_min = 0.1 * lr`.
- **Mixed precision**: `torch.amp.autocast` with `bfloat16` on CUDA.
- **EMA**: Exponential moving average of all parameters with `decay = 0.998`. The EMA model is evaluated on the validation set after training and is used for final text generation.
- **Gradient scaling**: disabled (bfloat16 does not require GradScaler).

### 3.3 Training Loop
1. Sample a clean batch `x0`.
2. Sample antithetic time pairs `(t, 1-t)` for balanced noise coverage.
3. Perturb `x0` to `x_t` using the active noise schedule.
4. Forward `model(x_t, sigma_bar) -> log_score`.
5. Compute loss, backprop, step optimizers + schedulers, update EMA.
6. After each epoch, evaluate average validation loss.
7. After training, evaluate EMA validation loss and generate a qualitative sample.

### 3.4 Current Training Config
```yaml
trainer:
  n_epochs: 2
  lr: 4.0e-3
  use_muon: True
  prob_sampling: False

data:
  batch_size: 96
  context_length: 512
```
Typical runtime on a single RTX 5080 Ti: **~26–32 minutes** for 2 epochs.

---

## 4. Generation / Inference (`inference_helpers.py`)

Three sampling algorithms are implemented. The active one is selected automatically based on `cfg.noise.type`.

### 4.1 Masking-based Generation (`sample_masking`)
For `MaskingNoise`.
1. Start from a fully masked sequence (`[MASK]` everywhere).
2. Iterate for `steps` (default **256**):
   - Forward the model to get logits.
   - Sample candidate tokens from the softmax.
   - Compute per-token confidence (softmax probability of the sampled token).
   - Unmask the top-`k` most confident tokens, where `k` grows as the fraction of remaining masks divided by remaining steps.
3. Return the fully or partially unmasked sequence.

This is an iterative **confidence-based unmasking** process, analogous to MaskGIT.

### 4.2 Substitution-based Generation (`sample_substitution`)
For `GeometricNoise` and `LogLinearNoise`.
1. Start from pure uniform noise.
2. Discretize time from `t=1` down to `t=eps` in `steps` increments.
3. At each step:
   - Compute `delta_sigma = sigma_bar(t) - sigma_bar(t - dt)`.
   - Get model score `s = exp(log_score)`.
   - Apply the **staggered score** correction: `exp(-delta_sigma * Q) * s`.
   - Multiply by the forward transition kernel `P(y | x_t, delta_sigma)`.
   - Sample the next token from the resulting categorical distribution.
4. Final argmax step at `t = eps`.

### 4.3 Discrete Flow Sampling (`sample_discrete_flow`)
A simpler Euler-style sampler for the flow-matching formulation:
1. Start from uniform noise at `t=0`.
2. For each step `i`:
   - Predict clean data logits `p(x1)`.
   - Mix the current one-hot state with `p(x1)` using `alpha = 1 / (steps - i)`.
   - Sample the next state from the mixture.
3. Final argmax at the last step.

This is currently **not** the default but can be selected by adapting the inference dispatch logic.

---

## 5. Evaluation (`run_eval.py`)

`run_eval.py` orchestrates a standard benchmark run:
1. Executes `train.py` with `trainer.n_epochs=2`.
2. Parses the most recent Hydra output directory.
3. Extracts the **final validation loss** from `val_loss_log.csv`.
4. Prints the generated text from `summary.txt` for qualitative inspection.

### 5.1 Current Baseline (after DynTanh adoption)
```
Final Loss:     0.7194
Val Loss (EMA): 0.8128
Runtime:        26m 27s
```
This is the configuration to beat for future runtime/quality experiments.

---

## 6. Key Files & Responsibilities

| File | Role |
|---|---|
| `model.py` | Full model definition: `GPT`, `DDiTBlock`, `SelfAttention`, `DynTanh`, noise schedules. |
| `train.py` | Training loop, optimizer setup, EMA, validation, qualitative generation. |
| `losses.py` | `score_entropy`, `flow_loss`, `loss_function` dispatch. |
| `dataset.py` | `ShakespeareDataset`, `StringHandler`, noising/perturbation functions. |
| `inference_helpers.py` | `sample_masking`, `sample_substitution`, `sample_discrete_flow`. |
| `run_eval.py` | Standardized evaluation script: train + parse results. |
| `conf/base_config.yaml` | Hydra configuration for hyperparameters. |

---

## 7. Design Evolution Notes

- **DynTanh replaced LayerNorm** across all normalization sites. This was a subtractive/throughput-oriented change that reduced runtime by ~18 % without degrading quality.
- **PolyDynTanh** (polynomial `tanh` approximation) was tested but reverted: `torch.tanh` is already GPU-optimized, so the polynomial did not improve runtime and slightly hurt convergence.
- **Per-block depthwise convolutions** (`block_convs`, `block_convs2`) are instantiated but **commented out** in the forward pass; they were pruned in earlier experiments to reduce overhead.
- **GatedDeltaNet** attention from `fla` is supported via config but disabled by default in the current baseline to keep memory usage predictable on a 16 GB card.
