import torch
import torch.nn.functional as F
from model import GPT, Noise, MaskingNoise
from dataset import StringHandler, perturb_batch, perturb_batch_with_distribution, perturb_batch_with_masking
from typing import Optional

def flow_loss(
        model: GPT,
        x1: torch.Tensor,
        noise=None, # Argument kept for compatibility, ignored
        sh=None,    # Argument kept for compatibility, ignored
        sampler=None, # Argument kept for compatibility, ignored
        sampling_eps=1e-4 # Minimum time to avoid numerical instability
    ) -> torch.Tensor:
    """
    Discrete Flow Matching Loss.
    We interpolate between Noise (x0) and Data (x1) based on time t.
    The model tries to predict x1.
    """
    b, t_len = x1.shape
    device = x1.device
    vocab_size = model.config.vocab_size

    # 1. Sample Time t ~ Uniform[0, 1]
    t = torch.rand(b, device=device)
    
    # 2. Sample Noise x0 ~ Uniform(Vocab)
    # (The Meta snippet uses uniform noise. You could use your unigram dataset.distribution if you prefer)
    x0 = torch.randint(0, vocab_size, (b, t_len), device=device)

    # 3. Interpolate (The Flow Step)
    # In discrete space, linear interpolation of probability means:
    # "With probability t, the token is clean (x1). With probability 1-t, it is noise (x0)"
    
    # Generate a mask where values < t are TRUE (keep x1)
    # We reshape t to [b, 1] for broadcasting
    mask_prob = t.view(-1, 1)
    # Sample a random map [0,1] for every token
    decision_map = torch.rand(b, t_len, device=device)
    
    # x_t = x1 where map < t, else x0
    x_t = torch.where(decision_map < mask_prob, x1, x0)

    # 4. Model Prediction
    # We pass x_t and t. The model predicts logits for x1.
    # Note: We pass 't' into the 'sigma' argument of your model.
    logits = model(x_t, t)
    
    # 5. Loss: Cross Entropy between prediction and real x1
    # Reshape for CrossEntropy: (Batch * Seq, Vocab) vs (Batch * Seq)
    loss = F.cross_entropy(logits.view(-1, vocab_size), x1.view(-1))

    return loss

def score_entropy(
    score_log: torch.Tensor,
    sigma_bar: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    clamp_exp: float = 30.0,
    eps: float = 1e-12,
):
    """
    Compute the Score Entropy Loss (Eq. 7) *without* the outer sigma_t multiplier.

    Args:
        score_log:  (B, L, V) tensor of model outputs = log s_theta(x_t, bar{sigma}_t)
                    for each position and vocabulary element.
        sigma_bar:  (B, 1) tensor for \bar\sigma_t (integrated noise).
        x_t:        (B, L) int tensor with current noised tokens.
        x0:         (B, L) int tensor with original clean tokens.
        vocab_size: int, vocabulary size.
        clamp_exp:  float, clamp for exponent to keep exp(score_log) stable.
        eps:        float, small constant for numerical stability in logs/divides.

    Returns:
        loss:       (B, L) tensor containing Eq. (7) per token position (no sigma_t).
        details:    dict with optional diagnostics for logging.
    """
    B, L, vocab_size = score_log.shape
    # 1) Precompute helpers
    # stably compute exp(bar_sigma) - 1
    esigm1 = torch.where(
        sigma_bar < 0.5,
        torch.expm1(sigma_bar),
        torch.exp(sigma_bar) - 1
    )

    # ratio = non-diagonal terms (move) / diagonal terms (no-move) in Eq. (3)
    ratio = esigm1 / (esigm1 + vocab_size)
    # Clamp ratio away from 0 to avoid divide by zero and log(0)
    ratio = torch.clamp(ratio, min=eps)

    # We need both model predicted log s_theta and s = exp(log s_theta)
    # Clamp the exponent to prevent overflow (safe since the loss uses first-order terms)
    score_log = torch.clamp(score_log, max=clamp_exp)
    s = torch.exp(score_log)
    # We'll often need to take the values at a particular token indices
    def take_at(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # logits: (B, L, V), idx: (B, L) -> (B, L)
        return torch.gather(logits, dim=-1, index=idx[..., None]).squeeze(-1)

    # 2) Build positive term in Eq. (7)
    # Mean over all y, and then subtract out the y = x_t contribution
    s_scaled   = s / (vocab_size - 1)  # scaled scores
    s_mean_all = s_scaled.sum(dim=-1)    # (B, L)
    s_at_xt    = take_at(s_scaled, x_t)  # (B, L)
    pos_term   = s_mean_all - s_at_xt    # averages over y != x_t

    # 3) Build negative term in Eq. (7)
    # We need to consider a total of (V - 1) terms, split into two mutually exclusive cases:
    # Case 1: x_t == x0 (no move); all y != x_t have the same a_y = ratio
    #   there are a total of V - 1 such terms
    # Case 2: x_t != x0 (move):
    #   this can be split into two parts:
    #   Part a: y == x0; with a_y = 1 / ratio
    #     there is exactly 1 such term
    #   Part b: y != x0 and y != x_t with a_y = 1
    #     there are (V - 2) such terms
    log_s_mean  = score_log.sum(dim=-1) / (vocab_size - 1)   # (B, L)
    log_s_at_xt = take_at(score_log, x_t) / (vocab_size - 1) # (B, L)
    base_neg    = log_s_mean - log_s_at_xt # averages over y != x_t

    # Case split: no-move (x_t == x0) vs move (x_t != x0).
    no_move = (x_t == x0)

    # Case 1: no move (x_t == x0):
    #   a_y = p(y|x0)/p(x_t|x0) = move / no-move = ratio
    neg_term_no_move = ratio * base_neg

    # Case 2: When x_t != x0:
    # a_y = p(y|x0)/p(x_t|x0) = 1 / ratio when y = x0
    # a_y = p(y|x0)/p(x_t|x0) = 1 otherwise
    neg_term_move = take_at(score_log, x0) / (ratio * (vocab_size - 1)) + (vocab_size - 2) * base_neg / (vocab_size - 1)

    neg_term = torch.where(no_move, neg_term_no_move, neg_term_move)

    # 4) Build constant term K(a) summed over y != x_t.
    # Again split into two mutually exclusive cases

    # Case 1: no move (x_t == x0)
    # y can be != x_t in V - 1 ways, each with a_y = ratio
    const_no_move = ratio * (torch.log(ratio) - 1.0)

    # Case 2: move (x_t != x0)
    # a_y = p(y|x0)/p(x_t|x0) = 1 / ratio when y = x0
    # a_y = p(y|x0)/p(x_t|x0) = 1 otherwise
    const_move = ((-torch.log(ratio) - 1.0) / ratio - (vocab_size - 2)) / (vocab_size - 1)

    const_term = torch.where(no_move, const_no_move, const_move)

    # Final per-position loss (without the outer sigma_t multiplier):
    loss = pos_term - neg_term + const_term  # (B, L)

    return loss

def masked_ce_loss(model, x0, x_t, sigma_bar, sh):
    """
    Time-Conditioned Masked Cross-Entropy for MDLM.
    Calculates loss ONLY on the masked tokens.
    """
    vocab_size = sh.get_vocab_size()
    mask_token_id = sh.mask_token_id

    # Find exactly which tokens were masked
    mask_decision = (x_t == mask_token_id)

    # Forward pass
    log_score = model(x_t, sigma_bar)

    # Flatten for CE calculation
    logits_flat = log_score.view(-1, vocab_size)
    x0_flat = x0.view(-1)
    mask_decision_flat = mask_decision.view(-1)

    # Calculate unreduced CE
    loss_per_token = F.cross_entropy(logits_flat, x0_flat, reduction='none')

    # Zero out loss for visible tokens
    loss_masked_only = loss_per_token * mask_decision_flat.float()

    # Average the loss over the number of masked tokens (avoid div by zero)
    num_masked = mask_decision_flat.sum().clamp(min=1.0)
    final_loss = loss_masked_only.sum() / num_masked

    return final_loss

def loss_function(
        model: GPT,
        x0: torch.Tensor,
        noise: Noise,
        sh: StringHandler,
        sampler: Optional[torch.distributions.Categorical],
        t: Optional[torch.Tensor]=None,
        sampling_eps=1e-3
    ) -> torch.Tensor:
    """
    Computes the loss for a batch of data.
    Args:
        model:          discrete diffusion model
        x0:             (B, L) Longtensor of original clean tokens.
        noise:          a GeometricNoise instance
        t:              (B,) float tensor with time steps in [0, 1]. If None, sampled uniformly.
        x_t:            (B, L) int tensor with perturbed tokens. If None, generated on-the-fly.
        sampling_eps:   float, small epsilon to avoid 0 or 1 time steps.
    Returns:
        loss:           scalar tensor with the loss.
    """

    if t is None: # time step — antithetic pairs (t, 1-t) for balanced noise coverage
        B = x0.shape[0]
        half = (B + 1) // 2
        u = torch.rand(half, device=x0.device)
        u = torch.cat([u, 1.0 - u])[:B]
        t = sampling_eps + (1.0 - 2.0 * sampling_eps) * u

    sigma_bar, sigma = noise(t)

    if isinstance(noise, MaskingNoise):
        # Clamp masking probability to minimum 30% to ensure consistent training signal
        sigma_bar = sigma_bar.clamp(min=0.30)
        fn = perturb_batch_with_masking(x0, sigma_bar[:, None], sh)
        loss = masked_ce_loss(model, x0, fn, sigma_bar, sh)
        return loss

    if sampler is not None:
        fn = perturb_batch_with_distribution(x0, sigma_bar[:, None], sh, sampler)
    else:
        fn = perturb_batch(x0, sigma_bar[:, None], sh)

    log_score = model(fn, sigma_bar)
    loss = score_entropy(log_score, sigma_bar[:, None], fn, x0)

    loss = (sigma[:, None] * loss).mean(dim=-1).mean()

    return loss
