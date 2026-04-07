import torch
from dataset import StringHandler
from tqdm import tqdm
from torch.nn import functional as F

def transition(x_t: torch.Tensor, delta_sigma: torch.Tensor, sh: StringHandler) -> torch.Tensor:
    """
    Forward transition kernel:
        exp(σ_t^Δt Q^{tok})(x_t, y)

    Approximates the finite-time forward diffusion probability of moving from token x_t to y
    after a noise increment of Δσ = σ_t^{Δt}.

    Args:
        x_t:          (B, L) integer tensor of current tokens.
        delta_sigma:  scalar tensor representing σ_t^{Δt}.

    Returns:
        trans_probs:  (B, L, V) tensor of categorical probabilities over next tokens.
    """
    # Uniform mixing term from exp(delta_sigma * Q^{tok})
    # with the help of Eq. (3), this translates to:
    vocab_size = sh.get_vocab_size()
    base_prob = (1 - torch.exp(-delta_sigma[..., None])) / vocab_size
    trans = torch.ones(*x_t.shape, vocab_size, device=x_t.device) * base_prob

    # Remove the uniform contribution for the current token
    trans = trans.scatter(-1, x_t[..., None], torch.zeros_like(trans))

    # Ensure that probabilities across the vocabulary sum to 1
    diag_fill = 1 - trans.sum(dim=-1, keepdim=True)
    trans = trans.scatter(-1, x_t[..., None], diag_fill)
    return trans

def distribution_transition(
    x_t: torch.Tensor,
    delta_sigma: torch.Tensor,
    sh: StringHandler,
    distribution: torch.Tensor
) -> torch.Tensor:
    """
    Forward transition kernel for a distribution-based corruption process.

    This reflects a forward process where corrupted tokens are replaced by
    sampling from the provided token distribution.

    Args:
        x_t:          (B, L) integer tensor of current tokens.
        delta_sigma:  scalar tensor representing noise increment.
        distribution: (V,) tensor of the dataset's token distribution.

    Returns:
        trans_probs:  (B, L, V) tensor of categorical probabilities.
    """
    B, L = x_t.shape
    vocab_size = sh.get_vocab_size()
    p_move = 1 - torch.exp(-delta_sigma[..., None])
    trans_base = p_move * distribution.view(1, 1, -1)
    trans = trans_base.expand(B, L, vocab_size).clone()
    diag_fill = 1 - trans.sum(dim=-1, keepdim=True) + torch.gather(trans, -1, x_t[..., None])
    trans = trans.scatter(-1, x_t[..., None], diag_fill)
    
    return trans

def staggered_score(score, delta_sigma):
    """
    Applies the inverse exponential operator:
        exp(-σ_t^Δt Q^{tok}) s_θ(x_t, t)

    This "staggered" score correction accounts for the finite time-step Δt.

    Args:
        score:        (B, L, V) tensor, model output s_θ(x_t, t)
        delta_sigma:  scalar tensor representing σ_t^{Δt}

    Returns:
        adjusted_score: (B, L, V) tensor, transformed score
    """
    vocab_size = score.shape[-1]
    exp_factor = torch.exp(-delta_sigma)[..., None]  # (B, L, 1)
    correction = ((exp_factor - 1) / (vocab_size * exp_factor)) * score.sum(dim=-1, keepdim=True)
    return correction + score / exp_factor


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample from a batch of categorical distributions using the Gumbel-max trick.

    Args:
        probs: (B, L, V) tensor of probabilities that sum to 1 along dim=-1.

    Returns:
        samples: (B, L) tensor of sampled token indices.
    """
    # Add a small epsilon for numerical stability
    eps = 1e-10
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + eps) + eps)
    return torch.argmax(torch.log(probs + eps) + gumbel_noise, dim=-1)


def sample_substitution(model, noise, sh, cfg, device, dataset):
    """Generates a sample using the substitution-reversal process."""
    vocab_size = sh.get_vocab_size()
    distribution = dataset.distribution.to(device) if cfg.trainer.prob_sampling else None

    steps = cfg.inference.steps
    eps = cfg.inference.eps
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    step_size = (1 - eps) / steps
    x = torch.randint(0, vocab_size, (1, cfg.data.context_length), device=device)

    with torch.no_grad():
        for i in tqdm(range(steps), desc="Generating (Substitution)"):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            curr_sigma_bar = noise(t)[0]
            next_sigma_bar = noise(t - step_size)[0]
            delta_sigma = curr_sigma_bar - next_sigma_bar
            
            log_score = model(x, curr_sigma_bar)
            score = torch.exp(log_score)

            stag_score = staggered_score(score, delta_sigma)
            
            # FAIRNESS BRANCH: Use the correct transition kernel
            if distribution is not None:
                probs = stag_score * distribution_transition(x, delta_sigma, sh ,distribution)
            else:
                probs = stag_score * transition(x, delta_sigma, sh)
                
            x = sample_categorical(probs)

    # Final denoising step
    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    curr_sigma_bar = noise(t)[0]
    log_score = model(x, curr_sigma_bar)
    x = torch.argmax(log_score, dim=-1)
    
    return x

def sample_masking(model, noise, sh, cfg, device):
    """Generates a sample using the iterative unmasking process."""
    steps = cfg.inference.steps
    mask_token_id = sh.mask_token_id
    x = torch.full((1, cfg.data.context_length), fill_value=mask_token_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1, 0, steps + 1, device=device)

    with torch.no_grad():
        for i in tqdm(range(steps), desc="Generating (Masking)"):
            num_masked = (x == mask_token_id).sum(dim=-1)
            if num_masked.max() == 0: break
            
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            sigma_bar, _ = noise(t)
            
            log_score = model(x, sigma_bar)
            probs = F.softmax(log_score, dim=-1)
            candidate_tokens = sample_categorical(probs)
            
            confidence = torch.gather(probs, -1, candidate_tokens[..., None]).squeeze(-1)
            confidence = torch.where(x != mask_token_id, -1.0, confidence)
            
            ratio_to_unmask = 1.0 / (steps - i)
            num_to_unmask = (num_masked * ratio_to_unmask).long().clamp(min=1)
            indices_to_unmask = torch.topk(confidence, k=num_to_unmask.item(), dim=-1).indices

            mask_update = torch.zeros_like(x, dtype=torch.bool).scatter_(1, indices_to_unmask, True)
            x = torch.where(mask_update, candidate_tokens, x)
            
    return x

def sample_discrete_flow(model, sh, cfg, device):
    """
    Euler Method Sampling for Discrete Flow Matching.
    Mathematically: p(x_{t+h}) = p(x_t) + h * velocity
    Where velocity = (p(x_1) - p(x_t)) / (1 - t)
    """
    model.eval()
    
    batch_size = 1 # Or cfg.inference.batch_size if you have it
    seq_len = cfg.data.context_length
    vocab_size = sh.get_vocab_size()
    steps = cfg.inference.steps
    
    # 1. Start from Pure Noise (t=0)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 2. Time steps
    indices = range(steps)
    
    with torch.no_grad():
        for i in tqdm(indices, desc="Flow Sampling"):
            # Current time t (scalar)
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=device)
            
            # Predict x1 (Data) from current x
            logits = model(x, t)
            p1 = F.softmax(logits, dim=-1) # The model's belief of what the clean data is
            
            # Calculate mixing weight alpha for this step
            # We want to move from p_t to p_{t+dt}
            # The simplified update rule derived from the ODE is:
            # p_next = (1 - alpha) * one_hot(x) + alpha * p1
            # where alpha = dt / (1 - t)
            # if we have N steps, dt = 1/N. 1-t = (N-i)/N.
            # alpha = (1/N) / ((N-i)/N) = 1 / (N - i)
            
            # Avoid division by zero at the very last step
            if i == steps - 1:
                # At the very end, just take the model's prediction
                x = torch.argmax(logits, dim=-1)
                break
                
            alpha = 1.0 / (steps - i)
            
            # Create the One-Hot vector of our CURRENT state
            x_one_hot = F.one_hot(x, num_classes=vocab_size).float()
            
            # Euler Step in Probability Space
            # "Mix the current state with the predicted goal"
            probs = (1 - alpha) * x_one_hot + alpha * p1
            
            # Renormalize just in case (though math says it sums to 1)
            # probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample next token
            # We use the Gumbel-Max trick or standard sampling
            x = torch.distributions.Categorical(probs=probs).sample()
            
    return x
