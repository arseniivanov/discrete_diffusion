import torch
from dataset import StringHandler

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
