import os
import numpy as np
import torch
import torch.utils.data as data
import pickle
import textwrap

class StringHandler():
    def __init__(self):
        data_dir = './shakespeare_char/'
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)

    def itos(self, idx):
        itos = self.meta['itos'] # index to string
        return itos[idx]

    def stoi(self, strng):
        stoi = self.meta['stoi'] # string to index
        return stoi[strng]

    def get_vocab_size(self):
        return self.meta['vocab_size']

def decode(indices_tensor: torch.Tensor, sh: StringHandler):
    '''Decodes a 1D tensor of indices to text'''
    indices = indices_tensor.cpu().numpy()
    return ''.join([sh.itos(i) for i in indices])


class ShakespeareDataset(data.Dataset):
    """
    Memory-mapped dataset for character-level sequences.

    Each item is a 1D tensor of indices (torch.long) of length `context_len`
    from a rolling window over the encoded Shakespeare corpus.

    Notes
    -----
    - Uses np.memmap to avoid loading the entire file into RAM.
    - Returns only `x` (the context window).
      This will serve as the clean target for denoising.
      Noising will be applied on-the-fly during the training.
    """
    def __init__(
        self,
        data_dir: str,
        sh: StringHandler,
        split: str = "train",
        context_len: int = 256,
        dtype: np.dtype = np.uint16,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got: {split!r}")
        if context_len <= 0:
            raise ValueError(f"context_len must be positive, got: {context_len}")

        self.split = split
        self.context_len = int(context_len)

        bin_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"Could not find {bin_path}")

        # Memory-map the encoded corpus. uint16 matches the preprocessing.
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")

        counts = np.bincount(self.data, minlength=sh.get_vocab_size())
        counts_tensor = torch.from_numpy(counts).float()
        self.distribution = counts_tensor / counts_tensor.sum()

        # Number of valid starting positions for a full context window
        self._n = max(0, len(self.data) - self.context_len)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self._n:
            raise IndexError(f"Index {index} out of range for dataset of length {self._n}.")
        # Slice a contiguous window and convert to torch.long (int64)
        x_np = self.data[index : index + self.context_len].astype(np.int64)
        x = torch.from_numpy(x_np)  # shape: [context_len], dtype: torch.long
        return x

def get_data_loader(data_dir, sh, split, batch_size, context_len=256):
    dataset = ShakespeareDataset(data_dir, sh, split, context_len)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0), dataset

def perturb_batch_with_distribution(
    batch: torch.Tensor,
    sigma_bar: torch.Tensor,
    sh: StringHandler,
    sampler: torch.distributions.Categorical
) -> torch.Tensor:
    """
    Diffuses each token by sampling from the training data distribution.

    Args:
        batch: LongTensor of shape [B, L], each entry in [0, vocab_size-1].
        sigma_bar: Scalar tensor representing the noise level.
        sh: The StringHandler instance.
        token_distribution: A 1D tensor of shape [vocab_size] with token probabilities.

    Returns:
        batch_pert: Perturbed batch of LongTensor.
    """
    B, L = batch.shape
    vocab_size = sh.get_vocab_size()
    stay_base = torch.exp(-sigma_bar)
    move_prob = (1 - stay_base) * (1 - 1 / vocab_size)

    move_mask = torch.rand(B, L, device=batch.device) < move_prob
    new_ids = sampler.sample(sample_shape=(B, L))
    batch_pert = torch.where(move_mask, new_ids, batch)
    return batch_pert

def perturb_batch(batch: torch.Tensor, sigma_bar: torch.Tensor, sh: StringHandler) -> torch.Tensor:
    """
    Diffuse each token independently according to Eq. (3).

      - With probability e^{-sigma_bar} + (1 - e^{-sigma_bar})/N, a token stays the same.
      - Otherwise, it jumps uniformly to one of the other N-1 tokens.
    Args:
        batch: LongTensor of shape [B, L], each entry in [0, vocab_size-1]
        sigma_bar: scalar tensor
    Returns:
        batch_pert: perturbed batch of LongTensor
    """
    B, L = batch.shape

    vocab_size = sh.get_vocab_size()
    # 1) Compute move probability: (1 - e^{-sigma}) * (1 - 1/N)
    stay_base = torch.exp(-sigma_bar)
    move_prob = (1 - stay_base) * (1 - 1 / vocab_size)

    # 2) Bernoulli: should this token move?
    move_mask = torch.rand(B, L, device=batch.device) < move_prob

    # 3) For tokens that move, sample a *different* id uniformly from the other N-1 ids.
    #    Sample r in [0, N-2], then map to [0..N-1]\{orig} by skipping the original.
    r = torch.randint(low=0, high=vocab_size - 1, size=(B, L), device=batch.device)
    # shift up by 1 wherever r >= original id, covering {0, .., k-1, k+1, .., N-1}
    new_ids = r + (r >= batch)

    # 4) Apply moves; else keep original
    batch_pert = torch.where(move_mask, new_ids, batch)
    return batch_pert


def print_wrapped(long_text, width=80, **kwargs):
    """
    Print text wrapped to a maximum line width, preserving paragraph breaks.
    """
    paragraphs = long_text.splitlines()
    wrapped = [textwrap.fill(p, width=width) if p else '' for p in paragraphs]
    final_text = "\n".join(wrapped)
    print(final_text, **kwargs)
