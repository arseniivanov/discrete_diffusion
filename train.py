from torch.distributions import distribution
from dataset import get_data_loader, StringHandler, print_wrapped, decode
from model import GPT, GeometricNoise, GPTConfig
import torch
import torch.optim as optim
from losses import loss_function
import os
from inference_helpers import staggered_score, transition, sample_categorical

# Initialise
batch_size = 256
context_length = 256
data_dir = './shakespeare_char/'

sh = StringHandler()

train_dataloader, dataset = get_data_loader(data_dir, sh, 'train', batch_size, context_length)
val_dataloader, _   = get_data_loader(data_dir, sh, 'val', batch_size, context_length)

distribution = dataset.distribution

# Peek at one batch to confirm shapes/types
vocab_size = sh.get_vocab_size()
batch = next(iter(train_dataloader))
print(batch.shape)
print(batch[0]) # A tensor of indices of length `context_length`
# A character-level baby GPT model :)
n_layer = 3
n_head = 2
n_embd = 384
cond_dim = 64
block_size = context_length
dropout = 0.1
bias = False # do we use bias inside LayerNorm and Linear layers?

sigma_min, sigma_max = 1e-4, 20
noise = GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, cond_dim=cond_dim,
                  bias=bias, vocab_size=vocab_size, block_size=block_size, dropout=dropout)

config = GPTConfig(**model_args)
model = GPT(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
PATH = "model_epoch_1.pth"

if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    steps = 128
    eps = 1e-5
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    step_size = (1 - eps) / steps

    x = torch.randint(0, vocab_size, (1, context_length), device=device)

    with torch.no_grad():
        for i in range(steps + 1):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            curr_sigma_bar = noise(t)[0]
            if i < steps:
                next_sigma_bar = noise(t - step_size)[0]
                delta_sigma = curr_sigma_bar - next_sigma_bar

                log_score = model(x, curr_sigma_bar)
                score = torch.exp(log_score)

                stag_score = staggered_score(score, delta_sigma)
                probs = stag_score * transition(x, delta_sigma, sh)
                x = sample_categorical(probs)

            else:
                # last denoising step
                # delta_sigma = curr_noise_bar - 0
                delta_sigma = curr_sigma_bar

                log_score = model(x, curr_sigma_bar)
                score = torch.exp(log_score)

                stag_score = staggered_score(score, delta_sigma)
                probs = stag_score * transition(x, delta_sigma, sh)

                x = sample_categorical(probs)

            print(f'Decoded Text at step {i}:', flush=True, end='\n\n')
            print_wrapped(decode(x[0], sh), end='\n\n', flush=True)

else:
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    n_epochs = 1

    for epoch in range(n_epochs):
        for i, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            loss = loss_function(model, batch, noise, sh, sampling_eps=sigma_min, token_distribution=distribution)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f"Epoch {epoch} loss: {loss.item()}")
        #if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
