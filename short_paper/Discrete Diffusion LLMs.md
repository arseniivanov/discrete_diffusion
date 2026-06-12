Project plan by: Arseni Ivanov
## 1. Project Overview
The intent with the project is to explore Discrete Diffusion/Flow matching paradigms for LLMs in order to improve text generation speed.
The initial goal is to make a tiny trainable Diffusion LLM Model, following Andrey Karpathy's NanoGPT and the Shakespeare dataset.
The latter goal is to experiment with various losses, masking/noise schemes in order to figure out what works well with dLLMs and why.
Finally, incorporation of MoE, linear attention such as KDA will be explored.
If time permits, I am also looking to see if normalization blocks can be adjusted to improve generalization.
## 2. Problem Statement
* Autoregressive LLM inference is slow and requires O(N) time to generate a sequence of length N.
* Causal LLMs can't change/inpaint/refine past tokens without re-generating latter tokens because of the causal chain.
* Autoregressive LLMs require KV-caches, which increases memory movement with context length and the decoding/inference quickly becomes memory-bound on the GPU.

dLLMs have the opportunity to relieve all these pressure points, however they have their own issues as unstable training, and bad growth with context size.
## 3. Proposed Method
I am planning to use Karpathy's NanoGPT implementation, adjusted to reflect more modern architectures by looking at the structures from Sebastian Raschkas blog.

For evaluation, I am looking to combine quantitative metrics, such as Perplexity/Cross-Entropy/Flow-loss/Score-loss as well as qualitative metrics such as looking at the output.
It can be that a model fits the distribution well, but has lost semantic meaning by for example spamming common tokens like "the", "as", "a", "with", etc.
## 4. Dataset & Resources
https://github.com/karpathy/nanogpt
https://sebastianraschka.com/llm-architecture-gallery/
## 5. Related Work
https://arxiv.org/pdf/2107.03006
https://arxiv.org/pdf/2310.16834
https://arxiv.org/pdf/2407.15595
https://arxiv.org/pdf/2412.10193v1
https://arxiv.org/pdf/2502.05314
https://arxiv.org/pdf/2503.09573
https://arxiv.org/pdf/2506.17298
https://arxiv.org/pdf/2601.18089