# Discrete Diffusion Optimization Worklog

## Baseline (default config)
- **Val Loss (epoch 1)**: 1.0268
- **Train Loss (final)**: 0.8674
- **Config**: n_layer=3, n_head=2, n_embd=384, masking cosine noise, Muon+AdamW lr=1e-3, timestep_embedding=False
- **Text sample**:
  ```
  you art to more hearts, and not you good  give, and you have you,
  That be you, for you have for your you:
  But your your you be not the are your your your sterd,
  And lord, you do, what you you, you be you your your come, to king,
  Where to well be you you
  ```

---

## Experiment 1: Cosine LR warmup (FAILED — reverted)
- **Hypothesis**: 5% linear warmup + cosine decay to 0.1×lr would improve convergence
- **Val Loss**: 1.1594 (WORSE)
- **Train Loss**: 1.0388
- **Verdict**: Warmup wasted early training steps; flat lr=1e-3 is well-calibrated for 2 epochs

---

## Experiment 2: Sinusoidal timestep embedding ✓ (COMMITTED)
- **Hypothesis**: Replace simple MLPEmbedder(1→cond_dim) with TimestepEmbedder using 256-dim sinusoidal features for richer noise-level conditioning
- **Change**: `model.timestep_embedding: True` in base_config.yaml
- **Val Loss**: 1.0105 (IMPROVED from 1.0268)
- **Train Loss**: 0.7836 (IMPROVED from 0.8674)
- **Run dir**: outputs/shakespeare_diffusion_base/2026-04-20_11-06-16
- **Text sample**:
  ```
  For medities, that not not so not, note,
  With  as was nou safter to the come to the good so,
  I have then, nor you, for nor man  now the words not done, that lomes, O, he that she shall home,
  for  may be shall home, for to the good to:
  Have to the word to
  ```

---
