# Discrete Diffusion Optimization Worklog

## Baseline (default config)
- **Val Loss (epoch 1)**: 1.0268
- **Train Loss (final)**: 0.8674
- **Config**: n_layer=3, n_head=2, n_embd=384, masking cosine noise, Muon+AdamW lr=1e-3, timestep_embedding=False, cond_dim=64
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
- **Verdict**: Warmup wasted early training steps; flat lr=1e-3 is well-calibrated for 2 epochs

---

## Experiment 2: Sinusoidal timestep embedding ✓ (COMMITTED)
- **Hypothesis**: Replace MLPEmbedder(1→cond_dim) with TimestepEmbedder using 256-dim sinusoidal features
- **Change**: `model.timestep_embedding: True`
- **Val Loss**: 1.0105 (IMPROVED -0.0163)
- **Train Loss**: 0.7836
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

## Experiment 3: cond_dim 64→128 ✓ (COMMITTED)
- **Hypothesis**: cond_dim=64 bottlenecks adaLN conditioning per block; doubling gives richer modulation
- **Change**: `model.cond_dim: 128`
- **Val Loss**: 1.0098 (IMPROVED -0.0007)
- **Train Loss**: 0.8053
- **Run dir**: outputs/shakespeare_diffusion_base/2026-04-20_11-15-33
- **Text sample**:
  ```
  the graces,  and come the worder, and for the of the name of your gordeness me to beseems to were were lord;
  And sof the for the come that your your gods
  You hear you are come you, you good for your good you, for et one to me you well you do well well, I s
  ```

---

## Experiment 4: Linear noise schedule (FAILED — reverted)
- **Hypothesis**: Linear schedule gives uniform loss weighting vs cosine weighting high-noise steps more
- **Val Loss**: 1.0697 (WORSE)
- **Verdict**: Cosine weighting of high-noise timesteps is beneficial for this task

---

## Experiment 5: Gradient clipping max_norm=1.0 (FAILED — reverted)
- **Hypothesis**: Score entropy loss produces large gradients; clipping would stabilize training
- **Val Loss**: 1.0103 (WORSE)
- **Verdict**: Training is already stable without clipping

---

## Experiment 6: lr 1e-3→2e-3 ✓ (COMMITTED)
- **Hypothesis**: With better conditioning, the model can tolerate a higher LR and converge faster in 2 epochs
- **Change**: `trainer.lr: 2.0e-3`
- **Val Loss**: 0.9931 (IMPROVED -0.0167, breaks below 1.0)
- **Train Loss**: 0.7822
- **Run dir**: outputs/shakespeare_diffusion_base/2026-04-20_11-38-36
- **Text sample**:
  ```
  olt arest not, for then, not that you not not, I see thou you do me, do you comest dost thou that you, I am not were you;
  If there not not you do well do not you that we have  you, for the take you have not me:
  If you more you do you do we not you do you h
  ```

---
