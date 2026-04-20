# Discrete Diffusion Optimization Worklog

## Baseline (default config)
- **Val Loss**: 1.0268 | **Train Loss**: 0.8674
- **Config**: n_layer=3, n_head=2, n_embd=384, cond_dim=64, bias=False, timestep_embedding=False, context=256, lr=1e-3, masking cosine noise, Muon+AdamW
- **Text sample**: `you art to more hearts, and not you good  give, and you have you, / That be you, for you have for your you:`

---

## Exp 1: Cosine LR warmup — FAILED (val=1.1594)
- 5% warmup + cosine decay wasted early steps; flat lr is better for 2 epochs

## Exp 2: timestep_embedding=True — ✓ COMMITTED (val=1.0105, -0.0163)
- 256-dim sinusoidal features vs raw scalar → better noise-level conditioning
- Run: outputs/shakespeare_diffusion_base/2026-04-20_11-06-16

## Exp 3: cond_dim 64→128 — ✓ COMMITTED (val=1.0098, -0.0007)
- Wider adaLN bottleneck (128→2304 per block vs 64→2304)
- Run: outputs/shakespeare_diffusion_base/2026-04-20_11-15-33

## Exp 4: Linear noise schedule — FAILED (val=1.0697)
- Uniform rate_noise weighting is worse than cosine (high-noise emphasis)

## Exp 5: Gradient clipping max_norm=1.0 — FAILED (val=1.0103)
- Training already stable; clipping marginally hurts

## Exp 6: lr 1e-3→2e-3 — ✓ COMMITTED (val=0.9931, -0.0167, **breaks <1.0**)
- With better conditioning, higher LR converges faster in 2 epochs
- Run: outputs/shakespeare_diffusion_base/2026-04-20_11-38-36
- Text: `olt arest not, for then, not that you not not, I see thou you do me...`

## Exp 7: lr=3e-3 — FAILED (val=1.0004)
- Overshot; lr=2e-3 is the sweet spot

## Exp 8: bias=True — ✓ COMMITTED (val=0.9872, -0.0059)
- Bias terms in Linear/LayerNorm improve expressivity
- Run: outputs/shakespeare_diffusion_base/2026-04-20_11-54-27
- Text:
  ```
  Wver were it were, and of the line of thy word
  Ah, that in these mine of thy have thy lord,
  To bear thee thee to the word  and for me preence:
  I will be not your horser me more to your lord,
  ```

## Exp 9: context_length=512 — FAILED (OOM: 13.4/16 GB)
- Too memory-intensive at batch_size=512

## Exp 10: context_length=384 — ✓ COMMITTED (val=0.9789, -0.0083)
- 50% longer context fits in memory (~9 GB); model learns longer-range patterns
- Run: outputs/shakespeare_diffusion_base/2026-04-20_13-12-33
- Text:
  ```
  If  he would not yot and your good your you:
  Nou, come, and with the have your hands,
  Your your good wor, your lord, your your good.
  ```

---

## Exp 11: dropout=0.05 — FAILED (val=0.9865)
- Dropout reduces effective learning capacity in 2-epoch training

## Exp 12: Muon on all 2D matrices — ✓ COMMITTED (val=0.9711, -0.0078)
- Extend Muon from transformer.h only → all 2D weight matrices (sigma_map, wpe, wte, lm_head)
- Val loss improved but text quality degraded (repetitive "thou hast" loop)
- Hypothesis: wte/wpe with Muon distorts embedding space
- Run: outputs/shakespeare_diffusion_base/2026-04-20_13-40-09

---

**Current best: val_loss=0.9711**
Config: n_layer=3, n_head=2, n_embd=384, cond_dim=128, bias=True, timestep_embedding=True, context=384, lr=2e-3, cosine masking noise, Muon on all 2D matrices
