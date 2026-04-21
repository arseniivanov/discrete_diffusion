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

## Exp 13: Muon excl. embeddings (wte, wpe) — ✓ COMMITTED (val=0.9709, -0.0002)
- Exclude wte/wpe from Muon to fix embedding distortion; val improves marginally
- Text still shows "your" repetition (was present before Muon extension too)
- Run: outputs/shakespeare_diffusion_base/2026-04-20_13-50-57

---

## Exp 14: lr=3e-3 with context=384 — ✓ COMMITTED (val=0.9671, -0.0038)
- With larger batch token count (196K vs 131K), gradients are more stable → lr=3e-3 now works
- Text quality continues to be repetitive (inference issue, separate from val_loss)
- Run: outputs/shakespeare_diffusion_base/2026-04-20_14-02-07

---

## Exp 15: lr=4e-3 — ✓ COMMITTED (val=0.9576, -0.0095)
- Improvement accelerating; Muon tolerates higher LR well
- Run: outputs/shakespeare_diffusion_base/2026-04-20_14-12-59

## Exp 16: SwiGLU MLP (hidden=8/3*n_embd=1024) — FAILED (val=0.9724)
- Replaced GELU MLP with SwiGLU; hidden dim dropped 1536→1024 hurt intermediate capacity
- Same param count but less compute; 2 epochs not enough for gate to learn

---

## Exp 17: RoPE positional embeddings — ✓ COMMITTED (val=0.9152, **-0.0424**)
- Replaced learned wpe with RoPE (precomputed sinusoidal freqs applied to Q,K in attention)
- Removes wpe from Muon entirely; relative position encoding improves attention quality
- Epoch 1: 0.9410 → Epoch 2: 0.9152
- Run: outputs/shakespeare_diffusion_base/2026-04-20_14-45-54
- Text:
  ```
   have your yourself,
  That respice of your you and your love you not your love you do your lords, you dry not you have you to you;
  For the come of you we come to your lords,
  You are you take you, and the your good your lords
  You are that that you have of your lords
  Are your consent with your corssel
  Your wortune your louds
  To your fhese your please you hf yout
  I should make your con
  ```

---

## Exp 18: Register tokens (n=4) — ✓ COMMITTED (val=0.9130, -0.0022)
- Prepend 4 learnable register tokens to the sequence, strip before output; gives attention heads a "global context highway"
- Text still shows some "thou" repetition but no worse than previous best
- Run: outputs/shakespeare_diffusion_base/2026-04-20_21-40-38
- Text:
  ```
  hou, hast thou hast thou not the love that thou have with thee that would have seen thee to see the name
  on this would be thou art thought of thee thou art thought thy ears thou wast thou not thyus aut what
  thou camest thou titter,
  As thou letters that babe none in thee.

  HOLBEENTS:
  Then,
  I have not with the wars of thy persont:
  I will we do be so much and that comes here will  tak
  ```

## Exp 19: Register tokens n=8 — ✓ COMMITTED (val=0.9098, -0.0032)
- Doubling registers from 4→8 continues to improve; more global context capacity helps
- Run: outputs/shakespeare_diffusion_base/2026-04-20_21-53-13
- Text:
  ```
  s the cour your hanes; Or your come to the no more words of the cour geness to me
  t at your come Into make the lords, and sake your fed  and they have colded what he hathebefore here
  come to have them to see that they have your wonders:
  And some here gone, I will not holp the world.
  JULIET: You have none that you have been, yor  have your hates of you have you not one in this  man
  ```

---

## Exp 20: Local depthwise conv on token embeddings — ✓ COMMITTED (val=0.9050, -0.0048)
- Kernel-3 depthwise conv (residual) over tok_emb before blocks; captures char-level local structure (word patterns)
- Text quality notably improved: character names, less repetition
- Run: outputs/shakespeare_diffusion_base/2026-04-20_22-35-16
- Text:
  ```
   and tell them in the worst of them.
  SICINIUS:
  Not stand teem that they give thee to them to them.
  Second Servant:
  Worthly the fairers, are not the gasses. Let them makes; and when they have been toes to hear,
  that they not in the loss, they sreed mothers, that they more words than that you have but that
  the east of your father, You woull have  you not when you put upon this  han
  ```

---

## Exp 21: Per-block depthwise conv (k=3 after each block) — ✓ COMMITTED (val=0.8967, -0.0083)
- After each DDiTBlock, apply k=3 depthwise conv residually to token positions; local context at every layer
- Big jump: text quality dramatically improved, coherent sentences and character names
- Run: outputs/shakespeare_diffusion_base/2026-04-20_23-36-04
- Text:
  ```
   leave me with your lords.
  PAULINA:
  I am your lord, take your lives this sum ly love.
  MAULINA:
  No, God, my lord, hold not yet  his face.
  MARIANA:
  O, come, that I would have tone of him,
  And I have tiln the time of the world
  Hath done nou to mine  for this power,
  ```

## Exp 22: Simplified per-block conv (full sequence incl. registers) — ✓ COMMITTED (val=0.8919, -0.0048)
- Remove register-stripping inside block conv loop; apply conv to full [registers+tokens] sequence
- Cleaner code AND better val loss: simpler residual path improves gradient flow
- Run: outputs/shakespeare_diffusion_base/2026-04-21_10-03-00
- Text:
  ```
  s the lives to bear them, than they were they come to their come to them their hates of the eases: they are come to the partness of they come to have done their
  the causes of the war, that is the pisterer, that
  they are a hence: they have not heard, nor they hay
  been to the sure of their measure fives of the
  desires of the marken.'

  Second Servenger:
  Servant an enter his master. rn
  ```

---

## Exp 23: Stacked input conv (two k=3 depthwise, GELU between) — ✓ COMMITTED (val=0.8886, -0.0033)
- Add second k=3 depthwise conv on top of first: tok_emb += local_conv2(gelu(tok_emb)); effective RF=5 with nonlinearity
- More expressive character-level feature extraction at input
- Run: outputs/shakespeare_diffusion_base/2026-04-21_12-46-10
- Text:
  ```
   leave to  ome thee gone.
  My lorge, that have come home to set the worst.
  QUEEN ELIZABETH:
  What shall we have thee? and tell thee, that have done to thee.
  QQUEEN MARGARET:
  Ay, that is thou donest thou not have no,
  Ant that thou dost yoke of mine.
  ```

---

## Exp 24: Stacked per-block conv (two k=3 depthwise, GELU between) — ✓ COMMITTED (val=0.8832, -0.0054)
- Mirror the successful input conv stacking (Exp23) at every transformer block: x += block_conv2[i](gelu(x)); effective RF=5 with nonlinearity per block
- Same pattern that worked for input conv applied to per-block convs
- Run: outputs/shakespeare_diffusion_base/2026-04-21_*
- Text:
  ```
   you, that you have pressed to be so,
  bod,
  To thee that house, that will you have drown'd
  To tear the last, that you have done to yome
  Found what I will come to your good words,
  Aod rest touc loss, that you make me notly son.

  PAULINA:
  My heart, my lord lord,
  I have none to your good corsession.
  Now, good nime, son, son, what it is hence,
  To like the noble, sor, that thou hast not
  ```

## Exp 25: ALiBi locality bias on top of RoPE — ✓ COMMITTED (val=0.8818, -0.0014)
- Added 2 learnable per-head slopes (init [0.1, 0.05]) to SelfAttention; bias = -slope * |i-j| added to attention logits
- Complements RoPE: RoPE provides directional relative encoding, ALiBi adds explicit distance penalty encouraging nearby-token attention
- Hypothesis confirmed: for char-level text, explicit locality bias helps attention focus on word-level context
- Run: outputs/shakespeare_diffusion_base/2026-04-21_*
- Text:
  ```
   they mo love, that they have done
  Toestand, they are not that that hath been done,
  And then they have see them to the lines
  Of the gates, say, they see ere they see, the streess these 
  ins in their lives in the other gates  names, that that have they seek nut their wives, that they have they see  the mistrees of man?
  Is no more, there is the head of the name?
  ```

## Exp 26: ALiBi einsum (bfloat16-consistent slopes) — ✓ COMMITTED (val=0.8809, -0.0009)
- Changed ALiBi bias computation from broadcast multiply to `torch.einsum` with explicit bfloat16 cast for slopes
- Original: float32 slopes × bfloat16 dist (mixed precision). New: slopes cast to bfloat16 first, einsum stays in bfloat16
- Consistent dtype improves gradient signal through ALiBi slopes; also fixes non-contiguous tensor error for non-8-divisible T
- Run: outputs/shakespeare_diffusion_base/2026-04-21_21-48-03
- Text:
  ```
   let them when they have been remived them
  to hear them that they have been their gates,
  We make her lives to them, when they with them, they are content o' the kindred statds,
  And theyrane one to make the noters seeming oyes,
  And waked them eldest wakes on the bones, that they have done and their tartnes
  ```

## Exp 27: Equal loss weighting (remove sigma multiplier) — FAILED (val=0.9970)
- Changed loss from `(sigma * loss).mean()` to `loss.mean()` (equal weight to all noise levels)
- Catastrophic regression: sigma weighting is load-bearing for the cosine masking derivation

## Exp 28: RoPE theta=500 (lower base for char-level context) — FAILED (val=0.8837)
- Hypothesis: lower base makes more frequencies active within 384-char context
- Regression: longer-range positional info still valuable even at context=384

## Exp 29: QK-Norm (LayerNorm on Q and K per-head, before RoPE) — ✓ COMMITTED (val=0.8770, -0.0039)
- Add `nn.LayerNorm(head_dim, bias=False)` to Q and K after projection, before RoPE
- Stabilizes attention logit scale (used in Gemma-2, Mistral Nemo); avoids logit explosion with depth
- Run: outputs/shakespeare_diffusion_base/2026-04-21_23-23-34
- Text:
  ```
  The flattering in the seats of your deeds,
  The babe deserved with the holes of the deviles,
  And that they were to make them in the life,
  They have titter'd in the state of their hends then...
  ```

---

**Current best: val_loss=0.8770**
Config: n_layer=3, n_head=2, n_embd=384, cond_dim=128, bias=True, timestep_embedding=True, context=384, lr=4e-3, cosine masking noise, Muon on all non-embedding 2D matrices, **RoPE** (no wpe), **8 register tokens**, **stacked input conv (2×k=3 depthwise with GELU)**, **stacked per-block depthwise conv (2×k=3 with GELU)**, **ALiBi locality bias (einsum, bfloat16)**, **QK-Norm (per-head LayerNorm on Q and K)**
