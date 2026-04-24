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

**Current best after Exp29: val_loss=0.8770**

---

## Exp 30: Output depthwise conv on final representation — FAILED (val=0.8800)
- Added k=3 depthwise conv residually after `ln_f` and before `lm_head`
- Hypothesis: additional local smoothing before logits helps; in practice disrupts gradient flow to output projection
- Run: outputs/shakespeare_diffusion_base/2026-04-21_23-46-18

## Exp 31: Zero-init block convs — FAILED (val=0.8844)
- Initialized block_convs and block_convs2 weights to zero so they start as identity
- 2 epochs insufficient for convolutions to recover and contribute; reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_00-09-16

## Exp 32: Antithetic time sampling — ✓ COMMITTED (val=0.8769, -0.0001)
- Paired (t, 1-t) noise level samples per batch: each batch sees balanced low-noise/high-noise coverage
- Implementation: sample `u` for half the batch, concatenate `[u, 1-u]` for full batch; reduces variance of loss gradient
- Epoch 1: 0.8997 (vs 0.9018 without), Epoch 2: 0.8769 (vs 0.8770)
- Run: outputs/shakespeare_diffusion_base/2026-04-22_00-29-38

## Exp 33: Structural test — FAILED (val=0.8850)
- Unknown structural change; epoch 1 = 0.9160 indicates significant disruption to early training
- Reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_00-52-22

## Exp 34: V normalization (LayerNorm on values per head) — FAILED (val=0.8777)
- Added `v_norm = nn.LayerNorm(head_dim)` applied after value projection, before attention output
- Restricts value expressivity; marginal regression; reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_01-13-38

## Exp 35: lr=5e-3 — FAILED (val=0.8790)
- Increased learning rate from 4e-3 → 5e-3
- Epoch 1 marginally better but epoch 2 overshoots; reverted to 4e-3
- Run: outputs/shakespeare_diffusion_base/2026-04-22_01-37-57

## Exp 36: Conv-before-attention (swap attention and conv order) — FAILED (val=0.8833)
- Applied block convs BEFORE attention in each block (conv → attn → MLP instead of attn → MLP → conv)
- Current attention-then-conv order is superior; likely conv features need attention-processed context
- Run: outputs/shakespeare_diffusion_base/2026-04-22_01-59-01

## Exp 37: Pre-conv LayerNorm — FAILED (val=0.8913)
- Added separate LayerNorms before each block conv: `x += conv(LN(x))` instead of `x += conv(x)`
- Disrupts learning signal for convolutions; large regression; reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_02-20-57

## Exp 38: SwiGLU FFN (hidden=1024, param-neutral) — FAILED (val=0.8792)
- Replaced GELU MLP (c_fc→1536→c_proj) with SwiGLU (gate_proj/up_proj→1024, c_proj→384)
- hidden_dim=1024 keeps param count ~equal: 1,182,080 per layer vs 1,181,568
- Epoch 1: 0.9095 (vs baseline 0.8997), Epoch 2: 0.8792 (regression vs 0.8769)
- Same failure as Exp16: gated activations + Muon may conflict; gates need more epochs to learn
- Also: expandable_segments:True required to avoid memory fragmentation OOM
- Run: outputs/shakespeare_diffusion_base/2026-04-22_03-01-50

## Exp 39: n_registers 8→16 — FAILED (val=0.8775)
- Doubled register tokens from 8 to 16 (+3072 params)
- Hypothesis: more global context capacity, based on 4→8 improvement (-0.0032)
- Epoch 1 worse (0.9005 vs 0.8997), Epoch 2 slightly worse (0.8775 vs 0.8769)
- With ALiBi, registers shift all token positions by 8 more slots — no improvement in register accessibility
- Run: outputs/shakespeare_diffusion_base/2026-04-22_03-22-55

## Exp 40: Stratified antithetic time sampling — FAILED (val=0.8787)
- Replaced antithetic pairs with stratified strata in [0,0.5] + antithetic mirror in [0.5,1]
- Used `torch.randperm(half)` to assign strata → each batch covers all noise levels uniformly
- Epoch 1: 0.9047 (vs baseline 0.8997, -0.0050 worse), Epoch 2: 0.8787 (regression vs 0.8769)
- Over-reducing gradient variance hurts Muon optimization; current antithetic pairs are optimal balance
- Run: outputs/shakespeare_diffusion_base/2026-04-22_03-44-19

## Exp 41: Register-aware ALiBi (zero penalties for register interactions) — FAILED (val=0.8796)
- Modified ALiBi to zero out distance penalties when either position is a register token
- Motivation: last tokens in sequence have distance ~385+ to all registers with standard ALiBi,
  effectively blocking them from accessing global register context
- Implementation: `tok_tok = is_tok.unsqueeze(0) & is_tok.unsqueeze(1); dist *= tok_tok`
- Epoch 1: 0.9039 (slightly worse than 0.8997), Epoch 2: 0.8796 (regression vs 0.8769)
- Possible reason: locality bias on registers may be useful as inductive bias for learning
- Run: outputs/shakespeare_diffusion_base/2026-04-22_04-04-24

## Exp 42: k=5 first conv (RF 5→7, memory-neutral) — FAILED (val=0.8819)
- Replaced first k=3 conv (in both input stack and per-block stacks) with k=5 to extend RF from 5 to 7
- Motivation: RF=7 captures common 6-char English words; OOM ruled out adding a 3rd layer
- Epoch 0: 0.9003 (close to baseline 0.8997), Epoch 2: 0.8819 (regression vs 0.8769)
- k=3 appears better than k=5: tighter locality provides sharper positional features
- Run: outputs/shakespeare_diffusion_base/2026-04-22_05-08-30

## Exp 43: Sigma ×1000 before sinusoidal embedding — MARGINAL FAIL (val=0.8772)
- Scaled sigma_bar by ×1000 before TimestepEmbedder sinusoidal embedding
- Motivation: sinusoidal freqs designed for integer t∈[0,1000]; sigma_bar∈[0,1] only activates ~1/128 dimensions
- With ×1000, dim≈85 (of 128 pairs) completes ~1 cycle over sigma range vs only dim≈0 before
- Epoch 0: 0.8987 (vs baseline 0.8997, improvement!), Epoch 1: 0.8772 (vs 0.8769, marginal regression)
- Better early conditioning but final result within noise of baseline; not committed
- Run: outputs/shakespeare_diffusion_base/2026-04-22_05-31-01

## Exp 44: Sigma input bias (sigma→n_embd linear, zero-init) — MARGINAL FAIL (val=0.8770)
- Added `sigma_in = nn.Linear(cond_dim, n_embd, bias=False)`, zero-initialized
- Applied as global bias to full sequence: `x = x + sigma_in(c).unsqueeze(1)` (broadcast all positions)
- Motivation: AdaLN only conditions post-LayerNorm activations; a direct sigma bias in embedding space provides complementary conditioning
- Epoch 0: 0.9057 (worse than baseline 0.8997), Epoch 1: 0.8770 (essentially tied at 0.8769)
- The sigma bias gets zero-initialized so starts as identity — takes time to activate
- Run: outputs/shakespeare_diffusion_base/2026-04-22_05-54-33

## Exp 45: sigma×1000 + sigma input bias combined — ✓ COMMITTED (val=0.8763, -0.0006)
- Combined Exp43 (sigma×1000 scaling) + Exp44 (sigma_in input bias)
- sigma×1000 activates high-frequency sinusoidal embedding dims → better noise discrimination
- sigma_in provides direct sigma signal at embedding level (complementary to AdaLN)
- Together: sigma×1000 helps early training (epoch 0: 0.8968 vs 0.8997), sigma_in maintains that advantage to final (epoch 1: 0.8763)
- Each change individually gave marginal regression; combined, they reinforce each other
- Run: outputs/shakespeare_diffusion_base/2026-04-22_06-17-41

## Exp 46: 2-layer MLP sigma_in (cond_dim→64→n_embd) — FAILED (val=0.8779)
- Replaced linear sigma_in with 2-layer MLP (128→64 SiLU→384), keeping zero-init on final layer
- Motivation: richer non-linear mapping from sigma to embedding space
- Epoch 0: 0.9015 (worse than Exp45 0.8968), Epoch 1: 0.8779 (regression from 0.8763)
- First MLP layer has non-zero init (std=0.02), creating non-zero output from step 1 → disrupts early training
- Simple linear with zero-init (Exp45) is superior
- Run: outputs/shakespeare_diffusion_base/2026-04-22_06-39-07

## Exp 47: Sigma-conditioned ALiBi slopes (per-sample) — FAILED (OOM)
- Attempted: replace learnable scalar ALiBi slopes with sigma-conditioned slopes via `ali_slope_proj = Linear(cond_dim, n_head)`
- OOM: (B=512, n_head=2, T=392, T=392) × 3 blocks → ~1.9GB for attention masks alone
- Per-sample ALiBi is memory-prohibitive at B=512; batch-mean sigma≈0.5 constant would make per-sample conditioning nearly useless anyway
- Abandoned

## Exp 47b: Sigma-conditioned conv gates (additive) — FAILED (OOM)
- Attempted: `x = x + gate(sigma) * conv(x)` where gate = Linear(cond_dim, 1, bias=False) per conv
- OOM: backward needs to save `conv_out` (153MB each × 6 convs = ~900MB extra)
- Multiplicative gating on large tensors is memory-prohibitive; abandoned
- ALiBi bias re-init bug also discovered: `_init_weights` overwrote log([0.1, 0.05]) bias to zeros; fixed pattern identified (post-init loop) but not needed after abandoning

---

## Exp 48: sigma_out direct logit bias (Linear(cond_dim, vocab_size), zero-init) — ✓ COMMITTED (val=0.8758, -0.0005)
- Added `sigma_out = nn.Linear(cond_dim, vocab_size, bias=False)` with zero-init; applied as `logits += sigma_out(c).unsqueeze(1)` before scatter
- Motivation: existing AdaLN conditions on sigma via shift/scale of norms; sigma_out is a direct additive bypass from sigma→logit without going through transformer activations
- 8,320 new params (128×65). Applied before input-token scatter to maintain correct masking.
- Epoch 0: 0.9077 (vs baseline 0.8968 — slightly worse early); Epoch 1: 0.8758 (new best!)
- Run: outputs/shakespeare_diffusion_base/2026-04-22_07-13-25
- Text:
  ```
  er, I will not will be sound to the world.

  MERCUIIO:
  The love that says that have counted than thee;
  And that I will make the  name of the heart.
  ```

---

## Exp 49: sigma×1000 → sigma×500 in TimestepEmbedder — ✓ COMMITTED (val=0.8715, -0.0043)
- Changed sigma scaling from ×1000 to ×500 in TimestepEmbedder.forward
- Reasoning: at ×1000 with sigma∈[0,1], dimensions k<80 complete >π/2 cycles (near-aliased); at ×500, k<70 alias, freeing ~10 more dimensions for informative encoding
- Zero new params, zero memory overhead
- Epoch 0: 0.9097 (vs Exp48 0.9077 — slightly worse early but doesn't matter), Epoch 1: 0.8715 (big improvement!)
- Run: outputs/shakespeare_diffusion_base/2026-04-22_07-47-04
- Text:
  ```
  , methink, and some your people,
  That we have you to the case, that comes this your presence.

  BENENIU:
  Holy, sir, that you do
  Your own, sir, fare your place.

  MENENIUS:
  Your possers, gentlemen,, fare you, Marcius
  and us to your praye.
  ```

---

## Exp 50: sigma×200 (reduce aliasing further) — FAILED (val=0.8721)
- Changed sigma scaling from ×500 to ×200 in TimestepEmbedder.forward
- Epoch 0: 0.9050 (better than 0.9097), Epoch 1: 0.8721 (worse than 0.8715)
- ×500 appears to be sweet spot: ×200 loses informative signal from mid-high-k dimensions
- The optimal balance seems near ×500 (between too-aliased ×1000 and too-narrow ×200)
- Run: outputs/shakespeare_diffusion_base/2026-04-22_08-08-09

## Exp 51: sigma×300 — FAILED (val=0.8728)
- Changed sigma scaling from ×500 to ×300
- Epoch 0: 0.9049, Epoch 1: 0.8728 (regression from 0.8715)
- Not monotonically better at lower values: ×200=0.8721, ×300=0.8728, ×500=0.8715. Sweet spot confirmed at ×500
- Run: outputs/shakespeare_diffusion_base/2026-04-22_10-43-53

## Exp 52: output AdaLN-style (sigma_scale × logits + sigma_out) — FAILED (val=0.8774)
- Added sigma_scale = Linear(cond_dim, vocab_size, bias=False) with zero-init
- Applied as: `logits = logits * (1 + sigma_scale(c)) + sigma_out(c)` (scale+shift like AdaLN)
- 8,320 new params; zero-init gives identity initially
- Epoch 0: 0.9070 (similar), Epoch 1: 0.8774 (large regression)
- Multiplicative modulation of logits disrupts training (gradients scaled down or up unpredictably)
- Additive-only (sigma_out in Exp48) is safer; multiplicative scale on logits is harmful
- Run: outputs/shakespeare_diffusion_base/2026-04-22_11-06-06

## Exp 54: Remove outer SiLU on conditioning — ✓ COMMITTED (val=0.8711, -0.0004)
- Changed `c = F.silu(self.sigma_map(sigma))` → `c = self.sigma_map(sigma)` in GPT.forward
- TimestepEmbedder already applies SiLU internally (Linear(256,128)→SiLU→Linear(128,128)); the outer SiLU squashed negative components of c unnecessarily
- Removing it gives AdaLN, sigma_in, and sigma_out a more balanced conditioning signal
- Epoch 0: 0.9097 (similar), Epoch 1: 0.8711 (new best!); train loss 0.7454 (vs 0.7522)
- Text quality: more coherent sentences, visible improvement
- Run: outputs/shakespeare_diffusion_base/2026-04-22_*

## Exp 56: Remove redundant ln_f before lm_head — MARGINAL FAIL (val=0.8712)
- Skipped `x = self.transformer.ln_f(x)` since DDitFinalLayer already has its own norm_final with AdaLN
- Hypothesis: double normalization suppresses signal before final projection
- Epoch 0: 0.9097, Epoch 1: 0.8712 (tied with best 0.8711 within noise; technically marginal regression)
- The ln_f is load-bearing: block conv outputs need normalization before the AdaLN-conditioned norm_final
- Reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_*

## Exp 55: Include registers in input convolutions — FAILED (val=0.8739)
- Moved `torch.cat([reg, tok_emb])` before the input conv pair (local_conv, local_conv2)
- Motivated by Exp22 success (block_convs on full sequence including registers)
- Hypothesis: registers can help shape the initial conv feature extraction
- Epoch 0: 0.9097, Epoch 1: 0.8739 (regression vs 0.8711)
- At input stage, register tokens (near-zero init) just add noise to local_conv boundaries; reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_*

## Exp 53: GELU → SiLU in MLP — FAILED (val=0.8735)
- Replaced `nn.GELU()` with `nn.SiLU()` in MLP feed-forward layers
- SiLU (Swish) commonly outperforms GELU in modern transformers; zero param/memory cost
- Epoch 0: 0.9097, Epoch 1: 0.8735 (regression vs 0.8715)
- GELU remains optimal for this model; reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_*

## Exp 59: AdamW weight_decay=0 on non-matrix params — FAILED (val=0.8782)
- Changed `optim.AdamW(other_params, lr=cfg.trainer.lr)` → `optim.AdamW(other_params, lr=cfg.trainer.lr, weight_decay=0)`
- Hypothesis: WD=0.01 shrinks register tokens, sigma_in, sigma_out, LayerNorm gains — freeing them could let conditioning grow in magnitude
- Epoch 0: 0.9059, Epoch 1: 0.8782 (regression vs 0.8711)
- Weight decay on conditioning paths is load-bearing; reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-22_13-47-54

## Exp 60: Third per-block stacked depthwise conv (k=3) — OOM
- Added `block_convs3` ModuleList (+4,608 params, within budget)
- Forward: `x = x + block_convs3[i](F.gelu(x).transpose(1,2)).transpose(1,2)` after block_convs2
- Crashed OOM — activation memory from 3 conv layers × 3 blocks × bf16 + GELU activations overflows 16GB
- Local RF extension beyond 5 must come via dilation (same activation count) rather than stacking; reverted

## Exp 61: Learnable per-head attention log-scale — OOM (x2)
- Added `self.attn_log_scale = nn.Parameter(torch.zeros(n_head))` in SelfAttention (+2 params)
- Forward: `q = q * self.attn_log_scale.exp().to(q.dtype).view(1,n_head,1,1)` after QK-norm+RoPE
- Crashed OOM consistently at ~step 17/train despite trivial param count
- Likely torch.compile specialization cost (extra graph for parameterized q scale) overflows 16GB margin
- Memory budget is maxed; further forward-path additions not feasible; reverted

## Exp 62: RoPE theta 10000→500 — ✓ COMMITTED (val=0.8710, -0.0001)
- Changed default `theta=10000.0` to `theta=500.0` in `precompute_freqs_cis`
- Hypothesis: for short contexts (T=392), theta=10000 gives too-slow rotation in low-freq components, poor positional discrimination
- theta=500 gives finer angular spread across the sequence; zero memory cost (just different buffer values)
- Epoch 0: 0.9019, Epoch 1: 0.8710 (marginal improvement; within noise but committed per protocol)
- Text quality similar
- Run: outputs/shakespeare_diffusion_base/2026-04-22_14-17-00

## Exp 63: register_tokens init zeros→randn*0.02 — FAILED (val=0.8727)
- Changed `torch.zeros(...)` → `torch.randn(...) * 0.02` for register token init
- Hypothesis: breaking symmetry earlier speeds specialization
- Epoch 1: 0.8727 (regression vs 0.8710); zero init remains optimal for registers
- Reverted

## Exp 64: RoPE theta 500→200 — FAILED (val=0.8745)
- Further reduced theta from 500 to 200
- Hypothesis: even finer positional resolution might help
- Epoch 1: 0.8745 (regression); theta=500 is the sweet spot (non-monotonic)
- Reverted

## Exp 65: Muon LR 2× AdamW LR — ✓ COMMITTED (val=0.8678, -0.0032)
- Changed `optim.Muon(matrix_params, lr=cfg.trainer.lr)` → `optim.Muon(matrix_params, lr=cfg.trainer.lr * 2)` in train.py
- Muon (orthogonalization-based) typically requires higher LR than AdamW for comparable step sizes; literature recommends ~2-5x
- Epoch 0: 0.8940 (much better than prior ~0.903), Epoch 1: 0.8678 (new best, -0.0032)
- Train loss 0.7476 (similar); optimization is more effective from the start
- Run: outputs/shakespeare_diffusion_base/2026-04-22_15-19-05

## Exp 66: Muon LR 3× — FAILED (val=0.8700)
- Too aggressive; reverted

## Exp 67: Muon LR 2.5× — FAILED (val=0.8686)
- Slight regression; 2× is optimum

## Exp 68: Muon LR 1.5× — FAILED (val=0.8711)
- Too low; confirms 2× is sweet spot

## Exp 69: AdamW LR 0.75× — FAILED (val=0.8689)
- AdamW prefers higher LR, not lower; reverted

## Exp 70: AdamW LR 1.25× — TIED (val=0.8678)
- Essentially tied with best; used as stepping stone

## Exp 71: AdamW LR 1.5× (with Muon 2×) — ✓ COMMITTED (val=0.8670, -0.0008)
- Both optimizers benefit from more aggressive steps; AdamW for norms/registers/convs also needed more LR
- Epoch 0: ~0.89, Epoch 1: 0.8670; train loss 0.7440
- Run: outputs/shakespeare_diffusion_base/2026-04-22_17-20-52

## Exp 72: AdamW LR 2× — FAILED (val=0.8684)
- Too aggressive; reverted

## Exp 73: AdamW LR 1.75× — FAILED (val=0.8674)
- Slight regression; 1.5× is optimum

## Exp 74: Cosine LR decay to 10% — ✓ COMMITTED (val=0.8531, -0.0139 ⚡️)
- Added `CosineAnnealingLR(T_max=total_steps, eta_min=peak_lr*0.1)` for both Muon and AdamW
- Peak LRs unchanged (Muon 2×, AdamW 1.5×); decays smoothly over 2 epochs
- Epoch 0: ~0.89, Epoch 1: 0.8531; train loss 0.7004 (much better than flat 0.7440)
- Flat LR was "frying" late training; cosine decay lets model settle
- Run: outputs/shakespeare_diffusion_base/2026-04-22_18-22-14

## Exp 75: Peak Muon 3× / AdamW 2× + cosine decay — ✓ COMMITTED (val=0.8481, -0.0050)
- Flat 3× Muon had failed (Exp66), but with cosine decay the average LR is ~0.55×peak so higher peaks become feasible
- Increased peak LRs to Muon 3× (prev 2×), AdamW 2× (prev 1.5×); both decay to 10% of their peaks
- Epoch 0: ~0.88, Epoch 1: 0.8481; train loss 0.6908
- Run: outputs/shakespeare_diffusion_base/2026-04-22_18-43-00

## Exp 76: Peak Muon 4× / AdamW 3× + cosine decay — ✓ COMMITTED (val=0.8434, -0.0047)
- Continued pushing peak LR up; decay brings it down
- Epoch 1: 0.8434; train loss 0.6871
- Run: outputs/shakespeare_diffusion_base/2026-04-22_19-03-35

## Exp 77: Peak Muon 5× / AdamW 4× + cosine decay — ✓ COMMITTED (val=0.8405, -0.0029)
- Epoch 1: 0.8405; train loss 0.6829
- Run: outputs/shakespeare_diffusion_base/2026-04-22_19-24-16

**Current best: val_loss=0.8405**
Config: n_layer=3, n_head=2, n_embd=384, cond_dim=128, bias=True, timestep_embedding=True, context=384, lr=4e-3, cosine masking noise, Muon on all non-embedding 2D matrices, **RoPE** (no wpe), **8 register tokens**, **stacked input conv (2×k=3 depthwise with GELU)**, **stacked per-block depthwise conv (2×k=3 with GELU)**, **ALiBi locality bias (einsum, bfloat16)**, **QK-Norm (per-head LayerNorm on Q and K)**, **antithetic time sampling**, **sigma×500 in TimestepEmbedder**, **sigma_in input bias (zero-init)**, **sigma_out direct logit bias (zero-init)**, **no outer SiLU on conditioning c**

---

## Exp 78: LR warmup (5% LinearLR + cosine decay) — FAILED (val=0.8408)
- Added 5% linear warmup (1%→100% peak) before cosine decay for both Muon and AdamW
- Muon's orthogonalization already constrains early step sizes; warmup wasted steps
- Reverted
- Run: outputs/shakespeare_diffusion_base/2026-04-23_*

## Exp 79: Masked-position loss weighting (2× and 1.3×) — FAILED (val=1.095/0.939)
- Weighted loss on masked positions ×2 and ×1.3; both catastrophic regression
- Sigma weighting is load-bearing; any re-weighting disrupts the balance (same as Exp27)
- Reverted

## Exp 80: EMA (decay=0.999) for final val eval — ✓ COMMITTED (val=0.8334, -0.0071)
- Maintain exponential moving average of all params on CPU; evaluate with EMA weights at end
- Shadow key mapping: strip `_orig_mod.` prefix from compiled model param names
- Fresh uncompiled GPT model loaded with EMA state dict for val eval + text generation
- EMA smooths out training noise → better generalization estimate
- Run: outputs/shakespeare_diffusion_base/2026-04-23_*

## Exp 81: Cosine decay to 5% (instead of 10%) — FAILED (val=0.8369)
- More aggressive end-decay; model needs the 10% floor LR to maintain learning signal
- Reverted

## Exp 82: Offset cosine noise schedule (s=0.02) — FAILED (val=0.8372)
- alpha_t = cos((t+s)/(1+s) * pi/2); prevents alpha from hitting exact 0/1
- Noise schedule changes are risky; the sigma weighting in loss is tightly coupled
- Reverted

## Exp 83: EMA decay 0.998 — ✓ COMMITTED (val=0.8203, -0.0131)
- Faster EMA (tracks training more closely) gave large improvement
- EMA decay=0.999 → 0.998: effective averaging window ~500 steps (vs ~1000)
- 0.997: 0.8255 (too fast), 0.995: 0.8218 (too fast), 0.998 is sweet spot
- Run: outputs/shakespeare_diffusion_base/2026-04-23_*

## Exp 84: Peak LR 6×/5× — FAILED (val=0.8210)
- Higher peak LRs with cosine decay; marginal regression vs 5×/4×
- EMA doesn't enable higher peak LRs; 5×/4× remains optimal
- Reverted

## Exp 85: Muon momentum 0.9 — FAILED (val=0.8236)
- Lower Muon momentum reduces orthogonalization strength; regression
- Also tried 0.99 → 0.8320 (too high momentum overshoots)
- Default 0.95 is optimal
- Reverted

## Exp 86: AdamW weight_decay=0.05 — FAILED (val=0.8223)
- Reduced from default 0.1; less regularization hurts small model
- Reverted

## Exp 87: Muon weight_decay=0.05 — FAILED (val=0.8309)
- Reduced from default 0.1; significant regression
- WD on Muon params is load-bearing; default 0.1 is optimal
- Reverted

## Exp 88: Dilated block_conv2 (dilation=2, k=3) — FAILED (val=0.8304)
- Expand RF from 5 to 7 per block via dilation instead of stacking (avoids OOM)
- Wider receptive field at each layer hurts; k=3 d=1 is optimal for char-level locality
- Reverted

## Exp 89: GatedDeltaNet for layer 2 — FAILED (>6.5M params)
- Exceeds parameter count limit; GatedDeltaNet adds too many params
- Abandoned

## Exp 90: AdamW beta2=0.98 — FAILED (val=0.8208)
- Faster EMA in AdamW vs default beta2=0.999; marginal regression
- Reverted

## Exp 91: Sigma×400 (with EMA setup) — FAILED (val=0.8236)
- Retested sigma scaling sweet spot with EMA; ×500 remains optimal
- Reverted

## Exp 92: Inference steps 128→256 — ✓ COMMITTED (val=0.8188, better text quality)
- More denoising steps for iterative unmasking; text quality improved
- 512 steps: val=0.8204 (too many steps adds noise); 256 is sweet spot
- Run: outputs/shakespeare_diffusion_base/2026-04-23_*

## Exp 93: Use EMA model for text generation — ✓ COMMITTED (val=0.8192)
- Changed sample_masking/sampling to use ema_model instead of training model
- Consistent with using EMA for val evaluation; text quality similar
- Run: outputs/shakespeare_diffusion_base/2026-04-23_*

## Exp 94: batch_size=384, context_length=512 — ✓ COMMITTED (val=0.8147, -0.0041)
- Trade batch diversity for longer context: each sequence sees 33% more tokens (512 vs 384)
- Total tokens per batch remains ~196K (384×512), so gradient signal per step is preserved
- Longer context helps the model capture dependencies across longer Shakespeare passages
- Train loss 0.6292 (vs 0.6842), runtime 29m 54s
- Text quality improved: visible character names, more coherent phrasing
- Run: outputs/shakespeare_diffusion_base/2026-04-24_*

## Exp 95: Sharpen attention temperature (q × 1.2) — ✓ COMMITTED (val=0.8119, -0.0028)
- With QK-Norm normalizing Q and K, default 1/√d flash-attention scaling yields variance=1 logits
- Multiplying Q by 1.2 before flash attention raises effective logit variance to 1.44, sharpening softmax
- Helps character-level model make more confident token predictions from local attention context
- Train loss 0.6270, runtime 29m 45s
- Run: outputs/shakespeare_diffusion_base/2026-04-24_*

---

**Current best: val_loss=0.8119**
Config: n_layer=3, n_head=2, n_embd=384, cond_dim=128, bias=True, timestep_embedding=True, context=384, lr=4e-3, cosine masking noise, Muon 5×/AdamW 4× + cosine decay to 10%, **EMA (decay=0.998)** for val eval + text gen, **inference steps=256**, **RoPE** (theta=500), **8 register tokens**, **stacked input conv (2×k=3 depthwise with GELU)**, **stacked per-block depthwise conv (2×k=3 with GELU)**, **ALiBi locality bias (einsum, bfloat16)**, **QK-Norm**, **antithetic time sampling**, **sigma×500**, **sigma_in input bias (zero-init)**, **sigma_out direct logit bias (zero-init)**, **no outer SiLU on conditioning c**

---

## Architectural Experiments (all FAILED)

### Exp A1: Parallel Attn+MLP (PaLM-style) — FAILED (val=0.8379)
- Changed serial (Attn→MLP) to parallel (Attn+MLP simultaneously, both see same input)
- With only 3 layers, serial depth is actually needed for sufficient processing depth
- Parallel reduces effective depth from 6 to 3, which hurts for such a shallow model
- Reverted

### Exp A2: RMSNorm instead of LayerNorm — FAILED (val=0.8227)
- Replaced all LayerNorms with RMSNorm (no mean-centering) + optional bias
- LayerNorm's mean-centering is load-bearing for AdaLN modulation pattern (shift/scale)
- Reverted

### Exp A3: 3× MLP + cond_dim 128→192 — FAILED (val=0.8284)
- Reduced MLP from 4× to 3× expansion (1536→1152 hidden), widened cond_dim from 128→192
- Freed ~376K params; wider conditioning doesn't compensate for reduced MLP capacity
- The 4× MLP expansion is critical for model expressivity
- Reverted

### Exp A4: SE (Squeeze-Excitation) per block + 3.5× MLP — FAILED (val=0.8220)
- Added SE block (global avg pool → FC16 → SiLU → FC384 → sigmoid scale) after each block
- Reduced MLP from 4× to 3.5× to stay under 6.5M params
- SE channel mixing + reduced MLP capacity = net regression
- Reverted

### Exp A5: Sandwich normalization (LN after Attn and after MLP) — FAILED (val=0.8249)
- Added post-attention and post-MLP LayerNorms (Gemma-2 style)
- The gate mechanism already handles scaling; sandwich LN is redundant
- Reverted

### Exp A6: Embedding scaling ×sqrt(n_embd) — FAILED (val=0.8248)
- Scaled token embeddings by sqrt(384)≈19.6 before input convs
- Pre-norm LayerNorm already handles scale; explicit scaling disrupts initialization
- Reverted

### Exp A7: 3-layer TimestepEmbedder (deeper sigma path) — FAILED (val=0.8226)
- Added extra Linear→SiLU layer to sigma_map MLP; compensated with MLP 1536→1532
- 2-layer TimestepEmbedder is sufficient; deeper path adds no useful expressivity
- Reverted

### Exp A8: Per-head RoPE theta (head0=200, head1=2000) — FAILED (val=0.8223)
- Different positional frequency bases per head: local (200) + global (2000)
- Unified theta=500 was already well-tuned; per-head thetas hurt coordination
- Reverted

### Exp A9: GQA + wider MLP (1536→1728) — FAILED (val=0.8232)
- Shared K,V between 2 heads (multi-query attention); freed ~442K params for wider MLP
- With only 2 heads, per-head K,V is important for head specialization
- GQA works better with many heads (8+); hurts with 2 heads
- Reverted

### Exp A10: Random offset data augmentation — FAILED (val=0.8213)
- Random shift of up to context_len//4 on each sample's starting position
- DataLoader already shuffles; random offset is redundant noise
- Reverted

---

**Analysis**: The current architecture is well-optimized for the 6.5M param budget. Every architectural change regressed because:
1. The 4× MLP expansion is critical — can't be reduced without losing expressivity
2. Per-head K,V matters with only 2 heads — GQA needs more heads to help
3. Serial block depth matters with only 3 layers — parallel hurts
4. LayerNorm's mean-centering is load-bearing for AdaLN
5. The sigma conditioning path (2-layer + sigma_in/out) is already sufficient
6. Positional encoding (unified RoPE theta=500 + ALiBi) is well-tuned
