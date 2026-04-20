# Autonomous Research Protocol: Discrete Diffusion

## Objective
Minimize the validation loss of the discrete diffusion model over a 5-epoch training run. Secondary metric: subjective quality of the output string in `summary.txt`.

## Architecture Constraints (HARD LIMITS)
Hardware: Single RTX 3080 Ti (12GB VRAM). 
You are strictly forbidden from increasing:
- `data.batch_size` (> 512)
- `model.n_layer` (> 4)
- `model.n_head` (> 4)
- `model.n_embd` (> 384)

## Workflow
1. Analyze the current codebase, focusing on `model.py`, `losses.py`, and `conf/agent_conf.yaml`.
2. Propose a single, testable hypothesis (e.g., modifying the noise schedule, adjusting dropout).
3. Implement the change. You may edit Python files or YAML configurations.
4. Execute the evaluation script: `python run_eval.py`
5. Read the final validation loss from the generated output logs.
6. If the validation loss improves (is lower than the previous best), commit the change via git.
7. If the validation loss degrades or the run crashes (OOM), revert the codebase to the previous commit using `git reset --hard`.
8. Repeat.


Here is the baseline:

--- Final Generated Text ---
e        ee ee e   e   ee eee ee  e ee   eeee eeeeeeeeeeeee ee    eeee   e  e
ee e e e e ee e e e e eeeee e       eee  ee    ee  e e e         eeeee e     e
eee   e  eee   e   eeeeee      e  e   ee e  e e e te    tee  ee   e a hteeht eee
eiee   t ee
hte

Training finished. Final loss: 1.8884. Duration: 29m 14s


