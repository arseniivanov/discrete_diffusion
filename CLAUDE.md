
# Autonomous Research Protocol: Discrete Diffusion

## Objective
Minimize the validation loss of the discrete diffusion model over a 5-epoch training run. Secondary metric: subjective quality of the output string in `summary.txt`.

## Architecture Constraints (HARD LIMITS)
Hardware: Single RTX 5080 Ti (16GB VRAM). 
For fairness of the search, you are strictly forbidden from increasing:
- `data.batch_size` (> 512)
- `model.n_layer` (> 3)
- `model.n_head` (> 2)
- `model.n_embd` (> 384)

## Workflow
1. Analyze the current codebase, focusing on `model.py`, `losses.py`, and `conf/agent_conf.yaml`.
2. Propose a single, testable hypothesis (e.g., modifying the noise schedule, adjusting dropout).
3. Implement the change. You are on your own branch, and may edit Python files or YAML configurations.
4. Execute the evaluation script: `python run_eval.py`
5. Read the final validation loss from the generated output logs.
6. If the validation loss improves (is lower than the previous best), commit the change via git.
7. If the validation loss degrades or the run crashes (OOM), revert the codebase to the previous commit using `git reset --hard`.
8. Repeat.

Current best solution to beat:

--- Final Generated Text ---
, that to hate that done,
To have the face of this heaven showes in the prestice of the woe of honour,
And this worth of the waster scope to be gone,
For that this conserved to a worserves.
A goe cops to me to come.

MENENIUS:
You swill not come with the woman's love,
And with the hearts  of Rome horses, I come to live in him, and make me not to sheep for gave you; for that the mar

Training finished. Final loss: 0.7488. Val loss: 0.8710. Duration: 19m 46s
