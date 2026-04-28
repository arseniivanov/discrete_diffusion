
# Autonomous Research Protocol: Discrete Diffusion

## Objective
Minimize the validation loss of the discrete diffusion model over a 2-epoch training run. Secondary metric: subjective quality of the output string in `summary.txt`.

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
 to take your honour to you, my lord.

MENENIUS:
I will not come to have them, for you have you need,
To give me with your honour to your consent,
And you are your honour and your mother,
Your honour save you yet thae more,
You have done to the matters of the cause;
But the time shall be crook'd upon my gage,
And make my hearted  the house of the nose,
That he is no more that mame 

Training finished. Final loss: 0.6842. Val loss: 0.8188. Duration: 20m 54s
