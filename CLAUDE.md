
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
- Adding parameters to the framework
- Changing the loss function 

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
ee and pleberal                                                                                                                                                                                                                                                                                                                                     
That he dyestundred brain at her grace bear.                                                                                                                                                                                                                                                                                                        
Shall not let him , cannot go be, you dare no her                                                                                                                                                                                                                                                                                                   
As Evok yef. Hear on, not a mean, that so will make a sister                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                    
DUKE  VINCENTIO:                                                                                                                                                                                                                                                                                                                                    
Rebellion of Yourso use thee this anc Away.                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                    
GLOUCESTER:                                                                                                                                                                                                                                                                                                                                         
Why, tho                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                    
Training finished. Final loss: 1.7652. Duration: 10m 50s                                                                                                                                                                                                                                                                                            

