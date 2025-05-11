# Cliff Walking with RL

This project explores and compares different reinforcement learning algorithms on the slippery variant of the CliffWalking-v0 environment. The study analyzes performance based on various hyperparameters and training metrics. The goal is to evaluate the strengths and weaknesses of each approach in this kind of environment.

## Project Structure

The project is organized as follows:

### Main Components
- `execution.py`: Main execution script for running the experiments. It has a loop for different parameter settings, execution script `experimentation.py` many times.
- `experimentation.py`: Contains the experimentation setup and configuration for different parameters with argparse.
- `set_params.py`: Script to define and manage parameter sets for experiments.

### Algorithms (`/algorithms`)
- `value_iteration.py`: Implementation of the Value Iteration algorithm
- `q_learning.py`: Implementation of Q-Learning algorithm
- `model_based.py`: Implementation of Model-Based RL approaches
- `actor_critic.py`: Implementation of Actor-Critic algorithm
- `REINFORCE.py`: Implementation of REINFORCE algorithm
- `utils.py`: Utility functions specific to the algorithms.

### Results (`/results`)
- Storage for experimental results, metrics, and data

### Output (`/output`)
- Storage for output files in csv format.
- Contains a directory for each algorithm's results

### Other
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `requirements.txt`: Lists the Python packages required for the project.
- `README.md`: This file.

## Execution