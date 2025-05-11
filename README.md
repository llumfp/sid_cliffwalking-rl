# Cliff Walking with RL

This project explores and compares different reinforcement learning algorithms on the slippery variant of the CliffWalking-v0 environment. The study analyzes performance based on various hyperparameters and training metrics. The goal is to evaluate the strengths and weaknesses of each approach in this kind of environment.

## Project Structure

The project is organized as follows:

### Main Components
- `REPORT.pdf`: Technical report comparing five reinforcement learning algorithms on the CliffWalking environment with experimental analysis.
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

### Output (`/output`)
- Storage for output files in csv format.
- Contains a directory for each algorithm's results

### Other
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `requirements.txt`: Lists the Python packages required for the project.
- `README.md`: This file.

## Execution

The project can be executed in two main ways:

1.  **Running a Full Experimentation Suite:**
    *   Execute the `execution.py` script directly:
        ```bash
        python execution.py
        ```
    *   This script reads parameter combinations defined in `set_params.py` (specifically the `PARAMETRES` list).
    *   It then iterates through each parameter set, calling `experimentation.py` to run an individual experiment for each.
    *   Results from all experiments are aggregated, and visualizations are created (handled by `CreateVisualizations` in `utils.py`).

2.  **Running a Single Experiment with Specific Parameters:**
    *   Execute the `experimentation.py` script directly, providing command-line arguments to specify the algorithm and its hyperparameters.
    *   The available arguments can be seen by running:
        ```bash
        python experimentation.py --help
        ```
    *   Example: To run the Q-learning algorithm for 500 episodes with a gamma of 0.9 and a learning rate of 0.01:
        ```bash
        python experimentation.py --alg q_learning --episodes 500 --gamma 0.9 --lr 0.01
        ```
    *   Other available parameters include `--exp_id`, `--rew` (reward signal), `--epsilon`, `--epsilon_decay`, and `--lr_decay`.
    *   The script will initialize the specified RL agent, train it, test it, and save the results (including average rewards and execution time) to a CSV file in the `output/` directory (e.g., `results_q_learning.csv`). It also saves policy visualizations and reward plots in the `results/<algorithm_name>/` directory, even though it is not included in the repository.

**To configure an experimentation with many different parameters:**

*   Modify the `PARAMETRES` list within the `set_params.py` file.
*   Each entry in this list is a dictionary representing a unique set of parameters for an experiment.
*   Define the desired `alg` (algorithm), `episodes`, `gamma`, `epsilon`, `epsilon_decay`, `reward_signal`, `lr` (learning rate), and `lr_decay` for each experiment you want to include in the suite.
*   Once `set_params.py` is configured, running `python execution.py` will execute all defined experiments sequentially.
