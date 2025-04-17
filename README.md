# Cliff Walking with RL

This project explores and compares different reinforcement learning algorithms on the slippery variant of the CliffWalking-v2 environment. The study analyzes performance based on various hyperparameters and training metrics. The goal is to evaluate the strengths and weaknesses of each approach in this kind of environment.

## Project Structure

The project is organized as follows:

### Main Components
- `execution.py`: Main execution script for running the experiments. It has a loop for different parameters settings, execution script `experimentation.py` many times.
- `experimentation.py`: Contains the experimentation setup and configuration for differents parameters with argparse.
- `analyze_results.py`: Script for analyzing and visualizing the results

### Algorithms (`/algorithms`)
- `value_iteration.py`: Implementation of the Value Iteration algorithm
- `q_learning.py`: Implementation of Q-Learning algorithm
- `model_based.py`: Implementation of Model-Based RL approaches

### Results (`/results`)
- Storage for experimental results, metrics, and data

