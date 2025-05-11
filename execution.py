from utils import CreateVisualizations
from set_params import PARAMETRES, ALGORITHMS, METRICS, CRITERIA_BEST_RESPONSE

import subprocess
import gymnasium as gym
import pandas as pd
import os, csv
import sys

# Get the path to the virtual environment Python
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "py")
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "py")
venv_python = "py"

script = "experimentation.py"

def experiment():            
    # Execute the script sequentially with each parameter combination
    for ide, params in enumerate(PARAMETRES):
        # Build the execution command
        cmd = [venv_python, script, 
              "--exp_id", str(ide),
              "--alg", params["alg"],
              "--episodes", params["episodes"],
              "--gamma", params["gamma"],
              "--epsilon", params["epsilon"],
              "--epsilon_decay", params["epsilon_decay"],
              "--rew", params["reward_signal"],
              "--lr", params["lr"],
              "--lr_decay", params["lr_decay"]
            ]

        print("Executing:", " ".join(cmd))
        # Execute the command
        subprocess.run(cmd)
        
    # Create visualizations for the results
    CreateVisualizations(ALGORITHMS, METRICS, CRITERIA_BEST_RESPONSE)


if __name__ == "__main__":
    experiment()