import subprocess
import gymnasium as gym
import pandas as pd
import os, csv
import sys

# Set UTF-8 encoding for output
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Get the path to the virtual environment Python
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
script = "experimentation.py"

def experiment_sample():
    # Define different parameter combinations to test
    parametres = [
        {
            "alg": alg,
            "episodes": str(episodes),
            "gamma": "0.95",
            "epsilon": str(epsilon),
            "epsilon_decay": decay
        }
        for alg in ["q_learning"] # "q_learning", "value_iteration", "model_based", "reinforce"
        for episodes in [500, 1500, 3000]
        for epsilon in [0.1, 0.3]
        for decay in ["none"] # , "linear", "exponential"
    ]

    # Execute the script sequentially with each parameter combination
    for ide, params in enumerate(parametres):
        # Build the execution command
        cmd = [venv_python, script, 
              "--exp_id", str(ide),
              "--alg", params["alg"],
              "--episodes", params["episodes"],
              "--gamma", params["gamma"],
              "--epsilon", params["epsilon"],
              "--epsilon_decay", params["epsilon_decay"]]

        print("Executing:", " ".join(cmd))
        # Execute the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        print(f"Output: {output}")
        if result.stderr:
            print(f"Error: {result.stderr}")

def experiment1():
    # Define different parameter combinations to test
    parametres = [
        {
            "alg": alg,
            "episodes": str(episodes),
            "gamma": str(gamma),
            "epsilon": str(epsilon),
            "epsilon_decay": epsilon_decay,
            "reward_signal": reward_signal,
            "lr": str(lr),
            "lr_decay": lr_decay
        }
        for alg in ["q_learning"]  # "value_iteration", "model_based", "reinforce"
        for episodes in [500, 1500] # , 1500
        for epsilon in [0.1] # , 0.3
        for epsilon_decay in ["none"]  # "linear", "exponential"
        for reward_signal in ["default"]  # "custom", "final_100"
        for gamma in [0.95]
        for lr in [0.1]  # , 0.5
        for lr_decay in ["none"]  # "linear", "exponential" si quieres probar
    ]

    # Create the output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Execute the script sequentially with each parameter combination
    for ide, params in enumerate(parametres):
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
               "--lr_decay", params["lr_decay"]]
        print("Executing:", " ".join(cmd))
        # Execute the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        print(f"Output: {output}")
        if result.stderr:
            print(f"Error: {result.stderr}")

if __name__ == "__main__":
    experiment1()