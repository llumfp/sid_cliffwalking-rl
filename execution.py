import subprocess

script = "experimentation.py"

def experiment_sample():
    # Define different parameter combinations to test
    parametres = [
        {
            "alg": alg,
            "eps": "1000",
            "disc": "0.99",
            "exp": str(exp),
            "exp_decay": decay
        }
        for alg in ["q_learning", "model_based", "value_iteration"]
        for exp in [0.1, 0.3]
        for decay in ["none", "linear", "exponential"]
    ]
    # Execute the script sequentially with each parameter combination
    for params in parametres:
        # Build the execution command
        cmd = ["python", script, 
              "--alg", params["alg"],
              "--eps", params["eps"],
              "--disc", params["disc"],
              "--exp", params["exp"],
              "--exp_decay", params["exp_decay"]]

        print("Executing:", " ".join(cmd))
        # Execute the command
        subprocess.run(cmd)

if __name__ == "__main__":
    experiment_sample()