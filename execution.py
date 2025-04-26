import subprocess
import gymnasium as gym
import pandas as pd
import os, csv

script = "experimentation.py"

def experiment_sample():
    # Define different parameter combinations to test
    parametres = [
        {
            "alg": alg,
            "eps": str(eps),
            "disc": "0.99",
            "exp": str(exp),
            "exp_decay": decay
        }
        for alg in ["value_iteration"] # "q_learning", "model_based", 
        for eps in [500, 1500, 3000]
        for exp in [0.1, 0.3]
        for decay in ["none", "linear", "exponential"]
    ]
    
    # Create the output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a CSV file to store the results
    results_file = os.path.join(output_dir, "results.csv")
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["alg", "eps", "disc", "exp", "exp_decay", "steps_train", "reward", "time"]) 
    
    # Execute the script sequentially with each parameter combination
    for params in parametres:
        # Build the execution command
        cmd = ["py", script, 
              "--alg", params["alg"],
              "--eps", params["eps"],
              "--disc", params["disc"],
              "--exp", params["exp"],
              "--exp_decay", params["exp_decay"]]

        print("Executing:", " ".join(cmd))
        # Execute the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.split("SOLUCION: ")[1]
        print(f"Output: {output}")
        
                
        # Check if the command was successful
        with open("output/results.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(output.strip().split(","))  # Append the output to the CSV file
        
        

if __name__ == "__main__":
    experiment_sample()