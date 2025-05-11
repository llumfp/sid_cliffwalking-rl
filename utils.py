import pandas as pd, os
import matplotlib.pyplot as plt




def add_BestResponse(data:pd.DataFrame, criteria:list, alg:bool = False) -> pd.DataFrame:
    """
    This function takes a DataFrame as input and returns the best response based on two criteria:
    1. The maximum value of the "reward_test" column.
    2. The minimum value of the "optimality" column.

    Args:
        data (pd.Dataframe): DataFrame containing the data to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing the best response based on the specified criteria.
    """
    
    BRESPONSES = []
    
    for i in criteria:
        if i not in data.columns:
            raise ValueError(f"Column '{i}' not found in the DataFrame.")
        
        if i == "reward_test":
            if alg:
                data = data[data["reward_signal"] == "default"]
            BRESPONSES.append(data.loc[data["reward_test"].idxmax()].to_frame().T)
        else:
            BRESPONSES.append(data.loc[data["optimality"].idxmin()].to_frame().T)
    
    return BRESPONSES


def CreateVisualizations(algs: list, metrics:list, criteria: list) -> None:
    """
    Create visualizations for the results of different algorithms.
    This function generates plots comparing the performance of different algorithms based on their reward signals and other parameters.

    Args:
        algs (list): List of algorithm names to visualize results for.
    """
    
    # Initialize list to store best response data for each algorithm
    Bresponses = []
    
    # Loop through each algorithm and read the corresponding CSV file
    for alg in algs:
        data = pd.read_csv(f"output/results_{alg}.csv")
        os.makedirs(f"output/results_{alg}", exist_ok=True)
        
        # Filter data based on reward signals
        data_default = data[data["reward_signal"] == "default"]
        data_custom = data[data["reward_signal"] == "custom"]

        # Get the best response of all the data
        br = add_BestResponse(data, criteria, alg= alg == "q_learning")
        Bresponses.append(br)

        # Chenge the X-axis depending on the algorithm (each algorithm has different parameters)       
        ejeX = ["gamma"] if alg in ["value_iteration", "model_based"] else ["gamma", "lr", "lr_decay"] if alg != "actor_critic" else ["gamma", "lr"]
        
        # Loop through each parameter in ejeX to create plots
        for X in ejeX:
            # set the unique values of the X axis
            gammas = data[X].unique()
            # Loop through each metric to create plots
            for metric in metrics:
                # Create a figure and axis for the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                # loop through each unique number of episodes to create diferent lines in the same plot
                for n_episodes in data["episodes"].unique():
                    # do the same to plot in the same plot the default and custom reward signals
                    for i, (reward_signal, data_subset) in enumerate([("default", data_default), ("custom", data_custom)]):
                        values = []
                        # Filter data for the current reward signal and number of episodes to get the values for the y axis
                        for gamma in gammas:
                            data_gamma = data_subset[data_subset[X] == gamma]
                            data_gamma = data_gamma[data_gamma["episodes"] == n_episodes]
                            mean_optimality = data_gamma[metric].mean() # mean of the metric (We use the mean to select the value, there is only one value for each combination of parameters) (there are some cases that there are 3 or 6, but its bcs existance of more hyperparameters)
                            values.append(mean_optimality)
                        
                        # plot line and put the correct label
                        ax.plot([i for i in range(len(gammas))], values, marker='o', label=f"{reward_signal.capitalize()} Reward, Episodes: {n_episodes}")

                # Set plot labels and title and save the plot
                ax.set_xlabel(X)
                ax.set_ylabel(f"{metric}")
                ax.set_title(f"{metric} Comparison by {X} and {metric}")
                ax.legend(title=f"{metric}")
                ax.set_xticks([i for i in range(len(gammas))])
                ax.set_xticklabels(gammas, rotation=45)
                plt.tight_layout()
                plt.savefig(f"output/results_{alg}/{X}_{metric}_{n_episodes}.png")
                plt.close(fig)



    # Concatenate the best response data for all algorithms
    all_concatenated = Bresponses[0]
    for br_prima in Bresponses[1:]:
        for i in range(len(br_prima)):
            all_concatenated[i] = pd.concat([all_concatenated[i], br_prima[i]], ignore_index=True)
            
            
    # Create the output directory for all algorithms
    os.makedirs(f"output/results_all_algorithms", exist_ok=True)

    # Iterate through the metrics to create bar plots for the best responses
    for metric, concatenated in zip(criteria, all_concatenated):

        # Create bar plots for reward_test for each algorithm
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract reward_test values for each algorithm
        reward_test_values = concatenated[metric].values
        ax.bar(concatenated["exp_id"].values, reward_test_values, color='skyblue')

        # Set plot labels and title
        ax.set_xlabel("Algorithms")
        ax.set_ylabel(metric)
        ax.set_title(f"Best {metric} Comparison Across Algorithms")
        plt.tight_layout()

        # Save the bar plot
        plt.savefig(f"output/results_all_algorithms/best_{metric}_comparison.png")
        plt.close(fig)
        
        # Save the concatenated DataFrame to a CSV file
        concatenated.to_csv(f"output/results_all_algorithms/best_{metric}_comparison.csv", index=False)


