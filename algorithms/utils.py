import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import pandas as pd
import numpy as np

def revert_state_to_row_col(state:int) -> tuple[int, int]:
    """
    Convert a state number to its corresponding row and column in a 4x12 grid.

    Args:
        state (int): The state number to convert.

    Returns:
        tuple[int, int]: The row and column corresponding to the state number.
    """
    
    row = state // 12
    col = state % 12
    return row,col  

def print_policy(policy: list[int]) -> None:
    """
    Print the policy in a human-readable format.

    Args:
        policy (list[int]): The policy to print, represented as a list of integers.
    """

    visual_help = {0: '^', 1: '>', 2: 'v', 3: '<'}
    actual_policy = np.zeros((4, 12)).tolist()
    for i in range(len(policy)):
        row, col = revert_state_to_row_col(i)
        actual_policy[row][col] = visual_help[policy[i]]
    
    for row in actual_policy:
        print(" | ".join(row))
 
def draw_rewards(rewards:list[int], show:bool=True, path:str = "") -> None:
    """
    Draw a line plot of rewards over episodes.

    Args:
        rewards (list[int]): List of rewards to plot.
        show (bool, optional): Show the plot or not. Defaults to True.
        path (str, optional): The path to save the figure. Defaults to "".
    """
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(path)

def draw_history(history:list[int], title:str) -> None:
    """
    Draw a line plot of the moving average of a list of numbers.

    Args:
        history (list[int]): List of numbers to plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(history), label=f"{title} promedio (ventana=50)")
    plt.xlabel("Episodio")
    plt.ylabel(title)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
    
def moving_average(x:list[int], w=50) -> np.ndarray:
    """
    Calculate the moving average of a list of numbers.

    Args:
        x (list[int]): List of numbers to calculate the moving average.
        w (int, optional): Moving average. Defaults to 50.

    Returns:
        np.ndarray: The moving average of the input list.
    """
    
    return np.convolve(x, np.ones(w)/w, mode='valid')




class RewardWrapperFinal100(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        reward = reward if state != 47 else 100
        
        
        if state in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
            print(f'DONE: {done}')
            done = True
            print(f'DONE: {done}')
        
        return state, reward, done, truncated, info

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        env.unwrapped.P[47] = {0: [(0.3333333333333333, np.int64(36), -100, False), (0.3333333333333333, np.int64(35), -1, False), (0.3333333333333333, np.int64(47), 0, True)], 1: [(0.3333333333333333, np.int64(35), -1, False), (0.3333333333333333, np.int64(47), 0, True), (0.3333333333333333, np.int64(47), 0, True)], 2: [(0.3333333333333333, np.int64(47), 0, True), (0.3333333333333333, np.int64(47), 0, True), (0.3333333333333333, np.int64(36), -100, False)], 3: [(0.3333333333333333, np.int64(47), 0, True), (0.3333333333333333, np.int64(36), -100, False), (0.3333333333333333, np.int64(35), -1, False)]}
        super().__init__(env)
    
    def step(self, action):        
        state, reward, is_done, truncated, info = self.env.step(action)
        
        if state == 47:
            reward = 0
            is_done = True
        
        return state, reward, is_done, truncated, info
