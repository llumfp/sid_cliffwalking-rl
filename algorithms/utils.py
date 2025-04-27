import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import pandas as pd
import numpy as np

def revert_state_to_row_col(state):
    row = state // 12
    col = state % 12
    return row,col  

def print_policy(policy):
    # Using ASCII characters instead of Unicode arrows
    visual_help = {0: '^', 1: '>', 2: 'v', 3: '<'}
    actual_policy = np.zeros((4, 12)).tolist()
    for i in range(len(policy)):
        row, col = revert_state_to_row_col(i)
        actual_policy[row][col] = visual_help[policy[i]]
    
    for row in actual_policy:
        print(" | ".join(row))
 
def draw_rewards(rewards, show=True, path = ""):
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

def draw_history(history, title, split=200):
    average_rewards = [np.mean(history[i:i + split]) for i in range(0, len(history), split)]
    std_rewards = [np.std(history[i:i + split]) for i in range(0, len(history), split)]
    lower_bound = [avg - std for avg, std in zip(average_rewards, std_rewards)]
    upper_bound = [avg + std for avg, std in zip(average_rewards, std_rewards)]
    plt.figure(figsize=(10, 6))
    plt.fill_between(range(1, len(average_rewards) + 1), lower_bound, upper_bound, color='b', alpha=0.2, label='±1 Std Dev')
    plt.plot(range(1, len(average_rewards) + 1), average_rewards, marker='o', linestyle='-')
    plt.title(f'Average {title} Every {split} Episodes')
    plt.xlabel(f'Episode Group ({split} episodes each)')
    plt.ylabel('Average ' + title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
        super().__init__(env)
    
    def step(self, action):
        state, reward, is_done, truncated, info = self.env.step(action)
        
        if state in [47]:
            is_done = True
        
        return state, reward, is_done, is_done, info
