import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

class ValueIteration:
    def __init__(self, env: gym.Env, gamma:float=0.9, num_episodes:int=1000, t_max:int=100, reward_threshold:float=50):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.V[47] = 10
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.t_max = t_max
        self.reward_threshold = reward_threshold
        self.terminal_state = [47]
        
    def calc_action_value(self, state, action):
        action_value = sum([prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.unwrapped.P[state][action]]) 
        return action_value

    def select_action(self, state):
        best_action = 0
        best_value = -np.inf
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        max_diff = 0
        for state in range(self.env.observation_space.n):
            if state in self.terminal_state:
                continue

            state_values = []
            for action in range(self.env.action_space.n):  
                state_values.append(self.calc_action_value(state, action))
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    
    def policy(self):   
        policy = np.zeros(self.env.observation_space.n) 
        for s in range(self.env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy
    
    def check_improvements(self):
        reward_test = 0.0
        for i in range(self.num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            for i in range(self.t_max):
                action = self.select_action(state)
                new_state, new_reward, is_done, truncated, _ = self.env.step(action)
                total_reward += new_reward
                if is_done: 
                    break
                state = new_state
            reward_test += total_reward
        reward_avg = reward_test / self.num_episodes
        return reward_avg

    def train(self): 
        rewards = []
        max_diffs = []
        t = 0
        best_reward = -1000
        
        while len(max_diffs) < 1 or max_diff > 0.0001:
            _, max_diff = self.value_iteration()
            max_diffs.append(max_diff)
            t += 1
            reward_test = self.check_improvements()
            print(f"After value iteration,reward_test of {reward_test}, max_diff = " + str(max_diff))
            rewards.append(reward_test)
                
            if reward_test > best_reward:
                print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
                best_reward = reward_test
                
            print_policy(self.policy())
        
        return rewards, max_diffs
    
    
def print_policy(policy):
    visual_help = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    actual_policy = np.zeros((4, 12)).tolist()
    for i in range(len(policy)):
        row, col = revert_state_to_row_col(i)
        actual_policy[row][col] = visual_help[policy[i]]
    
    for row in actual_policy:
        print(" | ".join(row))


def revert_state_to_row_col(state):
    row = state // 12
    col = state % 12
    return row,col   

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()


class CustomFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        state, reward, is_done, truncated, info = self.env.step(action)
        
        if state in [47]:
            is_done = True
        
        return state, reward, is_done, is_done, info


env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
env = CustomFrozenLakeWrapper(env)

GAMMA = 0.9999
NUM_EPISODES = 200
T_MAX = 100

agent = ValueIteration(env, gamma=GAMMA, num_episodes=NUM_EPISODES, t_max=T_MAX, reward_threshold=-20)
rewards, max_diffs = agent.train()

policy = agent.policy()
print_policy(policy)

draw_rewards(rewards)