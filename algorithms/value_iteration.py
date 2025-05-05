import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 

class ValueIteration:
    def __init__(self, env: gym.Env, gamma:float=0.9, num_episodes:int=1000, t_max:int=100, reward_threshold:float=-60):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.t_max = t_max
        self.reward_threshold = reward_threshold
        
        
        
    def calc_action_value(self, state, action):
        action_value = sum([prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.P[state][action]]) 
        
        action_value = 0
        for prob, next_state, reward, _ in self.env.P[state][action]:
            reward = reward if next_state != 47 else 0
            action_value += prob * (reward + self.gamma * self.V[next_state])        
        
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
                state_values = []
                for action in range(self.env.action_space.n):  
                    state_values.append(self.calc_action_value(state, action))
                new_V = max(state_values)
                diff = abs(new_V - self.V[state])
                if diff > max_diff:
                    max_diff = diff
                self.V[state] = new_V
            
        return self.V, max_diff
    

        #print(self.V.reshape((4, 12)))  # Ver la forma como una cuadrícula
    
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
        ready_to_compare = False
        
        init_time = time.time()
        
        while not ready_to_compare or max_diffs[-1] > 0.05:
            _, max_diff = self.value_iteration()
            #print(f"Iteration {t}, max_diff = " + str(max_diff))
            max_diffs.append(max_diff)
            t += 1
            reward_test = self.check_improvements()
            print(f"Iteration {t}, reward_test of {reward_test}, max_diff = " + str(max_diff))
            rewards.append(reward_test)
                
            if reward_test > best_reward:
                #print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
                best_reward = reward_test
                
            #self.print_policy(self.policy())
            
            if len(max_diffs) > 1:
                ready_to_compare = True
        
        end_time = time.time() - init_time
        #self.draw_rewards(rewards)
        print(f"SOLUCION: value_iteration, {self.num_episodes}, {self.gamma}, 0, 0, {t}, {best_reward}, {end_time:.2f}")
        
        return rewards, max_diffs
    

from utils import CustomWrapper, print_policy, draw_rewards, draw_history

env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
#env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False)
#env = CustomWrapper(env)

env = env.unwrapped

agent = ValueIteration(env, gamma=0.995, num_episodes=50, t_max=50, reward_threshold=-20)
rewards, max_diffs = agent.train()

draw_history(rewards, "Reward")
draw_history(max_diffs, "Max Diff")
print_policy(agent.policy())