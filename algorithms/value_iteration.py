import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 

class ValueIteration:
    def __init__(self, env: gym.Env, gamma:float=0.9, num_episodes:int=1000, t_max:int=100):
        self.env = env.unwrapped
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.t_max = t_max
        
        
        
        
    def calc_action_value(self, state, action):
        action_value = sum([prob * ((reward if state != 47 else 0) + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.P[state][action]])   
        
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
        
    def policy(self):   
        policy = np.zeros(self.env.observation_space.n) 
        for s in range(self.env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy
    
    def check_improvements(self):
        reward_test = 0.0
        for i in range(20):
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
        reward_avg = reward_test / 20
        return reward_avg

    def train(self, num_episodes:int=1000): 
        rewards = []
        max_diffs = []
        t = 0
        best_reward = -1000
        ready_to_compare = False
                
        while not ready_to_compare or max_diffs[-1] > 0.05:
            _, max_diff = self.value_iteration()
            
            max_diffs.append(max_diff)
            t += 1
            reward_test = self.check_improvements()
            print(f"Iteration {t}, reward_test of {reward_test}, max_diff = " + str(max_diff))
            rewards.append(reward_test)

            if reward_test > best_reward:
                best_reward = reward_test
            
            if len(max_diffs) > 1:
                ready_to_compare = True
        
        
        return rewards
    
    def test(self, num_episodes:int=1000):
        rewards = []
        for i in range(num_episodes):
            total_reward = 0.0
            state, _ = self.env.reset()
            for i in range(self.t_max):
                action = self.select_action(state)
                new_state, new_reward, is_done, truncated, _ = self.env.step(action)
                total_reward += new_reward
                if is_done: 
                    break
                state = new_state
            rewards.append(total_reward)
        return rewards