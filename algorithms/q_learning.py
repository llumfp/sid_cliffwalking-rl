import numpy as np
import random
import gymnasium as gym
from .utils import print_policy, draw_history, draw_rewards
import math

class Qlearning:
    """
    Algoritmo Q-learning
    """
    def __init__(self, env, gamma, learning_rate, epsilon, t_max,
                 epsilon_decay="none", lr_decay="none"):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.lr0 = learning_rate
        self.epsilon0 = epsilon
        self.t_max = t_max
        self.epsilon_decay = epsilon_decay
        self.lr_decay = lr_decay
    
    def lr_fn(self, t):
        """Devuelve learning rate en el episodio t."""
        if self.lr_decay == "none":
            return self.lr0
        elif self.lr_decay == "hyperbolic":
            return self.lr0 / t
        elif self.lr_decay == "exponential":
            return self.lr0 / math.exp(t)
        else:
            raise ValueError(f"lr_decay desconocido: {self.lr_decay}")

    def epsilon_fn(self, t):
        """Devuelve epsilon en el episodio t."""
        if self.epsilon_decay == "none":
            return self.epsilon0
        elif self.epsilon_decay == "hyperbolic":
            return self.epsilon0 / t
        elif self.epsilon_decay == "exponential":
            return self.epsilon0 / math.exp(t)
        else:
            raise ValueError(f"epsilon_decay desconocido: {self.epsilon_decay}")

    def select_action(self, state, t, training=True):
        if training and random.random() <= self.epsilon_fn(t):
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state,])

    def update_Q(self, state, action, reward, next_state, t):
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr_fn(t) * td_error

    def learn_from_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        for i in range(self.t_max):
            action = self.select_action(state, i+1)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state, i+1)
            if is_done:
                break
            state = new_state
        return total_reward

    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))
        return policy

    def train(self, num_episodes):
        rewards = []
        for i in range(num_episodes):
            reward = self.learn_from_episode()
            rewards.append(reward)
            if i % 500 == 499:
                print(f"Episode {i+1}: {sum(rewards[-500:-1])/len(rewards[-500:-1])} ")
        return rewards

    def test(self, num_episodes):
        is_done = False
        rewards = []
        for n_ep in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for i in range(self.t_max):
                action = self.select_action(state, 1, training=False)
                state, reward, is_done, truncated, _ = self.env.step(action)
                total_reward = total_reward + reward
                self.env.render()
                if is_done:
                    break
            rewards.append(total_reward)
            # if n_ep % 500 == 499:
            #     print(f"Episode {n_ep}: {sum(rewards[-500:-1])/len(rewards[-500:-1])} ")
            print(f"Episode {n_ep}: {rewards[-1]} ")
        return rewards